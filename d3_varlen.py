"""A variable-length text path for the vLLM d3 decision model on ROCm, written for RDNA 4 (gfx12).

- Prompts are joined into one row without padding. Their boundaries are kept by the Gated DeltaNet's cu_seqlens, the
  conv positions and per-prompt attention, so one forward answers the whole batch.
- int4 weights and activations are converted to FP8 right before each product. GPUs without a working FP8 product
  multiply in BF16.
- The prompt text before the state is computed once and reused. It cannot read the state that follows it, so this is
  an approximation of the model.

``exact=True`` turns off FP8 and reuse and only joins the prompts. Import this module with the model folder on sys.path.

Adapted from the d3 package (Apache-2.0): ``_unpack_fp8`` from ``d3q/kernels.py`` writes FP8 and scales by a power of
two, ``_gdn_prep`` from ``d3_kernels.py`` handles several prompts in one row and sums the conv in FP32.
"""
from __future__ import annotations

import contextlib
import json
import math
import types

import d3_fast
import d3_runtime as rt
import torch
import triton
import triton.language as tl
from d3q import kernels as d3k
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from triton.language.extra import libdevice

FP8 = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8).max)
INT4_RANGE = 15  # largest int4 magnitude after the zero point is removed
TOKEN_BUDGET = 8192  # most tokens computed in one forward
MIN_PREFIX = 16  # shorter common prefixes are not reused
PREFIX_KEEP = 4  # prefix results kept on the GPU


@triton.jit
def _gdn_prep(x_ptr, w_ptr, b_ptr, a_ptr, alog_ptr, dtb_ptr, pos_ptr, tail_ptr, q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, T, C, NK, NV,
              DK: tl.constexpr, REP: tl.constexpr, BT: tl.constexpr, KW: tl.constexpr, HAS_TAIL: tl.constexpr):
    """Conv + SiLU + q/k/v split + beta + g. At a prompt boundary the conv reads the end of the prefix or zero, not the previous prompt."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    j = tl.program_id(1)
    tmask = t < T
    pos = tl.load(pos_ptr + t, mask=tmask, other=0)
    d = tl.arange(0, DK)
    c = j * DK + d
    row = t.to(tl.int64)
    acc = tl.zeros([BT, DK], dtype=tl.float32)
    for w in tl.static_range(KW):
        back = KW - 1 - w
        inside = (pos >= back) & tmask
        xv = tl.load(x_ptr + (row - back)[:, None] * C + c[None, :], mask=inside[:, None], other=0.0).to(tl.float32)
        if HAS_TAIL:
            outside = (pos < back) & tmask
            xv = xv + tl.load(tail_ptr + (KW - 1 + pos - back)[:, None] * C + c[None, :], mask=outside[:, None], other=0.0).to(tl.float32)
        acc = acc + tl.load(w_ptr + c * KW + w).to(tl.float32)[None, :] * xv
    conv = acc.to(tl.bfloat16).to(tl.float32)
    y = (conv / (1.0 + libdevice.exp(-conv))).to(tl.bfloat16)
    if j < NK:
        for r in tl.static_range(REP):
            tl.store(q_ptr + row[:, None] * (NV * DK) + ((j * REP + r) * DK + d)[None, :], y, mask=tmask[:, None])
    elif j < 2 * NK:
        for r in tl.static_range(REP):
            tl.store(k_ptr + row[:, None] * (NV * DK) + (((j - NK) * REP + r) * DK + d)[None, :], y, mask=tmask[:, None])
    else:
        hv = j - 2 * NK
        tl.store(v_ptr + row[:, None] * (NV * DK) + (hv * DK + d)[None, :], y, mask=tmask[:, None])
        bb = tl.load(b_ptr + row * NV + hv, mask=tmask, other=0.0).to(tl.float32)
        tl.store(beta_ptr + row * NV + hv, (1.0 / (1.0 + libdevice.exp(-bb))).to(tl.bfloat16), mask=tmask)
        s = tl.load(a_ptr + row * NV + hv, mask=tmask, other=0.0).to(tl.float32) + tl.load(dtb_ptr + hv)
        sp = tl.where(s > 20.0, s, libdevice.log1p(libdevice.exp(s)))
        tl.store(g_ptr + row * NV + hv, -libdevice.exp(tl.load(alog_ptr + hv)) * sp, mask=tmask)


def gdn_prep(mixed, b, a, conv_w, a_log, dt_bias, k_heads, head_dim, pos, tail):
    _, T, C = mixed.shape
    nv = (C - 2 * k_heads * head_dim) // head_dim
    q = torch.empty(1, T, nv, head_dim, dtype=torch.bfloat16, device=mixed.device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    g = torch.empty(1, T, nv, dtype=torch.float32, device=mixed.device)
    beta = torch.empty(1, T, nv, dtype=torch.bfloat16, device=mixed.device)
    _gdn_prep[(triton.cdiv(T, 16), 2 * k_heads + nv)](
        mixed, conv_w, b, a, a_log, dt_bias, pos, tail if tail is not None else mixed, q, k, v, g, beta, T, C, k_heads, nv,
        DK=head_dim, REP=nv // k_heads, BT=16, KW=conv_w.shape[-1], HAS_TAIL=tail is not None, num_warps=4)
    return q, k, v, g, beta


@triton.jit
def _to_fp8(x_ptr, q_ptr, n, BLOCK: tl.constexpr):
    """Moves activations to FP8, clamping only the range. Being floating point, FP8 keeps the same relative precision without a per-row scale."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(q_ptr + offs, tl.minimum(tl.maximum(x, -448.0), 448.0).to(tl.float8e4nv), mask=mask)


@triton.jit
def _unpack_fp8(packed_ptr, scale_ptr, zero_ptr, out_ptr, n_words, words_per_row, groups_per_row, factor,
                GROUP: tl.constexpr, BLOCK: tl.constexpr, ZERO: tl.constexpr):
    """Unpacks int4 weights to FP8, shifted by factor (a power of two)."""
    words = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = words < n_words
    word = tl.load(packed_ptr + words, mask=mask, other=0)
    row = words // words_per_row
    column = (words % words_per_row) * 8
    scale = tl.load(scale_ptr + row * groups_per_row + column // GROUP, mask=mask, other=0.0).to(tl.float32)
    nibbles = (word[:, None] >> (tl.arange(0, 8) * 4)[None, :]) & 0xF
    values = (nibbles - 8).to(tl.float32)
    if ZERO:
        zero_word = tl.load(zero_ptr + (row // 8) * groups_per_row + column // GROUP, mask=mask, other=0)
        values -= (((zero_word >> ((row % 8) * 4)) & 0xF) - 8).to(tl.float32)[:, None]
    values = values * scale[:, None] * factor
    tl.store(out_ptr + words[:, None] * 8 + tl.arange(0, 8)[None, :], values.to(tl.float8e4nv), mask=mask[:, None])


def _fp8_forward(self, x):
    shape = x.shape
    rows = x.reshape(-1, shape[-1]).contiguous()
    device = rows.device
    q = torch.empty(rows.shape, device=device, dtype=FP8)
    n = rows.numel()
    _to_fp8[(triton.cdiv(n, 1024),)](rows, q, n, BLOCK=1024)
    height, words = self.weight_packed.shape
    weight = torch.empty(height, words * 8, device=device, dtype=FP8)
    packed = self.weight_packed.numel()
    zero = self.weight_zero_point
    _unpack_fp8[(triton.cdiv(packed, d3k.UNPACK_BLOCK),)](
        self.weight_packed, self.weight_scale, self.weight_packed if zero is None else zero, weight, packed, words,
        self.weight_scale.shape[1], self.fp8_factor, GROUP=self.group_size, BLOCK=d3k.UNPACK_BLOCK, ZERO=zero is not None)
    out = torch._scaled_mm(q, weight.t(), scale_a=self.fp8_one, scale_b=self.fp8_back, out_dtype=x.dtype)
    if self.bias is not None:
        out = out + self.bias
    return out.reshape(*shape[:-1], self.out_features)


def fp8_available(device):
    """Checks on a small input that the FP8 conversion and product give the right value on this GPU."""
    try:
        x = torch.ones(16, 16, device=device, dtype=torch.bfloat16)  # the FP8 product needs an input width divisible by 16
        q = torch.empty(x.shape, device=device, dtype=FP8)
        _to_fp8[(1,)](x, q, x.numel(), BLOCK=1024)
        one = torch.ones((), device=device, dtype=torch.float32)
        out = torch._scaled_mm(q, q.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        return float(out[0, 0]) == 16.0
    except (RuntimeError, ValueError, NotImplementedError):
        return False


def common_prefix(sequences, limit):
    length = 0
    first = sequences[0]
    while length < limit and all(sequence[length] == first[length] for sequence in sequences):
        length += 1
    return length


def token_chunks(lengths, budget=TOKEN_BUDGET):
    """Splits prompt indices in order so each chunk stays within budget tokens. A longer prompt gets a chunk of its own."""
    chunks, current, total = [], [], 0
    for index, length in enumerate(lengths):
        if current and total + length > budget:
            chunks.append(current)
            current, total = [], 0
        current.append(index)
        total += length
    if current:
        chunks.append(current)
    return chunks


class VarlenDecider:
    """Answers text requests through one forward over their prompts joined without padding. Requests run one at a time
    per loaded model.

    state_prefix is the text every state starts with, if any. The fixed text reused across requests then extends to the
    end of it.
    """

    def __init__(self, model, exact=False, state_prefix=""):
        self.model = model
        self.state_prefix = state_prefix
        self.lm = model.backbone.language_model
        self.kernels = d3_fast.kernels_next_to_this_file()
        self.fused = rt.FusedStateOnce(self.lm, self.kernels)
        self.embed = model.backbone.get_input_embeddings()
        self.reuse = not exact
        self.fp8 = not exact and fp8_available(model.device)
        self.layers = [module for module in model.backbone.modules() if isinstance(module, d3k.Int4Linear) and module.in_features % 16 == 0]
        if self.fp8:
            for module in self.layers:
                shift = math.floor(math.log2(FP8_MAX / (INT4_RANGE * float(module.weight_scale.float().abs().max()))))
                module.fp8_factor = 2.0 ** shift
                module.fp8_back = torch.tensor(2.0 ** -shift, device=model.device, dtype=torch.float32)
                module.fp8_one = torch.ones((), device=model.device, dtype=torch.float32)
                module.fp8_bound = types.MethodType(_fp8_forward, module)
        self.fixed = {}
        self.caches = {}

    @contextlib.contextmanager
    def multiply_in_fp8(self):
        """Switches the products to FP8 only while this path runs. Other model calls keep their own products."""
        if not self.fp8:
            yield
            return
        for module in self.layers:
            module.forward = module.fp8_bound
        try:
            yield
        finally:
            for module in self.layers:
                module.__dict__.pop("forward", None)

    def fixed_tokens(self, questions):
        """The prompt tokens before the state and its state_prefix, which every request with these questions shares."""
        key = json.dumps(questions, ensure_ascii=False, sort_keys=True)
        if key not in self.fixed:
            question = next(iter(questions))
            first, second = (self.model.prepare(f"{self.state_prefix}{word}", questions).sequences.get(question, []) for word in "AB")
            self.fixed[key] = first[:common_prefix([first, second], min(len(first), len(second)))]
        return self.fixed[key]

    def shared_tokens(self, sequences, states, questions):
        """The leading tokens this chunk reuses."""
        if not self.reuse:
            return []
        fixed = self.fixed_tokens(questions)
        if fixed and all(sequence[:len(fixed)] == fixed for sequence in sequences) and len(fixed) <= min(states):
            return fixed
        length = common_prefix(sequences, min(states))
        return sequences[0][:length] if length >= MIN_PREFIX else []

    def prefix(self, tokens):
        """Computes the KV and scan state of a prefix once. The PREFIX_KEEP most recent are kept."""
        key = tuple(tokens)
        cache = self.caches.pop(key, None)
        if cache is None:
            embeds = self.embed(torch.as_tensor([tokens], device=self.model.device))
            positions = torch.arange(len(tokens), device=embeds.device).view(1, 1, -1).expand(3, 1, -1)
            cache = self.fused.state(embeds, self.lm.rotary_emb(embeds, positions))
            while len(self.caches) >= PREFIX_KEEP:
                self.caches.pop(next(iter(self.caches)))
        self.caches[key] = cache
        return cache

    def attention(self, attn, p, x, rope, entry, bounds):
        T = x.shape[1]
        hd = attn.head_dim
        qp, kp = attn.q_proj(x), attn.k_proj(x)
        heads, kv_heads = qp.shape[-1] // (2 * hd), kp.shape[-1] // hd
        cos, sin = rope
        query, key = self.kernels.attn_prep(qp, kp, p["qw1"], p["kw1"], cos, sin, heads, kv_heads, hd, attn.q_norm.eps)
        value = qwen.repeat_kv(attn.v_proj(x).view(1, T, kv_heads, hd).transpose(1, 2), heads // kv_heads)
        out = query.new_empty(query.shape)
        sdpa = torch.nn.functional.scaled_dot_product_attention
        for start, middle, end in bounds:
            k_state, v_state = key[:, :, start:middle], value[:, :, start:middle]
            if entry is not None:
                k_state, v_state = torch.cat([entry[0], k_state], dim=2), torch.cat([entry[1], v_state], dim=2)
            if middle > start:
                out[:, :, start:middle] = sdpa(query[:, :, start:middle], k_state, v_state, scale=attn.scaling)
            if end > middle:
                k_all = torch.cat([k_state, key[:, :, middle:end]], dim=2)
                v_all = torch.cat([v_state, value[:, :, middle:end]], dim=2)
                out[:, :, middle:end] = sdpa(query[:, :, middle:end], k_all, v_all, scale=attn.scaling)
        gate = qp.view(1, T, heads, 2 * hd)[..., hd:]
        return attn.o_proj(self.kernels.sigmoid_gate(out.transpose(1, 2), gate))

    def delta(self, m, p, x, entry, pos, cu, cu_cpu):
        T = x.shape[1]
        mixed, b, a = m.in_proj_qkv(x), m.in_proj_b(x), m.in_proj_a(x)
        tail, initial = (None, None) if entry is None else entry
        q, k, v, g, beta = gdn_prep(mixed, b, a, p["conv_w"], p["A_log"], p["dt_bias"], m.num_k_heads, m.head_k_dim, pos,
                                    None if tail is None else tail[0].contiguous())
        if initial is not None:
            initial = initial.expand(cu.shape[0] - 1, -1, -1, -1).contiguous()
        core, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=initial, output_final_state=False,
                                         use_qk_l2norm_in_kernel=True, cu_seqlens=cu, cu_seqlens_cpu=cu_cpu)
        out = self.kernels.gated_rmsnorm(core.reshape(-1, m.head_v_dim), m.in_proj_z(x).reshape(-1, m.head_v_dim), m.norm.weight, m.norm.variance_epsilon)
        return m.out_proj(out.reshape(1, T, -1))

    def forward(self, rows, states, cache, offset):
        """rows: tokens per prompt without the prefix. states: state block tokens per prompt without the prefix. cache: the result of the offset-token prefix.

        Returns the hidden state of each prompt's last token.
        """
        device = self.model.device
        cache = cache or [None] * len(self.fused.stack)
        lengths = [len(row) for row in rows]
        cu_list = [0]
        for length in lengths:
            cu_list.append(cu_list[-1] + length)
        cu_cpu = torch.as_tensor(cu_list, dtype=torch.long)
        cu = cu_cpu.to(device)
        bounds = [(cu_list[n], cu_list[n] + states[n], cu_list[n + 1]) for n in range(len(rows))]
        pos = torch.cat([torch.arange(length, dtype=torch.int32) for length in lengths]).to(device)
        embeds = self.embed(torch.as_tensor([[token for row in rows for token in row]], device=device))
        rope = self.lm.rotary_emb(embeds, (pos.long() + offset).view(1, 1, -1).expand(3, 1, -1))
        k = self.kernels
        hidden, delta = embeds, None
        for (layer, softmax), p, entry in zip(self.fused.stack, self.fused.params, cache):
            hidden, x = k.add_rmsnorm(hidden, delta, p["w1_in"], p["eps"])
            if softmax:
                delta = self.attention(layer.self_attn, p, x, rope, entry, bounds)
            else:
                delta = self.delta(layer.linear_attn, p, x, entry, pos, cu, cu_cpu)
            hidden, x = k.add_rmsnorm(hidden, delta, p["w1_post"], p["eps"])
            mlp = layer.mlp
            delta = mlp.down_proj(k.silu_mul(mlp.gate_proj(x), mlp.up_proj(x)))
        return k.add_rmsnorm(hidden, delta, self.fused.final[0], self.fused.final[1])[1][0, cu[1:] - 1]

    def readout(self, hidden, counts):
        model = self.model
        if model.readout_dtype == "float32":
            logits = hidden.float() @ model.readout.T
        else:
            logits = torch.nn.functional.linear(hidden, model.readout).float()
        limit = torch.as_tensor(counts, device=model.device)[:, None]
        logits = logits.masked_fill(torch.arange(rt.MAX_OPTIONS, device=model.device)[None] >= limit, float("-inf"))
        values = (logits / model.temperature).softmax(-1).cpu().tolist()
        return [value[:count] for value, count in zip(values, counts)]

    def probabilities(self, sequences, states, counts, questions, n_questions=1):
        """Option probabilities per prompt. questions locate the fixed text, n_questions is the question count of one request."""
        shared = self.shared_tokens(sequences, states, questions)
        with torch.inference_mode(), self.model.request_temperature(n_questions), self.multiply_in_fp8():
            cache = self.prefix(shared) if shared else None
            hidden = self.forward([s[len(shared):] for s in sequences], [s - len(shared) for s in states], cache, len(shared))
            return self.readout(hidden, counts)

    def decide_questions(self, request):
        """Answers one text request with several questions.

        A short state reuses the fixed question list and repeats only the state for each question. Otherwise the state
        block is computed once, since every question reads the same one.
        """
        model = self.model
        questions = request["questions"]
        prepared = model.prepare(request["state"], questions)
        keys = prepared.runnable
        if not keys:
            return model.respond(prepared, {}, 0)
        state = prepared.states[keys[0]]
        fixed = self.fixed_tokens(questions) if self.reuse else []
        worth_it = len(keys) * (state - len(fixed)) <= len(fixed)  # repeating the state must cost less than the list it avoids
        if fixed and worth_it and all(prepared.sequences[key][:len(fixed)] == fixed and len(fixed) <= prepared.states[key] for key in keys):
            sequences = [prepared.sequences[key] for key in keys]
            found = {}
            for chunk in token_chunks([len(sequence) - len(fixed) for sequence in sequences]):
                members = [keys[index] for index in chunk]
                rows = self.probabilities([prepared.sequences[key] for key in members], [prepared.states[key] for key in members],
                                          [len(prepared.questions[key].keys) for key in members], questions, len(prepared.questions))
                found.update(zip(members, rows))
            return model.respond(prepared, found, len(fixed) + sum(len(sequence) - len(fixed) for sequence in sequences))
        shared = prepared.sequences[keys[0]][:state]
        pieces = [prepared.sequences[key][state:] for key in keys]
        out = {}
        with torch.inference_mode(), model.request_temperature(len(prepared.questions)), self.multiply_in_fp8():
            embeds = self.embed(torch.as_tensor([shared], device=model.device))
            positions = torch.arange(state, device=embeds.device).view(1, 1, -1).expand(3, 1, -1)
            cache = self.fused.state(embeds, self.lm.rotary_emb(embeds, positions))
            for chunk in token_chunks([len(piece) for piece in pieces]):
                hidden = self.forward([pieces[index] for index in chunk], [0] * len(chunk), cache, state)
                counts = [len(prepared.questions[keys[index]].keys) for index in chunk]
                for index, row in zip(chunk, self.readout(hidden, counts)):
                    out[keys[index]] = row
        return model.respond(prepared, out, state + sum(len(piece) for piece in pieces))

    def decide(self, batch):
        """Answers a list of single-question text requests in order. An invalid question comes back as an error in that request's answer."""
        model = self.model
        prepared = [model.prepare(request["state"], request["questions"]) for request in batch]
        runnable = [n for n, item in enumerate(prepared) if item.sequences]
        key_of = {n: prepared[n].runnable[0] for n in runnable}
        sequences = {n: prepared[n].sequences[key_of[n]] for n in runnable}
        answers = {}
        for chunk in token_chunks([len(sequences[n]) for n in runnable]):
            members = [runnable[index] for index in chunk]
            rows = [sequences[n] for n in members]
            states = [prepared[n].states[key_of[n]] for n in members]
            counts = [len(prepared[n].questions[key_of[n]].keys) for n in members]
            for n, row in zip(members, self.probabilities(rows, states, counts, batch[members[0]]["questions"])):
                answers[n] = {key_of[n]: row}
        return [model.respond(item, answers.get(n, {}), len(sequences.get(n, ()))) for n, item in enumerate(prepared)]
