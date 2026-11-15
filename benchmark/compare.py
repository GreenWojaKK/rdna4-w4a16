"""W4A16 linear products on one GPU, implementation against implementation, rows of input against time.

Implementations, all fed the same 4-bit values, scales and zero points (group 32, asymmetric, BF16 input):
  ours    this repository's kernel library (dot product kernel for 1 row, matrix kernel for 2-512 rows)
  vllm    vLLM RDNAHybridW4A16 apply (HIP skinny kernel for few rows, Triton for the rest), as installed
  unpack  the d3 checkpoint's Int4Linear: unpacks the whole weight to BF16 on every call, then F.linear
  dense   F.linear on weights dequantized to BF16 once (4x the memory of the packed weights)

Timing: each implementation cycles through copies of its weights larger in total than the device caches, so every
call reads its weights from device memory; the device is synchronized once per timed run. A few row counts are also
timed hot, calling one copy repeatedly. Correctness is measured against a float64 product of the exact dequantized
weights.

Arguments: the d3-ultra-INT4 folder (for the checkpoint's Int4Linear), dist/librdna4_w4a16.so, the output JSON path.
The Python environment needs ROCm PyTorch and vLLM. report.py prints the result.
"""
import ctypes
import json
import math
import statistics
import sys
import time

sys.path.insert(0, sys.argv[1])
import torch  # noqa: E402
from d3q.kernels import Int4Linear  # noqa: E402

import vllm  # noqa: E402
from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as hybrid  # noqa: E402
from vllm.utils.platform_utils import num_compute_units  # noqa: E402

device = torch.device("cuda:0")
library = ctypes.CDLL(sys.argv[2])
library.rdna4_w4a16_linear.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4
library.rdna4_w4a16_linear.restype = ctypes.c_int
GROUP = 32
OURS_MAX_ROWS = 512
COLD_BYTES = 768 << 20
TARGET_MS = 40.0
REPEATS = 5
LAYERS = [  # name, N (outputs), K (inputs), layers of this size in one pass of d3-ultra
    ("gate_proj, up_proj", 17408, 5120, 128), ("down_proj", 5120, 17408, 64), ("in_proj_qkv", 10240, 5120, 48),
    ("q_proj", 12288, 5120, 16), ("in_proj_z", 6144, 5120, 48), ("out_proj, o_proj", 5120, 6144, 64),
    ("k_proj, v_proj", 1024, 5120, 32),
]
ROWS = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048]
HOT_ROWS = (1, 16, 256)
CHECK_ROWS = (1, 4, 16, 128, 512)
IMPLS = ("ours", "vllm", "unpack", "dense")
CU = num_compute_units()


def to_int32(values):
    """int64 holding 32-bit patterns -> int32 with the same bits."""
    return torch.where(values >= 2**31, values - 2**32, values).to(torch.int32)


def pack_ct(nibbles):
    """[N, K] values 0..15 -> compressed-tensors words [N, K/8]: column 8j+i in bits 4i..4i+3."""
    n, k = nibbles.shape
    groups = nibbles.view(n, k // 8, 8).long()
    words = torch.zeros(n, k // 8, dtype=torch.long, device=nibbles.device)
    for i in range(8):
        words |= groups[:, :, i] << (4 * i)
    return to_int32(words)


def pack_zero(zero):
    """[N, G] values 0..15 -> compressed-tensors zero points [N/8, G]: row 8r+i in bits 4i..4i+3."""
    n, g = zero.shape
    rows = zero.view(n // 8, 8, g).long()
    words = torch.zeros(n // 8, g, dtype=torch.long, device=zero.device)
    for i in range(8):
        words |= rows[:, i, :] << (4 * i)
    return to_int32(words)


def clones(tensors, count):
    return [tuple(t.clone() for t in tensors) for _ in range(count)]


def nbytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


class Shape:
    def __init__(self, n, k):
        self.n, self.k = n, k
        g = k // GROUP
        torch.manual_seed(n * 7 + k)
        nibbles = torch.randint(0, 16, (n, k), dtype=torch.int32, device=device)
        zero = torch.randint(0, 16, (n, g), dtype=torch.int32, device=device)
        scale = (torch.rand(n, g, device=device) * 0.02 + 0.001).to(torch.bfloat16)
        # Exact dequantized weights; every implementation uses them rounded to BF16.
        exact = (nibbles - zero.repeat_interleave(GROUP, dim=1)).double() * scale.double().repeat_interleave(GROUP, dim=1)
        self.exact = exact
        ours = (pack_ct(nibbles), scale.contiguous(), pack_zero(zero))
        vl = (hybrid.pack_int4_exllama_shuffle(nibbles).contiguous().view(torch.int8), scale.contiguous(),
              zero.to(torch.bfloat16).contiguous())
        dense = (exact.to(torch.bfloat16).contiguous(),)
        del nibbles
        self.copies = {
            "ours": clones(ours, math.ceil(COLD_BYTES / nbytes(ours))),
            "vllm": clones(vl, math.ceil(COLD_BYTES / nbytes(vl))),
            "dense": clones(dense, math.ceil(COLD_BYTES / nbytes(dense))),
        }
        self.copies["unpack"] = self.copies["ours"]
        self.modules = [Int4Linear(w, s, z, k, n) for w, s, z in self.copies["ours"]]
        self.packed_bytes = nbytes(ours)

    def calls(self, impl, x, y, hot):
        """One callable per weight copy (one only when hot)."""
        m, n, k = x.shape[0], self.n, self.k
        copies = self.copies[impl][:1] if hot else self.copies[impl]
        if impl == "ours":
            def make(w, s, z):
                args = (w.data_ptr(), s.data_ptr(), z.data_ptr(), x.data_ptr(), y.data_ptr(), n, k, GROUP, m)
                return lambda: library.rdna4_w4a16_linear(*args)
            return [make(*c) for c in copies]
        if impl == "vllm":
            return [lambda w=w, s=s, z=z: hybrid._rdna_hybrid_w4a16_apply_impl(x, w, s, z, None, CU, GROUP) for w, s, z in copies]
        if impl == "unpack":
            modules = self.modules[:1] if hot else self.modules
            return [lambda module=module: module(x) for module in modules]
        return [lambda w=w[0]: torch.nn.functional.linear(x, w) for w in copies]


def timed(calls, count):
    """Device milliseconds per call, and host milliseconds per call spent queueing them."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    began = time.perf_counter()
    start.record()
    for i in range(count):
        calls[i % len(calls)]()
    end.record()
    queued = time.perf_counter() - began
    end.synchronize()
    return start.elapsed_time(end) / count, queued * 1000 / count


def measure(shape, m, hot=False):
    """Median device time per call of every implementation that takes m rows, interleaved over repeats."""
    x = torch.randn(m, shape.k, device=device).to(torch.bfloat16)
    y = torch.empty(m, shape.n, device=device, dtype=torch.bfloat16)
    impls = [impl for impl in IMPLS if impl != "ours" or m <= OURS_MAX_ROWS]
    calls = {impl: shape.calls(impl, x, y, hot) for impl in impls}
    counts = {}
    for impl in impls:  # warm up (Triton compiles here) and size the timed runs
        for call in calls[impl]:
            call()
        estimate, _ = timed(calls[impl], len(calls[impl]))
        counts[impl] = max(len(calls[impl]), math.ceil(TARGET_MS / max(estimate, 1e-3)))
    runs = {impl: [] for impl in impls}
    for repeat in range(REPEATS):
        order = impls[repeat % len(impls):] + impls[:repeat % len(impls)]
        for impl in order:
            runs[impl].append(timed(calls[impl], counts[impl]))
    out = {}
    for impl, values in runs.items():
        device_ms = [v[0] for v in values]
        host_ms = [v[1] for v in values]
        out[impl] = {
            "ms": statistics.median(device_ms), "min_ms": min(device_ms), "max_ms": max(device_ms),
            "host_ms": statistics.median(host_ms), "calls": counts[impl], "copies": len(calls[impl]),
        }
    return out


def correctness(shape, m):
    x = torch.randn(m, shape.k, device=device).to(torch.bfloat16)
    reference = x.double() @ shape.exact.T
    rms = reference.pow(2).mean().sqrt().item()
    y = torch.empty(m, shape.n, device=device, dtype=torch.bfloat16)
    results = {}
    outputs = {}
    if m <= OURS_MAX_ROWS:
        for call in shape.calls("ours", x, y, True):
            code = call()
            assert code == 0, f"the kernel returned {code}"
        outputs["ours"] = y.clone()
    outputs["vllm"] = shape.calls("vllm", x, y, True)[0]()
    outputs["unpack"] = shape.calls("unpack", x, y, True)[0]()
    outputs["dense"] = shape.calls("dense", x, y, True)[0]()
    for impl, out in outputs.items():
        error = (out.double() - reference).abs()
        results[impl] = {"max": error.max().item() / rms, "mean": error.mean().item() / rms}
    rounding = (reference.to(torch.bfloat16).double() - reference).abs()
    results["bf16 output rounding"] = {"max": rounding.max().item() / rms, "mean": rounding.mean().item() / rms}
    return results


def memory_sweep():
    """Read bandwidth against buffer size: one reduction reads the same buffer many times over."""
    out = []
    for mib in (2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024):
        n = (mib << 20) // 2
        buffer = torch.randn(n, device=device).to(torch.bfloat16)
        reps = max(1, (4 << 30) // (mib << 20))
        view = torch.as_strided(buffer, (reps, n), (0, 1))
        view.sum(dtype=torch.float32)
        best = math.inf
        for _ in range(3):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            view.sum(dtype=torch.float32)
            end.record()
            end.synchronize()
            best = min(best, start.elapsed_time(end))
        out.append({"mib": mib, "gbps": reps * (mib << 20) / (best * 1e-3) / 1e9})
        del buffer, view
    return out


def main():
    properties = torch.cuda.get_device_properties(device)
    environment = {
        "device": properties.name, "arch": getattr(properties, "gcnArchName", ""),
        "compute_units": properties.multi_processor_count, "l2_bytes": getattr(properties, "L2_cache_size", None),
        "total_memory": properties.total_memory, "torch": torch.__version__, "vllm": vllm.__version__,
        "triton": __import__("triton").__version__, "cu_count_vllm": CU,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(json.dumps(environment), flush=True)
    sweep = memory_sweep()
    print("read bandwidth by buffer size:", " ".join(f"{s['mib']}MiB:{s['gbps']:.0f}" for s in sweep), flush=True)
    results = {"environment": environment, "memory_sweep": sweep, "layers": []}
    for name, n, k, count in LAYERS:
        shape = Shape(n, k)
        entry = {"name": name, "n": n, "k": k, "count": count, "packed_bytes": shape.packed_bytes,
                 "copies": {impl: len(v) for impl, v in shape.copies.items()}, "rows": {}, "hot": {}, "correctness": {}}
        for m in CHECK_ROWS:
            entry["correctness"][m] = correctness(shape, m)
        for m in ROWS:
            entry["rows"][m] = measure(shape, m)
            line = " ".join(f"{impl}:{v['ms']:.4f}{'*' if v['host_ms'] >= 0.9 * v['ms'] else ''}" for impl, v in entry["rows"][m].items())
            print(f"{name:<18} M={m:<5} {line}", flush=True)
        for m in HOT_ROWS:
            entry["hot"][m] = measure(shape, m, hot=True)
        results["layers"].append(entry)
        with open(sys.argv[3], "w") as file:
            json.dump(results, file)
        del shape
        torch.cuda.empty_cache()
    print("done", flush=True)


main()
