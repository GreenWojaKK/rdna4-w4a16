"""Checks the developer mode of the device kernels on layers of d3-ultra's sizes filled with random weights.

It runs products in spans named after the phases of a model worker, with every second span timing its int4 products,
and compares what the record file says with what the script knows: the calls with their M, K, N and path, the time of
a timed product against the same call measured alone by waiting for the device, and the time of a span that does not
time its int4 products against the same calls with the mode off.

Arguments: the d3-ultra-INT4 folder, the kernel library, and the record file to write.
"""

import ctypes
import json
import sys
import time

sys.path.insert(0, sys.argv[1])
import torch  # noqa: E402
from d3q.kernels import Int4Linear  # noqa: E402

library = ctypes.CDLL(sys.argv[2])
library.rdna4_w4a16_linear.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4
library.rdna4_w4a16_route.argtypes = [ctypes.c_int] * 6
library.rdna4_w4a16_dev_start.argtypes = [ctypes.c_char_p, ctypes.c_int]
library.rdna4_w4a16_dev_span_begin.argtypes = [ctypes.c_char_p, ctypes.c_longlong, ctypes.c_longlong]
library.rdna4_w4a16_dev_op_begin.argtypes = [ctypes.c_int] * 7
library.rdna4_w4a16_dev_finish.restype = ctypes.c_char_p
device = torch.device("cuda:0")
GROUP = 32
BF16, F32 = 1, 3
INT4, HEAD = 0, 1
UNPACK, DENSE = 0, 4
PATHS = {1: "dot", 2: "tile1", 3: "tile2"}
LAYERS = [(17408, 5120), (5120, 17408), (1024, 5120)]
HEAD_SHAPE = (248320, 5120)
REPEATS = 40
# The phase and rows of each pair of spans, and whether its spans end with the head.
CASES = [("decode", 1, True), ("prefill", 250, True), ("decision_suffix", 250, False), ("decision_state", 1500, False)]


def layer(out, inputs):
    words = torch.randint(-(2**31), 2**31 - 1, (out, inputs // 8), dtype=torch.int32, device=device)
    scale = (torch.rand(out, inputs // GROUP, device=device) * 0.02 + 0.001).to(torch.bfloat16)
    zero = torch.randint(-(2**31), 2**31 - 1, (out // 8, inputs // GROUP), dtype=torch.int32, device=device)
    return Int4Linear(words, scale, zero, inputs, out)


def packed(module, x):
    y = torch.empty(x.shape[0], module.out_features, dtype=torch.bfloat16, device=device)
    code = library.rdna4_w4a16_linear(
        module.weight_packed.data_ptr(), module.weight_scale.data_ptr(), module.weight_zero_point.data_ptr(),
        x.data_ptr(), y.data_ptr(), module.out_features, module.in_features, GROUP, x.shape[0],
    )
    if code:
        raise RuntimeError(f"the kernel returned {code}")
    return y


def product(module, x):
    """One int4 product the way the model worker runs it: by a kernel, or by the checkpoint's layer with its reason."""
    route = library.rdna4_w4a16_route(module.out_features, module.in_features, GROUP, x.shape[0], BF16, 0)
    if route > 0:
        return packed(module, x)
    library.rdna4_w4a16_dev_op_begin(INT4, UNPACK, -route, x.shape[0], module.in_features, module.out_features, BF16)
    y = module(x)
    library.rdna4_w4a16_dev_op_end()
    return y


def head_product(hidden, head):
    library.rdna4_w4a16_dev_op_begin(HEAD, DENSE, 0, hidden.shape[0], head.shape[1], head.shape[0], BF16)
    y = hidden @ head.T
    library.rdna4_w4a16_dev_op_end()
    return y


def waited(function, repeats=REPEATS):
    """Milliseconds one call takes, measured by waiting for the device."""
    function()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / repeats * 1000


failures = []


def expect(condition, message):
    if not condition:
        failures.append(message)


def close(measured, reference):
    """Whether a time agrees with its reference: within a third, or within what a pair of events can be off by."""
    return abs(measured - reference) <= max(reference / 3, 0.08)


torch.manual_seed(0)
with torch.inference_mode():
    # The routes, which need no device.
    for (out, inputs, rows, dtype, bias), want in {
        (17408, 5120, 1, BF16, 0): 1, (17408, 5120, 16, BF16, 0): 2, (17408, 5120, 250, BF16, 0): 3,
        (17408, 5120, 513, BF16, 0): -1, (17408, 5120, 16, F32, 0): -2, (17408, 5120, 16, BF16, 1): -3,
        (17408, 5000, 16, BF16, 0): -4,
    }.items():
        got = library.rdna4_w4a16_route(out, inputs, GROUP, rows, dtype, bias)
        expect(got == want, f"route of {rows} rows on {out}x{inputs}, dtype {dtype}, bias {bias}: {got}, expected {want}")

    modules = {shape: layer(*shape) for shape in LAYERS}
    head = torch.randn(*HEAD_SHAPE, device=device).to(torch.bfloat16)
    inputs = {(shape, rows): torch.randn(rows, shape[1], device=device).to(torch.bfloat16) for shape in LAYERS for _, rows, _ in CASES}
    hidden = {rows: torch.randn(rows, HEAD_SHAPE[1], device=device).to(torch.bfloat16) for _, rows, with_head in CASES if with_head}

    # What each call takes alone with the mode off.
    alone = {(shape, rows): waited(lambda: product(modules[shape], inputs[(shape, rows)])) for shape in LAYERS for _, rows, _ in CASES}
    alone_head = {rows: waited(lambda: hidden[rows] @ head.T) for rows in hidden}

    expect(library.rdna4_w4a16_dev_start(sys.argv[3].encode(), 2) == 0, "the mode starts")
    expect(library.rdna4_w4a16_dev_start(sys.argv[3].encode(), 2) == -2, "the mode does not start twice")
    expect(library.rdna4_w4a16_dev_span_begin(b"", 1, 1) == -1, "a span needs a name")
    for phase, rows, with_head in CASES:
        for _ in range(2):  # The first span is timed as a whole, the second also times its int4 products.
            library.rdna4_w4a16_dev_span_begin(phase.encode(), rows, rows)
            for _ in range(REPEATS):
                for shape in LAYERS:
                    product(modules[shape], inputs[(shape, rows)])
            if with_head:
                head_product(hidden[rows], head)
            library.rdna4_w4a16_dev_span_end()
    product(modules[LAYERS[0]], inputs[(LAYERS[0], 1)])  # Outside every span: counted only.
    summary = json.loads(library.rdna4_w4a16_dev_finish().decode())
    expect(library.rdna4_w4a16_dev_finish() == b"", "a finished mode has nothing more to give")

lines = [json.loads(line) for line in open(sys.argv[3], encoding="utf-8")]
run, spans = lines[0], [line for line in lines if line["record"] == "span"]
expect(run["record"] == "run" and run["schema"] == 2 and run["every"] == 2, f"the file starts with the run: {run}")
expect(lines[-1] == summary, "the file ends with the summary")
expect([p["phase"] for p in summary["phases"]] == [case[0] for case in CASES], "phases in the order their first span opened")
expect(len(spans) == 2 * len(CASES) and summary["spans"] == len(spans), "one line for each span")
expect([span["timed"] for span in spans] == [False, True] * len(CASES), "every second span of a phase is timed")
expect(all(summary[name] == run[name] for name in ("every", "event_ms", "pair_ms")), "the summary repeats the run's costs")
print(f"device {run['device']} ({run['arch']}), a pair of events adds {run['event_ms']:.4f} ms to a product and {run['pair_ms']:.4f} ms to its span")

found = {(p["phase"], p["op"], p["path"], p["m"], p["k"], p["n"]): p for p in summary["products"]}
print("products, per call: timed by events in a span | alone, waiting for the device")
for phase, rows, with_head in CASES:
    whole, timed = spans[2 * [case[0] for case in CASES].index(phase) :][:2]
    reference = 0.0
    for out, k in LAYERS:
        route = library.rdna4_w4a16_route(out, k, GROUP, rows, BF16, 0)
        path = PATHS.get(route, "unpack")
        entry = found.get((phase, "int4", path, rows, k, out))
        expect(entry is not None, f"no product {phase} {path} {rows}x{k}x{out}")
        if entry is None:
            continue
        expect(entry["calls"] == 2 * REPEATS and entry["timed_calls"] == REPEATS, f"{phase} {path} {rows}x{k}x{out}: {entry['calls']} calls, {entry['timed_calls']} timed")
        expect(entry["dtype"] == "bf16" and entry.get("reason") == ("rows" if route < 0 else None), f"{phase} {path} {rows}x{k}x{out}: {entry}")
        each, want = entry["ms"] / entry["timed_calls"], alone[((out, k), rows)]
        reference += want * REPEATS
        print(f"  {phase:<15} int4 {path:<6} M={rows:<5} K={k:<6} N={out:<6} {each:8.4f} ms | {want:8.4f} ms")
        expect(close(each, want), f"{phase} {path} {rows}x{k}x{out}: {each:.4f} ms by events, {want:.4f} ms alone")
    if with_head:
        entry = found.get((phase, "head", "dense", rows, HEAD_SHAPE[1], HEAD_SHAPE[0]))
        expect(entry is not None and entry["calls"] == 2 and entry["timed_calls"] == 2, f"the head of {phase}: {entry}")
        if entry is not None:
            each, want = entry["ms"] / entry["timed_calls"], alone_head[rows]
            reference += want
            print(f"  {phase:<15} head {'dense':<6} M={rows:<5} K={HEAD_SHAPE[1]:<6} N={HEAD_SHAPE[0]:<6} {each:8.4f} ms | {want:8.4f} ms")
            expect(close(each, want), f"the head of {phase}: {each:.4f} ms by events, {want:.4f} ms alone")
    # A span that does not time its int4 products takes what its calls take with the mode off.
    print(f"  {phase:<15} span of {REPEATS * len(LAYERS)} products: {whole['gpu_ms']:9.3f} ms | {reference:9.3f} ms alone ({(whole['gpu_ms'] / reference - 1) * 100:+.1f}%)")
    expect(0.85 < whole["gpu_ms"] / reference < 1.25, f"{phase}: a span of {whole['gpu_ms']:.3f} ms for calls of {reference:.3f} ms")
    expect(all("ms" in p for p in whole["products"]), "every product of a span has its fields")
    # A timed span takes the same without what its events cost.
    left = timed["gpu_ms"] - run["pair_ms"] * sum(p["timed_calls"] for p in timed["products"] if p["op"] == "int4")
    print(f"  {phase:<15} timed span: {timed['gpu_ms']:9.3f} ms, {left:9.3f} ms without its events ({(left / whole['gpu_ms'] - 1) * 100:+.1f}% of the span above)")
    expect(0.7 < left / whole["gpu_ms"] < 1.4, f"{phase}: a timed span of {left:.3f} ms without its events beside one of {whole['gpu_ms']:.3f} ms")

counters = summary["counters"]
kernel = sum(2 * REPEATS for _, rows, _ in CASES for out, k in LAYERS if library.rdna4_w4a16_route(out, k, GROUP, rows, BF16, 0) > 0) + 1
fallback = sum(2 * REPEATS for _, rows, _ in CASES for out, k in LAYERS if library.rdna4_w4a16_route(out, k, GROUP, rows, BF16, 0) < 0)
expect(counters["kernel_calls"] == kernel, f"{counters['kernel_calls']} kernel calls, expected {kernel}")
expect(counters["fallback_rows"] == fallback and counters["fallback_calls"] == fallback, f"fallbacks {counters}, expected {fallback}")
expect(counters["head_calls"] == 2 * len(hidden) and counters["outside_calls"] == 1 and counters["unresolved"] == 0, f"counters {counters}")
for phase in summary["phases"]:
    expect(phase["spans"] == 2 and phase["timed_spans"] == 1, f"{phase['phase']}: {phase['spans']} spans, {phase['timed_spans']} timed")
    expect(phase["timed_int4_calls"] == REPEATS * len(LAYERS), f"{phase['phase']}: {phase['timed_int4_calls']} timed int4 products")
    timed = sum(p["ms"] for p in summary["products"] if p["phase"] == phase["phase"] and p["op"] == "int4")
    expect(abs(timed - phase["timed_int4_ms"]) < 0.01, f"{phase['phase']}: int4 products {timed}, phase {phase}")
    heads = sum(p["ms"] for p in summary["products"] if p["phase"] == phase["phase"] and p["op"] == "head")
    expect(abs(heads - phase["head_ms"] - phase["timed_head_ms"]) < 0.01, f"{phase['phase']}: heads {heads}, phase {phase}")
for failure in failures:
    print("FAILED:", failure)
sys.exit(1 if failures else 0)
