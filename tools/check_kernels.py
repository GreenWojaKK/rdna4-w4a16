"""Checks the packed int4 kernel against the checkpoint's own layer, which unpacks the weight to BF16 and multiplies.

Layers of the sizes d3-ultra has are filled with random weights, so no model is loaded. For each size and number of
rows it prints the share of outputs that differ, the largest difference, and the time one call of each takes; then the
time for the 400 layers one pass goes through.

Arguments: the d3-ultra-INT4 folder and the kernel library.
"""

import ctypes
import sys
import time

sys.path.insert(0, sys.argv[1])
import torch  # noqa: E402
from d3q.kernels import Int4Linear  # noqa: E402

library = ctypes.CDLL(sys.argv[2])
library.rdna4_w4a16_linear.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4
library.rdna4_w4a16_linear.restype = ctypes.c_int
device = torch.device("cuda:0")
GROUP = 32
# Name, output width, input width, and how many such layers one token passes through.
LAYERS = [
    ("gate_proj, up_proj", 17408, 5120, 128), ("down_proj", 5120, 17408, 64), ("in_proj_qkv", 10240, 5120, 48),
    ("in_proj_z", 6144, 5120, 48), ("out_proj, o_proj", 5120, 6144, 64), ("q_proj", 12288, 5120, 16),
    ("k_proj, v_proj", 1024, 5120, 32),
]
ROWS = (1, 2, 16, 17, 48, 49, 128, 250, 512)


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


def milliseconds(function, repeats=20):
    function()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / repeats * 1000


torch.manual_seed(0)
token = {rows: [0.0, 0.0] for rows in ROWS}
failed = False
with torch.inference_mode():
    for name, out, inputs, count in LAYERS:
        module = layer(out, inputs)
        for rows in ROWS:
            x = torch.randn(rows, inputs, device=device).to(torch.bfloat16)
            want, got = module(x).float(), packed(module, x).float()
            differ = float((want != got).float().mean())
            largest = float((got - want).abs().max() / want.pow(2).mean().sqrt())
            # Sums taken in another order may round the last bit of an output the other way, and nothing more.
            failed = failed or differ > 0.01 or largest > 0.05 or bool(got.isnan().any())
            unpacking, straight = milliseconds(lambda: module(x)), milliseconds(lambda: packed(module, x))
            token[rows][0] += unpacking * count
            token[rows][1] += straight * count
            print(f"{name:<18} {rows:>4} rows: {differ:.3%} of outputs differ, by at most {largest:.4f} of a typical output; "
                  f"unpack and multiply {unpacking:.3f} ms, packed {straight:.3f} ms")
        del module
print("the 400 layers of one pass:")
for rows, (unpacking, straight) in token.items():
    print(f"  {rows:>4} rows: unpack and multiply {unpacking:6.1f} ms, packed {straight:6.1f} ms")
sys.exit(1 if failed else 0)
