"""Reads a result of compare.py: per layer size, time against rows for every implementation, the 400-layer pass, the lower bound
from measured bandwidth and compute, correctness, and hot against cold. Argument: the result JSON."""
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
env = data["environment"]
sweep = data["memory_sweep"]
dram = sweep[-1]["gbps"] * 1e9
print(f"{env['device']} {env['arch']}, {env['compute_units']} CUs, L2 {env['l2_bytes']}, torch {env['torch']}, vllm {env['vllm']}, triton {env['triton']}")
print("read bandwidth GB/s by buffer size: " + ", ".join(f"{s['mib']}MiB {s['gbps']:.0f}" for s in sweep))
GROUP = 32
peak = 0.0
for layer in data["layers"]:
    for m, row in layer["rows"].items():
        if "dense" in row:
            peak = max(peak, 2 * int(m) * layer["n"] * layer["k"] / (row["dense"]["ms"] * 1e-3))
print(f"device memory bandwidth (1 GiB reads) {dram / 1e9:.0f} GB/s; best dense BF16 product {peak / 1e12:.1f} TFLOP/s")


def vllm_path(m, k):
    return "hip" if m <= 5 and k * m <= 32768 else "tri"


def bound_ms(layer, m):
    n, k = layer["n"], layer["k"]
    data_bytes = layer["packed_bytes"] + m * k * 2 + m * n * 2
    return max(data_bytes / dram, 2 * m * n * k / peak) * 1e3


totals = {}
for layer in data["layers"]:
    n, k, count = layer["n"], layer["k"], layer["count"]
    print(f"\n== {layer['name']}  N={n} K={k}  x{count} per pass  copies {layer['copies']}")
    print(f"{'M':>5} {'bound':>8} {'ours':>9} {'vllm':>9} {'path':>4} {'unpack':>9} {'dense':>9}  ours/vllm  best   ours eff  vllm eff")
    for m_text, row in layer["rows"].items():
        m = int(m_text)
        bound = bound_ms(layer, m)

        def cell(impl):
            if impl not in row:
                return f"{'-':>9}"
            v = row[impl]
            flag = "*" if v["host_ms"] >= 0.9 * v["ms"] else " "
            return f"{v['ms']:8.4f}{flag}"

        best = min(row, key=lambda impl: row[impl]["ms"])
        ratio = f"{row['ours']['ms'] / row['vllm']['ms']:9.2f}" if "ours" in row else f"{'-':>9}"
        eff = lambda impl: f"{bound / row[impl]['ms']:8.0%}" if impl in row else f"{'-':>8}"
        print(f"{m:>5} {bound:8.4f} {cell('ours')} {cell('vllm')} {vllm_path(m, k):>4} {cell('unpack')} {cell('dense')}  {ratio}  {best:<6} {eff('ours')} {eff('vllm')}")
        for impl, v in row.items():
            totals.setdefault(m, {}).setdefault(impl, 0.0)
            totals[m][impl] += v["ms"] * count
    hot = layer["hot"]
    print("  hot/cold time ratio: " + "; ".join(
        f"M={m}: " + " ".join(f"{impl} {hot[m][impl]['ms'] / layer['rows'][m][impl]['ms']:.2f}" for impl in hot[m]) for m in hot))
    print("  error / output rms (max, mean):")
    for m, results in layer["correctness"].items():
        print(f"    M={m:<4} " + " | ".join(f"{impl} {r['max']:.4f} {r['mean']:.5f}" for impl, r in results.items()))

print("\n== one pass of the 400 int4 layers (sum of count x cold time), ms")
print(f"{'M':>5} {'ours':>9} {'vllm':>9} {'unpack':>9} {'dense':>9}  ours/vllm")
for m, row in totals.items():
    ours = row.get("ours")
    print(f"{m:>5} " + " ".join(f"{row[impl]:9.1f}" if impl in row else f"{'-':>9}" for impl in ("ours", "vllm", "unpack", "dense"))
          + (f"  {ours / row['vllm']:9.2f}" if ours else ""))
