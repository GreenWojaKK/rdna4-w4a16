# rdna4-w4a16

[한국어](docs-translate/ko/README.md)

W4A16 matrix-product kernels for AMD RDNA 4 GPUs, with a developer-mode recorder and a comparison benchmark, and a ROCm decision path for the [d3-ultra-INT4 decision model](https://huggingface.co/vllm-sr/d3-ultra-INT4).

## Components

| Module | Role | Interface |
|---|---|---|
| [d3_varlen.py](d3_varlen.py) | Text decision execution with unpadded prompt batches, FP8 matrix products and cached prompt prefixes | `VarlenDecider` |
| [kernels](kernels) | Packed INT4 by BF16 matrix products for 1–512 input rows, and the developer-mode recorder | C ABI of `dist/librdna4_w4a16.so` on Linux and `dist/rdna4_w4a16.dll` on Windows |
| [record](record) | Reads and checks developer-mode records | Rust crate and program `rdna4-w4a16-record` |
| [benchmark](benchmark) | Times W4A16 matrix-product implementations by input row count | `compare.py`, `report.py` |

`VarlenDecider` accepts text requests; route image, video, Box and Track requests to `model.system_one`.

In the d3 package used for this work, fused kernels and FP8 layers are enabled for AMD Instinct architectures `gfx942` and `gfx950`. Its RDNA 4 (`gfx1201`) path dequantizes INT4 projections to BF16 before multiplication and pads requests. This repository provides unpadded decision execution and products computed directly from packed weights.

## Environment and model setup

Use an existing Python environment containing ROCm PyTorch and the dependencies in the model's `requirements.txt`. `VarlenDecider` additionally requires `triton` and `flash-linear-attention`. Building the kernel library requires the ROCm SDK installed in that environment and, on Linux, a C++20 host compiler; on Windows, the Visual Studio C++ build tools and the Windows SDK. Its device checks also import the model package's `d3q`.

Run shell commands from this repository's root. Set `PYTHON` to the existing environment's interpreter, replacing the example path:

```sh
export PYTHON=/path/to/existing/environment/bin/python
```

The Python examples below share this model setup. Run them from the repository root so that `d3_varlen` is importable. Set `MODEL` to the checkpoint directory; adding it to `sys.path` makes the package's `d3q` and runtime modules importable.

```python
import os
import sys

# Configure ROCm allocation before importing torch through the model package.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")

MODEL = "models/d3-ultra-INT4"
sys.path.insert(0, MODEL)

import transformers.modeling_utils as modeling_utils
from d3q.load import load_d3

# Skip the loader warmup that ROCm rejects in this setup.
modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

model = load_d3(MODEL, device="cuda:0")
```

## Text decisions with VarlenDecider

`VarlenDecider` combines three mechanisms:

- **Unpadded prompt execution.** Prompt token sequences are concatenated for each execution chunk. Gated DeltaNet's `cu_seqlens`, convolution positions and per-prompt attention preserve request boundaries.
- **FP8 matrix products.** INT4 weights and activations are converted to `float8_e4m3fn` immediately before each product. hipBLASLt executes these products on `gfx1201`. The constructor checks FP8 execution on the device and uses BF16 if that check fails.
- **Prompt prefix caching.** Requests with the same questions share the prompt prefix before the request-specific state. When every state starts with the same text, `state_prefix` names that text and the shared prefix extends to its end. The computed prefix state is cached on the GPU and reused when the token prefix matches.

FP8 conversion reduces numerical precision. Prefix reuse also changes the computation: the cached prefix is computed without attending to the request-specific state that follows it. Both mechanisms are enabled by default. Use `VarlenDecider(model, exact=True)` to disable FP8 and cross-request prefix reuse while retaining unpadded prompt execution.

After loading `model` with the setup above:

```python
from d3_varlen import VarlenDecider

decider = VarlenDecider(model)
# To disable FP8 and cross-request prefix reuse:
# decider = VarlenDecider(model, exact=True)

question = {
    "type": "score",
    "instructions": "Does this document match the query?",
    "criteria": ["No", "Partly", "Yes"],
}
replies = decider.decide([
    {
        "state": "Query: forklift\n\nDocument:\nForklift safety guide",
        "questions": {"relevance": question},
    },
    {
        "state": "Query: forklift\n\nDocument:\nHeat illness prevention",
        "questions": {"relevance": question},
    },
])
print([reply["answers"]["relevance"]["score"] for reply in replies])

labels = decider.decide_questions({
    "state": "Query: forklift",
    "questions": {
        "object": {
            "type": "choice",
            "instructions": "What does the query ask about?",
            "criteria": {"forklift": "Forklift", "crane": "Crane", "none": "None of these"},
        },
        "accident": {
            "type": "choice",
            "instructions": "Which accident type?",
            "criteria": {"caught": "Caught in", "fall": "Fall", "none": "None of these"},
        },
    },
})
print(labels["answers"]["object"]["choice"])
```

| Method | Input | Result |
|---|---|---|
| `decide(batch)` | A list of requests, each with one question | Replies in input order |
| `decide_questions(request)` | One request containing several questions | One reply containing the question results |

`decide_questions` repeats only the state after the cached prefix for each question when that costs no more tokens than computing the prefix again. Otherwise it computes the state block once and only the part after it for each question.

Replies use the same structure as `model.system_one`. Serialize calls for each loaded model: `VarlenDecider` temporarily overrides layer methods while executing FP8 requests.

## Packed INT4 matrix products

The reference `Int4Linear` implementation dequantizes the entire weight matrix to BF16 on each call, then multiplies it by the input. At small input row counts, writing and reading the dequantized matrix dominates that path.

The [kernel library](kernels/int4_linear.hip) computes `y = x Wᵀ` from the packed weights. `w4a16_route` selects the implementation from the input rows, layer shape, input dtype and bias:

| Input rows | Implementation |
|---|---|
| 1 | Dot-product kernel. A wave takes one weight row; each lane reads one 32-column group per step, so the wave reads 512 contiguous bytes of the row. Lanes dequantize in registers, accumulate with the BF16 dot-product instruction and reduce at the end |
| 2–512 | Matrix kernel. A workgroup dequantizes 16-row weight tiles, 512 columns at a time, into shared memory: one tile below 48 input rows, two from 48 rows when the output width is a multiple of 32. Each wave multiplies a 16-row input tile with 16×16×16 BF16 matrix instructions |
| 48–128 | The matrix kernel with two weight tiles, reading ahead. A workgroup reads the next unit's packed weights while it multiplies the current unit, each wave reads its input before the products that use it, and barriers wait for shared memory only. Each wave takes three input tiles (48 rows) or two (49–128 rows), skipping a last tile with no rows, so each decoded weight fragment feeds several products; eight waves (48 rows) or four (49–128 rows) split each group's columns and add their sums at the end. When a weight row is a multiple of 8192 columns (of 2048 up to 64 rows), each workgroup starts at its own unit so that workgroups do not read the same memory channels at once. Dequantization goes through FP16 |
| Other | Not taken; the route returns the reason, and the caller runs its reference path |

Both kernels compute each weight as `(nibble - zero_point) * scale` rounded to BF16, as the reference layer does, and accumulate in FP32. Outputs differ from the reference layer only by accumulation order.

| Property | Requirement |
|---|---|
| Weight layout | compressed-tensors pack-quantized: packed `[out, in/8]` int32 with the first column in the lowest bits, scale `[out, in/32]` BF16, zero point `[out/8, in/32]` int32 or none for symmetric weights |
| Group size | 32 columns |
| Input width | A multiple of 512 |
| Output width | A multiple of 16 |
| Input dtype | BF16 |
| Bias | None |

### C ABI

[kernels/api.h](kernels/api.h) is the public C-compatible header. It defines every exported function, status, path, dtype and fallback reason without requiring HIP headers. The library reports **ABI version 2** through `w4a16_abi_version()`. Clients check that version before calling other functions. The developer record format remains schema 2.

| Function | Result |
|---|---|
| `w4a16_abi_version()` | ABI version |
| `w4a16_status_string(status)` | Static English description of a status |
| `w4a16_route(out, in, group, rows, dtype, bias)` | Positive kernel path, or a negative status explaining rejection |
| `w4a16_linear(packed, scale, zero, x, y, out, in, group, rows)` | Enqueues the product; returns `STATUS_OK` or a negative status |
| `w4a16_record_begin(path, every)` | Starts the recorder and opens its file |
| `w4a16_span_begin(name, rows, tokens)`, `w4a16_span_end()` | Opens and closes a caller-named span |
| `w4a16_op_begin(...)`, `w4a16_op_end()` | Records a caller-run fallback or dense head |
| `w4a16_record_end(&summary)` | Closes recording; returns a status and a library-owned JSON string |

Status values are shared across functions: `0` is success; `-1` through `-5` identify rows, dtype, bias, shape and launch failures; `-6` is an invalid argument, `-7` an invalid recorder state, `-8` file I/O failure and `-9` device initialization or synchronization failure. Route's positive values identify implementations. Callers may fall back for unsupported products; malformed calls require correction.

BF16 buffers use `uint16_t*` to express their stored bits. The header specifies each buffer's dimensions and alignment. Callers own the contiguous device storage, keep it alive until execution completes, and provide non-overlapping output storage. All GPU operations use the current device's **default stream**; callers explicitly order work on other streams. A successful product call means the launch was enqueued. Asynchronous GPU errors are checked when the caller synchronizes.

The recorder is process-global. While recording, use one host thread, keep the current device unchanged, and serialize library calls. Spans cannot nest. Nested products belong to the outer product; balance begin/end before ending a span. `record_end` closes an open span and returns text valid until the next recorder start. Copy that text before starting another recording.

[rdna4_w4a16.py](rdna4_w4a16.py) provides the Python `Library` binding and raises errors with the library's status description. Its declarations and enum values in `_rdna4_w4a16_abi.py` are generated from the public header by `tools/generate_binding.py`; its `--check` option reports declarations that no longer match the header. Distribute both Python files with the shared library. Dplus's build script copies all three files beside its executable.

### Developer mode

The recorder times each span on the device with HIP events and counts the products inside it by input rows M, inputs K, outputs N, dtype, implementation and fallback reason. Spans of one name form a phase. Per-product timings are sampled in one span of every `every` per phase, and the event overhead measured at startup is subtracted. On Linux, spans and timed products are also ROCTx ranges; the ROCm SDK for Windows holds no ROCTx.

The record file is JSONL: the run, one line per span, then the summary. [kernels/instrument.h](kernels/instrument.h), [kernels/instrument.cpp](kernels/instrument.cpp) and [record/src/spec.rs](record/src/spec.rs) define the format. The [record](record) crate reads a file and checks its summary against the totals of its spans; its program prints the phases and the products that took the most time:

```sh
cargo run --manifest-path record/Cargo.toml -- run.dev.jsonl
```

### Build

```sh
RDNA4_W4A16_PYTHON="$PYTHON" tools/build_kernels.sh
```

On Windows, in PowerShell:

```powershell
$env:RDNA4_W4A16_PYTHON = $PYTHON
tools\build_kernels.ps1
```

| Script | Work |
|---|---|
| [tools/build_kernels.sh](tools/build_kernels.sh) | Builds `dist/librdna4_w4a16.so` with the HIP compiler of the ROCm SDK in the selected environment. `RDNA4_W4A16_GPU_ARCH` selects the architecture; the default is `gfx1201` |
| [tools/build_kernels.ps1](tools/build_kernels.ps1) | Builds `dist/rdna4_w4a16.dll` and its import library on Windows in the same way. A process that loads the DLL needs the SDK's `bin` folder, which holds the HIP runtime and its DLLs, on its `PATH` |

## Measurements and experiment history

The recorded development environment was one AMD Radeon AI PRO R9700 with the `d3-ultra-INT4` package, ROCm 10 and torch 2.12. FP8 `float8_e4m3fn` products ran on `gfx1201` through hipBLASLt even though that architecture was not listed by the d3 package's FP8 path.

### Kernel comparison

[benchmark/compare.py](benchmark/compare.py) times this library against vLLM's `RDNAHybridW4A16` kernel, the reference `Int4Linear` and a BF16 product on dequantized weights, for the d3-ultra-INT4 layer shapes and 1–2048 input rows. Each implementation cycles through weight copies larger than the device caches, and errors are measured against a float64 product. [benchmark/report.py](benchmark/report.py) prints a result file from [benchmark/results](benchmark/results) as tables.

### Execution history

- The decision server of KOSHArXiv answers its reranking and query-label requests through `VarlenDecider`.
- The kernel library runs in the model worker of Dplus for generation decode steps and prefills and for decision passes of up to 512 rows. That use does not go through `VarlenDecider`.

## Runtime compatibility

`VarlenDecider` integrates with the d3 package's internal `d3_runtime.py`, `d3_fast.py` and `d3q/kernels.py`. Changes to those interfaces can require corresponding changes in this module.

## License

Apache-2.0. Parts are adapted from the d3 package; see [NOTICE](NOTICE).
