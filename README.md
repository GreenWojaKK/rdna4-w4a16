# d3-rdna4

ROCm kernels for running the [vLLM d3 decision model](https://huggingface.co/vllm-sr/d3-ultra-INT4) faster on RDNA 4.

The d3 package's fused kernels and its FP8 layers are gated to AMD Instinct GPUs (`gfx942`, `gfx950`). On RDNA 4 (`gfx1201`), each int4 projection unpacks to BF16 before a BF16 product, and requests run with padding. This module answers text requests through a different path.

## What it does

- **One flat row.** Prompts are joined without padding and computed in one forward. Boundaries are kept by the Gated DeltaNet's `cu_seqlens`, the conv positions and per-prompt attention.
- **FP8 products.** `float8_e4m3fn` matrix products run on `gfx1201` through hipBLASLt, although the package does not list it. int4 weights and activations are converted to FP8 right before each product. GPUs where this fails fall back to BF16.
- **Fixed text reuse.** The prompt text before the state is the same for every request with the same questions, so it is computed once and kept on the GPU.

FP8 and reuse are approximations: FP8 rounds harder, and reused text cannot read the state that follows it. `exact=True` turns both off and only joins the prompts.

Image, video, Box and Track requests are not handled here. Send those to `model.system_one`.

## Use

Put `d3_rdna4.py` on the path, along with the model folder, and install the model's `requirements.txt`, `triton` and `flash-linear-attention`.

```python
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")  # set before torch starts on ROCm

MODEL = "models/d3-ultra-INT4"
sys.path.insert(0, MODEL)

import transformers.modeling_utils as modeling_utils
from d3q.load import load_d3

modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None  # ROCm rejects this warmup

from d3_rdna4 import FlatDecider

model = load_d3(MODEL, device="cuda:0")
flat = FlatDecider(model)

question = {"type": "score", "instructions": "Does this document match the query?", "criteria": ["No", "Partly", "Yes"]}
replies = flat.decide([
    {"state": "Query: forklift\n\nDocument:\nForklift safety guide", "questions": {"relevance": question}},
    {"state": "Query: forklift\n\nDocument:\nHeat illness prevention", "questions": {"relevance": question}},
])
print([reply["answers"]["relevance"]["score"] for reply in replies])

labels = flat.decide_questions({"state": "Query: forklift", "questions": {
    "object": {"type": "choice", "instructions": "What does the query ask about?", "criteria": {"forklift": "Forklift", "crane": "Crane", "none": "None of these"}},
    "accident": {"type": "choice", "instructions": "Which accident type?", "criteria": {"caught": "Caught in", "fall": "Fall", "none": "None of these"}},
}})
print(labels["answers"]["object"]["choice"])
```

- `decide(batch)` takes a list of requests with one question each.
- `decide_questions(request)` takes one request with several questions.

Replies have the same shape as `model.system_one`. Run requests one at a time per loaded model: FP8 is switched on the model's layers while a request runs.

## Limits

- This module relies on the internals of the package's `d3_runtime.py`, `d3_fast.py` and `d3q/kernels.py`. A runtime change can break it.
- It was built on one AMD Radeon AI PRO R9700 with the `d3-ultra-INT4` package, ROCm 10 and torch 2.12.
- This standalone version has not been run yet. It was extracted from working code, and the fixed-text detection and class name changed in the process.

## License

Apache-2.0. Parts are adapted from the d3 package; see [NOTICE](NOTICE).
