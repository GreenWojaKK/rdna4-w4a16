# rdna4-w4a16

[English](../../README.md)

AMD RDNA 4 GPU용 W4A16 행렬곱 kernel과 개발자 모드 recorder, 비교 benchmark, 그리고 [d3-ultra-INT4 판정 모델](https://huggingface.co/vllm-sr/d3-ultra-INT4)의 ROCm 판정 경로다.

## 구성 요소

| 모듈 | 역할 | 인터페이스 |
|---|---|---|
| [d3_varlen.py](../../d3_varlen.py) | Padding 없는 prompt batch, FP8 행렬곱과 prompt prefix cache를 사용하는 텍스트 판정 실행 | `VarlenDecider` |
| [kernels](../../kernels) | 입력 1–512행의 packed INT4 × BF16 행렬곱과 개발자 모드 recorder | Linux의 `dist/librdna4_w4a16.so`, Windows의 `dist/rdna4_w4a16.dll`의 C ABI |
| [record](../../record) | 개발자 모드 기록을 읽고 검증한다 | Rust crate와 프로그램 `rdna4-w4a16-record` |
| [benchmark](../../benchmark) | W4A16 행렬곱 구현을 입력 행 수별로 측정한다 | `compare.py`, `report.py` |

`VarlenDecider`는 텍스트 요청을 처리한다. 이미지·비디오·Box·Track 요청은 `model.system_one`으로 전달한다.

이 작업에 사용한 d3 패키지는 AMD Instinct architecture인 `gfx942`와 `gfx950`에서 fused kernel과 FP8 layer를 활성화한다. RDNA 4(`gfx1201`) 경로는 INT4 projection을 BF16으로 dequantize한 뒤 곱하고, 요청에 padding을 적용한다. 이 저장소는 padding 없는 판정 실행과 packed 가중치에서 직접 계산하는 행렬곱을 제공한다.

## 환경과 모델 준비

ROCm PyTorch와 모델의 `requirements.txt`에 명시된 의존성이 설치된 기존 Python 환경을 사용한다. `VarlenDecider`에는 `triton`과 `flash-linear-attention`도 필요하다. Kernel 라이브러리를 빌드하려면 그 환경에 설치된 ROCm SDK가 필요하다. Linux에서는 C++20 host compiler, Windows에서는 Visual Studio C++ build tools와 Windows SDK도 필요하다. 장치 검사는 모델 패키지의 `d3q`도 import한다.

Shell 명령은 저장소 루트에서 실행한다. 아래 예시 경로를 기존 환경의 interpreter 경로로 바꿔 `PYTHON`을 설정한다.

```sh
export PYTHON=/path/to/existing/environment/bin/python
```

아래 Python 예제는 다음 모델 준비 코드를 공통으로 사용한다. `d3_varlen`을 import할 수 있도록 저장소 루트에서 실행한다. `MODEL`은 checkpoint 디렉터리로 지정한다. 이 경로를 `sys.path`에 추가하면 모델 패키지의 `d3q`와 runtime 모듈을 import할 수 있다.

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

## VarlenDecider로 텍스트 판정

`VarlenDecider`는 세 가지 방식을 결합한다.

- **Padding 없는 prompt 실행.** 실행 chunk마다 prompt의 token sequence를 연결한다. Gated DeltaNet의 `cu_seqlens`, convolution position과 prompt별 attention으로 요청 경계를 유지한다.
- **FP8 행렬곱.** 각 곱셈 직전에 INT4 가중치와 activation을 `float8_e4m3fn`으로 변환한다. `gfx1201`에서는 hipBLASLt로 이 행렬곱을 실행한다. 생성자는 해당 장치의 FP8 실행을 확인하며, 확인에 실패하면 BF16을 사용한다.
- **Prompt prefix cache.** 질문이 같은 요청은 요청별 state 앞의 prompt prefix를 공유한다. 모든 state가 같은 문구로 시작하면 `state_prefix`에 그 문구를 지정하고, 공유 prefix는 그 문구의 끝까지 늘어난다. 계산한 prefix state를 GPU에 저장하고 token prefix가 일치하면 재사용한다.

FP8 변환은 수치 정밀도를 낮춘다. Prefix 재사용도 계산을 바꾼다. Cache의 prefix는 뒤따르는 요청별 state에 대한 attention 없이 계산된다. 두 기능은 기본으로 활성화된다. `VarlenDecider(model, exact=True)`를 사용하면 padding 없는 prompt 실행을 유지하면서 FP8과 요청 간 prefix 재사용을 비활성화한다.

공통 준비 코드로 `model`을 로드한 뒤 다음과 같이 사용한다.

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

| 메서드 | 입력 | 결과 |
|---|---|---|
| `decide(batch)` | 질문이 하나씩 있는 요청의 목록 | 입력 순서의 응답 목록 |
| `decide_questions(request)` | 여러 질문이 있는 요청 하나 | 질문별 결과를 담은 응답 하나 |

`decide_questions`는 질문마다 cache한 prefix 뒤의 state만 반복하는 비용이 prefix를 다시 계산하는 token 수 이하이면 그렇게 한다. 그렇지 않으면 state block을 한 번 계산하고 질문마다 그 뒤의 부분만 계산한다.

응답 구조는 `model.system_one`과 같다. `VarlenDecider`는 FP8 요청 실행 중 layer method를 임시로 교체하므로, 로드한 모델 하나에 대한 호출은 직렬로 실행한다.

## Packed INT4 행렬곱

기준 `Int4Linear` 구현은 호출마다 가중치 행렬 전체를 BF16으로 dequantize한 뒤 입력과 곱한다. 입력 행 수가 작으면 dequantize한 행렬을 쓰고 읽는 시간이 이 경로의 대부분을 차지한다.

[Kernel 라이브러리](../../kernels/int4_linear.hip)는 packed 가중치에서 `y = x Wᵀ`를 계산한다. `w4a16_route`는 입력 행 수, layer shape, 입력 dtype과 bias로 구현을 고른다.

| 입력 행 수 | 구현 |
|---|---|
| 1 | Dot-product kernel. Wave 하나가 가중치 한 행을 맡고, 각 lane이 step마다 32열 group 하나를 읽어 wave가 그 행의 연속 512바이트를 읽는다. Lane은 register에서 dequantize하고 BF16 dot-product instruction으로 누산한 뒤 마지막에 합산한다 |
| 2–512 | Matrix kernel. Workgroup이 16행 weight tile을 512열씩 공유 메모리로 dequantize한다. 입력 48행 미만에서는 tile 하나, 48행부터는 출력 폭이 32의 배수일 때 tile 둘이다. 각 wave는 16행 입력 tile을 16×16×16 BF16 matrix instruction으로 곱한다 |
| 48–128 | Tile 둘을 쓰는 matrix kernel이 미리 읽는다. Workgroup은 현재 unit을 곱하는 동안 다음 unit의 packed 가중치를 읽고, 각 wave는 입력을 그 입력을 쓰는 곱셈보다 먼저 읽으며, barrier는 공유 메모리만 기다린다. Wave 하나가 입력 tile 셋(48행)이나 둘(49~128행)을 맡고 행이 없는 마지막 tile은 건너뛰어, decode한 가중치 조각 하나를 여러 곱셈에 쓴다. Wave 여덟(48행)이나 넷(49~128행)이 묶음의 열을 나눠 맡고 마지막에 합을 더한다. 가중치 한 행이 8192열의 배수(64행까지는 2048열의 배수)면, workgroup들이 같은 메모리 channel을 동시에 읽지 않도록 workgroup마다 다른 unit에서 시작한다. Dequantize는 FP16을 거친다 |
| 그 밖 | 사용하지 않는다. Route가 사유를 반환하고 호출자가 기준 경로를 실행한다 |

두 kernel 모두 기준 layer와 같이 각 가중치를 `(nibble - zero_point) * scale`로 계산해 BF16으로 반올림하고 FP32로 누산한다. 출력은 누산 순서에 따른 차이만 있다.

| 항목 | 조건 |
|---|---|
| Weight 배치 | compressed-tensors pack-quantized: packed `[out, in/8]` int32(첫 열이 가장 낮은 bit), scale `[out, in/32]` BF16, zero point `[out/8, in/32]` int32 또는 대칭 가중치에서는 없음 |
| Group size | 32열 |
| 입력 너비 | 512의 배수 |
| 출력 너비 | 16의 배수 |
| 입력 dtype | BF16 |
| Bias | 없음 |

### C ABI

[kernels/api.h](../../kernels/api.h)는 HIP 헤더 없이 사용할 수 있는 공개 C 헤더다. 모든 export 함수, status, 실행 경로, dtype과 fallback 사유를 정의한다. 라이브러리는 `w4a16_abi_version()`으로 **ABI version 2**를 반환한다. 호출자는 다른 함수를 쓰기 전에 이 버전을 확인한다. 개발자 기록 형식은 schema 2를 유지한다.

| 함수 | 결과 |
|---|---|
| `w4a16_abi_version()` | ABI 버전 |
| `w4a16_status_string(status)` | 정적 영어 status 설명 |
| `w4a16_route(out, in, group, rows, dtype, bias)` | 양수 kernel 경로 또는 거절 사유를 나타내는 음수 status |
| `w4a16_linear(packed, scale, zero, x, y, out, in, group, rows)` | 행렬곱을 enqueue하고 `STATUS_OK` 또는 음수 status 반환 |
| `w4a16_record_begin(path, every)` | Recorder 시작 및 기록 파일 열기 |
| `w4a16_span_begin(name, rows, tokens)`, `w4a16_span_end()` | 호출자가 이름 붙인 span 시작·종료 |
| `w4a16_op_begin(...)`, `w4a16_op_end()` | 호출자가 실행한 fallback 또는 dense head 기록 |
| `w4a16_record_end(&summary)` | 기록을 종료하고 status와 라이브러리 소유 JSON 문자열 반환 |

모든 함수가 같은 status 값을 쓴다. `0`은 성공, `-1`부터 `-5`는 rows·dtype·bias·shape·launch 실패, `-6`은 잘못된 인자, `-7`은 잘못된 recorder 상태, `-8`은 파일 I/O 실패, `-9`는 장치 초기화 또는 동기화 실패다. Route의 양수는 구현을 나타낸다. 지원하지 않는 행렬곱은 호출자가 fallback할 수 있으며, 잘못된 호출은 인자를 수정해야 한다.

BF16 버퍼는 저장된 비트를 나타내는 `uint16_t*`로 선언한다. 각 버퍼의 크기와 정렬 조건은 헤더에 명시한다. 호출자는 연속된 device storage를 소유하고 실행이 끝날 때까지 유지하며, 출력 버퍼는 입력과 겹치지 않게 제공한다. 모든 GPU 연산은 현재 장치의 **default stream**을 사용한다. 다른 stream과의 실행 순서는 호출자가 명시적으로 연결한다. 행렬곱 호출 성공은 launch가 enqueue됐음을 나타내며, 비동기 GPU 오류는 호출자가 동기화할 때 확인한다.

Recorder는 프로세스 전체에서 하나다. 기록 중에는 한 host thread에서 호출을 직렬화하고 현재 장치를 유지한다. Span은 중첩하지 않는다. 중첩된 product는 바깥 product에 포함되며, span 종료 전에 begin/end를 맞춘다. `record_end`는 열린 span을 닫고 다음 recorder 시작까지 유효한 문자열을 반환한다. 다음 기록을 시작하기 전에 문자열을 복사한다.

[rdna4_w4a16.py](../../rdna4_w4a16.py)의 Python `Library` binding은 라이브러리의 status 설명을 담은 오류를 발생시킨다. `_rdna4_w4a16_abi.py`의 선언과 enum 값은 `tools/generate_binding.py`가 공개 헤더에서 생성하며, 그 `--check` 옵션은 헤더와 맞지 않는 선언을 알린다. 두 Python 파일을 공유 라이브러리와 함께 배포한다. Dplus의 빌드 스크립트는 세 파일을 실행 파일 옆에 복사한다.

### 개발자 모드

Recorder는 span마다 HIP event로 장치 시간을 재고, 그 안의 곱셈을 입력 행 수 M, 입력 크기 K, 출력 크기 N, dtype, 구현과 fallback 사유별로 센다. 같은 이름의 span이 하나의 phase를 이룬다. 곱셈별 시간은 phase마다 `every`개 span 중 하나에서 재고, 시작할 때 측정한 event overhead를 뺀다. Linux에서는 span과 시간을 잰 곱셈을 ROCTx range로도 표시한다. Windows용 ROCm SDK에는 ROCTx가 없다.

기록 파일은 JSONL이다. 실행 정보, span마다 한 줄, 마지막에 요약이 온다. [kernels/instrument.h](../../kernels/instrument.h), [kernels/instrument.cpp](../../kernels/instrument.cpp), [record/src/spec.rs](../../record/src/spec.rs)가 형식을 정의한다. [record](../../record) crate는 기록 파일을 읽고 요약을 span 합계와 대조하며, 프로그램은 phase와 시간이 가장 많이 든 곱셈을 출력한다.

```sh
cargo run --manifest-path record/Cargo.toml -- run.dev.jsonl
```

### 빌드

```sh
RDNA4_W4A16_PYTHON="$PYTHON" tools/build_kernels.sh
```

Windows에서는 PowerShell에서 실행한다.

```powershell
$env:RDNA4_W4A16_PYTHON = $PYTHON
tools\build_kernels.ps1
```

| 스크립트 | 작업 |
|---|---|
| [tools/build_kernels.sh](../../tools/build_kernels.sh) | 선택한 환경의 ROCm SDK HIP compiler로 `dist/librdna4_w4a16.so`를 빌드한다. `RDNA4_W4A16_GPU_ARCH`로 architecture를 지정하며 기본값은 `gfx1201`이다 |
| [tools/build_kernels.ps1](../../tools/build_kernels.ps1) | Windows에서 같은 방식으로 `dist/rdna4_w4a16.dll`과 import library를 빌드한다. DLL을 로드하는 프로세스는 HIP runtime과 그 DLL이 있는 SDK의 `bin` 폴더를 `PATH`에 둔다 |

## 측정과 실험 경위

기록된 개발 환경은 AMD Radeon AI PRO R9700 한 대, `d3-ultra-INT4` 패키지, ROCm 10, torch 2.12다. d3 패키지의 FP8 경로에 `gfx1201`이 명시돼 있지 않았지만, 이 architecture에서 hipBLASLt를 통한 `float8_e4m3fn` 행렬곱을 실행했다.

### Kernel 비교

[benchmark/compare.py](../../benchmark/compare.py)는 d3-ultra-INT4 layer shape과 입력 1–2048행에서 이 라이브러리를 vLLM의 `RDNAHybridW4A16` kernel, 기준 `Int4Linear`, dequantize해 둔 가중치의 BF16 곱셈과 비교한다. 각 구현은 장치 cache보다 큰 가중치 사본을 번갈아 사용하고, 오차는 float64 곱셈과 비교한다. [benchmark/report.py](../../benchmark/report.py)는 [benchmark/results](../../benchmark/results)의 결과 파일을 표로 출력한다.

### 실행 이력

- KOSHArXiv의 판정 서버는 검색 결과 재정렬과 검색어 분류 요청을 `VarlenDecider`로 처리한다.
- Kernel 라이브러리는 Dplus의 model worker에서 generation의 decode step과 prefill, 512행 이하의 판정 pass에 쓰인다. 이 사용은 `VarlenDecider`를 거치지 않는다.

## Runtime 호환성

`VarlenDecider`는 d3 패키지 내부의 `d3_runtime.py`, `d3_fast.py`, `d3q/kernels.py`와 연동한다. 이 인터페이스가 변경되면 해당 모듈도 이에 맞게 수정해야 할 수 있다.

## 라이선스

Apache-2.0. 일부 코드는 d3 패키지를 바탕으로 작성했다. [NOTICE](../../NOTICE)를 참조한다.
