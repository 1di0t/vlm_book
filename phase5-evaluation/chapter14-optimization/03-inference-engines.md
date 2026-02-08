---
---

# 14.3 추론 엔진 (Inference Engines)

> **LLM을 프로덕션에서 효율적으로 서빙하기 위한 최적화된 추론 프레임워크**

---

## 핵심 용어

| 용어 | 정의 |
|------|------|
| **vLLM** | PagedAttention 기반 고처리량 LLM 서빙 엔진. UC Berkeley 개발 |
| **SGLang** | RadixAttention + 구조적 생성 프로그래밍 인터페이스 제공하는 서빙 엔진 |
| **TensorRT-LLM** | NVIDIA의 LLM 전용 추론 최적화 라이브러리. 커널 퓨전, 양자화 내장 |
| **llama.cpp** | C/C++ 기반 경량 LLM 추론 엔진. CPU/Metal/CUDA 지원 |
| **ONNX Runtime** | Microsoft의 범용 ML 추론 엔진. ONNX 포맷 모델 실행 |
| **Continuous Batching** | 요청 완료 시 즉시 교체하는 동적 배칭 기법 |
| **Kernel Fusion** | 여러 GPU 연산을 하나의 커널로 합쳐 오버헤드를 줄이는 최적화 |
| **RadixAttention** | Radix Tree로 KV Cache prefix를 관리하여 재사용을 극대화하는 기법 |
| **GGUF** | llama.cpp에서 사용하는 양자화 모델 포맷 (GGML Universal Format) |
| **Speculative Decoding** | 작은 모델이 초안을 생성하고 큰 모델이 검증하여 속도를 높이는 기법 |

---

## 각 엔진의 핵심 최적화

### 1. vLLM

vLLM은 **PagedAttention**을 핵심으로 한 고처리량 서빙 엔진이다.

**핵심 최적화 기법**:

| 기법 | 설명 |
|------|------|
| PagedAttention | 비연속 KV Cache 관리, 메모리 낭비 60~80% 제거 |
| Continuous Batching | 요청 완료 시 즉시 새 요청 삽입 |
| Prefix Caching | 공통 prefix KV 재사용 (APC: Automatic Prefix Caching) |
| Chunked Prefill | 긴 프롬프트를 청크 단위 처리, decode와 인터리빙 |
| Tensor Parallelism | 다중 GPU에 모델 분산 |
| Speculative Decoding | Draft 모델로 초안 → Target 모델로 검증 |
| KV Cache Quantization | FP8/INT8 KV Cache로 메모리 50% 절약 |
| CUDA Graph | GPU 커널 런치 오버헤드 제거 |

**아키텍처**:

```
Client Request
    ↓
Scheduler (Continuous Batching)
    ↓
Tokenizer → Prefill/Decode → Detokenizer
    ↓               ↓
KV Cache Manager (PagedAttention)
    ↓
GPU Worker(s) [Tensor Parallel]
```

**장점**: 처리량(throughput) 최강. 대규모 배치 서빙에 최적.
**단점**: 단일 요청 지연시간은 최소가 아닐 수 있음. NVIDIA GPU 필수.

### 2. SGLang

SGLang은 **RadixAttention**과 **프로그래밍 인터페이스**를 결합한 서빙 엔진이다.

**핵심 최적화 기법**:

| 기법 | 설명 |
|------|------|
| RadixAttention | Radix Tree로 모든 가능한 prefix의 KV Cache를 자동 관리 |
| 구조적 생성 | JSON, 정규식 기반 출력 제약 (Constrained Decoding) |
| Zero-Overhead Batching | 스케줄링 오버헤드 최소화 |
| FlashInfer Backend | 고효율 attention 커널 |
| Cache-Aware Scheduling | KV Cache 재사용을 극대화하는 스케줄링 |

**RadixAttention vs Prefix Caching**:

vLLM의 Prefix Caching은 **정확히 일치하는 prefix만** 재사용한다. SGLang의 RadixAttention은 Radix Tree(압축 트라이)로 **부분 prefix**도 자동으로 매칭하고 재사용한다.

```
Radix Tree 예시:

Root
├── "You are a medical AI assistant..." (시스템 프롬프트)
│   ├── "다음 처방전을 분석해..." (태스크 A)
│   │   ├── "처방전 이미지: [img1]" → KV_A1
│   │   └── "처방전 이미지: [img2]" → KV_A2
│   └── "다음 검사 결과를 정리해..." (태스크 B)
│       └── "검사지 이미지: [img3]" → KV_B1
```

모든 요청이 시스템 프롬프트 KV를 공유하고, 같은 태스크 유형이면 태스크 프롬프트 KV까지 공유한다.

**장점**: KV Cache 재사용률 최고. 구조적 출력(JSON 등) 생성 성능 우수.
**단점**: 생태계가 vLLM보다 작음.

### 3. TensorRT-LLM

NVIDIA의 공식 LLM 추론 최적화 라이브러리.

**핵심 최적화 기법**:

| 기법 | 설명 |
|------|------|
| Kernel Fusion | Multi-Head Attention, LayerNorm + Add 등 커널 결합 |
| FP8/INT8/INT4 양자화 | SmoothQuant, AWQ, GPTQ 통합 지원 |
| In-Flight Batching | Continuous Batching의 NVIDIA 구현 |
| KV Cache Reuse | 반복 prefix KV 재사용 |
| Paged KV Cache | vLLM과 유사한 페이지 기반 관리 |
| Multi-GPU | Tensor/Pipeline Parallelism |
| Custom CUDA Kernels | H100/A100 최적화 커널 |

**Kernel Fusion 예시**:

일반적인 Transformer 레이어에서:

```
# 퓨전 전 (6개 커널 런치)
x = layer_norm(x)           # 커널 1
q = linear_q(x)             # 커널 2
k = linear_k(x)             # 커널 3
v = linear_v(x)             # 커널 4
attn = attention(q, k, v)   # 커널 5
out = linear_out(attn)      # 커널 6

# 퓨전 후 (2개 커널 런치)
q, k, v = fused_qkv(layer_norm(x))  # 커널 1 (LN + QKV 결합)
out = fused_attention_output(q, k, v) # 커널 2 (Attention + Output 결합)
```

커널 런치 횟수 감소 → GPU idle 시간 감소 → 지연시간 개선.

**장점**: 단일 요청 지연시간 최소. NVIDIA 하드웨어에서 최고 성능.
**단점**: NVIDIA 전용. 빌드 복잡. 모델 지원 범위가 제한적.

### 4. llama.cpp

C/C++ 구현의 경량 추론 엔진. **CPU 추론** 및 **개인 PC** 환경에 최적화.

**핵심 최적화 기법**:

| 기법 | 설명 |
|------|------|
| GGUF 양자화 | Q2_K ~ Q8_0까지 다양한 양자화 포맷 |
| SIMD 최적화 | AVX2/AVX-512/NEON 벡터 연산 |
| mmap | 메모리 매핑으로 빠른 모델 로드 |
| Metal/CUDA/Vulkan | 다양한 GPU 백엔드 지원 |
| KV Cache Quantization | Q4_0/Q8_0 KV Cache |
| Grammar-based Sampling | BNF 문법 기반 출력 제약 |

**GGUF 양자화 포맷 비교**:

| 포맷 | bits/weight | 상대 품질 | 7B 모델 크기 |
|------|------------|----------|-------------|
| Q2_K | 2.6 | 낮음 | ~2.7 GB |
| Q3_K_M | 3.4 | 보통 | ~3.5 GB |
| Q4_K_M | 4.8 | 좋음 | ~4.4 GB |
| Q5_K_M | 5.7 | 매우 좋음 | ~5.1 GB |
| Q6_K | 6.6 | 우수 | ~5.9 GB |
| Q8_0 | 8.0 | 거의 무손실 | ~7.2 GB |
| F16 | 16.0 | 무손실 | ~14.0 GB |

**장점**: CPU만으로 추론 가능. 설치 매우 간단. 다양한 플랫폼(Windows/Mac/Linux).
**단점**: GPU 서빙 성능은 vLLM/TensorRT-LLM에 미치지 못함.

### 5. ONNX Runtime

Microsoft의 범용 ML 추론 엔진. LLM 전용은 아니지만 ONNX 포맷을 통한 범용성이 강점.

**핵심 최적화 기법**:

| 기법 | 설명 |
|------|------|
| Graph Optimization | 연산 그래프 수준 최적화 (불필요 노드 제거, 퓨전) |
| Execution Provider | CUDA, TensorRT, DirectML, OpenVINO 등 다양한 백엔드 |
| Quantization | Dynamic/Static INT8 양자화 |
| IO Binding | 호스트-디바이스 데이터 전송 최소화 |

**장점**: 하드웨어 중립. ONNX 변환만 하면 어디서든 실행.
**단점**: LLM 특화 최적화(PagedAttention 등)가 없어 처리량 한계.

---

## 수학적 원리

### 추론 지연시간 분해

LLM 추론의 총 지연시간:

$$
T_{\text{total}} = T_{\text{prefill}} + (n_{\text{output}} - 1) \times T_{\text{decode}}
$$

**Prefill** (Compute-bound):

$$
T_{\text{prefill}} = \frac{2 \times N_{\text{params}} \times n_{\text{input}}}{\text{GPU FLOPS}}
$$

**Decode** (Memory-bound):

$$
T_{\text{decode}} = \frac{2 \times N_{\text{params}} \times b_{\text{param}}}{\text{Memory Bandwidth}}
$$

여기서 $b_{\text{param}}$은 파라미터당 바이트 수.

### Arithmetic Intensity 분석

$$
\text{AI} = \frac{\text{FLOPs}}{\text{Bytes Transferred}}
$$

- **Prefill**: $\text{AI} \propto n_{\text{input}}$ → 시퀀스가 길수록 compute-bound
- **Decode**: $\text{AI} \approx 1$ → 항상 memory-bound

이 차이가 추론 엔진 최적화의 핵심이다:
- Prefill → 커널 퓨전, Tensor Parallelism으로 연산 처리량 극대화
- Decode → 메모리 대역폭 활용 극대화, 배칭으로 AI 증가

### Continuous Batching의 처리량 개선

Static Batching:

$$
\text{Throughput}_{\text{static}} = \frac{B}{\max_i(T_i)}
$$

모든 요청이 가장 긴 요청이 끝날 때까지 대기.

Continuous Batching:

$$
\text{Throughput}_{\text{continuous}} = \frac{B}{\text{avg}(T_i)} \times \eta
$$

$\eta$는 스케줄링 효율 (보통 0.85~0.95). 짧은 요청이 빨리 빠지고 새 요청이 즉시 들어온다.

**처리량 개선**: Static 대비 **2~5x** 향상.

### Speculative Decoding

Draft 모델(작음) $M_d$가 $k$개 토큰 초안 생성 → Target 모델(큼) $M_t$가 한 번에 검증:

$$
\text{Speedup} = \frac{k + 1}{1 + k \cdot \frac{T_{M_d}}{T_{M_t}}} \times \alpha
$$

$\alpha$는 수용률(acceptance rate). Draft 모델의 예측이 Target과 일치하는 비율.

$\alpha = 0.8$, $k = 5$, $T_{M_d}/T_{M_t} = 0.1$인 경우:

$$
\text{Speedup} = \frac{6}{1.5} \times 0.8 = 3.2\times
$$

---

## 추론 엔진 비교

| 항목 | vLLM | SGLang | TensorRT-LLM | llama.cpp | ONNX Runtime |
|------|------|--------|-------------|-----------|-------------|
| **주요 목표** | 고처리량 서빙 | KV 재사용 + 구조적 생성 | 최저 지연시간 | 경량/CPU 추론 | 범용 추론 |
| **핵심 기법** | PagedAttention | RadixAttention | 커널 퓨전 | GGUF 양자화 | 그래프 최적화 |
| **배칭** | Continuous | Continuous | In-Flight | 제한적 | Static |
| **처리량** | 최고 수준 | 최고 수준 | 높음 | 낮음 | 보통 |
| **단일 요청 지연** | 보통 | 보통 | 최저 | 보통 | 보통 |
| **GPU 필수** | Yes (NVIDIA) | Yes (NVIDIA) | Yes (NVIDIA) | No (CPU 가능) | No |
| **양자화 지원** | AWQ/GPTQ/FP8 | AWQ/GPTQ/FP8 | FP8/INT8/INT4 | GGUF (Q2~Q8) | INT8 |
| **모델 포맷** | HuggingFace | HuggingFace | TensorRT 엔진 | GGUF | ONNX |
| **설치 난이도** | 쉬움 | 쉬움 | 어려움 | 매우 쉬움 | 보통 |
| **API 호환** | OpenAI API | OpenAI API | Triton | 자체 API | 자체 API |
| **Multi-GPU** | TP/PP | TP/PP | TP/PP | 제한적 | 제한적 |
| **구조적 출력** | 지원 | 최고 수준 | 제한적 | Grammar | 미지원 |
| **개발 활발도** | 매우 활발 | 활발 | 활발 | 매우 활발 | 안정적 |

### 의료 문서 OCR에서의 엔진 선택

| 시나리오 | 추천 엔진 | 이유 |
|---------|----------|------|
| 대규모 배치 처리 (수천 건/일) | vLLM 또는 SGLang | 처리량 최우선 |
| 동일 양식 반복 처리 | SGLang | RadixAttention으로 KV 재사용 극대화 |
| 실시간 단건 (진료실 OCR) | TensorRT-LLM | 최저 지연시간 |
| 엣지/로컬 (병원 내부 서버) | llama.cpp | GPU 없이도 가동, 보안 |
| JSON 구조화 출력 필수 | SGLang | Constrained Decoding 최고 성능 |
| 다양한 하드웨어 지원 | ONNX Runtime | CPU/Intel GPU/DirectML |

---

## 코드

### 1. vLLM 서빙 설정

```python
"""
vLLM을 사용한 의료 문서 OCR 모델 서빙
- OpenAI 호환 API 서버
- AWQ 양자화 모델 로드
- Prefix Caching 활성화
"""

import os
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VLLMServerConfig:
    """vLLM 서버 설정"""
    model: str
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 64
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 8192
    quantization: Optional[str] = None    # "awq", "gptq", None
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"          # "fp8_e5m2" for FP8 KV
    enforce_eager: bool = False
    disable_log_requests: bool = True

    def to_cli_args(self) -> str:
        """CLI 인수 문자열 생성"""
        args = [
            f"--model {self.model}",
            f"--host {self.host}",
            f"--port {self.port}",
            f"--tensor-parallel-size {self.tensor_parallel_size}",
            f"--max-model-len {self.max_model_len}",
            f"--gpu-memory-utilization {self.gpu_memory_utilization}",
            f"--max-num-seqs {self.max_num_seqs}",
            f"--dtype {self.dtype}",
            f"--kv-cache-dtype {self.kv_cache_dtype}",
        ]

        if self.enable_prefix_caching:
            args.append("--enable-prefix-caching")
        if self.enable_chunked_prefill:
            args.append("--enable-chunked-prefill")
            args.append(f"--max-num-batched-tokens {self.max_num_batched_tokens}")
        if self.quantization:
            args.append(f"--quantization {self.quantization}")
        if self.enforce_eager:
            args.append("--enforce-eager")
        if self.disable_log_requests:
            args.append("--disable-log-requests")

        return " ".join(args)


def generate_vllm_launch_script(config: VLLMServerConfig) -> str:
    """vLLM 서버 실행 스크립트 생성"""
    return f"""#!/bin/bash
# vLLM 서버 실행 스크립트
# 의료 문서 OCR 모델 서빙용

export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

python -m vllm.entrypoints.openai.api_server \\
    {config.to_cli_args()}
"""


# === vLLM Python API 사용 예시 ===

def vllm_offline_inference_example():
    """vLLM 오프라인 추론 (배치 처리)"""
    from vllm import LLM, SamplingParams

    # 모델 로드 (AWQ 양자화)
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        quantization="awq",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
    )

    # 샘플링 설정
    sampling_params = SamplingParams(
        temperature=0.0,        # Greedy (의료 문서는 결정론적 생성)
        max_tokens=2048,
        top_p=1.0,
        repetition_penalty=1.05,
    )

    # 의료 문서 OCR 프롬프트
    system_prompt = (
        "당신은 의료 문서 OCR 전문가입니다. "
        "이미지에서 추출한 텍스트를 구조화된 JSON으로 변환하세요. "
        "약물명, 용량, 투여 방법은 정확히 기재하세요."
    )

    prompts = [
        f"[System] {system_prompt}\n[User] 다음 처방전을 분석해:\n처방 내용: Metformin 500mg 1일 2회 식후",
        f"[System] {system_prompt}\n[User] 다음 검사 결과를 정리해:\nWBC 6500, RBC 480만, Hb 14.2",
        f"[System] {system_prompt}\n[User] 다음 진단서를 분석해:\n진단명: 제2형 당뇨병 (E11.9)",
    ]

    # 배치 추론 (Prefix Caching 자동 적용)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt[:80]
        generated = output.outputs[0].text
        logger.info(f"Prompt: {prompt}...")
        logger.info(f"Output: {generated[:200]}")
        logger.info(f"Tokens: {len(output.outputs[0].token_ids)}")
        logger.info("---")


# === OpenAI 호환 클라이언트 ===

async def vllm_api_client_example():
    """vLLM OpenAI 호환 API 클라이언트"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",  # vLLM은 기본적으로 API 키 불필요
    )

    response = await client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        messages=[
            {
                "role": "system",
                "content": "의료 문서 OCR 전문가. JSON 출력.",
            },
            {
                "role": "user",
                "content": "처방전: Aspirin 100mg 1일 1회 조식후",
            },
        ],
        temperature=0.0,
        max_tokens=1024,
    )

    print(response.choices[0].message.content)


# 서버 설정 생성
if __name__ == "__main__":
    # 의료 문서 OCR용 vLLM 설정
    config = VLLMServerConfig(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        max_num_seqs=64,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        quantization="awq",
        kv_cache_dtype="auto",
    )

    print("=== vLLM 서버 실행 스크립트 ===")
    print(generate_vllm_launch_script(config))
    print()
    print(f"=== CLI 인수 ===")
    print(f"python -m vllm.entrypoints.openai.api_server {config.to_cli_args()}")
```

### 2. SGLang 서빙 설정

```python
"""
SGLang을 사용한 의료 문서 OCR 모델 서빙
- RadixAttention 기반 KV Cache 재사용
- 구조적 출력(JSON) 생성
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SGLangServerConfig:
    """SGLang 서버 설정"""
    model: str
    host: str = "0.0.0.0"
    port: int = 30000
    tp_size: int = 1                    # Tensor Parallelism
    mem_fraction_static: float = 0.88   # GPU 메모리 할당 비율
    max_total_tokens: int = 65536       # 최대 토큰 풀 크기
    context_length: int = 8192
    chunked_prefill_size: int = 4096
    schedule_policy: str = "lpm"        # lpm: longest prefix match
    quantization: Optional[str] = None  # "awq", "gptq"
    disable_radix_cache: bool = False
    enable_torch_compile: bool = False

    def to_cli_args(self) -> str:
        args = [
            f"--model-path {self.model}",
            f"--host {self.host}",
            f"--port {self.port}",
            f"--tp-size {self.tp_size}",
            f"--mem-fraction-static {self.mem_fraction_static}",
            f"--max-total-tokens {self.max_total_tokens}",
            f"--context-length {self.context_length}",
            f"--chunked-prefill-size {self.chunked_prefill_size}",
            f"--schedule-policy {self.schedule_policy}",
        ]

        if self.quantization:
            args.append(f"--quantization {self.quantization}")
        if self.disable_radix_cache:
            args.append("--disable-radix-cache")
        if self.enable_torch_compile:
            args.append("--enable-torch-compile")

        return " \\\n    ".join(args)


def generate_sglang_launch_script(config: SGLangServerConfig) -> str:
    """SGLang 서버 실행 스크립트"""
    return f"""#!/bin/bash
# SGLang 서버 실행 스크립트
# RadixAttention 기반 의료 문서 OCR 서빙

export CUDA_VISIBLE_DEVICES=0

python -m sglang.launch_server \\
    {config.to_cli_args()}
"""


# === SGLang 프론트엔드 프로그래밍 ===

def sglang_frontend_example():
    """
    SGLang 프론트엔드 언어를 활용한 구조적 생성
    - 의료 문서 OCR에서 JSON 구조로 출력
    """
    import sglang as sgl

    @sgl.function
    def medical_ocr_extract(s, document_text: str):
        """의료 문서에서 구조화된 정보 추출"""
        s += sgl.system(
            "당신은 의료 문서 OCR 전문가입니다. "
            "문서에서 정보를 추출하여 정확한 JSON으로 반환하세요."
        )
        s += sgl.user(
            f"다음 문서에서 정보를 추출해:\n\n{document_text}\n\n"
            "JSON 형식으로 반환해."
        )
        s += sgl.assistant(
            sgl.gen(
                "extraction",
                max_tokens=1024,
                temperature=0.0,
                regex=r'\{[^}]+\}',   # JSON 객체 형태 제약
            )
        )

    @sgl.function
    def medical_ocr_batch(s, documents: list):
        """여러 문서를 순차적으로 처리 (KV Cache 재사용 극대화)"""
        s += sgl.system(
            "당신은 의료 문서 OCR 전문가입니다. "
            "각 문서에서 환자명, 진단명, 처방 정보를 추출하세요."
        )
        # 동일 system prompt → RadixAttention이 KV를 자동 재사용
        for i, doc in enumerate(documents):
            s += sgl.user(f"문서 {i+1}:\n{doc}")
            s += sgl.assistant(
                sgl.gen(
                    f"result_{i}",
                    max_tokens=512,
                    temperature=0.0,
                )
            )

    # 실행
    runtime = sgl.Runtime(
        model_path="Qwen/Qwen2.5-7B-Instruct-AWQ",
        tp_size=1,
    )
    sgl.set_default_backend(runtime)

    # 단건 처리
    result = medical_ocr_extract.run(
        document_text="처방전\n환자: 김철수\n진단: 고혈압\n처방: Amlodipine 5mg 1일 1회"
    )
    print("추출 결과:", result["extraction"])

    runtime.shutdown()


# === OpenAI 호환 API 클라이언트 (JSON 모드) ===

async def sglang_json_output_example():
    """SGLang의 JSON 구조적 출력"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="http://localhost:30000/v1",
        api_key="not-needed",
    )

    # JSON Schema 정의
    medical_schema = {
        "type": "object",
        "properties": {
            "patient_name": {"type": "string"},
            "diagnosis": {"type": "string"},
            "diagnosis_code": {"type": "string"},
            "medications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "dose": {"type": "string"},
                        "frequency": {"type": "string"},
                    },
                    "required": ["name", "dose", "frequency"],
                },
            },
        },
        "required": ["patient_name", "diagnosis", "medications"],
    }

    response = await client.chat.completions.create(
        model="default",
        messages=[
            {
                "role": "system",
                "content": "의료 문서에서 정보를 추출하여 JSON으로 반환.",
            },
            {
                "role": "user",
                "content": (
                    "처방전 내용:\n"
                    "환자명: 박영희\n"
                    "진단: 제2형 당뇨병 (E11.9)\n"
                    "처방:\n"
                    "1. Metformin 500mg 1일 2회 식후\n"
                    "2. Glimepiride 2mg 1일 1회 조식전"
                ),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "medical_extraction",
                "schema": medical_schema,
            },
        },
        temperature=0.0,
        max_tokens=512,
    )

    result = json.loads(response.choices[0].message.content)
    print(json.dumps(result, indent=2, ensure_ascii=False))


# 설정 생성
if __name__ == "__main__":
    config = SGLangServerConfig(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        tp_size=1,
        mem_fraction_static=0.88,
        context_length=8192,
        chunked_prefill_size=4096,
        schedule_policy="lpm",  # Longest Prefix Match → RadixAttention 극대화
        quantization="awq",
    )

    print("=== SGLang 서버 실행 스크립트 ===")
    print(generate_sglang_launch_script(config))
```

### 3. 엔진 선택 가이드 유틸리티

```python
"""
추론 엔진 선택 가이드
- 워크로드 특성에 따른 최적 엔진 추천
"""

from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum


class Priority(Enum):
    THROUGHPUT = "throughput"           # 처리량 우선
    LATENCY = "latency"                # 지연시간 우선
    COST = "cost"                      # 비용 우선
    EDGE = "edge"                      # 엣지/로컬 배포
    STRUCTURED_OUTPUT = "structured"   # 구조적 출력


@dataclass
class WorkloadProfile:
    """워크로드 프로파일"""
    avg_input_tokens: int           # 평균 입력 토큰 수
    avg_output_tokens: int          # 평균 출력 토큰 수
    requests_per_minute: int        # 분당 요청 수
    shared_prefix_ratio: float      # 공통 prefix 비율 (0~1)
    needs_json_output: bool         # JSON 구조적 출력 필요 여부
    gpu_available: bool             # GPU 사용 가능 여부
    gpu_type: str = "unknown"       # GPU 종류
    gpu_count: int = 1              # GPU 수
    priority: Priority = Priority.THROUGHPUT


def recommend_engine(profile: WorkloadProfile) -> List[Tuple[str, float, str]]:
    """워크로드에 적합한 엔진 추천 (엔진명, 점수, 이유)"""
    scores = {}

    # vLLM 점수
    vllm_score = 50.0
    vllm_reasons = []
    if profile.gpu_available:
        vllm_score += 20
    if profile.priority == Priority.THROUGHPUT:
        vllm_score += 20
        vllm_reasons.append("처리량 최적화 우수")
    if profile.requests_per_minute > 100:
        vllm_score += 15
        vllm_reasons.append("대규모 배치에 강함")
    if profile.shared_prefix_ratio > 0.3:
        vllm_score += 10
        vllm_reasons.append("Prefix Caching 지원")
    if not profile.gpu_available:
        vllm_score = 0
    scores["vLLM"] = (vllm_score, ", ".join(vllm_reasons) or "범용 서빙 엔진")

    # SGLang 점수
    sglang_score = 50.0
    sglang_reasons = []
    if profile.gpu_available:
        sglang_score += 20
    if profile.shared_prefix_ratio > 0.5:
        sglang_score += 25
        sglang_reasons.append("RadixAttention으로 KV 재사용 극대화")
    if profile.needs_json_output:
        sglang_score += 20
        sglang_reasons.append("구조적 출력 최고 성능")
    if profile.priority == Priority.STRUCTURED_OUTPUT:
        sglang_score += 15
    if not profile.gpu_available:
        sglang_score = 0
    scores["SGLang"] = (sglang_score, ", ".join(sglang_reasons) or "범용 서빙 엔진")

    # TensorRT-LLM 점수
    trt_score = 40.0
    trt_reasons = []
    if profile.gpu_available and "A100" in profile.gpu_type or "H100" in profile.gpu_type:
        trt_score += 25
        trt_reasons.append("NVIDIA 고급 GPU 최적화")
    if profile.priority == Priority.LATENCY:
        trt_score += 25
        trt_reasons.append("단일 요청 지연시간 최저")
    if not profile.gpu_available:
        trt_score = 0
    scores["TensorRT-LLM"] = (trt_score, ", ".join(trt_reasons) or "NVIDIA 전용")

    # llama.cpp 점수
    llama_score = 40.0
    llama_reasons = []
    if not profile.gpu_available:
        llama_score += 30
        llama_reasons.append("CPU 추론 가능")
    if profile.priority == Priority.EDGE:
        llama_score += 25
        llama_reasons.append("엣지/로컬 배포 최적")
    if profile.priority == Priority.COST:
        llama_score += 20
        llama_reasons.append("저비용 운영")
    if profile.requests_per_minute > 100:
        llama_score -= 20
        llama_reasons.append("대규모 배치에 부적합")
    scores["llama.cpp"] = (llama_score, ", ".join(llama_reasons) or "경량 추론")

    # ONNX Runtime 점수
    onnx_score = 30.0
    onnx_reasons = []
    if "Intel" in profile.gpu_type or "AMD" in profile.gpu_type:
        onnx_score += 20
        onnx_reasons.append("비NVIDIA 하드웨어 지원")
    scores["ONNX Runtime"] = (onnx_score, ", ".join(onnx_reasons) or "범용 런타임")

    # 점수순 정렬
    ranked = [
        (name, score, reason)
        for name, (score, reason) in sorted(
            scores.items(), key=lambda x: x[1][0], reverse=True
        )
        if score > 0
    ]

    return ranked


if __name__ == "__main__":
    # 의료 문서 OCR 배치 처리 시나리오
    profile = WorkloadProfile(
        avg_input_tokens=3000,
        avg_output_tokens=500,
        requests_per_minute=200,
        shared_prefix_ratio=0.6,    # 동일 양식 60%
        needs_json_output=True,
        gpu_available=True,
        gpu_type="A100",
        gpu_count=1,
        priority=Priority.THROUGHPUT,
    )

    print("=== 의료 문서 OCR 배치 처리 - 엔진 추천 ===\n")
    for rank, (name, score, reason) in enumerate(recommend_engine(profile), 1):
        print(f"  {rank}. {name:20s} (점수: {score:.0f}) - {reason}")

    print()

    # 엣지 배포 시나리오
    edge_profile = WorkloadProfile(
        avg_input_tokens=2000,
        avg_output_tokens=300,
        requests_per_minute=10,
        shared_prefix_ratio=0.3,
        needs_json_output=False,
        gpu_available=False,
        priority=Priority.EDGE,
    )

    print("=== 병원 내부 엣지 서버 - 엔진 추천 ===\n")
    for rank, (name, score, reason) in enumerate(recommend_engine(edge_profile), 1):
        print(f"  {rank}. {name:20s} (점수: {score:.0f}) - {reason}")
```

---

## Docker 배포 설정

### vLLM Docker Compose

```yaml
# docker-compose.vllm.yml
version: "3.8"

services:
  vllm-server:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - VLLM_ATTENTION_BACKEND=FLASH_ATTN
      - HF_TOKEN=${HF_TOKEN}
    ports:
      - "8000:8000"
    volumes:
      - ${MODEL_CACHE_DIR:-./model-cache}:/root/.cache/huggingface
    command: >
      --model ${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct-AWQ}
      --quantization awq
      --max-model-len 8192
      --gpu-memory-utilization 0.90
      --enable-prefix-caching
      --enable-chunked-prefill
      --max-num-seqs 64
      --dtype auto
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
```

---

## 용어 체크리스트

| 용어 | 이해 여부 | 핵심 포인트 |
|------|:---------:|-------------|
| vLLM | [ ] | PagedAttention 기반 고처리량 서빙. 업계 표준 |
| SGLang | [ ] | RadixAttention + 구조적 생성. KV 재사용 극대화 |
| TensorRT-LLM | [ ] | NVIDIA 커널 퓨전 최적화. 최저 지연시간 |
| llama.cpp | [ ] | C++ 경량 엔진. CPU 추론, GGUF 양자화 |
| ONNX Runtime | [ ] | 범용 ML 런타임. 하드웨어 중립 |
| Continuous Batching | [ ] | 동적 요청 교체. Static 대비 2~5x 처리량 |
| Kernel Fusion | [ ] | 여러 연산을 하나의 GPU 커널로 결합 |
| RadixAttention | [ ] | Radix Tree로 부분 prefix까지 자동 KV 재사용 |
| GGUF | [ ] | llama.cpp 양자화 포맷. Q2~Q8까지 다양 |
| Speculative Decoding | [ ] | Draft+Target 모델 조합으로 디코딩 가속 |
| Arithmetic Intensity | [ ] | FLOPs/Bytes. Prefill은 높고 Decode는 낮음 |
| Constrained Decoding | [ ] | JSON Schema/정규식으로 출력 형식 강제 |
