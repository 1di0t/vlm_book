# 3.4 Inference 특성

LLM 추론은 학습과 다른 특성을 가진다. 효율적인 서빙을 위해 이 특성을 이해해야 한다.

## 3.4.1 Compute-bound vs Memory-bound

### 기본 개념

- **Compute-bound**: GPU 연산 능력이 병목
- **Memory-bound**: GPU 메모리 대역폭이 병목

### LLM 추론의 특성

| 단계 | 특성 | 이유 |
|------|------|------|
| Prefill | Compute-bound | 큰 행렬 곱 (batch × seq × hidden) |
| Decode | Memory-bound | 작은 행렬 곱 (batch × 1 × hidden) |

```python
# Prefill: 많은 연산
# Q: (batch, seq_len, d)  - seq_len이 큼
# K: (batch, seq_len, d)
# Attention: O(seq_len² × d)

# Decode: 적은 연산
# Q: (batch, 1, d)  - 1개 토큰만
# K: (batch, seq_len+1, d)
# Attention: O(seq_len × d)
```

### Arithmetic Intensity

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Memory Bytes}}
$$

- 높으면 Compute-bound
- 낮으면 Memory-bound

```python
# Decode 시 Arithmetic Intensity
# 연산: 2 × d × d (Linear layer)
# 메모리: d × d × 2 (Weight 읽기, FP16)
# AI = (2 × d × d) / (2 × d × d) = 1

# AI가 낮으면 메모리 대역폭에 막힘
```

## 3.4.2 Batch Size와 Throughput

### Batching의 효과

```python
# 단일 요청
# GPU 활용률 낮음 (Memory-bound)

# Batch 처리
# 여러 요청의 가중치 공유 → GPU 활용률 증가
```

### Throughput vs Latency

| 메트릭 | 설명 | 최적화 방향 |
|--------|------|------------|
| Latency | 단일 요청 응답 시간 | 배치 크기 ↓ |
| Throughput | 초당 처리 토큰 수 | 배치 크기 ↑ |

```
                    Throughput
                        ▲
                        │     ┌─────────────
                        │    /
                        │   /
                        │  /
                        │ /
                        │/
                        └────────────────────▶ Batch Size
                             Memory Limit

배치 크기 증가 → Throughput 증가 (어느 시점까지)
              → Latency도 증가
```

### 최적 배치 크기 찾기

```python
def find_optimal_batch_size(model, max_seq_len, gpu_memory_gb):
    """메모리 기반 최대 배치 크기 추정"""

    # 모델 파라미터 메모리
    model_params = sum(p.numel() for p in model.parameters())
    model_memory = model_params * 2  # FP16

    # KV Cache per request
    kv_cache_per_request = estimate_kv_cache(model, max_seq_len)

    # 남은 메모리
    available = gpu_memory_gb * 1024**3 - model_memory

    # 최대 배치 크기
    max_batch = available // kv_cache_per_request

    return max_batch
```

## 3.4.3 Continuous Batching

### Static Batching의 문제

```
Request 1: [■■■■■■■■■■■■■■■]  (긴 응답)
Request 2: [■■■■]□□□□□□□□□□□  (짧은 응답 → 대기)
Request 3: [■■■■■■]□□□□□□□□□  (중간 응답 → 대기)

□ = GPU 유휴
```

### Continuous Batching (Iteration-level)

```
시간 t1: [R1 R2 R3] 처리
시간 t2: [R1 R2 R3] 처리, R2 완료
시간 t3: [R1 R4 R3] 처리 (R4 새로 추가!)
시간 t4: [R1 R4 R3] 처리, R3 완료
시간 t5: [R1 R4 R5] 처리 (R5 새로 추가!)

완료된 요청 즉시 제거, 새 요청 즉시 추가
→ GPU 활용률 최대화
```

### vLLM의 Continuous Batching

```python
# vLLM 서버 예시
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")

# 여러 요청 동시 처리
prompts = [
    "What is AI?",
    "Explain quantum computing",
    "Write a poem about nature"
]

sampling_params = SamplingParams(temperature=0.8, max_tokens=100)

# Continuous batching으로 효율적 처리
outputs = llm.generate(prompts, sampling_params)
```

## 3.4.4 Time To First Token (TTFT)

### 정의

프롬프트 입력 → 첫 토큰 생성까지 시간

```
[Prompt] ─────Prefill─────▶ [First Token] ───Decode───▶ [...]
         └── TTFT ──────┘
```

### TTFT에 영향을 미치는 요소

| 요소 | 영향 |
|------|------|
| 프롬프트 길이 | 길수록 TTFT ↑ |
| 모델 크기 | 클수록 TTFT ↑ |
| GPU 성능 | 좋을수록 TTFT ↓ |
| Batch 대기 | 대기 중이면 TTFT ↑ |

### 최적화 방법

1. **Speculative Decoding**: 작은 모델로 추측 후 검증
2. **Prompt Caching**: 공통 프롬프트 캐시
3. **Chunked Prefill**: Prefill을 나눠서 Decode와 인터리브

## 3.4.5 Time Per Output Token (TPOT)

### 정의

각 출력 토큰 생성에 걸리는 시간

```
[Token 1] ──TPOT──▶ [Token 2] ──TPOT──▶ [Token 3] ──TPOT──▶ ...
```

### TPOT에 영향을 미치는 요소

| 요소 | 영향 |
|------|------|
| 모델 크기 | 클수록 TPOT ↑ |
| Batch 크기 | 클수록 TPOT ↑ |
| KV Cache 크기 | 클수록 TPOT ↑ |
| 메모리 대역폭 | 높을수록 TPOT ↓ |

### 총 응답 시간

$$
\text{Total Time} = \text{TTFT} + (\text{num\_tokens} - 1) \times \text{TPOT}
$$

## 3.4.6 GPU 메모리 구성

```
┌─────────────────────────────────────┐
│           GPU Memory                │
├─────────────────────────────────────┤
│  Model Weights (고정)               │
│  - FP16: ~14GB for 7B model        │
│  - INT4: ~3.5GB for 7B model       │
├─────────────────────────────────────┤
│  KV Cache (동적, 요청별)            │
│  - seq_len × layers × heads × d_k  │
│  - 예: 4K context → ~2GB/request   │
├─────────────────────────────────────┤
│  Activations (임시)                 │
│  - Forward pass 중간 결과           │
│  - 배치 크기에 비례                 │
├─────────────────────────────────────┤
│  Overhead (CUDA, Framework)        │
│  - ~1-2GB                          │
└─────────────────────────────────────┘
```

### 메모리 계산 예시

```python
def estimate_memory_usage(
    model_params_b,  # 10억 단위 파라미터
    precision_bytes,  # 2 for FP16, 1 for INT8
    batch_size,
    seq_len,
    num_layers,
    num_heads,
    d_k
):
    """GPU 메모리 사용량 추정 (GB)"""

    # 모델 가중치
    model_memory = model_params_b * 1e9 * precision_bytes / 1e9

    # KV Cache
    kv_cache = (
        2 *  # K and V
        batch_size *
        num_layers *
        seq_len *
        num_heads *
        d_k *
        2  # FP16
    ) / 1e9

    # Overhead
    overhead = 2.0

    total = model_memory + kv_cache + overhead

    return {
        "model": model_memory,
        "kv_cache": kv_cache,
        "overhead": overhead,
        "total": total
    }

# 예시: LLaMA-7B, batch=1, seq=4096
mem = estimate_memory_usage(
    model_params_b=7,
    precision_bytes=2,
    batch_size=1,
    seq_len=4096,
    num_layers=32,
    num_heads=32,
    d_k=128
)
print(f"Model: {mem['model']:.1f} GB")
print(f"KV Cache: {mem['kv_cache']:.1f} GB")
print(f"Total: {mem['total']:.1f} GB")
```

## 3.4.7 최적화 기법 요약

### Quantization

| 방식 | 메모리 절약 | 품질 영향 |
|------|------------|----------|
| FP16 | 기준 | 기준 |
| INT8 | 50% | 거의 없음 |
| INT4 | 75% | 약간 |
| INT4 (GPTQ/AWQ) | 75% | 최소 |

### Attention 최적화

| 기법 | 효과 |
|------|------|
| Flash Attention | 메모리 O(n) → O(1), 속도 2-4x |
| Multi-Query Attention | KV Cache 1/h |
| Grouped-Query Attention | KV Cache 1/g |
| Sliding Window | KV Cache 고정 |

### 서빙 최적화

| 기법 | 효과 |
|------|------|
| Continuous Batching | Throughput ↑ |
| PagedAttention | 메모리 효율 ↑ |
| Speculative Decoding | TPOT ↓ |
| Tensor Parallelism | 큰 모델 분산 |

## 3.4.8 추론 엔진 비교

| 엔진 | 특징 |
|------|------|
| vLLM | PagedAttention, Continuous Batching |
| TensorRT-LLM | NVIDIA 최적화, INT4/INT8 |
| llama.cpp | CPU 추론, 경량 |
| SGLang | RadixAttention, 프롬프트 캐싱 |

```python
# vLLM 예시
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9
)

# TensorRT-LLM 예시
# (별도 빌드 필요)

# llama.cpp 예시 (CPU)
# ./main -m model.gguf -p "Hello" -n 100
```

## 3.4.9 실습 체크리스트

- [ ] Prefill vs Decode 시간 측정
- [ ] 배치 크기별 Throughput 측정
- [ ] TTFT, TPOT 측정
- [ ] GPU 메모리 사용량 모니터링
- [ ] vLLM vs HuggingFace 성능 비교

## 3.4.10 핵심 요약

| 개념 | 설명 |
|------|------|
| Prefill | Compute-bound, 프롬프트 처리 |
| Decode | Memory-bound, 토큰 생성 |
| TTFT | 첫 토큰까지 시간 |
| TPOT | 토큰당 시간 |
| Continuous Batching | 동적 배치로 GPU 활용률 ↑ |

## Phase 1 정리

Phase 1에서는 VLM을 이해하기 위한 기초를 다뤘다:

1. **수학적 기초**: 선형대수, 미적분, 확률
2. **Transformer**: Self-Attention, Positional Encoding, Block 구조
3. **LLM**: Tokenization, Generation, KV Cache, Inference

다음 Phase에서는 Vision과 VLM 아키텍처를 다룬다.
