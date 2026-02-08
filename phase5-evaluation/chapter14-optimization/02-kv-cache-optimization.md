---
---

# 14.2 KV Cache 최적화

> **Transformer 추론에서 Key-Value 캐시의 메모리 효율성과 재사용성을 극대화하는 기법**

---

## 핵심 용어

| 용어 | 정의 |
|------|------|
| **KV Cache** | 이전 토큰의 Key/Value 벡터를 저장해 중복 계산을 방지하는 캐시 |
| **PagedAttention** | OS의 가상 메모리 개념을 차용하여 KV Cache를 비연속 블록(page)으로 관리하는 기법 |
| **Prefix Caching** | 공통 접두사(system prompt 등)의 KV Cache를 여러 요청 간 공유하는 기법 |
| **Chunked Prefill** | 긴 프롬프트의 prefill을 청크 단위로 분할해 decode 요청과 인터리빙하는 기법 |
| **Sliding Window Attention** | 최근 W 토큰에 대해서만 attention을 수행하여 KV Cache를 고정 크기로 유지 |
| **Continuous Batching** | 완료된 요청을 즉시 제거하고 새 요청을 추가하는 동적 배칭 |
| **Prefill** | 프롬프트 전체를 한 번에 처리해 KV Cache를 채우는 단계 |
| **Decode** | KV Cache를 참조하며 토큰을 하나씩 생성하는 단계 |
| **MQA** (Multi-Query Attention) | KV Head를 1개로 줄여 KV Cache 크기를 대폭 축소 |
| **GQA** (Grouped-Query Attention) | KV Head를 그룹으로 공유 (MHA와 MQA의 중간) |

---

## 수학적 원리

### 1. KV Cache 메모리 계산

Transformer 모델의 KV Cache가 차지하는 메모리:

$$
M_{\text{KV}} = 2 \times L \times n_{\text{kv\_heads}} \times d_h \times n_{\text{seq}} \times b_{\text{param}}
$$

여기서:
- $2$: Key와 Value 두 종류
- $L$: 레이어 수
- $n_{\text{kv\_heads}}$: KV Head 수 (GQA에서는 Query Head보다 적음)
- $d_h$: Head 차원 ($d_{\text{model}} / n_{\text{heads}}$)
- $n_{\text{seq}}$: 시퀀스 길이 (현재까지 생성된 토큰 수)
- $b_{\text{param}}$: 파라미터당 바이트 수 (FP16=2, FP32=4)

**예시: Llama-3-8B (GQA, 8 KV heads)**

$$
M_{\text{KV}} = 2 \times 32 \times 8 \times 128 \times 4096 \times 2 = 536\text{ MB (per sequence)}
$$

배치 크기 $B$일 때 총 KV Cache:

$$
M_{\text{total}} = B \times M_{\text{KV}}
$$

배치 32면 **17.2 GB**. 모델 자체 크기(16 GB)를 초과한다. **KV Cache가 메모리 병목의 핵심**이다.

### 2. Attention 유형별 KV Cache 비교

| Attention | KV Head 수 | KV Cache 비율 |
|-----------|-----------|--------------|
| MHA (Multi-Head) | $n_h$ | 1.0x (기준) |
| GQA (Grouped-Query) | $n_h / g$ | $1/g$ |
| MQA (Multi-Query) | 1 | $1/n_h$ |

GQA에서 그룹 수 $g$가 클수록 KV Cache 절약이 커진다:

$$
M_{\text{KV}}^{\text{GQA}} = \frac{n_{\text{kv\_heads}}}{n_{\text{q\_heads}}} \times M_{\text{KV}}^{\text{MHA}}
$$

Llama-3-8B: $n_q = 32$, $n_{\text{kv}} = 8$ → KV Cache가 MHA 대비 **25%**만 필요.

### 3. PagedAttention

기존 방식의 문제:

- 각 시퀀스에 **최대 길이만큼** 연속 메모리를 사전 할당
- 실제 사용량 < 할당량 → **내부 단편화(internal fragmentation)**
- 서로 다른 시퀀스 간 공유 불가 → **외부 단편화(external fragmentation)**

vLLM의 논문에 따르면 기존 방식은 **메모리의 60~80%가 낭비**된다.

**PagedAttention 해결 방식**:

OS의 가상 메모리와 동일한 개념:

$$
\text{Virtual KV Block} \xrightarrow{\text{Page Table}} \text{Physical KV Block}
$$

- KV Cache를 고정 크기 **블록(page)** 단위로 관리 (보통 16 토큰/블록)
- 각 시퀀스는 **페이지 테이블**로 비연속 물리 블록에 매핑
- 블록 단위 할당/해제 → 단편화 최소화

**메모리 효율**:

$$
\text{Waste} = \text{Block Size} - (n_{\text{seq}} \mod \text{Block Size})
$$

블록 크기가 16일 때 최대 낭비는 15토큰분 → 시퀀스가 길수록 낭비 비율이 0에 수렴한다.

### 4. Prefix Caching

동일한 system prompt를 사용하는 $N$개의 요청이 있을 때:

**기존**: 각 요청이 독립적으로 KV Cache 보유

$$
M_{\text{naive}} = N \times (M_{\text{prefix}} + M_{\text{unique}})
$$

**Prefix Caching**: 공통 prefix의 KV를 한 번만 저장하고 공유

$$
M_{\text{cached}} = M_{\text{prefix}} + N \times M_{\text{unique}}
$$

**절약량**:

$$
\Delta M = (N - 1) \times M_{\text{prefix}}
$$

의료 문서 OCR에서 system prompt가 2K 토큰이고 동시 요청이 32개라면:
- Llama-3-8B 기준 prefix KV ≈ 256 MB
- 절약량 = 31 × 256 MB ≈ **7.9 GB**

**Prefix Caching 조건**:
- 요청들이 동일한 접두사를 공유해야 함
- 접두사가 길수록 효과 큼
- KV Cache의 접두사 부분이 정확히 일치해야 재사용 가능

### 5. Sliding Window Attention

최근 $W$ 토큰에 대해서만 attention을 수행:

$$
\text{Attention}(Q_t, K, V) = \text{softmax}\left(\frac{Q_t K_{[t-W:t]}^T}{\sqrt{d_h}}\right) V_{[t-W:t]}
$$

**KV Cache 메모리**:

$$
M_{\text{SW}} = 2 \times L \times n_{\text{kv}} \times d_h \times W \times b_{\text{param}}
$$

시퀀스 길이 $n_{\text{seq}}$와 무관하게 **고정 메모리** $O(W)$:

$$
\frac{M_{\text{SW}}}{M_{\text{full}}} = \frac{W}{n_{\text{seq}}}
$$

$W = 4096$, $n_{\text{seq}} = 128K$이면 KV Cache가 **3.1%**로 줄어든다.

**단점**: 윈도우 밖의 토큰 정보 손실. 긴 문맥 의존성이 필요한 태스크에서 성능 저하 가능.

Mistral 계열 모델은 Sliding Window + Full Attention 레이어를 혼합하여 사용한다.

### 6. Chunked Prefill

긴 프롬프트의 prefill 연산은 GPU를 오래 점유해서 decode 요청의 지연(latency)을 증가시킨다.

**Chunked Prefill 전략**:

프롬프트를 $C$ 토큰 크기의 청크로 분할:

$$
\text{Prefill Chunks} = \left\lceil \frac{n_{\text{prompt}}}{C} \right\rceil
$$

각 청크 사이에 decode 요청을 끼워넣어(interleaving) **TTFT를 희생하지 않으면서 전체 시스템 처리량 유지**.

**트레이드오프**:

$$
\text{TTFT}_{\text{chunked}} = \text{TTFT}_{\text{unchunked}} + \text{decode\_interleave\_overhead}
$$

prefill 중인 요청의 TTFT는 약간 증가하지만, 이미 decode 중인 요청의 TPOT은 안정적으로 유지된다.

---

## 의료 문서 OCR에서의 KV Cache 최적화

### 워크로드 특성 분석

| 특성 | 의료 문서 OCR | 일반 챗봇 |
|------|-------------|-----------|
| 프롬프트 길이 | 길다 (문서 전체 + 이미지 토큰) | 짧다~보통 |
| 생성 길이 | 보통 (구조화된 텍스트) | 다양함 |
| System Prompt | 길고 고정적 (양식 규칙, 필드 정의) | 짧고 고정적 |
| 동시 요청 | 배치 처리 다수 | 실시간 단건 |
| 문서 양식 | 반복 패턴 높음 (처방전, 검사지 등) | 패턴 없음 |

### 최적화 전략

1. **Prefix Caching 최우선 적용**
   - 의료 문서 OCR의 system prompt는 보통 1K~4K 토큰으로 길다
   - 동일 양식의 문서를 배치 처리할 때 prefix가 완전히 동일
   - 효과: 배치 32 기준 KV Cache **50~70% 절약**

2. **PagedAttention 필수**
   - 문서 길이가 가변적 (처방전 짧음, 퇴원 요약서 김)
   - 고정 할당 시 메모리 낭비가 극심
   - vLLM 또는 SGLang 사용 권장

3. **Sliding Window 주의**
   - 의료 문서는 앞부분 정보(환자 정보)가 뒷부분(진단)에 영향
   - 순수 Sliding Window는 부적합
   - Sliding Window + Sink Token 조합 또는 Full Attention 모델 권장

4. **KV Cache 양자화**
   - KV Cache를 FP16 → INT8로 양자화하면 메모리 50% 절약
   - 정확도 손실이 미미한 경우가 많음 (KV는 가중치보다 양자화에 강인)

---

## 코드

### 1. KV Cache 메모리 계산기

```python
"""
KV Cache 메모리 계산기
- 다양한 모델/설정에 대한 KV Cache 메모리 추정
- 최적화 기법 적용 시 절약량 계산
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import math


@dataclass
class ModelConfig:
    """모델 아키텍처 설정"""
    name: str
    num_layers: int
    num_q_heads: int
    num_kv_heads: int          # GQA: < num_q_heads, MQA: 1
    head_dim: int
    max_seq_len: int
    dtype_bytes: int = 2       # FP16=2, FP32=4, INT8=1


# 대표 모델 프리셋
MODEL_PRESETS: Dict[str, ModelConfig] = {
    "llama-3-8b": ModelConfig(
        name="Llama-3-8B",
        num_layers=32,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=8192,
    ),
    "llama-3-70b": ModelConfig(
        name="Llama-3-70B",
        num_layers=80,
        num_q_heads=64,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=8192,
    ),
    "qwen2.5-7b": ModelConfig(
        name="Qwen2.5-7B",
        num_layers=28,
        num_q_heads=28,
        num_kv_heads=4,
        head_dim=128,
        max_seq_len=32768,
    ),
    "qwen2.5-72b": ModelConfig(
        name="Qwen2.5-72B",
        num_layers=80,
        num_q_heads=64,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=32768,
    ),
    "mistral-7b": ModelConfig(
        name="Mistral-7B",
        num_layers=32,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=32768,
    ),
}


@dataclass
class OptimizationConfig:
    """KV Cache 최적화 설정"""
    paged_attention: bool = False
    page_size: int = 16                     # 블록당 토큰 수
    prefix_caching: bool = False
    prefix_length: int = 0                  # 공유 prefix 토큰 수
    sliding_window: Optional[int] = None    # Sliding Window 크기
    kv_cache_dtype_bytes: Optional[int] = None  # KV Cache 양자화 (예: 1 for INT8)


class KVCacheCalculator:
    """KV Cache 메모리 계산 및 최적화 분석"""

    def __init__(self, model: ModelConfig):
        self.model = model

    def per_token_kv_bytes(self, dtype_bytes: Optional[int] = None) -> float:
        """토큰 1개당 KV Cache 바이트 수"""
        b = dtype_bytes or self.model.dtype_bytes
        # 2(K+V) × L × n_kv_heads × d_h × bytes
        return 2 * self.model.num_layers * self.model.num_kv_heads * self.model.head_dim * b

    def sequence_kv_bytes(
        self,
        seq_len: int,
        dtype_bytes: Optional[int] = None,
    ) -> float:
        """시퀀스 1개의 KV Cache 바이트 수"""
        return self.per_token_kv_bytes(dtype_bytes) * seq_len

    def batch_kv_bytes(
        self,
        batch_size: int,
        seq_len: int,
        optimization: Optional[OptimizationConfig] = None,
    ) -> Dict[str, float]:
        """배치 전체의 KV Cache 메모리 (최적화 적용)"""

        opt = optimization or OptimizationConfig()
        effective_dtype = opt.kv_cache_dtype_bytes or self.model.dtype_bytes

        # 실제 KV 유지 시퀀스 길이 결정
        if opt.sliding_window is not None:
            effective_seq_len = min(seq_len, opt.sliding_window)
        else:
            effective_seq_len = seq_len

        # 기본 KV Cache (최적화 없음, 원본 dtype)
        naive_total = batch_size * self.sequence_kv_bytes(seq_len)

        # 최적화 적용 KV Cache
        per_seq_kv = self.sequence_kv_bytes(effective_seq_len, effective_dtype)

        if opt.prefix_caching and opt.prefix_length > 0:
            prefix_kv = self.sequence_kv_bytes(
                min(opt.prefix_length, effective_seq_len), effective_dtype
            )
            unique_kv = per_seq_kv - prefix_kv
            # prefix는 1번만, unique는 배치 수만큼
            optimized_total = prefix_kv + batch_size * unique_kv
        else:
            optimized_total = batch_size * per_seq_kv

        # PagedAttention 오버헤드 (페이지 테이블 + 내부 단편화)
        paged_overhead = 0.0
        if opt.paged_attention:
            # 각 시퀀스의 마지막 블록 내부 단편화 (평균 절반)
            avg_waste_per_seq = (opt.page_size / 2) * self.per_token_kv_bytes(effective_dtype)
            paged_overhead = batch_size * avg_waste_per_seq
            # 페이지 테이블 메모리 (무시할 수준이지만 계산)
            pages_per_seq = math.ceil(effective_seq_len / opt.page_size)
            page_table_bytes = batch_size * pages_per_seq * 8  # int64 포인터
            paged_overhead += page_table_bytes

        return {
            "naive_bytes": naive_total,
            "naive_gb": naive_total / (1024 ** 3),
            "optimized_bytes": optimized_total + paged_overhead,
            "optimized_gb": (optimized_total + paged_overhead) / (1024 ** 3),
            "saved_bytes": naive_total - (optimized_total + paged_overhead),
            "saved_gb": (naive_total - optimized_total - paged_overhead) / (1024 ** 3),
            "saving_pct": (1 - (optimized_total + paged_overhead) / naive_total) * 100
            if naive_total > 0 else 0,
        }

    def max_batch_size(
        self,
        gpu_memory_gb: float,
        model_size_gb: float,
        seq_len: int,
        overhead_gb: float = 1.0,
        optimization: Optional[OptimizationConfig] = None,
    ) -> int:
        """주어진 GPU 메모리에서 가능한 최대 배치 크기"""
        available_gb = gpu_memory_gb - model_size_gb - overhead_gb
        if available_gb <= 0:
            return 0

        opt = optimization or OptimizationConfig()
        effective_dtype = opt.kv_cache_dtype_bytes or self.model.dtype_bytes

        if opt.sliding_window is not None:
            effective_seq_len = min(seq_len, opt.sliding_window)
        else:
            effective_seq_len = seq_len

        per_seq_bytes = self.sequence_kv_bytes(effective_seq_len, effective_dtype)

        if opt.prefix_caching and opt.prefix_length > 0:
            prefix_bytes = self.sequence_kv_bytes(
                min(opt.prefix_length, effective_seq_len), effective_dtype
            )
            unique_bytes = per_seq_bytes - prefix_bytes
            available_after_prefix = available_gb * (1024 ** 3) - prefix_bytes
            if available_after_prefix <= 0:
                return 0
            return int(available_after_prefix / unique_bytes)

        available_bytes = available_gb * (1024 ** 3)
        return int(available_bytes / per_seq_bytes)

    def report(
        self,
        batch_size: int,
        seq_len: int,
        gpu_memory_gb: float,
        model_size_gb: float,
        optimization: Optional[OptimizationConfig] = None,
    ) -> str:
        """종합 리포트 생성"""
        result = self.batch_kv_bytes(batch_size, seq_len, optimization)
        max_bs = self.max_batch_size(
            gpu_memory_gb, model_size_gb, seq_len, optimization=optimization
        )

        lines = [
            f"=== KV Cache 메모리 리포트: {self.model.name} ===",
            f"",
            f"모델 설정:",
            f"  레이어: {self.model.num_layers}, "
            f"Q Heads: {self.model.num_q_heads}, "
            f"KV Heads: {self.model.num_kv_heads}, "
            f"Head Dim: {self.model.head_dim}",
            f"  GQA 비율: {self.model.num_kv_heads}/{self.model.num_q_heads} "
            f"= {self.model.num_kv_heads/self.model.num_q_heads:.2f}x",
            f"",
            f"추론 설정:",
            f"  Batch Size: {batch_size}, Seq Len: {seq_len}",
            f"  GPU: {gpu_memory_gb} GB, 모델: {model_size_gb} GB",
            f"",
            f"KV Cache 메모리:",
            f"  토큰당: {self.per_token_kv_bytes() / 1024:.1f} KB",
            f"  최적화 전: {result['naive_gb']:.2f} GB",
            f"  최적화 후: {result['optimized_gb']:.2f} GB",
            f"  절약: {result['saved_gb']:.2f} GB ({result['saving_pct']:.1f}%)",
            f"",
            f"최대 배치 크기: {max_bs}",
        ]
        return "\n".join(lines)


# 사용 예시
if __name__ == "__main__":
    # 의료 문서 OCR 시나리오
    model = MODEL_PRESETS["llama-3-8b"]
    calc = KVCacheCalculator(model)

    # 시나리오 1: 최적화 없음
    print(calc.report(
        batch_size=32,
        seq_len=4096,
        gpu_memory_gb=80,       # A100 80GB
        model_size_gb=16,       # FP16
    ))

    print("\n" + "=" * 60 + "\n")

    # 시나리오 2: Prefix Caching + PagedAttention + KV INT8
    opt = OptimizationConfig(
        paged_attention=True,
        page_size=16,
        prefix_caching=True,
        prefix_length=2048,     # 2K 토큰 system prompt
        kv_cache_dtype_bytes=1, # KV Cache INT8
    )
    print(calc.report(
        batch_size=32,
        seq_len=4096,
        gpu_memory_gb=80,
        model_size_gb=16,
        optimization=opt,
    ))

    print("\n" + "=" * 60 + "\n")

    # 모델별 비교
    print("=== 모델별 토큰당 KV Cache ===")
    for name, preset in MODEL_PRESETS.items():
        c = KVCacheCalculator(preset)
        per_tok = c.per_token_kv_bytes()
        seq_4k = c.sequence_kv_bytes(4096)
        print(f"  {preset.name:20s}: {per_tok/1024:6.1f} KB/token, "
              f"4K seq = {seq_4k/(1024**2):7.1f} MB")
```

### 2. PagedAttention 개념 구현

```python
"""
PagedAttention 개념 시뮬레이터
- 실제 vLLM의 구현을 단순화한 교육용 코드
- 페이지 테이블 기반 KV Cache 관리 원리 이해
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PagedKVBlock:
    """물리 KV 블록"""
    block_id: int
    block_size: int             # 블록당 토큰 수
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype = torch.float16

    key_cache: torch.Tensor = field(init=False)
    value_cache: torch.Tensor = field(init=False)
    num_filled: int = field(default=0, init=False)

    def __post_init__(self):
        shape = (self.block_size, self.num_kv_heads, self.head_dim)
        self.key_cache = torch.zeros(shape, dtype=self.dtype)
        self.value_cache = torch.zeros(shape, dtype=self.dtype)

    @property
    def is_full(self) -> bool:
        return self.num_filled >= self.block_size

    @property
    def num_free(self) -> int:
        return self.block_size - self.num_filled

    def append(self, key: torch.Tensor, value: torch.Tensor) -> int:
        """KV 쌍 추가. 추가된 토큰 수 반환"""
        n_tokens = key.shape[0]
        n_to_fill = min(n_tokens, self.num_free)

        if n_to_fill == 0:
            return 0

        start = self.num_filled
        end = start + n_to_fill
        self.key_cache[start:end] = key[:n_to_fill]
        self.value_cache[start:end] = value[:n_to_fill]
        self.num_filled = end

        return n_to_fill


class BlockAllocator:
    """물리 블록 할당/해제 관리"""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # 물리 블록 풀 생성
        self.free_blocks: List[int] = list(range(num_blocks))
        self.blocks: Dict[int, PagedKVBlock] = {}

        for i in range(num_blocks):
            self.blocks[i] = PagedKVBlock(
                block_id=i,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )

        logger.info(
            f"BlockAllocator 초기화: {num_blocks} blocks × {block_size} tokens"
        )

    def allocate(self) -> Optional[int]:
        """빈 블록 1개 할당"""
        if not self.free_blocks:
            logger.warning("남은 블록 없음!")
            return None
        block_id = self.free_blocks.pop(0)
        return block_id

    def free(self, block_id: int) -> None:
        """블록 해제"""
        block = self.blocks[block_id]
        block.key_cache.zero_()
        block.value_cache.zero_()
        block.num_filled = 0
        self.free_blocks.append(block_id)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return len(self.blocks) - len(self.free_blocks)


@dataclass
class SequenceKVManager:
    """시퀀스 1개의 KV Cache 관리 (페이지 테이블)"""
    seq_id: int
    page_table: List[int] = field(default_factory=list)  # 논리→물리 매핑
    total_tokens: int = 0


class PagedAttentionManager:
    """PagedAttention 기반 KV Cache 매니저"""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        self.num_layers = num_layers
        self.block_size = block_size

        # 레이어별 블록 할당자
        self.allocators: List[BlockAllocator] = [
            BlockAllocator(num_blocks, block_size, num_kv_heads, head_dim)
            for _ in range(num_layers)
        ]

        # 시퀀스별 KV 관리자 (레이어별)
        self.sequences: Dict[int, List[SequenceKVManager]] = {}

    def add_sequence(self, seq_id: int) -> bool:
        """새 시퀀스 등록"""
        if seq_id in self.sequences:
            logger.warning(f"시퀀스 {seq_id} 이미 존재")
            return False

        self.sequences[seq_id] = [
            SequenceKVManager(seq_id=seq_id)
            for _ in range(self.num_layers)
        ]
        return True

    def append_kv(
        self,
        seq_id: int,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> bool:
        """시퀀스에 KV 토큰 추가"""
        if seq_id not in self.sequences:
            return False

        seq_mgr = self.sequences[seq_id][layer_idx]
        allocator = self.allocators[layer_idx]
        remaining = key.shape[0]
        offset = 0

        while remaining > 0:
            # 현재 마지막 블록에 공간이 있는지 확인
            if seq_mgr.page_table:
                last_block_id = seq_mgr.page_table[-1]
                last_block = allocator.blocks[last_block_id]
                if not last_block.is_full:
                    filled = last_block.append(
                        key[offset:], value[offset:]
                    )
                    offset += filled
                    remaining -= filled
                    seq_mgr.total_tokens += filled
                    continue

            # 새 블록 할당
            block_id = allocator.allocate()
            if block_id is None:
                logger.error(f"시퀀스 {seq_id} 레이어 {layer_idx}: 블록 부족")
                return False

            seq_mgr.page_table.append(block_id)
            block = allocator.blocks[block_id]
            filled = block.append(key[offset:], value[offset:])
            offset += filled
            remaining -= filled
            seq_mgr.total_tokens += filled

        return True

    def remove_sequence(self, seq_id: int) -> None:
        """시퀀스 제거 및 블록 해제"""
        if seq_id not in self.sequences:
            return

        for layer_idx, seq_mgr in enumerate(self.sequences[seq_id]):
            for block_id in seq_mgr.page_table:
                self.allocators[layer_idx].free(block_id)

        del self.sequences[seq_id]
        logger.info(f"시퀀스 {seq_id} 제거 완료")

    def get_kv(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """시퀀스의 전체 KV Cache 조회 (연속 텐서로 결합)"""
        if seq_id not in self.sequences:
            return None

        seq_mgr = self.sequences[seq_id][layer_idx]
        allocator = self.allocators[layer_idx]

        if not seq_mgr.page_table:
            return None

        keys = []
        values = []
        for block_id in seq_mgr.page_table:
            block = allocator.blocks[block_id]
            keys.append(block.key_cache[:block.num_filled])
            values.append(block.value_cache[:block.num_filled])

        return torch.cat(keys, dim=0), torch.cat(values, dim=0)

    def stats(self) -> Dict:
        """메모리 사용 통계"""
        total_blocks = sum(
            len(a.blocks) for a in self.allocators
        )
        used_blocks = sum(
            a.num_used_blocks for a in self.allocators
        )
        total_tokens = sum(
            seq_mgr.total_tokens
            for seq_mgrs in self.sequences.values()
            for seq_mgr in seq_mgrs
        )

        return {
            "num_sequences": len(self.sequences),
            "total_blocks": total_blocks,
            "used_blocks": used_blocks,
            "free_blocks": total_blocks - used_blocks,
            "utilization_pct": (used_blocks / total_blocks * 100) if total_blocks > 0 else 0,
            "total_cached_tokens": total_tokens // self.num_layers,  # 레이어 중복 제거
        }


# 사용 예시
if __name__ == "__main__":
    manager = PagedAttentionManager(
        num_blocks=256,     # 레이어당 256개 물리 블록
        block_size=16,      # 블록당 16 토큰
        num_layers=4,       # 테스트용 4 레이어
        num_kv_heads=8,
        head_dim=64,
    )

    # 시퀀스 3개 추가
    for seq_id in range(3):
        manager.add_sequence(seq_id)

        # 각 시퀀스에 토큰 추가 (시뮬레이션)
        seq_len = 50 + seq_id * 30  # 50, 80, 110 토큰
        for layer_idx in range(4):
            k = torch.randn(seq_len, 8, 64, dtype=torch.float16)
            v = torch.randn(seq_len, 8, 64, dtype=torch.float16)
            manager.append_kv(seq_id, layer_idx, k, v)

    # 통계 출력
    stats = manager.stats()
    print("\n=== PagedAttention 통계 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 시퀀스 0 완료 → 블록 해제
    manager.remove_sequence(0)
    stats = manager.stats()
    print("\n=== 시퀀스 0 제거 후 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
```

---

## KV Cache 최적화 기법 비교

| 기법 | 메모리 절약 | 구현 복잡도 | 정확도 영향 | 적용 조건 |
|------|-----------|-----------|-----------|----------|
| PagedAttention | 60~80% 낭비 제거 | 높음 (vLLM 내장) | 없음 | 항상 |
| Prefix Caching | $(N-1)/N \times$ prefix | 보통 | 없음 | 공통 prefix 있을 때 |
| Sliding Window | $1 - W/n_{\text{seq}}$ | 낮음 (모델 자체) | 있음 (장거리 손실) | 모델이 지원할 때 |
| KV INT8 | 50% | 보통 | 미미 | 항상 |
| GQA/MQA | $1/g$ 또는 $1/n_h$ | 없음 (모델 자체) | 약간 | 모델 선택 시 |
| Chunked Prefill | 간접적 (처리량 ↑) | 보통 | 없음 | 긴 프롬프트 |

---

## 용어 체크리스트

| 용어 | 이해 여부 | 핵심 포인트 |
|------|:---------:|-------------|
| KV Cache | [ ] | Key/Value 저장으로 중복 계산 방지. 메모리 병목의 주범 |
| PagedAttention | [ ] | 비연속 블록으로 KV 관리. 메모리 단편화 60~80% 제거 |
| Prefix Caching | [ ] | 공통 접두사 KV 공유. 동일 system prompt 배치에 효과적 |
| Chunked Prefill | [ ] | 긴 prefill을 청크 분할. decode 지연 방지 |
| Sliding Window | [ ] | 최근 W 토큰만 KV 유지. 고정 메모리 O(W) |
| Continuous Batching | [ ] | 동적으로 요청 추가/제거. 처리량 극대화 |
| Prefill / Decode | [ ] | 추론의 두 단계. Prefill은 compute-bound, Decode는 memory-bound |
| MQA / GQA | [ ] | KV Head 공유로 Cache 크기 축소. 모델 아키텍처 레벨 최적화 |
| Block Allocator | [ ] | 물리 블록 풀에서 블록 할당/해제 관리 |
| Page Table | [ ] | 논리 블록 → 물리 블록 매핑. OS 가상 메모리와 동일 개념 |
| KV Cache Quantization | [ ] | KV를 INT8로 양자화. 50% 메모리 절약, 정확도 손실 미미 |
| Internal Fragmentation | [ ] | 블록 내부 미사용 공간. PagedAttention으로 최소화 |
