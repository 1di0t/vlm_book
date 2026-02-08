# 15.3 Batching 전략

LLM 서빙에서 배칭은 처리량(throughput)을 극대화하는 핵심 기법이다. 의료 문서 OCR처럼 입력 길이 편차가 큰 워크로드에서는 배칭 전략의 선택이 시스템 성능을 좌우한다.

---

> **핵심 용어 박스**
>
> | 용어 | 정의 |
> |------|------|
> | **Static Batching** | 고정 크기 배치를 구성한 후, 모든 요청이 완료될 때까지 대기하는 방식 |
> | **Dynamic Batching** | 대기 큐에서 일정 시간/개수 조건을 만족하면 배치를 구성하는 방식 |
> | **Continuous Batching** | 개별 요청의 완료 시점에 바로 새 요청을 삽입하는 방식 (iteration-level scheduling) |
> | **Little's Law** | 정상 상태에서 시스템 내 평균 요청 수 $L = \lambda W$ |
> | **Padding** | 배치 내 시퀀스 길이를 통일하기 위해 짧은 시퀀스에 더미 토큰을 추가하는 것 |
> | **Prefill Phase** | 입력 토큰을 한 번에 처리하는 초기 단계 (compute-bound) |
> | **Decode Phase** | 토큰을 하나씩 자기회귀적으로 생성하는 단계 (memory-bound) |
> | **KV Cache** | Attention 계산에 필요한 Key, Value 텐서를 캐싱하여 재계산을 방지 |

---

## 15.3.1 Static Batching

### 동작 원리

가장 단순한 배칭 방식. 고정된 크기 $B$의 배치를 구성한 후, 모든 요청이 동시에 시작하고 가장 느린 요청이 끝날 때까지 전체가 대기한다.

```
시간 축 →
요청 1: |████████|                    (800 tokens)
요청 2: |████████████████|            (1600 tokens)
요청 3: |████|                        (400 tokens)
요청 4: |████████████████████████|    (2400 tokens) ← 병목
         ├────────────────────────┤
         전체 배치가 이 시점에서야 완료
         요청 1, 3은 이미 끝났지만 GPU가 놀고 있다
```

### 수학적 분석

배치 크기 $B$일 때, 각 요청 $i$의 처리 시간을 $t_i$라 하면:

$$
\text{Batch Time} = \max(t_1, t_2, \ldots, t_B)
$$

$$
\text{Throughput}_{\text{static}} = \frac{B}{\max(t_1, t_2, \ldots, t_B)}
$$

**GPU 활용률:**

$$
\text{GPU Utilization} = \frac{\sum_{i=1}^{B} t_i}{B \times \max(t_1, \ldots, t_B)}
$$

**예시:**

4개 요청, 처리 시간 = [2s, 5s, 1s, 8s]

$$
\text{Throughput} = \frac{4}{8} = 0.5 \text{ req/s}
$$

$$
\text{GPU Util} = \frac{2 + 5 + 1 + 8}{4 \times 8} = \frac{16}{32} = 50\%
$$

GPU 절반이 낭비된다.

### 구현

```python
"""
Static Batching 구현
"""

import time
import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceRequest:
    """추론 요청"""
    request_id: str
    input_tokens: list[int]
    max_output_tokens: int
    arrival_time: float = field(default_factory=time.time)
    result: Any = None
    completion_time: float = 0.0


class StaticBatcher:
    """
    Static Batching: 고정 크기 배치.

    장점: 구현 단순
    단점: 패딩 낭비, 긴 요청이 짧은 요청을 블로킹
    """

    def __init__(self, batch_size: int = 8, max_wait_seconds: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_seconds = max_wait_seconds
        self.queue: list[InferenceRequest] = []
        self._lock = asyncio.Lock()

    async def add_request(self, request: InferenceRequest) -> None:
        async with self._lock:
            self.queue.append(request)

    async def collect_batch(self) -> list[InferenceRequest]:
        """배치 수집: batch_size에 도달하거나 max_wait 초과 시 반환"""
        start = time.time()

        while True:
            async with self._lock:
                if len(self.queue) >= self.batch_size:
                    batch = self.queue[:self.batch_size]
                    self.queue = self.queue[self.batch_size:]
                    return batch

            elapsed = time.time() - start
            if elapsed >= self.max_wait_seconds:
                async with self._lock:
                    if self.queue:
                        batch = self.queue[:]
                        self.queue.clear()
                        return batch

            await asyncio.sleep(0.01)

    def compute_padding_waste(self, batch: list[InferenceRequest]) -> dict:
        """패딩으로 인한 연산 낭비 계산"""
        lengths = [len(r.input_tokens) for r in batch]
        max_len = max(lengths)
        total_tokens = sum(lengths)
        padded_tokens = max_len * len(batch)
        waste_ratio = 1 - (total_tokens / padded_tokens) if padded_tokens > 0 else 0

        return {
            "batch_size": len(batch),
            "max_length": max_len,
            "total_real_tokens": total_tokens,
            "total_padded_tokens": padded_tokens,
            "waste_ratio": round(waste_ratio, 4),
        }
```

---

## 15.3.2 Dynamic Batching

### 동작 원리

요청이 도착할 때마다 큐에 넣고, 일정 조건(크기 또는 시간)을 만족하면 배치를 구성한다. Static보다 유연하지만, 여전히 배치 내에서 가장 긴 요청에 의한 블로킹 문제가 있다.

```
시간 축 →
┌─ Batch 1 ─────────────┐  ┌─ Batch 2 ────────────┐
│ 요청 A: |████████|     │  │ 요청 D: |██████|      │
│ 요청 B: |████|         │  │ 요청 E: |████████████| │
│ 요청 C: |██████████|   │  │ 요청 F: |████|        │
└────────────────────────┘  └───────────────────────┘
  배치 타임아웃 or 크기 도달      다음 배치 즉시 시작
```

### 수학적 분석

동적 배치의 평균 대기 시간:

$$
W_{\text{queue}} = \min\left(\frac{B - 1}{2\lambda}, \, T_{\text{timeout}}\right)
$$

여기서:
- $B$: 배치 크기
- $\lambda$: 요청 도착률 (req/s)
- $T_{\text{timeout}}$: 최대 대기 시간

총 응답 시간:

$$
T_{\text{total}} = W_{\text{queue}} + T_{\text{inference}}
$$

**트레이드오프:**

- $T_{\text{timeout}}$이 크면: 배치가 커져 throughput ↑, 하지만 latency ↑
- $T_{\text{timeout}}$이 작으면: 배치가 작아져 latency ↓, 하지만 throughput ↓

### 구현

```python
"""
Dynamic Batching 구현
"""

import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DynamicBatchConfig:
    """동적 배칭 설정"""
    max_batch_size: int = 16
    max_wait_ms: float = 100.0         # 최대 대기 시간 (ms)
    max_total_tokens: int = 16384      # 배치 내 최대 총 토큰 수
    prefer_similar_lengths: bool = True # 비슷한 길이끼리 그룹핑


class DynamicBatcher:
    """
    Dynamic Batching with Token Budget.

    배치 구성 조건 (OR):
    1. 큐 내 요청 수 >= max_batch_size
    2. 대기 시간 >= max_wait_ms
    3. 큐 내 총 토큰 수 >= max_total_tokens
    """

    def __init__(self, config: DynamicBatchConfig):
        self.config = config
        self.queue: list[InferenceRequest] = []
        self._lock = asyncio.Lock()
        self._event = asyncio.Event()

    async def add_request(self, request: InferenceRequest) -> None:
        async with self._lock:
            self.queue.append(request)
        self._event.set()

    def _total_tokens(self, requests: list[InferenceRequest]) -> int:
        return sum(len(r.input_tokens) + r.max_output_tokens for r in requests)

    async def collect_batch(self) -> list[InferenceRequest]:
        """조건 기반 배치 수집"""
        deadline = time.time() + self.config.max_wait_ms / 1000.0

        while True:
            async with self._lock:
                # 조건 1: 배치 크기 충족
                if len(self.queue) >= self.config.max_batch_size:
                    return self._extract_batch()

                # 조건 3: 토큰 예산 충족
                if self.queue and self._total_tokens(self.queue) >= self.config.max_total_tokens:
                    return self._extract_batch()

            # 조건 2: 타임아웃
            remaining = deadline - time.time()
            if remaining <= 0:
                async with self._lock:
                    if self.queue:
                        return self._extract_batch()
                    else:
                        deadline = time.time() + self.config.max_wait_ms / 1000.0

            try:
                await asyncio.wait_for(self._event.wait(), timeout=max(remaining, 0.001))
                self._event.clear()
            except asyncio.TimeoutError:
                pass

    def _extract_batch(self) -> list[InferenceRequest]:
        """배치 추출 (길이 기반 정렬 옵션)"""
        if self.config.prefer_similar_lengths:
            self.queue.sort(key=lambda r: len(r.input_tokens))

        batch_size = min(len(self.queue), self.config.max_batch_size)

        # 토큰 예산 내에서 최대한 많이 넣기
        batch = []
        total_tokens = 0
        for req in self.queue[:batch_size]:
            req_tokens = len(req.input_tokens) + req.max_output_tokens
            if total_tokens + req_tokens > self.config.max_total_tokens and batch:
                break
            batch.append(req)
            total_tokens += req_tokens

        # 큐에서 제거
        for req in batch:
            self.queue.remove(req)

        logger.info(
            f"배치 구성: size={len(batch)}, total_tokens={total_tokens}"
        )
        return batch
```

---

## 15.3.3 Continuous Batching

### 동작 원리

Continuous Batching(iteration-level scheduling)은 각 디코딩 스텝마다 스케줄링을 수행한다. 요청이 완료되면 즉시 GPU 슬롯에서 제거하고 새 요청을 삽입한다. **GPU가 쉬는 시간이 없다.**

```
시간 축 →
Slot 1: |█A██|█D████|█F██|█H█|...
Slot 2: |█B████████|█E██|█G████|...
Slot 3: |█C██████|█D'███|█I█|...
         ↑        ↑       ↑
         A 완료 → D 삽입   D' 완료 → I 삽입

모든 슬롯이 항상 활성 상태
→ GPU 활용률 ≈ 100%
```

### 수학적 분석

Continuous Batching의 이론적 throughput:

$$
\text{Throughput}_{\text{continuous}} \approx \sum_{i=1}^{N} \frac{1}{t_i} \times \frac{B}{N}
$$

더 정확하게는, 정상 상태에서의 throughput:

$$
\text{Throughput}_{\text{continuous}} = \frac{B}{\bar{t}}
$$

여기서 $\bar{t}$는 평균 처리 시간, $B$는 동시 처리 슬롯 수.

**Static vs Continuous 비교:**

$$
\frac{\text{Throughput}_{\text{continuous}}}{\text{Throughput}_{\text{static}}} = \frac{B / \bar{t}}{B / \max(t_i)} = \frac{\max(t_i)}{\bar{t}}
$$

이 비율은 요청 간 처리 시간 편차가 클수록 커진다. 의료 문서 OCR에서는:

- 처방전: $t \approx 1$s (짧은 문서)
- 병원 차트: $t \approx 10$s (긴 문서)
- $\max / \bar{t} \approx 10/3 \approx 3.3\times$ 개선

### Little's Law

$$
L = \lambda W
$$

여기서:
- $L$: 시스템 내 평균 요청 수 (동시에 처리 중인 요청)
- $\lambda$: 요청 도착률 (req/s)
- $W$: 평균 체류 시간 (대기 + 처리)

**활용:**

목표 throughput $\lambda = 30$ req/s, 평균 처리 시간 $W = 2$s이면:

$$
L = 30 \times 2 = 60
$$

동시에 60개 요청을 처리해야 한다. 각 GPU가 8개 동시 처리하면 $60 / 8 = 8$대의 GPU가 필요하다.

### 구현

```python
"""
Continuous Batching Scheduler 구현
"""

import time
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class RequestState(Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class SequenceState:
    """시퀀스 상태 추적"""
    request_id: str
    input_tokens: list[int]
    generated_tokens: list[int] = field(default_factory=list)
    max_output_tokens: int = 4096
    state: RequestState = RequestState.WAITING
    arrival_time: float = field(default_factory=time.time)
    start_time: float = 0.0
    end_time: float = 0.0
    kv_cache_blocks: int = 0

    @property
    def total_tokens(self) -> int:
        return len(self.input_tokens) + len(self.generated_tokens)

    @property
    def is_finished(self) -> bool:
        return (
            self.state == RequestState.FINISHED
            or len(self.generated_tokens) >= self.max_output_tokens
        )

    @property
    def latency(self) -> float:
        if self.end_time > 0 and self.start_time > 0:
            return self.end_time - self.start_time
        return 0.0


@dataclass
class SchedulerConfig:
    """스케줄러 설정"""
    max_num_seqs: int = 32               # 동시 처리 최대 시퀀스 수
    max_num_batched_tokens: int = 16384  # 배치 당 최대 토큰 수
    max_paddings: int = 256              # 허용 최대 패딩 토큰 수
    preemption_mode: str = "recompute"   # 선점 방식: recompute or swap
    enable_chunked_prefill: bool = True  # 청크 프리필 활성화


class ContinuousBatchScheduler:
    """
    Continuous Batching 스케줄러.

    각 iteration에서:
    1. 완료된 시퀀스 제거
    2. Waiting 큐에서 새 시퀀스 추가 (KV Cache 여유분만큼)
    3. Running 시퀀스 배치 구성
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.waiting_queue: list[SequenceState] = []
        self.running: list[SequenceState] = []
        self.finished: list[SequenceState] = []
        self._total_blocks_used = 0
        self._max_blocks = 1000  # 예시: 총 KV Cache 블록 수

        # 메트릭
        self._total_processed = 0
        self._total_latency = 0.0

    def add_request(self, seq: SequenceState) -> None:
        """새 요청을 Waiting 큐에 추가"""
        seq.state = RequestState.WAITING
        self.waiting_queue.append(seq)
        logger.debug(f"요청 추가: {seq.request_id}, waiting={len(self.waiting_queue)}")

    def schedule(self) -> dict:
        """
        한 iteration의 스케줄링 수행.

        반환:
        - prefill_batch: 이번 스텝에서 프리필할 시퀀스들
        - decode_batch: 이번 스텝에서 디코딩할 시퀀스들
        - preempted: 선점된 시퀀스들
        """
        # Step 1: 완료된 시퀀스 제거
        newly_finished = []
        still_running = []
        for seq in self.running:
            if seq.is_finished:
                seq.state = RequestState.FINISHED
                seq.end_time = time.time()
                self._release_blocks(seq)
                newly_finished.append(seq)
                self._total_processed += 1
                self._total_latency += seq.latency
            else:
                still_running.append(seq)

        self.running = still_running
        self.finished.extend(newly_finished)

        # Step 2: Waiting 큐에서 새 시퀀스 삽입
        prefill_batch = []
        available_slots = self.config.max_num_seqs - len(self.running)
        available_blocks = self._max_blocks - self._total_blocks_used

        while self.waiting_queue and available_slots > 0:
            seq = self.waiting_queue[0]
            required_blocks = self._estimate_blocks(seq)

            if required_blocks > available_blocks:
                # KV Cache 부족 → 선점 또는 대기
                if self.config.preemption_mode == "recompute" and self.running:
                    # 가장 최근 시퀀스를 선점
                    preempted = self._preempt_last()
                    if preempted:
                        available_blocks += preempted.kv_cache_blocks
                        continue
                break

            self.waiting_queue.pop(0)
            seq.state = RequestState.PREFILLING
            seq.start_time = time.time()
            seq.kv_cache_blocks = required_blocks
            self._total_blocks_used += required_blocks

            prefill_batch.append(seq)
            self.running.append(seq)
            available_slots -= 1
            available_blocks -= required_blocks

        # Step 3: 디코딩 배치 구성
        decode_batch = [
            seq for seq in self.running
            if seq.state == RequestState.DECODING
        ]

        # 프리필 완료 → 디코딩으로 전환
        for seq in prefill_batch:
            seq.state = RequestState.DECODING

        return {
            "prefill_batch": prefill_batch,
            "decode_batch": decode_batch,
            "newly_finished": newly_finished,
            "waiting": len(self.waiting_queue),
            "running": len(self.running),
        }

    def _estimate_blocks(self, seq: SequenceState) -> int:
        """시퀀스에 필요한 KV Cache 블록 수 추정"""
        total_tokens = len(seq.input_tokens) + seq.max_output_tokens
        block_size = 16  # 블록당 토큰 수
        return (total_tokens + block_size - 1) // block_size

    def _release_blocks(self, seq: SequenceState) -> None:
        """완료된 시퀀스의 KV Cache 블록 해제"""
        self._total_blocks_used -= seq.kv_cache_blocks
        seq.kv_cache_blocks = 0

    def _preempt_last(self) -> Optional[SequenceState]:
        """가장 최근 추가된 시퀀스를 선점(중단 후 재큐잉)"""
        if not self.running:
            return None

        victim = self.running.pop()
        victim.state = RequestState.WAITING
        victim.generated_tokens.clear()  # recompute 모드
        self._release_blocks(victim)
        self.waiting_queue.insert(0, victim)

        logger.warning(f"시퀀스 선점: {victim.request_id}")
        return victim

    def get_metrics(self) -> dict:
        """스케줄러 메트릭"""
        avg_latency = (
            self._total_latency / self._total_processed
            if self._total_processed > 0 else 0
        )
        return {
            "total_processed": self._total_processed,
            "avg_latency_s": round(avg_latency, 3),
            "waiting_queue_size": len(self.waiting_queue),
            "running_size": len(self.running),
            "kv_cache_utilization": round(
                self._total_blocks_used / self._max_blocks, 3
            ),
        }
```

---

## 15.3.4 Chunked Prefill

### 문제

VLM에서 의료 문서 이미지를 처리하면 prefill 단계에서 수천 개의 이미지 토큰이 생긴다. Prefill이 오래 걸리면 이미 디코딩 중인 요청의 latency가 증가한다(Head-of-Line blocking).

### 해결: Chunked Prefill

Prefill을 고정 크기 청크로 분할하여 디코딩 요청과 interleave한다:

$$
\text{Prefill Tokens} = [c_1, c_2, \ldots, c_K] \quad \text{where } |c_k| = C_{\text{chunk}}
$$

$$
K = \left\lceil \frac{N_{\text{prefill}}}{C_{\text{chunk}}} \right\rceil
$$

```python
"""
Chunked Prefill 구현
"""

from dataclasses import dataclass


@dataclass
class ChunkedPrefillConfig:
    """청크 프리필 설정"""
    chunk_size: int = 512             # 한 iteration에 처리할 프리필 토큰 수
    max_prefill_ratio: float = 0.5    # 배치 토큰 중 프리필 비율 상한


class ChunkedPrefillScheduler:
    """
    Chunked Prefill 스케줄러.

    각 iteration에서:
    - 디코딩 토큰: N_decode (각 시퀀스 1토큰)
    - 프리필 토큰: min(chunk_size, budget - N_decode)
    - 전체 토큰: N_decode + N_prefill <= max_batch_tokens
    """

    def __init__(self, config: ChunkedPrefillConfig, max_batch_tokens: int = 4096):
        self.config = config
        self.max_batch_tokens = max_batch_tokens
        self.prefill_queue: list[dict] = []  # (seq_id, remaining_tokens)

    def add_prefill(self, seq_id: str, total_tokens: int) -> None:
        """프리필 대기열에 추가"""
        self.prefill_queue.append({
            "seq_id": seq_id,
            "remaining": total_tokens,
            "processed": 0,
        })

    def schedule_iteration(self, num_decode_seqs: int) -> dict:
        """
        한 iteration 스케줄링.

        반환:
        - decode_budget: 디코딩에 할당된 토큰 수
        - prefill_chunks: [(seq_id, start, end), ...] 프리필 청크들
        """
        decode_tokens = num_decode_seqs  # 디코딩은 시퀀스당 1토큰

        # 프리필 예산 계산
        total_budget = self.max_batch_tokens
        prefill_budget = min(
            int(total_budget * self.config.max_prefill_ratio),
            total_budget - decode_tokens,
            self.config.chunk_size,
        )
        prefill_budget = max(prefill_budget, 0)

        # 프리필 청크 할당
        prefill_chunks = []
        remaining_budget = prefill_budget

        while self.prefill_queue and remaining_budget > 0:
            entry = self.prefill_queue[0]
            chunk_size = min(entry["remaining"], remaining_budget)

            prefill_chunks.append({
                "seq_id": entry["seq_id"],
                "start": entry["processed"],
                "end": entry["processed"] + chunk_size,
            })

            entry["processed"] += chunk_size
            entry["remaining"] -= chunk_size
            remaining_budget -= chunk_size

            if entry["remaining"] <= 0:
                self.prefill_queue.pop(0)

        return {
            "decode_budget": decode_tokens,
            "prefill_budget": prefill_budget - remaining_budget,
            "prefill_chunks": prefill_chunks,
            "total_tokens": decode_tokens + (prefill_budget - remaining_budget),
            "pending_prefills": len(self.prefill_queue),
        }
```

---

## 15.3.5 의료 문서 OCR에서의 배칭 전략 선택

### 문서 길이 분포 분석

의료 문서 OCR 워크로드의 특성:

| 문서 유형 | 평균 출력 토큰 | 표준 편차 | 최대 토큰 |
|-----------|--------------|-----------|----------|
| 처방전 | 200 | 80 | 500 |
| 진단서 | 800 | 300 | 2000 |
| 검사 결과 | 400 | 150 | 1000 |
| 병원 차트 | 2000 | 800 | 5000 |
| 혼합 워크로드 | 600 | 700 | 5000 |

변동 계수(Coefficient of Variation):

$$
CV = \frac{\sigma}{\mu} = \frac{700}{600} \approx 1.17
$$

$CV > 1$이면 편차가 매우 크다는 의미. **Continuous Batching이 필수적이다.**

### 전략별 성능 비교 시뮬레이션

```python
"""
배칭 전략 성능 비교 시뮬레이션
"""

import random
import statistics
from dataclasses import dataclass


@dataclass
class SimulationResult:
    """시뮬레이션 결과"""
    strategy: str
    total_time: float
    throughput: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    gpu_utilization: float


def simulate_static_batching(
    processing_times: list[float],
    batch_size: int = 8,
) -> SimulationResult:
    """Static Batching 시뮬레이션"""
    latencies = []
    total_time = 0.0
    total_compute = 0.0

    for i in range(0, len(processing_times), batch_size):
        batch = processing_times[i:i + batch_size]
        batch_time = max(batch)
        total_time += batch_time
        total_compute += sum(batch)

        for t in batch:
            latencies.append(batch_time)  # 모든 요청이 max 시간만큼 대기

    latencies.sort()
    n = len(latencies)

    return SimulationResult(
        strategy="Static Batching",
        total_time=round(total_time, 2),
        throughput=round(len(processing_times) / total_time, 2),
        avg_latency=round(statistics.mean(latencies), 3),
        p50_latency=round(latencies[int(n * 0.50)], 3),
        p95_latency=round(latencies[int(n * 0.95)], 3),
        p99_latency=round(latencies[int(n * 0.99)], 3),
        gpu_utilization=round(total_compute / (total_time * batch_size), 3),
    )


def simulate_continuous_batching(
    processing_times: list[float],
    max_concurrent: int = 8,
) -> SimulationResult:
    """Continuous Batching 시뮬레이션 (이벤트 기반)"""
    import heapq

    # 이벤트 큐: (완료 시각, 요청 인덱스)
    event_queue: list[tuple[float, int]] = []
    latencies = []
    current_time = 0.0
    next_request = 0
    active_slots = 0

    # 초기 슬롯 채우기
    while next_request < len(processing_times) and active_slots < max_concurrent:
        t = processing_times[next_request]
        heapq.heappush(event_queue, (current_time + t, next_request))
        active_slots += 1
        next_request += 1

    # 이벤트 처리
    while event_queue:
        finish_time, req_idx = heapq.heappop(event_queue)
        current_time = finish_time
        latencies.append(processing_times[req_idx])
        active_slots -= 1

        # 완료 즉시 새 요청 삽입
        if next_request < len(processing_times):
            t = processing_times[next_request]
            heapq.heappush(event_queue, (current_time + t, next_request))
            active_slots += 1
            next_request += 1

    total_time = current_time
    total_compute = sum(processing_times)

    latencies.sort()
    n = len(latencies)

    return SimulationResult(
        strategy="Continuous Batching",
        total_time=round(total_time, 2),
        throughput=round(len(processing_times) / total_time, 2),
        avg_latency=round(statistics.mean(latencies), 3),
        p50_latency=round(latencies[int(n * 0.50)], 3),
        p95_latency=round(latencies[int(n * 0.95)], 3),
        p99_latency=round(latencies[int(n * 0.99)], 3),
        gpu_utilization=round(total_compute / (total_time * max_concurrent), 3),
    )


def run_comparison():
    """의료 문서 OCR 워크로드로 배칭 전략 비교"""
    random.seed(42)

    # 의료 문서 OCR 처리 시간 분포 (초)
    # 혼합 워크로드: 처방전(짧음) ~ 병원 차트(긺)
    processing_times = []
    for _ in range(200):
        doc_type = random.choice(["prescription", "diagnosis", "test", "chart"])
        if doc_type == "prescription":
            t = max(0.5, random.gauss(1.0, 0.4))
        elif doc_type == "diagnosis":
            t = max(1.0, random.gauss(3.0, 1.2))
        elif doc_type == "test":
            t = max(0.8, random.gauss(2.0, 0.8))
        else:  # chart
            t = max(2.0, random.gauss(6.0, 2.5))
        processing_times.append(round(t, 2))

    print(f"총 요청 수: {len(processing_times)}")
    print(f"처리 시간: mean={statistics.mean(processing_times):.2f}s, "
          f"std={statistics.stdev(processing_times):.2f}s, "
          f"max={max(processing_times):.2f}s")
    print()

    # Static Batching
    static_result = simulate_static_batching(processing_times, batch_size=8)
    print(f"=== {static_result.strategy} ===")
    print(f"  총 처리 시간: {static_result.total_time}s")
    print(f"  Throughput: {static_result.throughput} req/s")
    print(f"  Avg Latency: {static_result.avg_latency}s")
    print(f"  p95 Latency: {static_result.p95_latency}s")
    print(f"  GPU 활용률: {static_result.gpu_utilization * 100:.1f}%")
    print()

    # Continuous Batching
    continuous_result = simulate_continuous_batching(processing_times, max_concurrent=8)
    print(f"=== {continuous_result.strategy} ===")
    print(f"  총 처리 시간: {continuous_result.total_time}s")
    print(f"  Throughput: {continuous_result.throughput} req/s")
    print(f"  Avg Latency: {continuous_result.avg_latency}s")
    print(f"  p95 Latency: {continuous_result.p95_latency}s")
    print(f"  GPU 활용률: {continuous_result.gpu_utilization * 100:.1f}%")
    print()

    # 개선율
    improvement = (
        continuous_result.throughput / static_result.throughput - 1
    ) * 100
    print(f"Continuous Batching Throughput 개선: +{improvement:.1f}%")


if __name__ == "__main__":
    run_comparison()
```

---

## 15.3.6 배칭 전략 요약

| 특성 | Static | Dynamic | Continuous |
|------|--------|---------|------------|
| GPU 활용률 | 낮음 (패딩 낭비) | 중간 | 높음 (≈100%) |
| 구현 복잡도 | 낮음 | 중간 | 높음 |
| Latency 변동 | 큼 (max에 종속) | 중간 | 낮음 |
| 가변 길이 처리 | 나쁨 | 보통 | 우수 |
| 메모리 관리 | 단순 | 단순 | 복잡 (PagedAttention 필요) |
| 적합한 워크로드 | 균일한 입력 | 일반적 | 편차 큰 입력 |
| 의료 문서 OCR | 부적합 | 보통 | **최적** |

---

## 용어 체크리스트

학습 후 아래 질문에 답할 수 있어야 한다:

- [ ] Static Batching에서 GPU 활용률 공식 $\sum t_i / (B \times \max t_i)$를 유도할 수 있는가?
- [ ] Continuous Batching이 Static 대비 $\max(t_i) / \bar{t}$ 배 개선되는 이유는?
- [ ] Little's Law $L = \lambda W$에서 각 변수의 의미와 용량 계획 활용법은?
- [ ] Dynamic Batching의 max_wait 파라미터가 throughput/latency 트레이드오프에 미치는 영향은?
- [ ] Chunked Prefill이 Head-of-Line blocking을 해결하는 원리는?
- [ ] 변동 계수(CV)가 1보다 클 때 Continuous Batching이 필수적인 이유는?
- [ ] KV Cache 블록 선점(preemption)의 recompute vs swap 방식 차이는?
- [ ] 의료 문서 OCR에서 문서 유형별 처리 시간 편차가 배칭 전략 선택에 미치는 영향은?
