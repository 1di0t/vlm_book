---
---

# 14.4 벤치마크 (Benchmark)

> **모델 추론 성능을 체계적으로 측정하고 비교하기 위한 방법론과 도구**

---

## 핵심 용어

| 용어 | 정의 |
|------|------|
| **Latency** | 요청 1건의 처리에 걸리는 총 시간 (초 단위) |
| **Throughput** | 단위 시간당 처리하는 토큰 또는 요청 수 (tokens/sec, requests/sec) |
| **TTFT** (Time To First Token) | 요청 전송 후 첫 번째 토큰이 생성될 때까지의 시간 |
| **TPOT** (Time Per Output Token) | 출력 토큰 하나를 생성하는 데 걸리는 평균 시간 |
| **ITL** (Inter-Token Latency) | 연속된 출력 토큰 사이의 시간 간격 |
| **Memory Footprint** | 모델 추론에 필요한 총 GPU/시스템 메모리 |
| **Roofline Model** | 하드웨어의 compute/memory 한계를 시각화하는 성능 모델 |
| **Arithmetic Intensity** | 연산 대비 메모리 접근 비율 (FLOPs/Byte) |
| **P50/P95/P99** | 지연시간의 50/95/99 백분위수 |
| **Warm-up** | 측정 전 GPU 캐시/JIT를 안정시키는 사전 실행 단계 |

---

## 수학적 원리

### 1. Roofline Model

하드웨어 성능의 상한선을 두 가지 자원으로 모델링한다:

$$
\text{Attainable Performance} = \min\left(\text{Peak FLOPS},\; \text{AI} \times \text{Peak Bandwidth}\right)
$$

여기서 AI(Arithmetic Intensity)는:

$$
\text{AI} = \frac{\text{FLOPs (연산량)}}{\text{Bytes (메모리 전송량)}}
$$

**경계점 (Ridge Point)**:

$$
\text{AI}_{\text{ridge}} = \frac{\text{Peak FLOPS}}{\text{Peak Bandwidth}}
$$

- $\text{AI} < \text{AI}_{\text{ridge}}$: **Memory-bound** → 메모리 대역폭이 병목
- $\text{AI} > \text{AI}_{\text{ridge}}$: **Compute-bound** → 연산 능력이 병목

**GPU별 Ridge Point 예시**:

| GPU | Peak FLOPS (FP16) | Bandwidth | Ridge Point |
|-----|-------------------|-----------|-------------|
| A100 80GB | 312 TFLOPS | 2.0 TB/s | 156 |
| H100 SXM | 990 TFLOPS | 3.35 TB/s | 296 |
| RTX 4090 | 165 TFLOPS | 1.0 TB/s | 165 |
| RTX 3060 12GB | 12.7 TFLOPS | 0.36 TB/s | 35 |

### 2. LLM 추론의 Arithmetic Intensity

**Prefill 단계**:

입력 토큰 $n$개에 대한 행렬 연산:

$$
\text{AI}_{\text{prefill}} = \frac{2 \times N_{\text{params}} \times n}{2 \times N_{\text{params}} \times b_{\text{param}}} = \frac{n}{b_{\text{param}}}
$$

FP16 ($b = 2$)일 때:
- $n = 128$: AI = 64 → memory-bound (A100 기준)
- $n = 512$: AI = 256 → compute-bound (A100 기준)
- $n = 2048$: AI = 1024 → 완전 compute-bound

**Decode 단계** (batch size $B$):

$$
\text{AI}_{\text{decode}} = \frac{2 \times N_{\text{params}} \times B}{2 \times N_{\text{params}} \times b_{\text{param}}} = \frac{B}{b_{\text{param}}}
$$

FP16 ($b = 2$)일 때:
- $B = 1$: AI = 0.5 → 완전 memory-bound
- $B = 32$: AI = 16 → 여전히 memory-bound (A100 기준)
- $B = 512$: AI = 256 → compute-bound 진입

**핵심 인사이트**: Decode는 배칭 없이는 **항상** memory-bound이다. 배치 크기를 키워서 AI를 높이는 것이 처리량 극대화의 핵심이다.

### 3. 처리량 계산

**Throughput (tokens/sec)**:

$$
\text{Throughput} = \frac{B \times n_{\text{output}}}{T_{\text{total}}}
$$

여기서:

$$
T_{\text{total}} = T_{\text{prefill}} + n_{\text{output}} \times T_{\text{decode\_per\_token}}
$$

**단순화된 Decode Throughput (memory-bound 가정)**:

$$
\text{Throughput}_{\text{decode}} = \frac{B \times \text{Bandwidth}}{2 \times N_{\text{params}} \times b_{\text{param}}}
$$

A100 80GB, 7B FP16 모델, $B = 32$:

$$
\text{Throughput} = \frac{32 \times 2.0 \times 10^{12}}{2 \times 7 \times 10^9 \times 2} \approx 2286 \;\text{tokens/sec}
$$

### 4. TTFT와 TPOT

**TTFT (Time To First Token)**:

$$
\text{TTFT} = T_{\text{network}} + T_{\text{queue}} + T_{\text{prefill}}
$$

- $T_{\text{network}}$: 네트워크 지연
- $T_{\text{queue}}$: 스케줄러 대기 시간
- $T_{\text{prefill}}$: Prefill 연산 시간

**TPOT (Time Per Output Token)**:

$$
\text{TPOT} = \frac{T_{\text{total}} - \text{TTFT}}{n_{\text{output}} - 1}
$$

**End-to-End Latency**:

$$
T_{\text{e2e}} = \text{TTFT} + (n_{\text{output}} - 1) \times \text{TPOT}
$$

### 5. 백분위수 지연시간

P50, P95, P99는 지연시간 분포의 안정성을 측정한다:

$$
P_k = \text{value at } k\text{-th percentile of sorted latencies}
$$

프로덕션에서는 **P99 < SLA 임계치**를 만족해야 한다.

**Tail Latency Ratio**:

$$
\text{TLR} = \frac{P_{99}}{P_{50}}
$$

TLR이 작을수록 지연시간이 안정적이다. 이상적으로 TLR < 2.

---

## 벤치마크 설계

### 테스트 시나리오

| 시나리오 | 설명 | 측정 지표 |
|---------|------|----------|
| **Single Request** | 배치 없이 단건 요청 | TTFT, TPOT, E2E Latency |
| **Fixed Batch** | 고정 배치 크기로 동시 처리 | Throughput, Memory |
| **Increasing Load** | 요청 빈도를 점진적으로 증가 | Max Throughput, P99 Latency |
| **Long Context** | 긴 입력 (4K~128K 토큰) | TTFT, Memory Scaling |
| **Prefix Sharing** | 동일 prefix 반복 요청 | KV Cache Hit Rate, Throughput |

### 워크로드 프로파일

| 파라미터 | 짧은 대화 | 문서 처리 | 의료 OCR |
|---------|----------|----------|----------|
| 입력 토큰 | 50~200 | 1000~4000 | 2000~8000 |
| 출력 토큰 | 100~500 | 200~1000 | 200~800 |
| 배치 크기 | 1~8 | 16~64 | 8~32 |
| 동시 요청 | 높음 | 보통 | 보통 |
| Prefix 공유 | 낮음 | 보통 | 높음 |

---

## 의료 문서 OCR 벤치마크

### 문서 유형별 벤치마크

| 문서 유형 | 평균 입력 토큰 | 평균 출력 토큰 | 특이사항 |
|----------|-------------|-------------|---------|
| 처방전 | 500~1500 | 200~400 | 약물명 정확도 중요 |
| 검사 결과지 | 1000~3000 | 300~600 | 수치 정확도 중요 |
| 진단서 | 800~2000 | 200~500 | 진단 코드 정확도 |
| 퇴원 요약서 | 3000~8000 | 500~1500 | 긴 문맥 처리 |
| 의무 기록 | 2000~6000 | 400~1000 | 필체 인식 난이도 |

### 해상도별 벤치마크

| 해상도 | 이미지 토큰 수 (추정) | TTFT 영향 | 정확도 영향 |
|--------|---------------------|----------|-----------|
| 72 DPI | ~500 | 낮음 | 낮음 (정보 손실) |
| 150 DPI | ~1500 | 보통 | 보통 |
| 300 DPI | ~4000 | 높음 | 높음 |
| 600 DPI | ~12000 | 매우 높음 | 최고 (한계 수익) |

### 벤치마크 체크리스트

```
□ 정확도 지표: CER, WER, 약물 정확도, 진단 코드 F1
□ 속도 지표: TTFT, TPOT, E2E Latency, Throughput
□ 자원 지표: GPU Memory Peak, GPU Utilization, Power Draw
□ 안정성 지표: P50/P95/P99 Latency, Tail Latency Ratio
□ 확장성 지표: Throughput vs Batch Size, Memory vs Sequence Length
```

---

## 코드

### 1. BenchmarkRunner 클래스

```python
"""
LLM 추론 벤치마크 러너
- TTFT, TPOT, Throughput, Memory 측정
- 의료 문서 OCR 워크로드 지원
- 통계적 신뢰도를 위한 반복 측정
"""

import time
import json
import statistics
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkMode(Enum):
    SINGLE = "single"           # 단건 요청 (지연시간 측정)
    BATCH = "batch"             # 고정 배치 (처리량 측정)
    SWEEP = "sweep"             # 배치 크기 스윕
    STRESS = "stress"           # 부하 테스트


@dataclass
class BenchmarkConfig:
    """벤치마크 설정"""
    name: str = "llm_benchmark"
    mode: BenchmarkMode = BenchmarkMode.SINGLE
    num_warmup: int = 3                  # 워밍업 횟수
    num_iterations: int = 10             # 측정 반복 횟수
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    input_lengths: List[int] = field(default_factory=lambda: [512, 1024, 2048, 4096])
    output_length: int = 256
    timeout_sec: float = 120.0
    output_dir: str = "./benchmark_results"


@dataclass
class TokenTiming:
    """토큰별 타이밍 정보"""
    token_idx: int
    timestamp: float            # 절대 시간
    elapsed_ms: float           # 요청 시작 후 경과 시간 (ms)


@dataclass
class RequestResult:
    """단일 요청 결과"""
    request_id: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float              # Time To First Token (ms)
    tpot_ms: float              # Time Per Output Token (ms)
    total_latency_ms: float     # 전체 지연시간 (ms)
    tokens_per_sec: float       # 토큰 생성 속도
    token_timings: List[TokenTiming] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class BatchResult:
    """배치 벤치마크 결과"""
    batch_size: int
    input_length: int
    output_length: int
    results: List[RequestResult]
    wall_time_sec: float                # 벽시계 시간
    throughput_tokens_per_sec: float    # 전체 처리량
    throughput_requests_per_sec: float  # 요청 처리량

    @property
    def ttft_stats(self) -> Dict[str, float]:
        vals = [r.ttft_ms for r in self.results if r.error is None]
        return self._compute_stats(vals, "ttft_ms")

    @property
    def tpot_stats(self) -> Dict[str, float]:
        vals = [r.tpot_ms for r in self.results if r.error is None]
        return self._compute_stats(vals, "tpot_ms")

    @property
    def latency_stats(self) -> Dict[str, float]:
        vals = [r.total_latency_ms for r in self.results if r.error is None]
        return self._compute_stats(vals, "latency_ms")

    @staticmethod
    def _compute_stats(values: List[float], name: str) -> Dict[str, float]:
        if not values:
            return {}
        values_sorted = sorted(values)
        n = len(values_sorted)
        return {
            f"{name}_mean": statistics.mean(values),
            f"{name}_median": statistics.median(values),
            f"{name}_std": statistics.stdev(values) if n > 1 else 0,
            f"{name}_min": min(values),
            f"{name}_max": max(values),
            f"{name}_p50": values_sorted[int(n * 0.50)],
            f"{name}_p95": values_sorted[int(n * 0.95)] if n >= 20 else values_sorted[-1],
            f"{name}_p99": values_sorted[int(n * 0.99)] if n >= 100 else values_sorted[-1],
        }


class BenchmarkRunner:
    """LLM 추론 벤치마크 실행기"""

    def __init__(
        self,
        config: BenchmarkConfig,
        inference_fn: Callable,
        memory_fn: Optional[Callable] = None,
    ):
        """
        Args:
            config: 벤치마크 설정
            inference_fn: 추론 함수. (prompt, max_tokens) -> (output_text, token_timings)
            memory_fn: GPU 메모리 측정 함수. () -> (used_gb, total_gb)
        """
        self.config = config
        self.inference_fn = inference_fn
        self.memory_fn = memory_fn
        self.results: List[BatchResult] = []

        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _generate_prompt(self, target_tokens: int) -> str:
        """지정 토큰 수에 근사하는 프롬프트 생성"""
        # 한국어 기준 1토큰 ≈ 2~3글자
        chars_per_token = 2.5
        target_chars = int(target_tokens * chars_per_token)
        base = "의료 문서 분석 요청. "
        repeated = base * (target_chars // len(base) + 1)
        return repeated[:target_chars]

    def _run_single_request(
        self,
        prompt: str,
        max_tokens: int,
        request_id: str,
    ) -> RequestResult:
        """단일 요청 실행 및 타이밍 측정"""
        start_time = time.perf_counter()

        try:
            output_text, token_timings = self.inference_fn(prompt, max_tokens)
            end_time = time.perf_counter()

            total_ms = (end_time - start_time) * 1000

            # TTFT 계산
            if token_timings and len(token_timings) > 0:
                ttft_ms = token_timings[0].elapsed_ms
            else:
                ttft_ms = total_ms  # 스트리밍 아닐 경우 전체 시간

            # 출력 토큰 수 (근사)
            output_tokens = len(token_timings) if token_timings else max(1, len(output_text) // 3)

            # TPOT 계산
            if output_tokens > 1:
                tpot_ms = (total_ms - ttft_ms) / (output_tokens - 1)
            else:
                tpot_ms = 0.0

            tokens_per_sec = output_tokens / (total_ms / 1000) if total_ms > 0 else 0

            return RequestResult(
                request_id=request_id,
                input_tokens=len(prompt) // 3,  # 근사
                output_tokens=output_tokens,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                total_latency_ms=total_ms,
                tokens_per_sec=tokens_per_sec,
                token_timings=token_timings or [],
            )

        except Exception as e:
            end_time = time.perf_counter()
            logger.error(f"요청 {request_id} 실패: {e}")
            return RequestResult(
                request_id=request_id,
                input_tokens=0,
                output_tokens=0,
                ttft_ms=0,
                tpot_ms=0,
                total_latency_ms=(end_time - start_time) * 1000,
                tokens_per_sec=0,
                error=str(e),
            )

    def run_warmup(self, input_length: int) -> None:
        """워밍업 실행"""
        logger.info(f"워밍업 {self.config.num_warmup}회 실행 중...")
        prompt = self._generate_prompt(input_length)
        for i in range(self.config.num_warmup):
            self._run_single_request(prompt, 32, f"warmup_{i}")
        logger.info("워밍업 완료")

    def run_single_benchmark(
        self,
        input_length: int,
        output_length: int,
    ) -> BatchResult:
        """단건 요청 벤치마크 (반복 측정)"""
        prompt = self._generate_prompt(input_length)
        results = []

        for i in range(self.config.num_iterations):
            result = self._run_single_request(
                prompt, output_length, f"single_{i}"
            )
            results.append(result)
            logger.info(
                f"  [{i+1}/{self.config.num_iterations}] "
                f"TTFT={result.ttft_ms:.1f}ms, "
                f"TPOT={result.tpot_ms:.1f}ms, "
                f"Total={result.total_latency_ms:.1f}ms"
            )

        total_tokens = sum(r.output_tokens for r in results if r.error is None)
        total_time = sum(r.total_latency_ms for r in results if r.error is None) / 1000

        return BatchResult(
            batch_size=1,
            input_length=input_length,
            output_length=output_length,
            results=results,
            wall_time_sec=total_time,
            throughput_tokens_per_sec=total_tokens / total_time if total_time > 0 else 0,
            throughput_requests_per_sec=len(results) / total_time if total_time > 0 else 0,
        )

    def run_batch_benchmark(
        self,
        batch_size: int,
        input_length: int,
        output_length: int,
    ) -> BatchResult:
        """배치 벤치마크"""
        prompts = [self._generate_prompt(input_length) for _ in range(batch_size)]
        results = []

        wall_start = time.perf_counter()

        for i, prompt in enumerate(prompts):
            result = self._run_single_request(prompt, output_length, f"batch_{i}")
            results.append(result)

        wall_end = time.perf_counter()
        wall_time = wall_end - wall_start

        total_tokens = sum(r.output_tokens for r in results if r.error is None)

        return BatchResult(
            batch_size=batch_size,
            input_length=input_length,
            output_length=output_length,
            results=results,
            wall_time_sec=wall_time,
            throughput_tokens_per_sec=total_tokens / wall_time if wall_time > 0 else 0,
            throughput_requests_per_sec=batch_size / wall_time if wall_time > 0 else 0,
        )

    def run_sweep(self) -> List[BatchResult]:
        """배치 크기 × 입력 길이 스윕"""
        all_results = []

        # 워밍업
        self.run_warmup(self.config.input_lengths[0])

        for input_len in self.config.input_lengths:
            for batch_size in self.config.batch_sizes:
                logger.info(
                    f"\n=== Batch={batch_size}, Input={input_len} ==="
                )

                if self.config.mode == BenchmarkMode.SINGLE:
                    result = self.run_single_benchmark(input_len, self.config.output_length)
                else:
                    result = self.run_batch_benchmark(
                        batch_size, input_len, self.config.output_length
                    )

                all_results.append(result)

                # 메모리 측정
                if self.memory_fn:
                    used_gb, total_gb = self.memory_fn()
                    logger.info(f"  GPU Memory: {used_gb:.1f}/{total_gb:.1f} GB")

                # 결과 요약
                logger.info(
                    f"  Throughput: {result.throughput_tokens_per_sec:.1f} tokens/sec"
                )
                stats = result.latency_stats
                if stats:
                    logger.info(
                        f"  Latency: mean={stats.get('latency_ms_mean', 0):.1f}ms, "
                        f"p50={stats.get('latency_ms_p50', 0):.1f}ms, "
                        f"p95={stats.get('latency_ms_p95', 0):.1f}ms"
                    )

        self.results = all_results
        return all_results

    def run(self) -> List[BatchResult]:
        """벤치마크 실행 (모드에 따라)"""
        logger.info(f"벤치마크 시작: {self.config.name} (모드: {self.config.mode.value})")

        if self.config.mode == BenchmarkMode.SWEEP:
            return self.run_sweep()
        elif self.config.mode == BenchmarkMode.SINGLE:
            self.run_warmup(self.config.input_lengths[0])
            result = self.run_single_benchmark(
                self.config.input_lengths[0], self.config.output_length
            )
            self.results = [result]
            return [result]
        elif self.config.mode == BenchmarkMode.BATCH:
            self.run_warmup(self.config.input_lengths[0])
            result = self.run_batch_benchmark(
                self.config.batch_sizes[0],
                self.config.input_lengths[0],
                self.config.output_length,
            )
            self.results = [result]
            return [result]
        else:
            raise ValueError(f"지원하지 않는 모드: {self.config.mode}")
```

### 2. 결과 리포트 생성기

```python
"""
벤치마크 결과 리포트 생성기
- 콘솔 텍스트 리포트
- JSON 결과 저장
- 비교 분석 테이블
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import asdict


class BenchmarkReporter:
    """벤치마크 결과 리포트 생성"""

    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_text_report(
        self,
        results: list,     # List[BatchResult]
        metadata: Optional[Dict] = None,
    ) -> str:
        """텍스트 형식 리포트 생성"""
        lines = []
        lines.append("=" * 80)
        lines.append("LLM 추론 벤치마크 리포트")
        lines.append(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)

        if metadata:
            lines.append("\n--- 환경 정보 ---")
            for k, v in metadata.items():
                lines.append(f"  {k}: {v}")

        # 요약 테이블
        lines.append("\n--- 결과 요약 ---")
        lines.append(
            f"{'Batch':>6} | {'Input':>6} | {'Output':>6} | "
            f"{'TTFT(ms)':>10} | {'TPOT(ms)':>10} | "
            f"{'Latency(ms)':>12} | {'Throughput':>12}"
        )
        lines.append("-" * 80)

        for r in results:
            ttft = r.ttft_stats
            tpot = r.tpot_stats
            lat = r.latency_stats

            lines.append(
                f"{r.batch_size:>6} | "
                f"{r.input_length:>6} | "
                f"{r.output_length:>6} | "
                f"{ttft.get('ttft_ms_mean', 0):>10.1f} | "
                f"{tpot.get('tpot_ms_mean', 0):>10.1f} | "
                f"{lat.get('latency_ms_mean', 0):>12.1f} | "
                f"{r.throughput_tokens_per_sec:>10.1f} t/s"
            )

        # 상세 통계
        lines.append("\n--- 상세 통계 (P50 / P95 / P99) ---")
        for r in results:
            lat = r.latency_stats
            ttft = r.ttft_stats

            lines.append(
                f"\n  [Batch={r.batch_size}, Input={r.input_length}]"
            )
            if lat:
                lines.append(
                    f"    Latency: P50={lat.get('latency_ms_p50', 0):.1f}ms, "
                    f"P95={lat.get('latency_ms_p95', 0):.1f}ms, "
                    f"P99={lat.get('latency_ms_p99', 0):.1f}ms"
                )
            if ttft:
                lines.append(
                    f"    TTFT:    P50={ttft.get('ttft_ms_p50', 0):.1f}ms, "
                    f"P95={ttft.get('ttft_ms_p95', 0):.1f}ms, "
                    f"P99={ttft.get('ttft_ms_p99', 0):.1f}ms"
                )

            # Tail Latency Ratio
            p50 = lat.get("latency_ms_p50", 0)
            p99 = lat.get("latency_ms_p99", 0)
            if p50 > 0:
                tlr = p99 / p50
                lines.append(f"    TLR (P99/P50): {tlr:.2f}")

        # 에러 요약
        total_errors = sum(
            sum(1 for req in r.results if req.error is not None)
            for r in results
        )
        total_requests = sum(len(r.results) for r in results)
        lines.append(f"\n--- 에러 ---")
        lines.append(f"  총 요청: {total_requests}, 에러: {total_errors}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def save_json_report(
        self,
        results: list,
        metadata: Optional[Dict] = None,
        filename: Optional[str] = None,
    ) -> str:
        """JSON 형식으로 결과 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"

        report = {
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "results": [],
        }

        for r in results:
            entry = {
                "batch_size": r.batch_size,
                "input_length": r.input_length,
                "output_length": r.output_length,
                "wall_time_sec": r.wall_time_sec,
                "throughput_tokens_per_sec": r.throughput_tokens_per_sec,
                "throughput_requests_per_sec": r.throughput_requests_per_sec,
                "ttft_stats": r.ttft_stats,
                "tpot_stats": r.tpot_stats,
                "latency_stats": r.latency_stats,
                "num_requests": len(r.results),
                "num_errors": sum(1 for req in r.results if req.error is not None),
            }
            report["results"].append(entry)

        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON 리포트 저장: {filepath}")
        return str(filepath)

    def compare_engines(
        self,
        engine_results: Dict[str, list],  # {engine_name: List[BatchResult]}
        input_length: int,
        batch_size: int,
    ) -> str:
        """여러 엔진의 결과 비교"""
        lines = []
        lines.append("=" * 90)
        lines.append(f"엔진 비교 (Input={input_length}, Batch={batch_size})")
        lines.append("=" * 90)

        lines.append(
            f"{'Engine':>20} | {'TTFT(ms)':>10} | {'TPOT(ms)':>10} | "
            f"{'E2E(ms)':>10} | {'Throughput':>12} | {'P99(ms)':>10}"
        )
        lines.append("-" * 90)

        for engine_name, results in engine_results.items():
            # 매칭되는 결과 찾기
            matching = [
                r for r in results
                if r.batch_size == batch_size and r.input_length == input_length
            ]

            if not matching:
                lines.append(f"{engine_name:>20} | {'N/A':>10} | {'N/A':>10} | "
                           f"{'N/A':>10} | {'N/A':>12} | {'N/A':>10}")
                continue

            r = matching[0]
            ttft = r.ttft_stats
            tpot = r.tpot_stats
            lat = r.latency_stats

            lines.append(
                f"{engine_name:>20} | "
                f"{ttft.get('ttft_ms_mean', 0):>10.1f} | "
                f"{tpot.get('tpot_ms_mean', 0):>10.1f} | "
                f"{lat.get('latency_ms_mean', 0):>10.1f} | "
                f"{r.throughput_tokens_per_sec:>10.1f} t/s | "
                f"{lat.get('latency_ms_p99', 0):>10.1f}"
            )

        lines.append("=" * 90)
        return "\n".join(lines)


# === GPU 메모리 유틸리티 ===

def get_gpu_memory() -> tuple:
    """현재 GPU 메모리 사용량 (used_gb, total_gb)"""
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            return used, total
    except ImportError:
        pass
    return 0.0, 0.0


def get_system_info() -> Dict:
    """시스템 정보 수집"""
    import platform

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
    }

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda or "N/A"
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            total_mem = torch.cuda.get_device_properties(0).total_mem
            info["gpu_memory_gb"] = f"{total_mem / (1024**3):.1f}"
    except ImportError:
        info["torch_version"] = "N/A"

    return info
```

### 3. Roofline Model 분석기

```python
"""
Roofline Model 분석기
- 하드웨어 성능 상한 계산
- Prefill/Decode 단계별 병목 분석
- 양자화/배칭 최적화 효과 시뮬레이션
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math


@dataclass
class HardwareSpec:
    """GPU 하드웨어 스펙"""
    name: str
    peak_flops_tflops: float        # Peak FP16 TFLOPS
    memory_bandwidth_tb_s: float    # Memory Bandwidth (TB/s)
    memory_gb: float                # GPU Memory (GB)

    @property
    def ridge_point(self) -> float:
        """Roofline 경계점 (FLOP/Byte)"""
        return (self.peak_flops_tflops * 1e12) / (self.memory_bandwidth_tb_s * 1e12)

    @property
    def peak_flops(self) -> float:
        """Peak FLOPS"""
        return self.peak_flops_tflops * 1e12

    @property
    def bandwidth(self) -> float:
        """Memory Bandwidth (Bytes/s)"""
        return self.memory_bandwidth_tb_s * 1e12


# GPU 프리셋
GPU_PRESETS = {
    "a100_80gb": HardwareSpec("A100 80GB", 312.0, 2.0, 80.0),
    "h100_sxm": HardwareSpec("H100 SXM", 990.0, 3.35, 80.0),
    "l40s": HardwareSpec("L40S", 366.0, 0.864, 48.0),
    "rtx_4090": HardwareSpec("RTX 4090", 165.0, 1.008, 24.0),
    "rtx_3060": HardwareSpec("RTX 3060 12GB", 12.7, 0.360, 12.0),
    "rtx_3090": HardwareSpec("RTX 3090", 35.6, 0.936, 24.0),
}


@dataclass
class ModelSpec:
    """모델 스펙 (추론 분석용)"""
    name: str
    num_params_b: float             # 파라미터 수 (Billions)
    bytes_per_param: int = 2        # FP16=2, INT8=1, INT4=0.5
    num_layers: int = 32
    hidden_dim: int = 4096
    num_heads: int = 32
    num_kv_heads: int = 8

    @property
    def model_size_bytes(self) -> float:
        return self.num_params_b * 1e9 * self.bytes_per_param


class RooflineAnalyzer:
    """Roofline Model 분석기"""

    def __init__(self, hardware: HardwareSpec, model: ModelSpec):
        self.hw = hardware
        self.model = model

    def compute_arithmetic_intensity(
        self,
        phase: str,           # "prefill" or "decode"
        batch_size: int = 1,
        seq_len: int = 1,     # prefill용 입력 길이
    ) -> float:
        """Arithmetic Intensity 계산"""
        if phase == "prefill":
            # FLOPs ≈ 2 × N_params × seq_len
            # Bytes ≈ 2 × N_params × bytes_per_param (모델 가중치 로드)
            flops = 2 * self.model.num_params_b * 1e9 * seq_len
            bytes_transferred = 2 * self.model.num_params_b * 1e9 * self.model.bytes_per_param
            return flops / bytes_transferred if bytes_transferred > 0 else 0

        elif phase == "decode":
            # FLOPs ≈ 2 × N_params × batch_size
            # Bytes ≈ 2 × N_params × bytes_per_param (모델 가중치 로드)
            flops = 2 * self.model.num_params_b * 1e9 * batch_size
            bytes_transferred = 2 * self.model.num_params_b * 1e9 * self.model.bytes_per_param
            return flops / bytes_transferred if bytes_transferred > 0 else 0

        else:
            raise ValueError(f"Unknown phase: {phase}")

    def compute_attainable_performance(self, ai: float) -> float:
        """주어진 AI에서 달성 가능한 성능 (FLOPS)"""
        return min(self.hw.peak_flops, ai * self.hw.bandwidth)

    def is_memory_bound(self, ai: float) -> bool:
        """Memory-bound 여부"""
        return ai < self.hw.ridge_point

    def estimate_latency(
        self,
        phase: str,
        batch_size: int = 1,
        seq_len: int = 1,
    ) -> float:
        """추론 지연시간 추정 (초)"""
        ai = self.compute_arithmetic_intensity(phase, batch_size, seq_len)
        perf = self.compute_attainable_performance(ai)

        if phase == "prefill":
            flops = 2 * self.model.num_params_b * 1e9 * seq_len
        else:
            flops = 2 * self.model.num_params_b * 1e9 * batch_size

        return flops / perf if perf > 0 else float("inf")

    def estimate_throughput(
        self,
        batch_size: int,
        output_tokens: int,
    ) -> float:
        """Decode 처리량 추정 (tokens/sec)"""
        decode_latency = self.estimate_latency("decode", batch_size)
        if decode_latency > 0:
            return batch_size / decode_latency
        return 0

    def analyze(
        self,
        batch_sizes: List[int] = None,
        input_lengths: List[int] = None,
    ) -> str:
        """종합 Roofline 분석 리포트"""
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32, 64]
        if input_lengths is None:
            input_lengths = [128, 512, 1024, 2048, 4096]

        lines = []
        lines.append("=" * 80)
        lines.append(f"Roofline 분석: {self.model.name} on {self.hw.name}")
        lines.append("=" * 80)

        lines.append(f"\n하드웨어:")
        lines.append(f"  Peak FLOPS: {self.hw.peak_flops_tflops} TFLOPS (FP16)")
        lines.append(f"  Bandwidth: {self.hw.memory_bandwidth_tb_s} TB/s")
        lines.append(f"  Ridge Point: {self.hw.ridge_point:.1f} FLOP/Byte")
        lines.append(f"  Memory: {self.hw.memory_gb} GB")

        lines.append(f"\n모델:")
        lines.append(f"  파라미터: {self.model.num_params_b}B")
        lines.append(f"  Precision: {'FP16' if self.model.bytes_per_param == 2 else 'INT8' if self.model.bytes_per_param == 1 else f'{self.model.bytes_per_param}B'}")
        lines.append(f"  모델 크기: {self.model.model_size_bytes / (1024**3):.1f} GB")

        # Prefill 분석
        lines.append(f"\n--- Prefill 분석 ---")
        lines.append(f"{'Input Len':>10} | {'AI':>8} | {'Bound':>12} | {'Latency':>10} | {'Tokens/s':>10}")
        lines.append("-" * 60)

        for seq_len in input_lengths:
            ai = self.compute_arithmetic_intensity("prefill", seq_len=seq_len)
            bound = "Memory" if self.is_memory_bound(ai) else "Compute"
            latency = self.estimate_latency("prefill", seq_len=seq_len)
            tps = seq_len / latency if latency > 0 else 0
            lines.append(
                f"{seq_len:>10} | {ai:>8.1f} | {bound:>12} | "
                f"{latency*1000:>8.1f}ms | {tps:>10.0f}"
            )

        # Decode 분석
        lines.append(f"\n--- Decode 분석 ---")
        lines.append(f"{'Batch':>10} | {'AI':>8} | {'Bound':>12} | {'Per-token':>10} | {'Throughput':>12}")
        lines.append("-" * 60)

        for bs in batch_sizes:
            ai = self.compute_arithmetic_intensity("decode", batch_size=bs)
            bound = "Memory" if self.is_memory_bound(ai) else "Compute"
            latency = self.estimate_latency("decode", batch_size=bs)
            throughput = self.estimate_throughput(bs, 1)
            lines.append(
                f"{bs:>10} | {ai:>8.1f} | {bound:>12} | "
                f"{latency*1000:>8.1f}ms | {throughput:>10.0f} t/s"
            )

        # 양자화 효과 시뮬레이션
        lines.append(f"\n--- 양자화별 Decode 처리량 (Batch=32) ---")
        for quant_name, bpp in [("FP16", 2), ("INT8", 1), ("INT4", 0.5)]:
            orig_bpp = self.model.bytes_per_param
            self.model.bytes_per_param = bpp
            throughput = self.estimate_throughput(32, 1)
            self.model.bytes_per_param = orig_bpp
            lines.append(f"  {quant_name:>6}: {throughput:>10.0f} tokens/sec")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


# 사용 예시
if __name__ == "__main__":
    # A100에서 7B 모델 분석
    analyzer = RooflineAnalyzer(
        hardware=GPU_PRESETS["a100_80gb"],
        model=ModelSpec(
            name="Qwen2.5-7B (FP16)",
            num_params_b=7.0,
            bytes_per_param=2,
        ),
    )
    print(analyzer.analyze())

    print("\n\n")

    # RTX 3060에서 7B INT4 모델 분석
    analyzer_edge = RooflineAnalyzer(
        hardware=GPU_PRESETS["rtx_3060"],
        model=ModelSpec(
            name="Qwen2.5-7B (INT4)",
            num_params_b=7.0,
            bytes_per_param=1,    # INT4 ≈ 0.5이지만 오버헤드 포함
        ),
    )
    print(analyzer_edge.analyze())
```

### 4. 의료 문서 OCR 벤치마크 통합

```python
"""
의료 문서 OCR 전용 벤치마크
- 문서 유형별 성능 측정
- 정확도 + 속도 통합 평가
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentType(Enum):
    PRESCRIPTION = "처방전"
    LAB_RESULT = "검사결과지"
    DIAGNOSIS = "진단서"
    DISCHARGE_SUMMARY = "퇴원요약서"
    MEDICAL_RECORD = "의무기록"


@dataclass
class DocumentProfile:
    """문서 유형별 프로파일"""
    doc_type: DocumentType
    avg_input_tokens: int
    avg_output_tokens: int
    critical_fields: List[str]      # 정확도 필수 필드
    acceptable_cer: float           # 허용 CER
    acceptable_latency_ms: float    # 허용 지연시간


# 문서 유형별 프로파일 정의
DOCUMENT_PROFILES = {
    DocumentType.PRESCRIPTION: DocumentProfile(
        doc_type=DocumentType.PRESCRIPTION,
        avg_input_tokens=1000,
        avg_output_tokens=300,
        critical_fields=["약물명", "용량", "투여횟수", "투여경로"],
        acceptable_cer=0.01,        # 1% 이하
        acceptable_latency_ms=3000,
    ),
    DocumentType.LAB_RESULT: DocumentProfile(
        doc_type=DocumentType.LAB_RESULT,
        avg_input_tokens=2000,
        avg_output_tokens=500,
        critical_fields=["검사항목", "수치", "단위", "참고치"],
        acceptable_cer=0.005,       # 0.5% 이하 (수치 정확도 중요)
        acceptable_latency_ms=5000,
    ),
    DocumentType.DIAGNOSIS: DocumentProfile(
        doc_type=DocumentType.DIAGNOSIS,
        avg_input_tokens=1500,
        avg_output_tokens=400,
        critical_fields=["환자명", "진단명", "진단코드"],
        acceptable_cer=0.01,
        acceptable_latency_ms=4000,
    ),
    DocumentType.DISCHARGE_SUMMARY: DocumentProfile(
        doc_type=DocumentType.DISCHARGE_SUMMARY,
        avg_input_tokens=5000,
        avg_output_tokens=1000,
        critical_fields=["입원일", "퇴원일", "주진단", "경과", "추후계획"],
        acceptable_cer=0.02,        # 긴 문서라 약간 높게
        acceptable_latency_ms=10000,
    ),
    DocumentType.MEDICAL_RECORD: DocumentProfile(
        doc_type=DocumentType.MEDICAL_RECORD,
        avg_input_tokens=4000,
        avg_output_tokens=800,
        critical_fields=["환자정보", "주소", "현병력", "과거력"],
        acceptable_cer=0.02,
        acceptable_latency_ms=8000,
    ),
}


@dataclass
class OCRBenchmarkResult:
    """OCR 벤치마크 결과"""
    doc_type: DocumentType
    num_samples: int
    avg_cer: float
    avg_wer: float
    critical_field_accuracy: Dict[str, float]
    avg_latency_ms: float
    avg_ttft_ms: float
    avg_tpot_ms: float
    throughput_docs_per_min: float
    meets_cer_sla: bool
    meets_latency_sla: bool


class MedicalOCRBenchmark:
    """의료 문서 OCR 벤치마크 실행기"""

    def __init__(
        self,
        inference_fn=None,
        evaluate_fn=None,
    ):
        """
        Args:
            inference_fn: OCR 추론 함수 (image_or_text) -> extracted_text
            evaluate_fn: 평가 함수 (prediction, ground_truth) -> (cer, wer, field_acc)
        """
        self.inference_fn = inference_fn
        self.evaluate_fn = evaluate_fn

    def run_document_type_benchmark(
        self,
        doc_type: DocumentType,
        test_samples: List[Dict],   # [{input, ground_truth}]
        num_warmup: int = 3,
    ) -> OCRBenchmarkResult:
        """문서 유형별 벤치마크"""
        profile = DOCUMENT_PROFILES[doc_type]

        # 워밍업
        for i in range(min(num_warmup, len(test_samples))):
            self.inference_fn(test_samples[i]["input"])

        cers = []
        wers = []
        latencies = []
        ttfts = []
        tpots = []
        field_accs: Dict[str, List[float]] = {
            f: [] for f in profile.critical_fields
        }

        for sample in test_samples:
            # 추론 및 타이밍
            start = time.perf_counter()
            prediction = self.inference_fn(sample["input"])
            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

            # 평가
            if self.evaluate_fn and "ground_truth" in sample:
                cer, wer, f_acc = self.evaluate_fn(
                    prediction, sample["ground_truth"]
                )
                cers.append(cer)
                wers.append(wer)
                for field, acc in f_acc.items():
                    if field in field_accs:
                        field_accs[field].append(acc)

        avg_cer = sum(cers) / len(cers) if cers else 0
        avg_wer = sum(wers) / len(wers) if wers else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        avg_field_acc = {
            f: sum(v) / len(v) if v else 0
            for f, v in field_accs.items()
        }

        total_time_min = sum(latencies) / 1000 / 60  # 분 단위
        docs_per_min = len(test_samples) / total_time_min if total_time_min > 0 else 0

        return OCRBenchmarkResult(
            doc_type=doc_type,
            num_samples=len(test_samples),
            avg_cer=avg_cer,
            avg_wer=avg_wer,
            critical_field_accuracy=avg_field_acc,
            avg_latency_ms=avg_latency,
            avg_ttft_ms=0,      # 스트리밍일 경우 별도 측정
            avg_tpot_ms=0,
            throughput_docs_per_min=docs_per_min,
            meets_cer_sla=avg_cer <= profile.acceptable_cer,
            meets_latency_sla=avg_latency <= profile.acceptable_latency_ms,
        )

    def generate_report(self, results: List[OCRBenchmarkResult]) -> str:
        """종합 리포트"""
        lines = []
        lines.append("=" * 90)
        lines.append("의료 문서 OCR 벤치마크 리포트")
        lines.append("=" * 90)

        lines.append(
            f"\n{'문서 유형':>14} | {'샘플':>4} | {'CER':>8} | "
            f"{'WER':>8} | {'Latency':>10} | {'처리량':>10} | "
            f"{'CER SLA':>8} | {'속도 SLA':>8}"
        )
        lines.append("-" * 90)

        all_pass = True
        for r in results:
            cer_pass = "PASS" if r.meets_cer_sla else "FAIL"
            lat_pass = "PASS" if r.meets_latency_sla else "FAIL"

            if not r.meets_cer_sla or not r.meets_latency_sla:
                all_pass = False

            lines.append(
                f"{r.doc_type.value:>14} | "
                f"{r.num_samples:>4} | "
                f"{r.avg_cer:>7.4f} | "
                f"{r.avg_wer:>7.4f} | "
                f"{r.avg_latency_ms:>8.1f}ms | "
                f"{r.throughput_docs_per_min:>7.1f}/min | "
                f"{cer_pass:>8} | "
                f"{lat_pass:>8}"
            )

        # 필드별 정확도
        lines.append("\n--- 핵심 필드 정확도 ---")
        for r in results:
            lines.append(f"\n  [{r.doc_type.value}]")
            for field, acc in r.critical_field_accuracy.items():
                status = "OK" if acc >= 0.95 else "WARN" if acc >= 0.90 else "FAIL"
                lines.append(f"    {field:>12}: {acc:.1%} ({status})")

        # 종합 판정
        lines.append(f"\n{'=' * 90}")
        if all_pass:
            lines.append("종합 판정: PASS - 모든 SLA 충족")
        else:
            lines.append("종합 판정: FAIL - SLA 미충족 항목 존재")
        lines.append("=" * 90)

        return "\n".join(lines)


# 사용 예시 (시뮬레이션)
if __name__ == "__main__":
    import random

    # 시뮬레이션용 추론 함수
    def mock_inference(input_text):
        time.sleep(random.uniform(0.5, 2.0))  # 0.5~2초 지연
        return "추출된 텍스트 시뮬레이션"

    # 시뮬레이션용 평가 함수
    def mock_evaluate(prediction, ground_truth):
        cer = random.uniform(0.005, 0.02)
        wer = random.uniform(0.01, 0.05)
        field_acc = {
            "약물명": random.uniform(0.93, 0.99),
            "용량": random.uniform(0.95, 0.99),
            "투여횟수": random.uniform(0.96, 1.0),
            "투여경로": random.uniform(0.97, 1.0),
        }
        return cer, wer, field_acc

    benchmark = MedicalOCRBenchmark(
        inference_fn=mock_inference,
        evaluate_fn=mock_evaluate,
    )

    # 처방전 벤치마크 (시뮬레이션)
    test_data = [
        {"input": f"처방전 {i}", "ground_truth": f"정답 {i}"}
        for i in range(20)
    ]

    result = benchmark.run_document_type_benchmark(
        DocumentType.PRESCRIPTION,
        test_data,
        num_warmup=2,
    )

    report = benchmark.generate_report([result])
    print(report)
```

---

## 벤치마크 실행 가이드

### 단계별 절차

```
1. 환경 설정
   □ GPU 드라이버/CUDA 확인
   □ 벤치마크 도구 설치 (vllm, sglang 등)
   □ 모델 다운로드/양자화 준비

2. 테스트 데이터 준비
   □ 문서 유형별 최소 50개 샘플
   □ Ground truth 라벨 검증
   □ 입력 길이 분포 확인

3. 워밍업
   □ 모델 로드 후 더미 요청 3~5회
   □ GPU 클럭 안정화 확인
   □ CUDA Graph 컴파일 대기

4. 벤치마크 실행
   □ 단건 지연시간 (10회 반복)
   □ 배치 처리량 (batch 1~64 스윕)
   □ 입력 길이별 성능 (512~8192)
   □ 장시간 안정성 (30분 연속)

5. 결과 분석
   □ P50/P95/P99 지연시간 확인
   □ Throughput 대 SLA 비교
   □ GPU Memory Peak 확인
   □ 정확도 지표 교차 검증
```

---

## 용어 체크리스트

| 용어 | 이해 여부 | 핵심 포인트 |
|------|:---------:|-------------|
| Latency | [ ] | 요청 1건의 전체 처리 시간. 사용자 경험의 핵심 |
| Throughput | [ ] | 단위 시간당 처리량. 시스템 효율성 지표 |
| TTFT | [ ] | 첫 토큰까지 시간. Prefill 성능 반영 |
| TPOT | [ ] | 토큰 간 생성 시간. Decode 성능 반영 |
| ITL | [ ] | 토큰 간 간격. 스트리밍 UX 품질 지표 |
| Memory Footprint | [ ] | 총 메모리 사용량. 모델 + KV Cache + Overhead |
| Roofline Model | [ ] | Compute/Memory bound 경계 분석. AI로 판단 |
| Arithmetic Intensity | [ ] | FLOPs/Bytes. Prefill은 높고 Decode는 낮음 |
| P50/P95/P99 | [ ] | 지연시간 백분위수. P99이 SLA 기준 |
| Tail Latency Ratio | [ ] | P99/P50. 작을수록 안정적. 2 이하가 이상적 |
| Warm-up | [ ] | 측정 전 시스템 안정화. JIT, 캐시 워밍 |
| SLA | [ ] | Service Level Agreement. 성능 보장 기준 |
| Continuous Batching | [ ] | 동적 배칭으로 처리량 극대화 |
| Ridge Point | [ ] | Roofline에서 compute/memory bound 전환점 |
