# 16.2 24시간 안정성

GPU 기반 모델 서빙은 짧은 벤치마크에서는 잘 동작하지만, 24시간 이상 장기 운영에서 다양한 문제가 발생한다. 메모리 누수, KV Cache 단편화, 쓰레드 누수, GPU OOM은 시간이 지남에 따라 누적되어 결국 서비스 장애로 이어진다.

---

> **핵심 용어 박스**
>
> | 용어 | 정의 |
> |------|------|
> | **Memory Leak** | 더 이상 사용하지 않는 메모리가 해제되지 않고 계속 증가하는 현상 |
> | **GPU OOM** | GPU Out Of Memory. VRAM이 부족하여 새 텐서 할당이 실패하는 상태 |
> | **KV Cache Fragmentation** | PagedAttention의 블록이 단편화되어 실제 가용 메모리보다 사용 가능한 블록이 적어지는 현상 |
> | **Graceful Degradation** | 시스템 과부하 시 일부 기능을 제한하되 전체 서비스는 유지하는 전략 |
> | **Circuit Breaker** | 연속 실패가 임계값을 초과하면 요청을 차단하여 하위 시스템을 보호하는 패턴 |
> | **Thread Leak** | 생성된 쓰레드가 종료되지 않고 누적되는 현상. 결국 OS 자원 고갈 |
> | **Watchdog** | 시스템 상태를 주기적으로 감시하고 이상 감지 시 자동 복구하는 프로세스 |
> | **Backpressure** | 소비자가 처리할 수 없는 속도로 요청이 들어올 때 생산자를 제한하는 메커니즘 |

---

## 16.2.1 장기 운영에서 발생하는 문제들

### 시간 경과에 따른 리소스 사용 패턴

```
메모리 사용량
  ↑
  │                                    ╱ OOM!
  │                              ╱───╱
  │                        ╱───╱
  │                  ╱───╱        ← 메모리 누수 (느린 증가)
  │            ╱───╱
  │      ╱───╱
  │╱───╱
  └────────────────────────────────────→ 시간
  0h    4h    8h    12h   16h   20h  24h
```

### 문제 유형 분류

| 문제 | 발생 시점 | 증상 | 심각도 |
|------|----------|------|--------|
| Python 메모리 누수 | 6-12h | RSS 지속 증가 | 중간 |
| GPU VRAM 누수 | 4-8h | VRAM 사용량 증가 | 높음 |
| KV Cache 단편화 | 8-24h | 처리량 점진적 감소 | 중간 |
| 쓰레드 누수 | 12-48h | 컨텍스트 스위칭 증가 | 낮음 |
| 파일 디스크립터 누수 | 24h+ | 새 연결 생성 실패 | 높음 |
| CUDA 컨텍스트 오염 | 비정기 | 추론 결과 오류/행 | 치명적 |
| 로그 파일 비대 | 24h+ | 디스크 풀 | 중간 |

---

## 16.2.2 Memory Watchdog

### 메모리 누수 감지의 수학적 원리

메모리 사용량을 시계열 $M(t)$로 모델링할 때, 누수 속도(leak rate)를 선형 회귀로 추정한다:

$$
M(t) = M_0 + r \cdot t + \epsilon(t)
$$

여기서:
- $M_0$: 초기 메모리 사용량
- $r$: 누수 속도 (MB/hour)
- $\epsilon(t)$: 노이즈 (정상적인 메모리 변동)

**OOM 예상 시각:**

$$
T_{\text{OOM}} = \frac{M_{\max} - M_{\text{current}}}{r}
$$

$r > 0$이 통계적으로 유의하면 메모리 누수로 판정한다.

### 구현

```python
"""
memory_watchdog.py
메모리 누수 감지 및 자동 복구
"""

import os
import gc
import time
import asyncio
import logging
import statistics
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable
from enum import Enum

logger = logging.getLogger(__name__)


class MemoryAction(Enum):
    NONE = "none"
    GC_COLLECT = "gc_collect"
    CLEAR_CACHE = "clear_cache"
    REJECT_REQUESTS = "reject_requests"
    RESTART = "restart"


@dataclass
class MemoryConfig:
    """메모리 워치독 설정"""
    check_interval_seconds: int = 60
    # Python RSS 메모리 임계값
    rss_warning_mb: float = 8000.0
    rss_critical_mb: float = 12000.0
    # GPU VRAM 임계값
    vram_warning_pct: float = 90.0
    vram_critical_pct: float = 97.0
    # 누수 감지
    leak_detection_window: int = 60     # 최근 60개 샘플로 추세 분석
    leak_rate_threshold_mb_per_hour: float = 50.0  # 시간당 50MB 이상 증가 → 누수
    # 자동 GC
    auto_gc_interval_seconds: int = 300
    # OOM 예방
    oom_prevention_buffer_pct: float = 5.0  # 5% 여유 유지


@dataclass
class MemorySample:
    """메모리 샘플"""
    timestamp: float
    rss_mb: float
    vram_used_mb: float
    vram_total_mb: float
    gc_objects: int
    python_allocated_mb: float


class MemoryWatchdog:
    """
    메모리 워치독: CPU/GPU 메모리를 주기적으로 감시하고
    누수 감지 시 자동 대응.

    대응 단계:
    1. GC 강제 실행 (경미)
    2. CUDA Cache 클리어 (중간)
    3. 새 요청 거부 (심각)
    4. 프로세스 재시작 (치명)
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        on_restart: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self.config = config or MemoryConfig()
        self.on_restart = on_restart
        self.samples: list[MemorySample] = []
        self.is_running = False
        self._reject_new_requests = False
        self._last_gc_time = 0.0
        self._actions_taken: list[dict] = []

    @property
    def should_reject_requests(self) -> bool:
        return self._reject_new_requests

    def _get_rss_mb(self) -> float:
        """현재 프로세스 RSS 메모리 (MB)"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # psutil 없으면 /proc에서 직접 읽기 (Linux)
            try:
                with open(f"/proc/{os.getpid()}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            return int(line.split()[1]) / 1024
            except (FileNotFoundError, PermissionError):
                pass
            return 0.0

    def _get_vram_info(self) -> tuple[float, float]:
        """GPU VRAM 사용량 (used_mb, total_mb)"""
        try:
            import torch
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / (1024 * 1024)
                total = torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)
                return used, total
        except ImportError:
            pass
        return 0.0, 0.0

    def _get_python_allocated(self) -> float:
        """Python 할당된 메모리 (MB) - tracemalloc 기반"""
        try:
            import tracemalloc
            if tracemalloc.is_tracing():
                current, _ = tracemalloc.get_traced_memory()
                return current / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def collect_sample(self) -> MemorySample:
        """현재 메모리 상태 샘플링"""
        rss = self._get_rss_mb()
        vram_used, vram_total = self._get_vram_info()

        sample = MemorySample(
            timestamp=time.time(),
            rss_mb=rss,
            vram_used_mb=vram_used,
            vram_total_mb=vram_total,
            gc_objects=len(gc.get_objects()),
            python_allocated_mb=self._get_python_allocated(),
        )

        self.samples.append(sample)
        # 최대 24시간치 보관 (1분 간격 → 1440개)
        max_samples = 24 * 60
        if len(self.samples) > max_samples:
            self.samples = self.samples[-max_samples:]

        return sample

    def detect_leak(self, metric: str = "rss_mb") -> dict:
        """
        메모리 누수 감지 (선형 회귀 기반).

        slope > threshold → 누수 판정
        OOM 예상 시각 계산
        """
        window = self.config.leak_detection_window
        recent = self.samples[-window:] if len(self.samples) >= window else self.samples

        if len(recent) < 10:
            return {"detected": False, "reason": "insufficient_samples"}

        values = [getattr(s, metric) for s in recent]
        timestamps = [s.timestamp for s in recent]

        # 시간 단위를 시간(hour)으로 변환
        t0 = timestamps[0]
        hours = [(t - t0) / 3600.0 for t in timestamps]

        # 선형 회귀
        n = len(values)
        x_mean = sum(hours) / n
        y_mean = sum(values) / n

        numerator = sum((hours[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((hours[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return {"detected": False, "reason": "no_time_variation"}

        slope = numerator / denominator  # MB/hour
        intercept = y_mean - slope * x_mean

        # R² 계산 (적합도)
        ss_res = sum((values[i] - (slope * hours[i] + intercept)) ** 2 for i in range(n))
        ss_tot = sum((values[i] - y_mean) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        is_leak = (
            slope > self.config.leak_rate_threshold_mb_per_hour
            and r_squared > 0.7  # 70% 이상 설명력
        )

        result = {
            "detected": is_leak,
            "metric": metric,
            "slope_mb_per_hour": round(slope, 2),
            "r_squared": round(r_squared, 4),
            "current_value_mb": round(values[-1], 1),
            "samples_analyzed": n,
        }

        if is_leak and metric == "rss_mb":
            remaining_mb = self.config.rss_critical_mb - values[-1]
            if slope > 0:
                hours_to_oom = remaining_mb / slope
                result["estimated_oom_hours"] = round(hours_to_oom, 1)
                result["estimated_oom_timestamp"] = time.time() + hours_to_oom * 3600

        return result

    def determine_action(self, sample: MemorySample) -> MemoryAction:
        """현재 메모리 상태에 기반한 대응 결정"""
        # 치명: VRAM 임계값 초과
        if sample.vram_total_mb > 0:
            vram_pct = (sample.vram_used_mb / sample.vram_total_mb) * 100
            if vram_pct > self.config.vram_critical_pct:
                return MemoryAction.RESTART

            if vram_pct > self.config.vram_warning_pct:
                return MemoryAction.CLEAR_CACHE

        # 심각: RSS 임계값 초과
        if sample.rss_mb > self.config.rss_critical_mb:
            return MemoryAction.REJECT_REQUESTS

        if sample.rss_mb > self.config.rss_warning_mb:
            return MemoryAction.GC_COLLECT

        # 누수 감지 시 예방적 GC
        leak_info = self.detect_leak("rss_mb")
        if leak_info["detected"]:
            oom_hours = leak_info.get("estimated_oom_hours", float("inf"))
            if oom_hours < 1:
                return MemoryAction.REJECT_REQUESTS
            elif oom_hours < 4:
                return MemoryAction.CLEAR_CACHE
            else:
                return MemoryAction.GC_COLLECT

        return MemoryAction.NONE

    async def execute_action(self, action: MemoryAction) -> None:
        """대응 액션 실행"""
        if action == MemoryAction.NONE:
            return

        logger.warning(f"메모리 워치독 액션: {action.value}")
        self._actions_taken.append({
            "action": action.value,
            "timestamp": time.time(),
        })

        if action == MemoryAction.GC_COLLECT:
            collected = gc.collect()
            logger.info(f"GC 수행: {collected}개 객체 수거")

        elif action == MemoryAction.CLEAR_CACHE:
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
                logger.info("CUDA cache 클리어 완료")
            except ImportError:
                pass

        elif action == MemoryAction.REJECT_REQUESTS:
            self._reject_new_requests = True
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.warning("새 요청 거부 모드 활성화")

        elif action == MemoryAction.RESTART:
            logger.critical("메모리 한계 도달, 프로세스 재시작 요청")
            if self.on_restart:
                await self.on_restart()
            else:
                os._exit(1)  # 강제 종료 → K8s가 재시작

    async def run(self) -> None:
        """워치독 메인 루프"""
        self.is_running = True
        logger.info("메모리 워치독 시작")

        while self.is_running:
            try:
                sample = self.collect_sample()
                action = self.determine_action(sample)
                await self.execute_action(action)

                # 주기적 GC (기본 5분)
                now = time.time()
                if now - self._last_gc_time > self.config.auto_gc_interval_seconds:
                    gc.collect()
                    self._last_gc_time = now

            except Exception as e:
                logger.error(f"메모리 워치독 에러: {e}", exc_info=True)

            await asyncio.sleep(self.config.check_interval_seconds)

    def stop(self) -> None:
        self.is_running = False

    def get_report(self) -> dict:
        """메모리 상태 보고서"""
        if not self.samples:
            return {"status": "no_data"}

        latest = self.samples[-1]
        rss_leak = self.detect_leak("rss_mb")
        vram_leak = self.detect_leak("vram_used_mb")

        return {
            "current": {
                "rss_mb": round(latest.rss_mb, 1),
                "vram_used_mb": round(latest.vram_used_mb, 1),
                "vram_total_mb": round(latest.vram_total_mb, 1),
                "gc_objects": latest.gc_objects,
            },
            "leak_detection": {
                "rss": rss_leak,
                "vram": vram_leak,
            },
            "actions_taken": self._actions_taken[-20:],
            "rejecting_requests": self._reject_new_requests,
            "uptime_hours": round(
                (time.time() - self.samples[0].timestamp) / 3600, 2
            ) if self.samples else 0,
        }
```

---

## 16.2.3 Circuit Breaker

### Circuit Breaker 상태 머신

```
         성공률 < threshold              타임아웃 후 시도
    ┌─────────────────┐            ┌──────────────────┐
    │                 ▼            │                  ▼
┌───────┐       ┌──────────┐  ┌──────────────┐
│ CLOSED│       │   OPEN   │  │  HALF-OPEN   │
│(정상) │       │(차단)    │  │(시험 통과)   │
└───────┘       └──────────┘  └──────────────┘
    ▲                              │
    │       성공 → CLOSED          │
    └──────────────────────────────┘
                실패 → OPEN
```

### 수학적 모델

Circuit Breaker의 상태 전이 조건:

$$
P_{\text{failure}} = \frac{N_{\text{fail}}}{N_{\text{total}}} > \theta_{\text{open}}
$$

여기서:
- $N_{\text{fail}}$: 윈도우 내 실패 횟수
- $N_{\text{total}}$: 윈도우 내 총 요청 수
- $\theta_{\text{open}}$: 개방 임계값 (예: 0.5)

Half-Open 상태에서 복구 조건:

$$
P_{\text{success\_trial}} = \frac{N_{\text{success\_trial}}}{N_{\text{trial}}} > \theta_{\text{close}}
$$

### 구현

```python
"""
circuit_breaker.py
Circuit Breaker 패턴 구현
"""

import time
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Awaitable, Any
from collections import deque

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"       # 정상: 요청 허용
    OPEN = "open"           # 차단: 요청 거부
    HALF_OPEN = "half_open" # 시험: 제한적 요청 허용


@dataclass
class CircuitBreakerConfig:
    """Circuit Breaker 설정"""
    failure_threshold: float = 0.5       # 실패율 50% 초과 → OPEN
    success_threshold: float = 0.8       # 성공률 80% 초과 → CLOSED
    window_size: int = 100               # 슬라이딩 윈도우 크기
    open_timeout_seconds: float = 30.0   # OPEN → HALF_OPEN 대기 시간
    half_open_max_requests: int = 5      # HALF_OPEN에서 허용할 시험 요청 수
    min_requests_for_decision: int = 10  # 최소 요청 수 (이하면 판정 안 함)


@dataclass
class RequestResult:
    """요청 결과 기록"""
    timestamp: float
    success: bool
    duration_ms: float
    error_type: Optional[str] = None


class CircuitBreaker:
    """
    Circuit Breaker: 연쇄 장애 방지.

    GPU OOM, 모델 행, 타임아웃 등이 연속 발생하면
    요청을 차단하여 시스템을 보호한다.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[Callable[..., Awaitable[Any]]] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback = fallback
        self.state = CircuitState.CLOSED
        self._results: deque[RequestResult] = deque(maxlen=self.config.window_size)
        self._open_since: float = 0.0
        self._half_open_count: int = 0
        self._half_open_successes: int = 0
        self._state_changes: list[dict] = []

    def _record_state_change(self, old: CircuitState, new: CircuitState, reason: str):
        self._state_changes.append({
            "from": old.value,
            "to": new.value,
            "reason": reason,
            "timestamp": time.time(),
        })
        logger.warning(
            f"[CircuitBreaker:{self.name}] {old.value} → {new.value}: {reason}"
        )

    def _failure_rate(self) -> float:
        """현재 윈도우의 실패율"""
        if len(self._results) < self.config.min_requests_for_decision:
            return 0.0
        failures = sum(1 for r in self._results if not r.success)
        return failures / len(self._results)

    def _should_open(self) -> bool:
        """CLOSED → OPEN 전이 조건"""
        return self._failure_rate() > self.config.failure_threshold

    def _should_try_half_open(self) -> bool:
        """OPEN → HALF_OPEN 전이 조건 (타임아웃 경과)"""
        return (time.time() - self._open_since) >= self.config.open_timeout_seconds

    def can_execute(self) -> bool:
        """요청 실행 가능 여부"""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self._should_try_half_open():
                old = self.state
                self.state = CircuitState.HALF_OPEN
                self._half_open_count = 0
                self._half_open_successes = 0
                self._record_state_change(old, self.state, "timeout expired")
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return self._half_open_count < self.config.half_open_max_requests

        return False

    def record_success(self, duration_ms: float) -> None:
        """성공 기록"""
        self._results.append(RequestResult(
            timestamp=time.time(),
            success=True,
            duration_ms=duration_ms,
        ))

        if self.state == CircuitState.HALF_OPEN:
            self._half_open_count += 1
            self._half_open_successes += 1

            # HALF_OPEN에서 충분히 성공하면 CLOSED로
            if self._half_open_count >= self.config.half_open_max_requests:
                success_rate = self._half_open_successes / self._half_open_count
                if success_rate >= self.config.success_threshold:
                    old = self.state
                    self.state = CircuitState.CLOSED
                    self._record_state_change(
                        old, self.state,
                        f"recovery confirmed (success_rate={success_rate:.2f})"
                    )

    def record_failure(self, duration_ms: float, error_type: str = "unknown") -> None:
        """실패 기록"""
        self._results.append(RequestResult(
            timestamp=time.time(),
            success=False,
            duration_ms=duration_ms,
            error_type=error_type,
        ))

        if self.state == CircuitState.CLOSED:
            if self._should_open():
                old = self.state
                self.state = CircuitState.OPEN
                self._open_since = time.time()
                self._record_state_change(
                    old, self.state,
                    f"failure_rate={self._failure_rate():.2f}"
                )

        elif self.state == CircuitState.HALF_OPEN:
            # HALF_OPEN에서 실패 → 즉시 OPEN
            old = self.state
            self.state = CircuitState.OPEN
            self._open_since = time.time()
            self._record_state_change(old, self.state, f"trial failed: {error_type}")

    async def execute(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """
        Circuit Breaker로 보호된 함수 실행.

        사용법:
            result = await breaker.execute(model.inference, image, prompt)
        """
        if not self.can_execute():
            if self.fallback:
                logger.info(f"[CircuitBreaker:{self.name}] Fallback 실행")
                return await self.fallback(*args, **kwargs)
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN "
                f"(failure_rate={self._failure_rate():.2f})"
            )

        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = (time.time() - start) * 1000
            self.record_success(duration)
            return result

        except Exception as e:
            duration = (time.time() - start) * 1000
            self.record_failure(duration, type(e).__name__)
            raise

    def get_status(self) -> dict:
        """현재 상태 보고"""
        recent_results = list(self._results)
        total = len(recent_results)
        failures = sum(1 for r in recent_results if not r.success)

        return {
            "name": self.name,
            "state": self.state.value,
            "failure_rate": round(self._failure_rate(), 4),
            "total_requests": total,
            "failures": failures,
            "recent_state_changes": self._state_changes[-5:],
        }


class CircuitBreakerOpenError(Exception):
    """Circuit Breaker가 OPEN 상태일 때 발생"""
    pass
```

---

## 16.2.4 Graceful Degradation 전략

### 부하 단계별 대응

```python
"""
graceful_degradation.py
단계적 서비스 품질 저하 전략
"""

import time
import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)


class LoadLevel(IntEnum):
    """부하 수준 (0=정상, 4=위험)"""
    NORMAL = 0
    ELEVATED = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4


@dataclass
class DegradationPolicy:
    """
    단계별 서비스 저하 정책.

    Level 0 (NORMAL): 모든 기능 정상
    Level 1 (ELEVATED): 낮은 우선순위 요청 지연
    Level 2 (HIGH): 배치 크기 축소, 최대 토큰 제한
    Level 3 (CRITICAL): 신규 요청 큐잉만, 실시간 처리 중단
    Level 4 (EMERGENCY): 신규 요청 거부, 진행 중인 요청만 완료
    """

    # Level 1 임계값
    elevated_queue_length: int = 50
    elevated_gpu_util_pct: float = 85.0

    # Level 2 임계값
    high_queue_length: int = 100
    high_gpu_util_pct: float = 92.0

    # Level 3 임계값
    critical_queue_length: int = 200
    critical_gpu_util_pct: float = 97.0

    # Level 4 임계값
    emergency_queue_length: int = 500
    emergency_vram_util_pct: float = 99.0


class GracefulDegradationManager:
    """
    Graceful Degradation 관리자.

    현재 시스템 부하에 따라 서비스 수준을 자동 조절한다.
    """

    def __init__(self, policy: Optional[DegradationPolicy] = None):
        self.policy = policy or DegradationPolicy()
        self.current_level = LoadLevel.NORMAL
        self._level_history: list[dict] = []

    def evaluate_load(
        self,
        queue_length: int,
        gpu_utilization_pct: float,
        vram_utilization_pct: float,
        error_rate: float = 0.0,
    ) -> LoadLevel:
        """현재 부하 수준 평가"""
        p = self.policy

        if (queue_length >= p.emergency_queue_length
                or vram_utilization_pct >= p.emergency_vram_util_pct):
            new_level = LoadLevel.EMERGENCY
        elif (queue_length >= p.critical_queue_length
                or gpu_utilization_pct >= p.critical_gpu_util_pct
                or error_rate > 0.1):
            new_level = LoadLevel.CRITICAL
        elif (queue_length >= p.high_queue_length
                or gpu_utilization_pct >= p.high_gpu_util_pct):
            new_level = LoadLevel.HIGH
        elif (queue_length >= p.elevated_queue_length
                or gpu_utilization_pct >= p.elevated_gpu_util_pct):
            new_level = LoadLevel.ELEVATED
        else:
            new_level = LoadLevel.NORMAL

        if new_level != self.current_level:
            self._level_history.append({
                "from": self.current_level.name,
                "to": new_level.name,
                "timestamp": time.time(),
            })
            logger.warning(
                f"부하 수준 변경: {self.current_level.name} → {new_level.name}"
            )
            self.current_level = new_level

        return new_level

    def get_constraints(self) -> dict:
        """현재 부하 수준에 따른 제약 조건"""
        level = self.current_level

        if level == LoadLevel.NORMAL:
            return {
                "accept_requests": True,
                "max_tokens": 8192,
                "max_batch_size": 32,
                "max_image_resolution": 4096,
                "priority_filter": None,
                "streaming_enabled": True,
            }
        elif level == LoadLevel.ELEVATED:
            return {
                "accept_requests": True,
                "max_tokens": 4096,          # 토큰 제한
                "max_batch_size": 24,
                "max_image_resolution": 4096,
                "priority_filter": None,
                "streaming_enabled": True,
            }
        elif level == LoadLevel.HIGH:
            return {
                "accept_requests": True,
                "max_tokens": 2048,          # 토큰 추가 제한
                "max_batch_size": 16,        # 배치 축소
                "max_image_resolution": 2048, # 해상도 제한
                "priority_filter": "high",    # 낮은 우선순위 거부
                "streaming_enabled": False,   # 스트리밍 비활성
            }
        elif level == LoadLevel.CRITICAL:
            return {
                "accept_requests": True,
                "max_tokens": 1024,
                "max_batch_size": 8,
                "max_image_resolution": 1024,
                "priority_filter": "critical",
                "streaming_enabled": False,
            }
        else:  # EMERGENCY
            return {
                "accept_requests": False,     # 신규 요청 거부
                "max_tokens": 512,
                "max_batch_size": 4,
                "max_image_resolution": 1024,
                "priority_filter": "critical",
                "streaming_enabled": False,
            }
```

---

## 16.2.5 주기적 재시작 전략

### 수학적 근거: 가용성 계산

주기적 재시작의 가용성:

$$
A_{\text{restart}} = 1 - \frac{T_{\text{restart}}}{T_{\text{period}}}
$$

여기서:
- $T_{\text{restart}}$: 재시작 소요 시간 (모델 로딩 포함)
- $T_{\text{period}}$: 재시작 주기

**예시:** 재시작 3분, 주기 6시간이면:

$$
A = 1 - \frac{3}{360} = 1 - 0.0083 = 99.17\%
$$

Rolling restart로 인스턴스를 순차 재시작하면, 항상 $N-1$개가 서빙 가능:

$$
A_{\text{rolling}} = \frac{N - 1}{N} + \frac{1}{N} \times A_{\text{restart}}
$$

$N=3$이면:

$$
A_{\text{rolling}} = \frac{2}{3} + \frac{1}{3} \times 0.9917 = 0.6667 + 0.3306 = 99.72\%
$$

```python
"""
periodic_restart.py
주기적 Rolling Restart 스케줄러
"""

import time
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RestartScheduleConfig:
    """재시작 스케줄 설정"""
    restart_interval_hours: float = 6.0    # 6시간마다 재시작
    restart_window_start_hour: int = 2     # 재시작 허용 시작 (02:00)
    restart_window_end_hour: int = 5       # 재시작 허용 끝 (05:00)
    drain_timeout_seconds: int = 60        # 요청 드레인 대기
    force_restart_after_hours: float = 24  # 24시간 초과 시 강제


class PeriodicRestartScheduler:
    """
    주기적 재시작 스케줄러.

    장기 운영에서 누적되는 문제를 예방하기 위해
    정해진 주기로 프로세스를 재시작한다.

    전략:
    1. 낮은 트래픽 시간대에 재시작
    2. 요청 드레인 후 안전하게 종료
    3. Kubernetes에서 Rolling Update로 무중단
    """

    def __init__(self, config: Optional[RestartScheduleConfig] = None):
        self.config = config or RestartScheduleConfig()
        self.start_time = time.time()
        self.last_restart_time = time.time()

    def should_restart(self) -> tuple[bool, str]:
        """재시작 필요 여부 판단"""
        uptime_hours = (time.time() - self.last_restart_time) / 3600

        # 강제 재시작 (최대 운영 시간 초과)
        if uptime_hours >= self.config.force_restart_after_hours:
            return True, f"강제 재시작: 운영 {uptime_hours:.1f}시간 초과"

        # 주기적 재시작 (시간대 확인)
        if uptime_hours >= self.config.restart_interval_hours:
            current_hour = time.localtime().tm_hour
            if (self.config.restart_window_start_hour
                    <= current_hour
                    < self.config.restart_window_end_hour):
                return True, f"주기적 재시작: {uptime_hours:.1f}시간 운영, 재시작 시간대"

        return False, ""

    def get_status(self) -> dict:
        uptime_hours = (time.time() - self.last_restart_time) / 3600
        next_restart = self.config.restart_interval_hours - uptime_hours

        return {
            "uptime_hours": round(uptime_hours, 2),
            "next_restart_in_hours": round(max(0, next_restart), 2),
            "force_restart_at_hours": self.config.force_restart_after_hours,
            "restart_window": f"{self.config.restart_window_start_hour:02d}:00-{self.config.restart_window_end_hour:02d}:00",
        }
```

---

## 16.2.6 통합: StabilityManager

```python
"""
stability_manager.py
24시간 안정성 관리 통합 모듈
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class StabilityManager:
    """
    24시간 안정성 관리자.

    모든 안정성 컴포넌트를 통합 관리:
    - MemoryWatchdog: 메모리 누수 감지/대응
    - CircuitBreaker: 연쇄 장애 방지
    - GracefulDegradation: 단계적 서비스 저하
    - PeriodicRestart: 주기적 재시작
    """

    def __init__(
        self,
        memory_watchdog: "MemoryWatchdog",
        circuit_breaker: "CircuitBreaker",
        degradation_manager: "GracefulDegradationManager",
        restart_scheduler: "PeriodicRestartScheduler",
    ):
        self.memory = memory_watchdog
        self.circuit_breaker = circuit_breaker
        self.degradation = degradation_manager
        self.restart = restart_scheduler
        self._background_tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """모든 안정성 컴포넌트 시작"""
        logger.info("StabilityManager 시작")
        self._background_tasks.append(
            asyncio.create_task(self.memory.run())
        )
        self._background_tasks.append(
            asyncio.create_task(self._restart_check_loop())
        )

    async def _restart_check_loop(self) -> None:
        """주기적 재시작 체크 루프"""
        while True:
            should, reason = self.restart.should_restart()
            if should:
                logger.warning(f"재시작 트리거: {reason}")
                # Kubernetes 환경에서는 exit → pod 재시작
                import os
                os._exit(0)
            await asyncio.sleep(300)  # 5분마다 체크

    async def stop(self) -> None:
        """모든 컴포넌트 정지"""
        self.memory.stop()
        for task in self._background_tasks:
            task.cancel()
        logger.info("StabilityManager 정지")

    def get_full_report(self) -> dict:
        """전체 안정성 보고서"""
        return {
            "memory": self.memory.get_report(),
            "circuit_breaker": self.circuit_breaker.get_status(),
            "degradation_level": self.degradation.current_level.name,
            "degradation_constraints": self.degradation.get_constraints(),
            "restart_schedule": self.restart.get_status(),
        }
```

---

## 용어 체크리스트

학습 후 아래 질문에 답할 수 있어야 한다:

- [ ] 메모리 누수를 선형 회귀로 감지하는 원리에서 $R^2$가 0.7 이상이어야 하는 이유는?
- [ ] GPU VRAM 누수와 Python RSS 메모리 누수의 원인 차이는?
- [ ] Circuit Breaker의 CLOSED → OPEN → HALF_OPEN → CLOSED 전이 조건을 각각 설명할 수 있는가?
- [ ] Graceful Degradation의 5단계에서 각 단계별 제약 조건의 근거는?
- [ ] 주기적 재시작의 가용성 공식 $A = 1 - T_{\text{restart}} / T_{\text{period}}$에서 Rolling Restart의 장점은?
- [ ] KV Cache 단편화가 발생하는 메커니즘과 해결 방법은?
- [ ] Backpressure와 Circuit Breaker의 역할 차이는?
- [ ] 의료 문서 OCR 서빙에서 24시간 안정성이 특히 중요한 이유(의료 시스템 특성)는?
