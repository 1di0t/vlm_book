# 16.1 Health Check

프로덕션 GPU 서빙 시스템에서 Health Check는 단순한 "살아있음" 확인이 아니다. GPU 상태, 모델 로딩 상태, KV Cache 여유, VRAM 온도까지 다차원으로 모니터링해야 한다. 문제를 조기에 탐지하지 못하면 사용자에게 오류가 전파된다.

---

> **핵심 용어 박스**
>
> | 용어 | 정의 |
> |------|------|
> | **Liveness Probe** | 프로세스가 살아있는지 확인. 실패 시 컨테이너를 재시작 |
> | **Readiness Probe** | 트래픽을 받을 준비가 됐는지 확인. 실패 시 로드밸런서에서 제외 |
> | **Startup Probe** | 초기 로딩이 완료됐는지 확인. 모델 로딩에 수 분 소요되는 경우 필수 |
> | **GPU Health** | GPU의 VRAM 사용량, 온도, ECC 에러, 전력 상태 등 하드웨어 건강도 |
> | **ECC Error** | GPU 메모리의 오류 정정 코드. 누적되면 하드웨어 장애 징후 |
> | **NVML** | NVIDIA Management Library. GPU 상태를 프로그래밍으로 조회하는 API |
> | **Graceful Shutdown** | 진행 중인 요청을 완료한 후 안전하게 종료하는 방식 |

---

## 16.1.1 Health Check 계층 구조

### 3단계 헬스 체크 모델

```
┌─────────────────────────────────────────────────┐
│              Kubernetes Probes                   │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Startup  │  │ Liveness │  │ Readiness│      │
│  │  Probe   │→ │  Probe   │  │  Probe   │      │
│  └──────────┘  └──────────┘  └──────────┘      │
│       │              │             │             │
│       ▼              ▼             ▼             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ 모델     │  │ 프로세스 │  │ GPU +    │      │
│  │ 로딩     │  │ 생존     │  │ 모델 +   │      │
│  │ 완료?    │  │ 확인     │  │ 큐 상태  │      │
│  └──────────┘  └──────────┘  └──────────┘      │
└─────────────────────────────────────────────────┘
```

| Probe | 확인 대상 | 실패 시 동작 | 주기 |
|-------|-----------|-------------|------|
| Startup | 모델 로딩 완료 여부 | 재시작 (failureThreshold 초과 시) | 10s |
| Liveness | 프로세스 응답 가능 여부 | 컨테이너 재시작 | 30s |
| Readiness | 추론 가능 + GPU 정상 | 로드밸런서에서 제외 (트래픽 차단) | 10s |

---

## 16.1.2 GPU 상태 모니터링

### NVIDIA GPU 헬스 메트릭

GPU 서빙에서 확인해야 할 핵심 지표:

| 메트릭 | 정상 범위 | 경고 임계값 | 위험 임계값 |
|--------|-----------|------------|------------|
| GPU 온도 | 30-70°C | > 80°C | > 90°C |
| VRAM 사용률 | 50-90% | > 95% | > 99% |
| GPU 활용률 | 30-95% | < 5% (유휴) | 0% (장애) |
| ECC 에러 (correctable) | 0 | > 10/hour | > 100/hour |
| ECC 에러 (uncorrectable) | 0 | > 0 | 즉시 교체 |
| 전력 사용량 | TDP의 50-100% | > TDP | 스로틀링 |
| PCIe 대역폭 | 정상 | 성능 저하 | 연결 끊김 |

### GPU Health Monitor 구현

```python
"""
gpu_health_monitor.py
NVIDIA GPU 상태 모니터링
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class GPUMetrics:
    """단일 GPU 메트릭"""
    gpu_id: int
    name: str
    temperature_c: float
    gpu_utilization_pct: float
    memory_used_mb: float
    memory_total_mb: float
    power_draw_w: float
    power_limit_w: float
    ecc_correctable: int
    ecc_uncorrectable: int
    pcie_gen: int
    pcie_width: int
    timestamp: float = field(default_factory=time.time)

    @property
    def memory_utilization_pct(self) -> float:
        if self.memory_total_mb == 0:
            return 100.0
        return (self.memory_used_mb / self.memory_total_mb) * 100.0

    @property
    def power_utilization_pct(self) -> float:
        if self.power_limit_w == 0:
            return 0.0
        return (self.power_draw_w / self.power_limit_w) * 100.0


@dataclass
class GPUHealthThresholds:
    """GPU 헬스 체크 임계값"""
    temp_warning_c: float = 80.0
    temp_critical_c: float = 90.0
    memory_warning_pct: float = 95.0
    memory_critical_pct: float = 99.0
    ecc_correctable_warning: int = 10
    ecc_uncorrectable_critical: int = 1
    utilization_idle_threshold: float = 5.0
    power_throttle_pct: float = 100.0


class GPUHealthMonitor:
    """
    GPU 상태를 주기적으로 확인하고 헬스 상태를 판정하는 모니터.

    판정 로직:
    - HEALTHY: 모든 메트릭이 정상 범위
    - DEGRADED: 경고 임계값 초과 (서빙 가능하지만 주의 필요)
    - UNHEALTHY: 위험 임계값 초과 (서빙 불가, 트래픽 차단)
    """

    def __init__(
        self,
        thresholds: Optional[GPUHealthThresholds] = None,
        history_size: int = 100,
    ):
        self.thresholds = thresholds or GPUHealthThresholds()
        self.history: list[list[GPUMetrics]] = []
        self.history_size = history_size
        self._nvml_initialized = False

    def initialize(self) -> bool:
        """NVML 초기화"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"NVML 초기화 완료: {device_count}개 GPU 감지")
            return True
        except ImportError:
            logger.warning("pynvml 미설치, 더미 모드로 동작")
            return False
        except Exception as e:
            logger.error(f"NVML 초기화 실패: {e}")
            return False

    def collect_metrics(self) -> list[GPUMetrics]:
        """모든 GPU의 메트릭 수집"""
        if not self._nvml_initialized:
            return self._collect_dummy_metrics()

        try:
            import pynvml

            device_count = pynvml.nvmlDeviceGetCount()
            metrics = []

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")

                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0

                # ECC 에러
                try:
                    ecc_correctable = pynvml.nvmlDeviceGetTotalEccErrors(
                        handle,
                        pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                        pynvml.NVML_VOLATILE_ECC,
                    )
                    ecc_uncorrectable = pynvml.nvmlDeviceGetTotalEccErrors(
                        handle,
                        pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                        pynvml.NVML_VOLATILE_ECC,
                    )
                except pynvml.NVMLError:
                    ecc_correctable = 0
                    ecc_uncorrectable = 0

                # PCIe 정보
                try:
                    pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
                    pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
                except pynvml.NVMLError:
                    pcie_gen = 0
                    pcie_width = 0

                metrics.append(GPUMetrics(
                    gpu_id=i,
                    name=name,
                    temperature_c=float(temp),
                    gpu_utilization_pct=float(util.gpu),
                    memory_used_mb=mem_info.used / (1024 * 1024),
                    memory_total_mb=mem_info.total / (1024 * 1024),
                    power_draw_w=power,
                    power_limit_w=power_limit,
                    ecc_correctable=ecc_correctable,
                    ecc_uncorrectable=ecc_uncorrectable,
                    pcie_gen=pcie_gen,
                    pcie_width=pcie_width,
                ))

            # 히스토리 저장
            self.history.append(metrics)
            if len(self.history) > self.history_size:
                self.history.pop(0)

            return metrics

        except Exception as e:
            logger.error(f"GPU 메트릭 수집 실패: {e}")
            return []

    def _collect_dummy_metrics(self) -> list[GPUMetrics]:
        """NVML 없을 때 더미 데이터 (테스트/개발용)"""
        return [GPUMetrics(
            gpu_id=0,
            name="Dummy GPU",
            temperature_c=65.0,
            gpu_utilization_pct=70.0,
            memory_used_mb=40000.0,
            memory_total_mb=81920.0,
            power_draw_w=250.0,
            power_limit_w=400.0,
            ecc_correctable=0,
            ecc_uncorrectable=0,
            pcie_gen=4,
            pcie_width=16,
        )]

    def evaluate_health(self, metrics: list[GPUMetrics]) -> dict:
        """
        GPU 헬스 종합 판정.

        판정 우선순위:
        1. Uncorrectable ECC → UNHEALTHY (즉시)
        2. 온도 > critical → UNHEALTHY
        3. 메모리 > critical → UNHEALTHY
        4. 경고 임계값 초과 → DEGRADED
        5. 전부 정상 → HEALTHY
        """
        overall_status = HealthStatus.HEALTHY
        gpu_statuses = []
        issues = []

        for gpu in metrics:
            gpu_status = HealthStatus.HEALTHY
            gpu_issues = []

            # ECC 에러 체크 (최우선)
            if gpu.ecc_uncorrectable >= self.thresholds.ecc_uncorrectable_critical:
                gpu_status = HealthStatus.UNHEALTHY
                gpu_issues.append(
                    f"GPU {gpu.gpu_id}: Uncorrectable ECC 에러 {gpu.ecc_uncorrectable}건"
                )

            if gpu.ecc_correctable >= self.thresholds.ecc_correctable_warning:
                if gpu_status == HealthStatus.HEALTHY:
                    gpu_status = HealthStatus.DEGRADED
                gpu_issues.append(
                    f"GPU {gpu.gpu_id}: Correctable ECC 에러 {gpu.ecc_correctable}건"
                )

            # 온도 체크
            if gpu.temperature_c >= self.thresholds.temp_critical_c:
                gpu_status = HealthStatus.UNHEALTHY
                gpu_issues.append(
                    f"GPU {gpu.gpu_id}: 온도 위험 {gpu.temperature_c}°C"
                )
            elif gpu.temperature_c >= self.thresholds.temp_warning_c:
                if gpu_status == HealthStatus.HEALTHY:
                    gpu_status = HealthStatus.DEGRADED
                gpu_issues.append(
                    f"GPU {gpu.gpu_id}: 온도 경고 {gpu.temperature_c}°C"
                )

            # 메모리 체크
            if gpu.memory_utilization_pct >= self.thresholds.memory_critical_pct:
                gpu_status = HealthStatus.UNHEALTHY
                gpu_issues.append(
                    f"GPU {gpu.gpu_id}: VRAM 위험 {gpu.memory_utilization_pct:.1f}%"
                )
            elif gpu.memory_utilization_pct >= self.thresholds.memory_warning_pct:
                if gpu_status == HealthStatus.HEALTHY:
                    gpu_status = HealthStatus.DEGRADED
                gpu_issues.append(
                    f"GPU {gpu.gpu_id}: VRAM 경고 {gpu.memory_utilization_pct:.1f}%"
                )

            gpu_statuses.append({
                "gpu_id": gpu.gpu_id,
                "status": gpu_status.value,
                "issues": gpu_issues,
            })
            issues.extend(gpu_issues)

            # 전체 상태 갱신 (최악 기준)
            if gpu_status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif gpu_status == HealthStatus.DEGRADED and overall_status != HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.DEGRADED

        return {
            "status": overall_status.value,
            "gpu_count": len(metrics),
            "gpus": gpu_statuses,
            "issues": issues,
            "timestamp": time.time(),
        }

    def get_trend(self, gpu_id: int, metric_name: str, window: int = 10) -> dict:
        """
        메트릭 추세 분석.

        최근 window개 샘플의 기울기(slope)를 계산하여
        메트릭이 악화 방향으로 진행 중인지 판단한다.

        선형 회귀: slope = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
        """
        recent = self.history[-window:] if len(self.history) >= window else self.history
        if len(recent) < 2:
            return {"trend": "insufficient_data", "slope": 0.0}

        values = []
        for snapshot in recent:
            for gpu in snapshot:
                if gpu.gpu_id == gpu_id:
                    val = getattr(gpu, metric_name, None)
                    if val is not None:
                        values.append(float(val))

        if len(values) < 2:
            return {"trend": "insufficient_data", "slope": 0.0}

        n = len(values)
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0.0

        if slope > 0.5:
            trend = "increasing"
        elif slope < -0.5:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": round(slope, 4),
            "current": values[-1],
            "samples": n,
        }
```

---

## 16.1.3 Health Check 엔드포인트

```python
"""
health_endpoints.py
FastAPI Health Check 엔드포인트
"""

import os
import time
import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, Response, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ============================================================
# 상태 모델
# ============================================================

class LivenessResponse(BaseModel):
    status: str
    uptime_seconds: float


class ReadinessResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_healthy: bool
    queue_length: int
    active_requests: int
    details: dict


class StartupResponse(BaseModel):
    status: str
    model_name: str
    loading_time_seconds: float


# ============================================================
# Health Check 서비스
# ============================================================

class HealthCheckService:
    """
    3단계 Health Check 서비스.

    의료 문서 OCR 서빙 특화:
    - GPU 상태를 Readiness에 포함
    - 모델 로딩 시간이 길어 Startup Probe 분리
    - 큐 길이 기반 부하 판단
    """

    def __init__(self, gpu_monitor: "GPUHealthMonitor"):
        self.gpu_monitor = gpu_monitor
        self.start_time = time.time()
        self.model_loaded = False
        self.model_name = ""
        self.model_load_time = 0.0
        self.active_requests = 0
        self.queue_length = 0
        self.max_queue_length = 100  # 큐 길이 상한
        self._last_inference_time = time.time()
        self._inference_timeout = 300  # 5분간 추론 없으면 이상

    def mark_model_loaded(self, model_name: str, load_time: float) -> None:
        """모델 로딩 완료 마킹"""
        self.model_loaded = True
        self.model_name = model_name
        self.model_load_time = load_time
        logger.info(f"모델 로딩 완료: {model_name} ({load_time:.1f}s)")

    def mark_inference(self) -> None:
        """추론 발생 기록"""
        self._last_inference_time = time.time()

    def check_liveness(self) -> tuple[bool, LivenessResponse]:
        """
        Liveness 체크: 프로세스가 응답할 수 있는가?

        실패 조건:
        - 프로세스 데드락 (이 함수 자체가 호출 안 되므로 타임아웃으로 감지)
        """
        uptime = time.time() - self.start_time
        return True, LivenessResponse(
            status="alive",
            uptime_seconds=round(uptime, 1),
        )

    def check_readiness(self) -> tuple[bool, ReadinessResponse]:
        """
        Readiness 체크: 트래픽을 받을 수 있는가?

        실패 조건:
        1. 모델 미로딩
        2. GPU unhealthy
        3. 큐 포화 (queue_length > max)
        4. 장시간 추론 없음 (모델 행 의심)
        """
        is_ready = True
        details = {}

        # 조건 1: 모델 로딩
        if not self.model_loaded:
            is_ready = False
            details["model"] = "not_loaded"
        else:
            details["model"] = "loaded"

        # 조건 2: GPU 상태
        gpu_metrics = self.gpu_monitor.collect_metrics()
        gpu_health = self.gpu_monitor.evaluate_health(gpu_metrics)
        gpu_healthy = gpu_health["status"] != "unhealthy"

        if not gpu_healthy:
            is_ready = False
        details["gpu"] = gpu_health

        # 조건 3: 큐 상태
        if self.queue_length > self.max_queue_length:
            is_ready = False
            details["queue"] = "overloaded"
        else:
            details["queue"] = "ok"

        # 조건 4: 추론 타임아웃 (모델이 로딩됐는데 한참 추론이 없으면)
        if self.model_loaded:
            idle_time = time.time() - self._last_inference_time
            if idle_time > self._inference_timeout and self.active_requests > 0:
                is_ready = False
                details["inference"] = f"stuck ({idle_time:.0f}s idle with active requests)"

        return is_ready, ReadinessResponse(
            status="ready" if is_ready else "not_ready",
            model_loaded=self.model_loaded,
            gpu_healthy=gpu_healthy,
            queue_length=self.queue_length,
            active_requests=self.active_requests,
            details=details,
        )

    def check_startup(self) -> tuple[bool, StartupResponse]:
        """
        Startup 체크: 초기 로딩 완료 여부.

        모델 로딩에 수 분이 소요될 수 있으므로
        Liveness와 분리하여 불필요한 재시작을 방지한다.
        """
        return self.model_loaded, StartupResponse(
            status="started" if self.model_loaded else "loading",
            model_name=self.model_name or "loading...",
            loading_time_seconds=round(self.model_load_time, 1),
        )


# ============================================================
# FastAPI 라우트 등록
# ============================================================

def register_health_routes(app: FastAPI, health_service: HealthCheckService):
    """FastAPI 앱에 헬스 체크 라우트를 등록"""

    @app.get("/health/live", response_model=LivenessResponse)
    async def liveness():
        is_alive, response = health_service.check_liveness()
        if not is_alive:
            return Response(
                content=response.model_dump_json(),
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                media_type="application/json",
            )
        return response

    @app.get("/health/ready", response_model=ReadinessResponse)
    async def readiness():
        is_ready, response = health_service.check_readiness()
        if not is_ready:
            return Response(
                content=response.model_dump_json(),
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                media_type="application/json",
            )
        return response

    @app.get("/health/startup", response_model=StartupResponse)
    async def startup():
        is_started, response = health_service.check_startup()
        if not is_started:
            return Response(
                content=response.model_dump_json(),
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                media_type="application/json",
            )
        return response

    @app.get("/health/gpu")
    async def gpu_health():
        """GPU 상세 헬스 정보"""
        metrics = health_service.gpu_monitor.collect_metrics()
        health = health_service.gpu_monitor.evaluate_health(metrics)

        # 추세 분석 추가
        for gpu_info in health["gpus"]:
            gpu_id = gpu_info["gpu_id"]
            gpu_info["trends"] = {
                "temperature": health_service.gpu_monitor.get_trend(
                    gpu_id, "temperature_c"
                ),
                "memory": health_service.gpu_monitor.get_trend(
                    gpu_id, "memory_utilization_pct"
                ),
            }

        return health
```

---

## 16.1.4 Kubernetes Probe 설정

```yaml
# k8s-health-probes.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ocr-model-server
  namespace: medical-ocr
spec:
  template:
    spec:
      containers:
      - name: vllm-server
        image: medical-ocr/vllm-server:latest

        # Startup Probe: 모델 로딩 대기 (최대 10분)
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 54    # 30 + 54*10 = 570초 ≈ 9.5분
          successThreshold: 1

        # Liveness Probe: 프로세스 생존 확인
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 0   # startupProbe 이후 즉시 시작
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3      # 3회 연속 실패 → 재시작
          successThreshold: 1

        # Readiness Probe: 트래픽 수신 가능 확인
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 0
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 2      # 2회 연속 실패 → 트래픽 차단
          successThreshold: 1      # 1회 성공 → 트래픽 복구
```

### Probe 타이밍 수학

실패 감지 최대 시간:

$$
T_{\text{detect}} = \text{periodSeconds} \times \text{failureThreshold}
$$

| Probe | 감지 시간 |
|-------|-----------|
| Startup | $10 \times 54 = 540$s (9분) |
| Liveness | $30 \times 3 = 90$s (1.5분) |
| Readiness | $10 \times 2 = 20$s |

Readiness 실패 시 사용자 영향을 최소화하려면 $T_{\text{detect}}$를 작게 설정해야 한다. 하지만 너무 민감하면 일시적 부하에도 노드가 트래픽에서 제외되는 flapping이 발생한다.

---

## 16.1.5 Graceful Shutdown

```python
"""
graceful_shutdown.py
진행 중인 요청을 안전하게 완료한 후 종료
"""

import signal
import asyncio
import logging
import time
from typing import Set

logger = logging.getLogger(__name__)


class GracefulShutdownManager:
    """
    Graceful Shutdown 관리자.

    종료 시퀀스:
    1. SIGTERM 수신
    2. Readiness Probe를 NOT READY로 변경 (새 요청 차단)
    3. 진행 중인 요청 완료 대기 (최대 grace_period)
    4. 타임아웃 시 강제 종료
    5. 리소스 정리 (GPU 메모리 해제)
    """

    def __init__(self, grace_period_seconds: int = 30):
        self.grace_period = grace_period_seconds
        self.is_shutting_down = False
        self.active_requests: Set[str] = set()
        self._shutdown_event = asyncio.Event()

    def register_signals(self, loop: asyncio.AbstractEventLoop) -> None:
        """시그널 핸들러 등록"""
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_signal)

    def _handle_signal(self) -> None:
        """종료 시그널 핸들러"""
        if self.is_shutting_down:
            logger.warning("강제 종료 시그널 수신, 즉시 종료")
            raise SystemExit(1)

        logger.info("종료 시그널 수신, graceful shutdown 시작")
        self.is_shutting_down = True
        self._shutdown_event.set()

    def track_request(self, request_id: str) -> None:
        """요청 추적 시작"""
        self.active_requests.add(request_id)

    def untrack_request(self, request_id: str) -> None:
        """요청 추적 종료"""
        self.active_requests.discard(request_id)

    async def wait_for_shutdown(self) -> None:
        """종료 시그널 대기 및 graceful shutdown 수행"""
        await self._shutdown_event.wait()

        logger.info(
            f"Graceful shutdown: {len(self.active_requests)}개 요청 완료 대기 "
            f"(최대 {self.grace_period}s)"
        )

        start = time.time()
        while self.active_requests:
            elapsed = time.time() - start
            if elapsed >= self.grace_period:
                logger.warning(
                    f"Grace period 초과, {len(self.active_requests)}개 요청 강제 중단"
                )
                break
            await asyncio.sleep(0.5)
            logger.info(f"대기 중: {len(self.active_requests)}개 요청 진행 중")

        logger.info("Graceful shutdown 완료")
```

---

## 용어 체크리스트

학습 후 아래 질문에 답할 수 있어야 한다:

- [ ] Liveness, Readiness, Startup Probe의 각각의 역할과 실패 시 동작 차이는?
- [ ] Startup Probe가 없으면 대형 모델 로딩 시 어떤 문제가 발생하는가?
- [ ] GPU의 Uncorrectable ECC 에러가 감지되면 어떻게 대응해야 하는가?
- [ ] VRAM 사용률 추세(slope) 분석으로 어떤 문제를 사전에 탐지할 수 있는가?
- [ ] Readiness Probe의 failureThreshold를 너무 낮게 설정하면 발생하는 flapping 문제란?
- [ ] Probe 실패 감지 시간 $T = \text{period} \times \text{failureThreshold}$에서 각 값의 트레이드오프는?
- [ ] Graceful Shutdown에서 grace_period가 너무 짧으면/길면 어떤 문제가 발생하는가?
- [ ] 의료 문서 OCR에서 GPU 온도 모니터링이 특히 중요한 이유는?
