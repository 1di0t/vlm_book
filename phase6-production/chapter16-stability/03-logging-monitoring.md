---
---

# 16.3 로깅 & 모니터링

> **핵심 용어 박스**
> | 용어 | 정의 | 관련 도구 |
> |------|------|-----------|
> | Prometheus | 시계열 메트릭 수집 및 저장 시스템 | prometheus-client |
> | Grafana | 메트릭 시각화 및 대시보드 도구 | Grafana OSS |
> | ELK Stack | Elasticsearch + Logstash + Kibana 로그 파이프라인 | Elastic |
> | Alert Rule | 임계값 기반 자동 알림 규칙 | Alertmanager |
> | SLO | Service Level Objective — 서비스 품질 목표 | — |
> | SLA | Service Level Agreement — 서비스 품질 계약 | — |
> | Percentile | 백분위수 — 응답 시간 분포 지표 (p50, p95, p99) | — |
> | Structured Logging | 키-값 쌍 기반 구조화된 로그 형식 | structlog |

---

## 개요

프로덕션 VLM 서비스에서 로깅과 모니터링은 안정적 운영의 핵심이다. 단순히 로그를 남기는 것이 아니라, **측정 가능한 메트릭을 정의하고, 이상 징후를 자동 탐지하며, 장애 발생 시 신속하게 대응**하는 체계를 구축해야 한다.

의료 문서 OCR 서비스는 정확도와 응답 시간 모두 엄격한 기준이 필요하다. 금액이나 진단코드 오류는 보험 청구 거절로 직결되기 때문이다.

---

## 핵심 메트릭 정의

### 1. Latency 메트릭

| 메트릭 | 설명 | 목표 (예시) |
|--------|------|------------|
| TTFT (Time to First Token) | 첫 토큰 생성까지 시간 | < 500ms |
| TPOT (Time Per Output Token) | 토큰당 생성 시간 | < 50ms |
| E2E Latency | 요청 → 응답 완료 | < 5s (p95) |
| Queue Wait | 대기열 대기 시간 | < 200ms (p95) |

### 2. Throughput 메트릭

| 메트릭 | 설명 | 목표 (예시) |
|--------|------|------------|
| RPS (Requests Per Second) | 초당 처리 요청 수 | > 10 |
| Tokens/s | 초당 생성 토큰 수 | > 500 |
| Documents/min | 분당 처리 문서 수 | > 30 |

### 3. 품질 메트릭

| 메트릭 | 설명 | 목표 |
|--------|------|------|
| Error Rate | 오류 응답 비율 | < 0.1% |
| Timeout Rate | 타임아웃 비율 | < 0.5% |
| OCR Accuracy (샘플) | 주기적 정확도 체크 | > 95% |

### 4. 리소스 메트릭

| 메트릭 | 설명 | 임계값 |
|--------|------|--------|
| GPU Utilization | GPU 연산 사용률 | 60~90% |
| GPU Memory | VRAM 사용량 | < 90% |
| GPU Temperature | GPU 온도 | < 80°C |
| CPU Utilization | CPU 사용률 | < 80% |
| RAM Usage | 시스템 메모리 사용량 | < 85% |

---

## 수학적 원리

### 백분위수 (Percentile) 계산

응답 시간의 분포를 이해하려면 평균보다 백분위수가 더 유용하다.

**정의**: $n$개의 정렬된 관측값 $x_1 \leq x_2 \leq \ldots \leq x_n$에서 $p$-번째 백분위수:

$$P_p = x_{\lceil (p/100) \cdot n \rceil}$$

보간법 사용 시:

$$P_p = x_{\lfloor k \rfloor} + (k - \lfloor k \rfloor)(x_{\lfloor k \rfloor + 1} - x_{\lfloor k \rfloor})$$

여기서 $k = (p/100)(n-1) + 1$

**왜 p99가 중요한가?**

100명의 사용자가 요청을 보내면, p50은 절반이 경험하는 지연이지만 p99는 가장 느린 1%의 경험이다. 높은 트래픽에서 p99가 나쁘면 상당수 사용자가 나쁜 경험을 하게 된다:

- 1000 RPS × 1% = 매초 10명이 느린 응답을 경험

### SLO/SLA 설계

**SLI (Service Level Indicator)**: 측정 가능한 메트릭

$$\text{SLI}_{latency} = \frac{|\{r : \text{latency}(r) < T\}|}{|\{r\}|}$$

**SLO (Service Level Objective)**: SLI의 목표값

$$\text{SLO}: \text{SLI}_{latency} \geq 99.5\% \quad (T = 3\text{s})$$

**Error Budget**: SLO에서 허용하는 실패 예산

$$\text{Error Budget} = 1 - \text{SLO Target}$$

30일 기준 error budget:

$$\text{Budget}_{minutes} = 30 \times 24 \times 60 \times (1 - 0.999) = 43.2 \text{ min}$$

### 이상 탐지: Z-Score 기반

실시간 메트릭에서 이상을 탐지하는 기본 방법:

$$z = \frac{x - \mu}{\sigma}$$

$|z| > 3$이면 이상으로 판단 (99.7% 신뢰구간 밖).

**EWMA (Exponentially Weighted Moving Average)** 기반 적응형 탐지:

$$\hat{\mu}_t = \alpha x_t + (1-\alpha) \hat{\mu}_{t-1}$$

$$\hat{\sigma}_t^2 = \alpha (x_t - \hat{\mu}_t)^2 + (1-\alpha) \hat{\sigma}_{t-1}^2$$

$\alpha$는 smoothing factor (보통 0.1~0.3). 최근 값에 더 큰 가중치를 부여한다.

---

## Prometheus 메트릭 수집

### 메트릭 정의 및 수집기 구현

```python
import time
import threading
from dataclasses import dataclass, field
from typing import Optional
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    start_http_server, generate_latest,
    CONTENT_TYPE_LATEST,
)

# --- Prometheus 메트릭 정의 ---

# 요청 관련
REQUEST_COUNT = Counter(
    "ocr_requests_total",
    "총 OCR 요청 수",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "ocr_request_duration_seconds",
    "요청 처리 시간 (초)",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0],
)

TTFT_LATENCY = Histogram(
    "ocr_ttft_seconds",
    "첫 토큰 생성 시간 (초)",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)

TOKENS_GENERATED = Counter(
    "ocr_tokens_generated_total",
    "생성된 총 토큰 수",
)

# GPU 관련
GPU_UTILIZATION = Gauge(
    "gpu_utilization_percent",
    "GPU 사용률 (%)",
    ["gpu_id"],
)

GPU_MEMORY_USED = Gauge(
    "gpu_memory_used_bytes",
    "GPU 메모리 사용량 (bytes)",
    ["gpu_id"],
)

GPU_MEMORY_TOTAL = Gauge(
    "gpu_memory_total_bytes",
    "GPU 총 메모리 (bytes)",
    ["gpu_id"],
)

GPU_TEMPERATURE = Gauge(
    "gpu_temperature_celsius",
    "GPU 온도 (°C)",
    ["gpu_id"],
)

# 모델 관련
MODEL_LOADED = Gauge(
    "model_loaded",
    "모델 로드 상태 (1=loaded, 0=not loaded)",
)

ACTIVE_REQUESTS = Gauge(
    "ocr_active_requests",
    "현재 처리 중인 요청 수",
)

QUEUE_SIZE = Gauge(
    "ocr_queue_size",
    "대기열 크기",
)

# 비즈니스 메트릭
OCR_ACCURACY_SAMPLE = Gauge(
    "ocr_accuracy_sample",
    "샘플 기반 OCR 정확도",
    ["document_type"],
)

CRITICAL_FIELD_ERROR = Counter(
    "ocr_critical_field_errors_total",
    "Critical Field 오류 수",
    ["field_type"],  # amount, kcd_code, date
)
```

### GPU 모니터링 수집기

```python
class GPUMetricsCollector:
    """주기적으로 GPU 메트릭을 수집하여 Prometheus에 노출."""

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def _collect_loop(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
        except Exception:
            return

        while self._running:
            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    gpu_id = str(i)

                    # 사용률
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    GPU_UTILIZATION.labels(gpu_id=gpu_id).set(util.gpu)

                    # 메모리
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    GPU_MEMORY_USED.labels(gpu_id=gpu_id).set(mem.used)
                    GPU_MEMORY_TOTAL.labels(gpu_id=gpu_id).set(mem.total)

                    # 온도
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    GPU_TEMPERATURE.labels(gpu_id=gpu_id).set(temp)
                except Exception:
                    pass

            time.sleep(self.interval)
```

### 요청 미들웨어 (FastAPI)

```python
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

class MetricsMiddleware(BaseHTTPMiddleware):
    """모든 요청에 대해 메트릭을 자동 수집하는 미들웨어."""

    async def dispatch(self, request: Request, call_next):
        method = request.method
        endpoint = request.url.path

        ACTIVE_REQUESTS.inc()
        start_time = time.time()

        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception as e:
            status = "500"
            logger.error(f"Request failed: {e}")
            raise
        finally:
            duration = time.time() - start_time
            ACTIVE_REQUESTS.dec()

            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status=status,
            ).inc()

            REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)

        return response


def create_app() -> FastAPI:
    app = FastAPI(title="Medical OCR Service")
    app.add_middleware(MetricsMiddleware)

    gpu_collector = GPUMetricsCollector(interval=5.0)

    @app.on_event("startup")
    async def startup():
        # Prometheus 메트릭 서버 (별도 포트)
        start_http_server(9090)
        gpu_collector.start()
        MODEL_LOADED.set(1)
        logger.info("Metrics server started on :9090")

    @app.on_event("shutdown")
    async def shutdown():
        gpu_collector.stop()

    @app.get("/metrics")
    async def metrics():
        """Prometheus scrape 엔드포인트 (같은 포트로도 제공)."""
        from starlette.responses import Response
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app
```

---

## 구조화 로깅 (Structured Logging)

### structlog 기반 로깅 설정

```python
import structlog
import logging
import sys
import json
from datetime import datetime, timezone


def setup_structured_logging(log_level: str = "INFO", json_output: bool = True):
    """프로덕션용 구조화 로깅 설정."""

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )


class OCRRequestLogger:
    """OCR 요청별 구조화 로그를 생성."""

    def __init__(self):
        self.logger = structlog.get_logger("ocr.request")

    def log_request_start(self, request_id: str, document_type: str,
                          image_size: tuple):
        self.logger.info(
            "ocr_request_started",
            request_id=request_id,
            document_type=document_type,
            image_width=image_size[0],
            image_height=image_size[1],
        )

    def log_request_complete(self, request_id: str, duration_ms: float,
                             token_count: int, field_count: int):
        self.logger.info(
            "ocr_request_completed",
            request_id=request_id,
            duration_ms=round(duration_ms, 2),
            token_count=token_count,
            field_count=field_count,
        )

    def log_critical_field_error(self, request_id: str, field_type: str,
                                  expected: str, predicted: str):
        self.logger.warning(
            "critical_field_mismatch",
            request_id=request_id,
            field_type=field_type,
            expected=expected,
            predicted=predicted,
        )
        CRITICAL_FIELD_ERROR.labels(field_type=field_type).inc()

    def log_request_error(self, request_id: str, error: str,
                          error_type: str):
        self.logger.error(
            "ocr_request_failed",
            request_id=request_id,
            error=error,
            error_type=error_type,
        )
```

---

## 알림 규칙 (Alerting)

### Prometheus Alertmanager 규칙

```yaml
# prometheus/alert_rules.yml

groups:
  - name: ocr_service_alerts
    rules:
      # --- Latency ---
      - alert: HighP95Latency
        expr: histogram_quantile(0.95, rate(ocr_request_duration_seconds_bucket[5m])) > 5
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "p95 latency가 5초를 초과"
          description: "최근 5분간 p95 latency: {{ $value }}s"

      - alert: HighP99Latency
        expr: histogram_quantile(0.99, rate(ocr_request_duration_seconds_bucket[5m])) > 10
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "p99 latency가 10초를 초과"

      # --- Error Rate ---
      - alert: HighErrorRate
        expr: |
          rate(ocr_requests_total{status=~"5.."}[5m])
          / rate(ocr_requests_total[5m]) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "에러율 1% 초과"
          description: "5xx 에러율: {{ $value | humanizePercentage }}"

      # --- GPU ---
      - alert: GPUMemoryHigh
        expr: gpu_memory_used_bytes / gpu_memory_total_bytes > 0.95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU {{ $labels.gpu_id }} 메모리 95% 초과"

      - alert: GPUTemperatureHigh
        expr: gpu_temperature_celsius > 85
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "GPU {{ $labels.gpu_id }} 온도 85°C 초과"

      # --- Throughput ---
      - alert: LowThroughput
        expr: rate(ocr_requests_total[5m]) < 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "처리량 저하 — 10분간 RPS < 1"

      # --- Queue ---
      - alert: QueueBacklog
        expr: ocr_queue_size > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "대기열 100건 초과"

      # --- Critical Field ---
      - alert: CriticalFieldErrors
        expr: rate(ocr_critical_field_errors_total[15m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Critical Field 오류율 증가"
          description: "필드 유형: {{ $labels.field_type }}"

      # --- Service Health ---
      - alert: ServiceDown
        expr: up{job="ocr-service"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "OCR 서비스 다운"

      - alert: ModelNotLoaded
        expr: model_loaded == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "모델 로드 실패"
```

### Alertmanager 설정

```yaml
# alertmanager/alertmanager.yml

global:
  resolve_timeout: 5m

route:
  receiver: default
  group_by: [alertname, severity]
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h

  routes:
    - match:
        severity: critical
      receiver: critical-alerts
      repeat_interval: 1h

    - match:
        severity: warning
      receiver: warning-alerts
      repeat_interval: 4h

receivers:
  - name: default
    webhook_configs:
      - url: "http://alert-handler:8080/webhook"

  - name: critical-alerts
    slack_configs:
      - api_url: "${SLACK_WEBHOOK_URL}"
        channel: "#ocr-alerts-critical"
        title: "[CRITICAL] {{ .GroupLabels.alertname }}"
        text: "{{ .CommonAnnotations.summary }}"

  - name: warning-alerts
    slack_configs:
      - api_url: "${SLACK_WEBHOOK_URL}"
        channel: "#ocr-alerts-warning"
        title: "[WARNING] {{ .GroupLabels.alertname }}"
        text: "{{ .CommonAnnotations.summary }}"
```

---

## Grafana 대시보드

### 핵심 패널 구성

```json
{
  "dashboard": {
    "title": "Medical OCR Service Dashboard",
    "panels": [
      {
        "title": "Request Rate (RPS)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(ocr_requests_total[1m])",
            "legendFormat": "{{ status }}"
          }
        ]
      },
      {
        "title": "Latency Percentiles",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(ocr_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(ocr_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(ocr_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p99"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(ocr_requests_total{status=~'5..'}[5m]) / rate(ocr_requests_total[5m])"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "gauge",
        "targets": [
          {
            "expr": "gpu_utilization_percent",
            "legendFormat": "GPU {{ gpu_id }}"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "type": "timeseries",
        "targets": [
          {
            "expr": "gpu_memory_used_bytes / gpu_memory_total_bytes * 100",
            "legendFormat": "GPU {{ gpu_id }}"
          }
        ]
      },
      {
        "title": "Active Requests & Queue",
        "type": "timeseries",
        "targets": [
          {
            "expr": "ocr_active_requests",
            "legendFormat": "Active"
          },
          {
            "expr": "ocr_queue_size",
            "legendFormat": "Queue"
          }
        ]
      },
      {
        "title": "Critical Field Errors",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(ocr_critical_field_errors_total[5m])",
            "legendFormat": "{{ field_type }}"
          }
        ]
      }
    ]
  }
}
```

---

## SLO 대시보드 구현

```python
@dataclass
class SLODefinition:
    """SLO 정의."""
    name: str
    sli_query: str          # Prometheus 쿼리
    target: float           # 예: 0.995 (99.5%)
    window_days: int = 30   # 측정 기간

    @property
    def error_budget(self) -> float:
        return 1 - self.target

    @property
    def error_budget_minutes(self) -> float:
        return self.window_days * 24 * 60 * self.error_budget


# SLO 정의 예시
SLOS = [
    SLODefinition(
        name="Latency SLO",
        sli_query='rate(ocr_request_duration_seconds_bucket{le="3.0"}[30d]) / rate(ocr_request_duration_seconds_count[30d])',
        target=0.995,
    ),
    SLODefinition(
        name="Availability SLO",
        sli_query='1 - rate(ocr_requests_total{status=~"5.."}[30d]) / rate(ocr_requests_total[30d])',
        target=0.999,
    ),
    SLODefinition(
        name="Accuracy SLO",
        sli_query='ocr_accuracy_sample{document_type="all"}',
        target=0.95,
    ),
]

# 각 SLO의 error budget
for slo in SLOS:
    print(f"{slo.name}:")
    print(f"  Target: {slo.target*100}%")
    print(f"  Error Budget: {slo.error_budget*100}% = {slo.error_budget_minutes:.1f} min/month")
```

출력 예시:
```
Latency SLO:
  Target: 99.5%
  Error Budget: 0.5% = 216.0 min/month
Availability SLO:
  Target: 99.9%
  Error Budget: 0.1% = 43.2 min/month
Accuracy SLO:
  Target: 95.0%
  Error Budget: 5.0% = 2160.0 min/month
```

---

## 이상 탐지 자동화

```python
import numpy as np
from collections import deque
from enum import Enum


class AnomalyLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class EWMAAnomalyDetector:
    """EWMA 기반 실시간 이상 탐지기."""

    def __init__(
        self,
        alpha: float = 0.2,
        warning_sigma: float = 2.0,
        critical_sigma: float = 3.0,
        warmup_samples: int = 50,
    ):
        self.alpha = alpha
        self.warning_sigma = warning_sigma
        self.critical_sigma = critical_sigma
        self.warmup_samples = warmup_samples

        self.ewma_mean: float = 0.0
        self.ewma_var: float = 0.0
        self.sample_count: int = 0

    def update(self, value: float) -> AnomalyLevel:
        self.sample_count += 1

        if self.sample_count == 1:
            self.ewma_mean = value
            self.ewma_var = 0.0
            return AnomalyLevel.NORMAL

        # EWMA 업데이트
        prev_mean = self.ewma_mean
        self.ewma_mean = self.alpha * value + (1 - self.alpha) * self.ewma_mean
        self.ewma_var = (
            self.alpha * (value - prev_mean) ** 2
            + (1 - self.alpha) * self.ewma_var
        )

        # warmup 기간에는 항상 NORMAL
        if self.sample_count < self.warmup_samples:
            return AnomalyLevel.NORMAL

        # Z-score 계산
        std = np.sqrt(self.ewma_var) + 1e-8
        z_score = abs(value - self.ewma_mean) / std

        if z_score > self.critical_sigma:
            return AnomalyLevel.CRITICAL
        elif z_score > self.warning_sigma:
            return AnomalyLevel.WARNING
        return AnomalyLevel.NORMAL


class MetricMonitor:
    """여러 메트릭을 동시에 모니터링."""

    def __init__(self):
        self.detectors: dict[str, EWMAAnomalyDetector] = {}
        self.anomaly_history: deque = deque(maxlen=1000)

    def register_metric(self, name: str, **kwargs):
        self.detectors[name] = EWMAAnomalyDetector(**kwargs)

    def observe(self, name: str, value: float) -> AnomalyLevel:
        if name not in self.detectors:
            raise KeyError(f"Unknown metric: {name}")

        level = self.detectors[name].update(value)

        if level != AnomalyLevel.NORMAL:
            self.anomaly_history.append({
                "metric": name,
                "value": value,
                "level": level.value,
                "mean": self.detectors[name].ewma_mean,
                "std": np.sqrt(self.detectors[name].ewma_var),
            })

        return level

    def get_recent_anomalies(self, n: int = 10) -> list:
        return list(self.anomaly_history)[-n:]


# 사용 예시
monitor = MetricMonitor()
monitor.register_metric("latency_p99", alpha=0.1, warning_sigma=2.0, critical_sigma=3.0)
monitor.register_metric("error_rate", alpha=0.2, warning_sigma=2.5, critical_sigma=3.5)
monitor.register_metric("gpu_memory_pct", alpha=0.05, warning_sigma=2.0, critical_sigma=2.5)
```

---

## docker-compose: 모니터링 스택

```yaml
# docker-compose.monitoring.yml

services:
  prometheus:
    image: prom/prometheus:v2.50.0
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus/alert_rules.yml:/etc/prometheus/alert_rules.yml
      - prometheus_data:/prometheus
    ports:
      - "9091:9090"
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.retention.time=30d"

  alertmanager:
    image: prom/alertmanager:v0.27.0
    volumes:
      - ./alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml
    ports:
      - "9093:9093"

  grafana:
    image: grafana/grafana:10.3.0
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}

  node-exporter:
    image: prom/node-exporter:v1.7.0
    ports:
      - "9100:9100"

volumes:
  prometheus_data:
  grafana_data:
```

```yaml
# prometheus/prometheus.yml

global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

scrape_configs:
  - job_name: "ocr-service"
    static_configs:
      - targets: ["ocr-service:9090"]

  - job_name: "node-exporter"
    static_configs:
      - targets: ["node-exporter:9100"]
```

---

## 로그 집계 파이프라인 (ELK 대안: Loki)

대규모 ELK 대신 Grafana Loki를 사용하면 리소스 효율적이다:

```yaml
# docker-compose.logging.yml (Loki 기반)

services:
  loki:
    image: grafana/loki:2.9.4
    ports:
      - "3100:3100"
    volumes:
      - loki_data:/loki

  promtail:
    image: grafana/promtail:2.9.4
    volumes:
      - /var/log:/var/log:ro
      - ./promtail/config.yml:/etc/promtail/config.yml
    command: -config.file=/etc/promtail/config.yml

volumes:
  loki_data:
```

---

## 운영 체크리스트

| 항목 | 설정 여부 | 비고 |
|------|-----------|------|
| Prometheus 메트릭 수집 | | 15초 간격 |
| GPU 메트릭 (pynvml) | | 5초 간격 |
| Grafana 대시보드 | | 핵심 7개 패널 |
| Latency SLO 알림 | | p95 > 5s, p99 > 10s |
| Error Rate 알림 | | > 1% |
| GPU 메모리/온도 알림 | | > 95% / > 85°C |
| Critical Field 오류 알림 | | 의료 문서 특화 |
| 구조화 로깅 | | JSON 형식 |
| Error Budget 추적 | | 월간 리셋 |
| 이상 탐지 자동화 | | EWMA 기반 |

---

## 용어 체크리스트

- [ ] Prometheus Counter, Histogram, Gauge의 차이를 설명할 수 있는가?
- [ ] p50, p95, p99 latency의 의미와 계산법을 알고 있는가?
- [ ] SLI, SLO, SLA의 관계를 설명할 수 있는가?
- [ ] Error Budget의 의미와 계산법 ($1 - \text{SLO Target}$)을 이해하는가?
- [ ] EWMA 기반 이상 탐지의 수식을 직접 작성할 수 있는가?
- [ ] Z-Score 기반 이상 판단 기준 ($|z| > 3$)의 통계적 근거를 설명할 수 있는가?
- [ ] Prometheus + Grafana + Alertmanager 스택을 구성할 수 있는가?
- [ ] 구조화 로깅이 왜 필요한지, 일반 텍스트 로그 대비 장점을 설명할 수 있는가?
- [ ] 의료 문서 OCR 서비스에서 Critical Field 모니터링이 왜 중요한지 설명할 수 있는가?
