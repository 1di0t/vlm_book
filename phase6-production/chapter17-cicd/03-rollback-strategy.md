---
---

# 17.3 Rollback 전략

모델을 배포한 뒤 문제가 발생하면 즉시 이전 버전으로 되돌려야 한다. 롤백이 느리면 사용자 경험이 나빠지고, 비즈니스 손실이 발생한다. 자동 롤백 조건 설정, 수동 롤백 절차, Feature Flag를 활용한 점진적 배포/롤백 전략을 다룬다.

> **핵심 용어**
>
> | 용어 | 정의 |
> |------|------|
> | **Rollback Trigger** | 롤백을 발동시키는 **조건**. accuracy 하락, latency 증가, error rate 증가 등 사전 정의된 임계값을 위반하면 자동으로 실행된다. |
> | **Version Pinning** | 특정 모델 버전을 **고정**하여 의도하지 않은 업데이트를 방지하는 기법. 프로덕션 환경에서 검증된 버전만 사용하도록 강제한다. |
> | **Feature Flag** | 코드 배포 없이 **런타임에 기능을 활성화/비활성화**할 수 있는 토글. 모델 버전 전환, 점진적 배포, 즉시 롤백에 활용한다. |
> | **Automated Rollback** | 모니터링 메트릭이 사전 정의된 임계값을 위반하면 **자동으로 이전 버전으로 되돌리는** 시스템. 사람 개입 없이 수 초 내 롤백이 가능하다. |
> | **Circuit Breaker** | 연속 실패가 임계값을 초과하면 **요청을 차단**하고 fallback 응답을 반환하는 패턴. 장애 전파를 방지한다. |
> | **Graceful Degradation** | 모델 서빙에 문제가 생겼을 때 완전히 중단하지 않고, **품질을 낮추더라도 서비스를 유지**하는 전략. 예: 고성능 모델 → 경량 fallback 모델. |
> | **Blast Radius** | 장애 발생 시 **영향받는 범위**. 롤백 전략은 Blast Radius를 최소화하는 방향으로 설계해야 한다. |

---

## 자동 롤백 조건

자동 롤백은 "무엇이 잘못되면 롤백할 것인가"를 사전에 정의하는 것에서 시작한다.

### 롤백 트리거 메트릭

| 메트릭 | 임계값 (예시) | 윈도우 | 설명 |
|--------|-------------|--------|------|
| **Accuracy 하락** | < 기존 대비 3% 이상 하락 | 5분 평균 | 모델 품질 저하 |
| **Latency P99 증가** | > 500ms (기존 대비 2배) | 1분 평균 | 응답 시간 급증 |
| **Error Rate 증가** | > 5% (기존 대비 5배) | 1분 합계 | 서버 에러, 추론 실패 |
| **Throughput 감소** | < 기존 대비 50% | 3분 평균 | 처리 능력 급감 |
| **Memory OOM** | 컨테이너 메모리 > 90% | 즉시 | 메모리 부족으로 crash 임박 |
| **GPU 에러** | CUDA OOM 또는 NaN 출력 | 즉시 | 하드웨어 레벨 장애 |

### 수학적 기준: 이상 탐지

롤백 트리거의 임계값을 단순 고정값이 아닌, **통계적 이상 탐지**로 설정할 수 있다.

**Z-score 기반 이상 탐지:**

$$
z_t = \frac{x_t - \mu}{\sigma}
$$

여기서:
- $x_t$: 시점 $t$의 메트릭 값
- $\mu$: 최근 $N$개 윈도우의 평균
- $\sigma$: 최근 $N$개 윈도우의 표준편차

$|z_t| > 3$이면 이상치로 판단 (99.7% 신뢰구간 밖).

**지수 가중 이동 평균 (EWMA):**

$$
\text{EWMA}_t = \alpha \cdot x_t + (1 - \alpha) \cdot \text{EWMA}_{t-1}
$$

EWMA 제어 한계(Control Limits):

$$
\text{UCL} = \mu_0 + L \cdot \sigma \sqrt{\frac{\alpha}{2 - \alpha} \left[1 - (1-\alpha)^{2t}\right]}
$$

$$
\text{LCL} = \mu_0 - L \cdot \sigma \sqrt{\frac{\alpha}{2 - \alpha} \left[1 - (1-\alpha)^{2t}\right]}
$$

여기서 $L$은 제어 한계 승수(보통 3), $\alpha$는 스무딩 계수(보통 0.2), $\mu_0$는 기준 평균이다. EWMA가 UCL/LCL을 벗어나면 롤백을 트리거한다.

---

## 코드: RollbackManager 클래스

```python
import os
import time
import json
import logging
import subprocess
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
from typing import Callable

import requests
import numpy as np

logger = logging.getLogger(__name__)


class RollbackReason(Enum):
    ACCURACY_DROP = "accuracy_drop"
    LATENCY_SPIKE = "latency_spike"
    ERROR_RATE_HIGH = "error_rate_high"
    THROUGHPUT_DROP = "throughput_drop"
    MEMORY_OOM = "memory_oom"
    HEALTH_CHECK_FAIL = "health_check_fail"
    MANUAL = "manual"


@dataclass
class RollbackThresholds:
    """롤백 트리거 임계값 설정."""
    # 절대 임계값
    min_accuracy: float = 0.92
    max_latency_p99_ms: float = 500.0
    max_error_rate: float = 0.05
    min_throughput_rps: float = 10.0
    max_memory_percent: float = 90.0

    # 상대 임계값 (기존 대비 비율)
    accuracy_drop_ratio: float = 0.03    # 3% 이상 하락
    latency_increase_ratio: float = 2.0  # 2배 이상 증가
    error_rate_increase_ratio: float = 5.0  # 5배 이상 증가

    # 윈도우 설정
    window_size: int = 60               # 메트릭 윈도우 크기 (초)
    consecutive_failures: int = 3        # 연속 N회 위반 시 롤백


@dataclass
class RollbackEvent:
    """롤백 이벤트 기록."""
    timestamp: str
    reason: RollbackReason
    from_version: str
    to_version: str
    metrics_snapshot: dict
    triggered_by: str  # "auto" | "manual"
    duration_seconds: float = 0.0
    success: bool = False


@dataclass
class MetricSnapshot:
    """특정 시점의 메트릭 스냅샷."""
    timestamp: float
    accuracy: float
    latency_p99: float
    error_rate: float
    throughput: float
    memory_percent: float


class RollbackManager:
    """모델 롤백을 관리하는 클래스.

    주요 기능:
    - 자동 롤백 조건 모니터링
    - 수동 롤백 실행
    - 롤백 이력 관리
    - Version Pinning
    """

    def __init__(
        self,
        model_name: str,
        namespace: str = "default",
        thresholds: RollbackThresholds | None = None,
    ):
        self.model_name = model_name
        self.namespace = namespace
        self.thresholds = thresholds or RollbackThresholds()

        # 상태 관리
        self.current_version: str = ""
        self.previous_version: str = ""
        self.pinned_version: str | None = None
        self.rollback_history: list[RollbackEvent] = []

        # 메트릭 버퍼 (이상 탐지용)
        self.metric_buffer: deque[MetricSnapshot] = deque(maxlen=300)
        self.baseline_metrics: MetricSnapshot | None = None

        # 연속 위반 카운터
        self._violation_counters: dict[str, int] = {
            "accuracy": 0,
            "latency": 0,
            "error_rate": 0,
            "throughput": 0,
            "memory": 0,
        }

    def set_baseline(self, metrics: MetricSnapshot):
        """기존 모델의 기준 메트릭을 설정한다."""
        self.baseline_metrics = metrics
        logger.info(
            f"기준 메트릭 설정: accuracy={metrics.accuracy:.4f}, "
            f"latency_p99={metrics.latency_p99:.1f}ms"
        )

    def pin_version(self, version: str):
        """특정 버전을 고정한다. 고정된 버전은 자동 업데이트 대상에서 제외된다."""
        self.pinned_version = version
        logger.info(f"버전 고정: {version}")

    def unpin_version(self):
        """버전 고정을 해제한다."""
        logger.info(f"버전 고정 해제: {self.pinned_version}")
        self.pinned_version = None

    def check_rollback_needed(self, current: MetricSnapshot) -> RollbackReason | None:
        """현재 메트릭을 기반으로 롤백 필요 여부를 판단한다.

        Args:
            current: 현재 시점의 메트릭 스냅샷

        Returns:
            롤백 사유 (None이면 정상)
        """
        self.metric_buffer.append(current)
        reasons = []

        # 1. 절대 임계값 검사
        if current.accuracy < self.thresholds.min_accuracy:
            self._violation_counters["accuracy"] += 1
            reasons.append(("accuracy", RollbackReason.ACCURACY_DROP))
        else:
            self._violation_counters["accuracy"] = 0

        if current.latency_p99 > self.thresholds.max_latency_p99_ms:
            self._violation_counters["latency"] += 1
            reasons.append(("latency", RollbackReason.LATENCY_SPIKE))
        else:
            self._violation_counters["latency"] = 0

        if current.error_rate > self.thresholds.max_error_rate:
            self._violation_counters["error_rate"] += 1
            reasons.append(("error_rate", RollbackReason.ERROR_RATE_HIGH))
        else:
            self._violation_counters["error_rate"] = 0

        if current.throughput < self.thresholds.min_throughput_rps:
            self._violation_counters["throughput"] += 1
            reasons.append(("throughput", RollbackReason.THROUGHPUT_DROP))
        else:
            self._violation_counters["throughput"] = 0

        if current.memory_percent > self.thresholds.max_memory_percent:
            self._violation_counters["memory"] += 1
            reasons.append(("memory", RollbackReason.MEMORY_OOM))
        else:
            self._violation_counters["memory"] = 0

        # 2. 상대 임계값 검사 (기준 대비)
        if self.baseline_metrics:
            baseline = self.baseline_metrics

            if baseline.accuracy - current.accuracy > self.thresholds.accuracy_drop_ratio:
                self._violation_counters["accuracy"] += 1
                if ("accuracy", RollbackReason.ACCURACY_DROP) not in reasons:
                    reasons.append(("accuracy", RollbackReason.ACCURACY_DROP))

            if current.latency_p99 > baseline.latency_p99 * self.thresholds.latency_increase_ratio:
                self._violation_counters["latency"] += 1
                if ("latency", RollbackReason.LATENCY_SPIKE) not in reasons:
                    reasons.append(("latency", RollbackReason.LATENCY_SPIKE))

        # 3. 연속 위반 검사
        for metric_name, reason in reasons:
            if self._violation_counters[metric_name] >= self.thresholds.consecutive_failures:
                logger.warning(
                    f"롤백 트리거: {reason.value} "
                    f"(연속 {self._violation_counters[metric_name]}회 위반)"
                )
                return reason

        return None

    def execute_rollback(
        self,
        reason: RollbackReason,
        target_version: str | None = None,
        triggered_by: str = "auto",
    ) -> bool:
        """롤백을 실행한다.

        Args:
            reason: 롤백 사유
            target_version: 롤백할 버전 (None이면 이전 버전)
            triggered_by: "auto" | "manual"

        Returns:
            성공 여부
        """
        target = target_version or self.previous_version
        if not target:
            logger.error("롤백 대상 버전이 없다.")
            return False

        start_time = time.time()
        logger.info(
            f"롤백 실행: {self.current_version} → {target} "
            f"(사유: {reason.value}, 트리거: {triggered_by})"
        )

        try:
            # 1. 이전 버전 이미지로 디플로이먼트 업데이트
            registry = os.environ.get("DOCKER_REGISTRY", "registry")
            image = f"{registry}/{self.model_name}:{target}"
            self._kubectl(
                f"set image deployment/{self.model_name}-stable model={image}"
            )

            # 2. Canary 트래픽 제거 (Canary 배포 중이었다면)
            try:
                self._kubectl(
                    f"patch virtualservice {self.model_name} "
                    f"--type=merge -p '{{\"spec\":{{\"http\":[{{\"route\":["
                    f"{{\"destination\":{{\"host\":\"{self.model_name}-stable\"}},\"weight\":100}},"
                    f"{{\"destination\":{{\"host\":\"{self.model_name}-canary\"}},\"weight\":0}}"
                    f"]}}]}}}}'"
                )
            except RuntimeError:
                logger.warning("VirtualService 업데이트 실패 (Canary 미사용 환경일 수 있음)")

            # 3. 롤아웃 완료 대기
            self._kubectl(
                f"rollout status deployment/{self.model_name}-stable --timeout=120s"
            )

            duration = time.time() - start_time

            # 4. 롤백 이벤트 기록
            event = RollbackEvent(
                timestamp=datetime.now().isoformat(),
                reason=reason,
                from_version=self.current_version,
                to_version=target,
                metrics_snapshot=self._get_current_metrics_dict(),
                triggered_by=triggered_by,
                duration_seconds=duration,
                success=True,
            )
            self.rollback_history.append(event)

            # 5. 버전 상태 업데이트
            self.current_version = target
            self._reset_violation_counters()

            logger.info(f"롤백 완료: {duration:.1f}초 소요")
            return True

        except Exception as e:
            logger.error(f"롤백 실패: {e}")

            event = RollbackEvent(
                timestamp=datetime.now().isoformat(),
                reason=reason,
                from_version=self.current_version,
                to_version=target,
                metrics_snapshot={},
                triggered_by=triggered_by,
                duration_seconds=time.time() - start_time,
                success=False,
            )
            self.rollback_history.append(event)
            return False

    def _kubectl(self, args: str) -> str:
        """kubectl 명령을 실행한다."""
        cmd = f"kubectl -n {self.namespace} {args}"
        try:
            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            return result.stdout
        except subprocess.TimeoutExpired:
            raise RuntimeError("kubectl 타임아웃")

    def _get_current_metrics_dict(self) -> dict:
        """현재 메트릭 버퍼의 최근 값을 dict로 반환한다."""
        if not self.metric_buffer:
            return {}
        latest = self.metric_buffer[-1]
        return {
            "accuracy": latest.accuracy,
            "latency_p99": latest.latency_p99,
            "error_rate": latest.error_rate,
            "throughput": latest.throughput,
            "memory_percent": latest.memory_percent,
        }

    def _reset_violation_counters(self):
        """위반 카운터를 초기화한다."""
        for key in self._violation_counters:
            self._violation_counters[key] = 0

    def get_rollback_history(self) -> list[dict]:
        """롤백 이력을 반환한다."""
        return [
            {
                "timestamp": e.timestamp,
                "reason": e.reason.value,
                "from": e.from_version,
                "to": e.to_version,
                "triggered_by": e.triggered_by,
                "duration": f"{e.duration_seconds:.1f}s",
                "success": e.success,
            }
            for e in self.rollback_history
        ]
```

---

## 코드: 자동 롤백 모니터

```python
import time
import signal
import threading
import logging
from typing import Callable

import requests
import numpy as np

logger = logging.getLogger(__name__)


class AutoRollbackMonitor:
    """프로덕션 메트릭을 실시간 모니터링하고 자동 롤백을 수행하는 클래스.

    Prometheus에서 메트릭을 수집하고, 이상이 감지되면
    RollbackManager를 통해 자동 롤백을 실행한다.
    """

    def __init__(
        self,
        rollback_manager: "RollbackManager",
        prometheus_url: str,
        check_interval: int = 30,
        cooldown: int = 300,
    ):
        """
        Args:
            rollback_manager: RollbackManager 인스턴스
            prometheus_url: Prometheus 서버 URL
            check_interval: 메트릭 수집 간격 (초)
            cooldown: 롤백 후 재검사까지 대기 시간 (초)
        """
        self.manager = rollback_manager
        self.prometheus_url = prometheus_url.rstrip("/")
        self.check_interval = check_interval
        self.cooldown = cooldown

        self._running = False
        self._thread: threading.Thread | None = None
        self._last_rollback_time: float = 0

        # 콜백 훅
        self.on_rollback: Callable[[RollbackEvent], None] | None = None
        self.on_alert: Callable[[str, float], None] | None = None

    def start(self):
        """모니터링을 시작한다 (백그라운드 스레드)."""
        if self._running:
            logger.warning("모니터가 이미 실행 중이다.")
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"자동 롤백 모니터 시작: 간격={self.check_interval}초, "
            f"쿨다운={self.cooldown}초"
        )

    def stop(self):
        """모니터링을 중지한다."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("자동 롤백 모니터 중지")

    def _monitor_loop(self):
        """메트릭 수집 → 이상 탐지 → 롤백 루프."""
        while self._running:
            try:
                # 쿨다운 기간 중에는 롤백 스킵
                if time.time() - self._last_rollback_time < self.cooldown:
                    time.sleep(self.check_interval)
                    continue

                # 메트릭 수집
                snapshot = self._collect_metrics()
                if snapshot is None:
                    time.sleep(self.check_interval)
                    continue

                # 롤백 필요 여부 검사
                reason = self.manager.check_rollback_needed(snapshot)
                if reason is not None:
                    logger.warning(f"자동 롤백 트리거: {reason.value}")

                    # 알림 콜백
                    if self.on_alert:
                        self.on_alert(reason.value, snapshot.accuracy)

                    # 롤백 실행
                    success = self.manager.execute_rollback(
                        reason=reason,
                        triggered_by="auto",
                    )

                    if success:
                        self._last_rollback_time = time.time()
                        if self.on_rollback and self.manager.rollback_history:
                            self.on_rollback(self.manager.rollback_history[-1])

            except Exception as e:
                logger.error(f"모니터링 루프 에러: {e}")

            time.sleep(self.check_interval)

    def _collect_metrics(self) -> MetricSnapshot | None:
        """Prometheus에서 메트릭을 수집한다."""
        try:
            accuracy = self._query_prometheus(
                f'avg(model_accuracy{{model="{self.manager.model_name}"}})'
            )
            latency_p99 = self._query_prometheus(
                f'histogram_quantile(0.99, rate(model_latency_seconds_bucket'
                f'{{model="{self.manager.model_name}"}}[1m])) * 1000'
            )
            error_rate = self._query_prometheus(
                f'rate(model_errors_total{{model="{self.manager.model_name}"}}[1m])'
                f' / rate(model_requests_total{{model="{self.manager.model_name}"}}[1m])'
            )
            throughput = self._query_prometheus(
                f'rate(model_requests_total{{model="{self.manager.model_name}"}}[1m])'
            )
            memory = self._query_prometheus(
                f'container_memory_usage_bytes{{container="{self.manager.model_name}"}}'
                f' / container_spec_memory_limit_bytes'
                f'{{container="{self.manager.model_name}"}} * 100'
            )

            return MetricSnapshot(
                timestamp=time.time(),
                accuracy=accuracy,
                latency_p99=latency_p99,
                error_rate=error_rate,
                throughput=throughput,
                memory_percent=memory,
            )

        except Exception as e:
            logger.error(f"메트릭 수집 실패: {e}")
            return None

    def _query_prometheus(self, query: str) -> float:
        """Prometheus instant query를 실행한다."""
        resp = requests.get(
            f"{self.prometheus_url}/api/v1/query",
            params={"query": query},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data["status"] != "success" or not data["data"]["result"]:
            raise ValueError(f"Prometheus 쿼리 실패: {query}")

        value = float(data["data"]["result"][0]["value"][1])
        return value


# ── 사용 예시 ──────────────────────────────────────────────────
# manager = RollbackManager(
#     model_name="ocr-receipt",
#     thresholds=RollbackThresholds(
#         min_accuracy=0.92,
#         max_latency_p99_ms=500,
#         max_error_rate=0.05,
#         consecutive_failures=3,
#     ),
# )
# manager.current_version = "1.3.0"
# manager.previous_version = "1.2.0"
#
# monitor = AutoRollbackMonitor(
#     rollback_manager=manager,
#     prometheus_url="http://prometheus:9090",
#     check_interval=30,
#     cooldown=300,
# )
#
# # 알림 콜백 설정
# def on_alert(reason: str, accuracy: float):
#     send_slack(f"[ALERT] 모델 이상 감지: {reason}, accuracy={accuracy:.4f}")
#
# monitor.on_alert = on_alert
# monitor.start()
```

---

## 수동 롤백 절차

자동 롤백이 실패하거나, 자동 트리거가 설정되지 않은 메트릭에서 문제가 발견된 경우 수동 롤백을 수행한다.

### 수동 롤백 체크리스트

```
┌────────────────────────────────────────────────────────────┐
│                    수동 롤백 절차                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. [확인] 문제 상황 파악                                    │
│     ├── 어떤 메트릭이 이상한가?                              │
│     ├── 언제부터 발생했는가?                                 │
│     └── 영향 범위(Blast Radius)는?                          │
│                                                            │
│  2. [판단] 롤백 vs 핫픽스 결정                               │
│     ├── 즉시 롤백: 원인 불명, 심각한 성능 저하               │
│     └── 핫픽스: 원인 명확, 빠른 수정 가능                    │
│                                                            │
│  3. [실행] 롤백 수행                                        │
│     ├── 모델 아티팩트 교체 (이전 버전 로드)                  │
│     ├── 설정 파일 복원 (configs, feature flags)             │
│     └── 캐시 무효화 (필요 시)                               │
│                                                            │
│  4. [검증] 롤백 후 확인                                     │
│     ├── Health Check 통과 확인                              │
│     ├── 주요 메트릭 정상 복귀 확인                           │
│     └── Smoke Test (샘플 추론) 통과 확인                    │
│                                                            │
│  5. [기록] 포스트모템                                       │
│     ├── 장애 타임라인 기록                                  │
│     ├── 근본 원인 분석 (RCA)                                │
│     └── 재발 방지 액션 아이템 수립                           │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 수동 롤백 스크립트

```python
import argparse
import sys
import logging

logger = logging.getLogger(__name__)


def manual_rollback(
    model_name: str,
    target_version: str,
    namespace: str = "default",
    skip_validation: bool = False,
) -> bool:
    """수동 롤백을 실행한다.

    Args:
        model_name: 모델 이름
        target_version: 롤백 대상 버전
        namespace: Kubernetes 네임스페이스
        skip_validation: 검증 단계 스킵 여부

    Returns:
        성공 여부
    """
    manager = RollbackManager(model_name=model_name, namespace=namespace)
    manager.current_version = "unknown"
    manager.previous_version = target_version

    # 1. 롤백 실행
    logger.info(f"수동 롤백 시작: {model_name} → {target_version}")
    success = manager.execute_rollback(
        reason=RollbackReason.MANUAL,
        target_version=target_version,
        triggered_by="manual",
    )

    if not success:
        logger.error("롤백 실패")
        return False

    # 2. 검증
    if not skip_validation:
        logger.info("롤백 후 검증 시작...")
        time.sleep(30)  # 안정화 대기

        # Health Check
        from deployment_automation import HealthChecker
        checker = HealthChecker(f"http://{model_name}-stable.{namespace}:8080")
        results = checker.full_check()

        if not results.get("overall", False):
            logger.error(f"검증 실패: {results}")
            return False

        logger.info("검증 통과")

    # 3. 이력 출력
    for event in manager.get_rollback_history():
        logger.info(f"롤백 이력: {event}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="수동 모델 롤백")
    parser.add_argument("--model", required=True, help="모델 이름")
    parser.add_argument("--version", required=True, help="롤백 대상 버전")
    parser.add_argument("--namespace", default="default", help="K8s 네임스페이스")
    parser.add_argument("--skip-validation", action="store_true", help="검증 스킵")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    success = manual_rollback(
        model_name=args.model,
        target_version=args.version,
        namespace=args.namespace,
        skip_validation=args.skip_validation,
    )
    sys.exit(0 if success else 1)
```

---

## Feature Flag를 활용한 점진적 배포/롤백

Feature Flag는 코드를 다시 배포하지 않고도 모델 버전을 전환할 수 있게 해준다. 롤백 시 "배포"가 필요 없으므로 속도가 극적으로 빨라진다.

### Feature Flag 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                   Feature Flag 기반 모델 서빙                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [클라이언트] ──→ [API Gateway] ──→ [Model Router]           │
│                                        │                     │
│                                   [Feature Flag Service]     │
│                                        │                     │
│                          ┌─────────────┼─────────────┐       │
│                          │             │             │       │
│                     [Model v1.2]  [Model v1.3]  [Fallback]  │
│                     (Production)  (Canary)      (경량 모델)  │
│                                                              │
│  Feature Flag 설정:                                          │
│  ┌──────────────────────────────────────────────┐            │
│  │ model_v1.3_enabled: true                      │            │
│  │ model_v1.3_percentage: 10                     │            │
│  │ model_v1.3_user_segments: ["beta_testers"]    │            │
│  │ fallback_enabled: false                       │            │
│  └──────────────────────────────────────────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Feature Flag 서비스 구현

```python
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import redis

logger = logging.getLogger(__name__)


@dataclass
class FeatureFlag:
    """Feature Flag 정의."""
    name: str
    enabled: bool = False
    percentage: float = 0.0        # 0~100, 점진적 배포 비율
    user_segments: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class FeatureFlagService:
    """Redis 기반 Feature Flag 서비스.

    모델 버전 전환, 점진적 배포, 즉시 롤백에 사용한다.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self._prefix = "ff:"
        self._cache: dict[str, FeatureFlag] = {}
        self._cache_ttl = 10  # 로컬 캐시 TTL (초)
        self._cache_updated: dict[str, float] = {}

    def create_flag(self, flag: FeatureFlag):
        """Feature Flag를 생성한다."""
        key = f"{self._prefix}{flag.name}"
        data = {
            "enabled": str(flag.enabled),
            "percentage": str(flag.percentage),
            "user_segments": json.dumps(flag.user_segments),
            "metadata": json.dumps(flag.metadata),
            "created_at": str(flag.created_at),
            "updated_at": str(flag.updated_at),
        }
        self.redis.hset(key, mapping=data)
        self._cache[flag.name] = flag
        logger.info(f"Feature Flag 생성: {flag.name}")

    def get_flag(self, name: str) -> FeatureFlag | None:
        """Feature Flag를 조회한다 (로컬 캐시 우선)."""
        # 캐시 확인
        if name in self._cache:
            cache_time = self._cache_updated.get(name, 0)
            if time.time() - cache_time < self._cache_ttl:
                return self._cache[name]

        # Redis에서 조회
        key = f"{self._prefix}{name}"
        data = self.redis.hgetall(key)
        if not data:
            return None

        flag = FeatureFlag(
            name=name,
            enabled=data["enabled"] == "True",
            percentage=float(data["percentage"]),
            user_segments=json.loads(data.get("user_segments", "[]")),
            metadata=json.loads(data.get("metadata", "{}")),
        )

        self._cache[name] = flag
        self._cache_updated[name] = time.time()
        return flag

    def update_flag(self, name: str, **kwargs):
        """Feature Flag를 업데이트한다."""
        flag = self.get_flag(name)
        if not flag:
            raise ValueError(f"존재하지 않는 Flag: {name}")

        for key, value in kwargs.items():
            if hasattr(flag, key):
                setattr(flag, key, value)

        flag.updated_at = time.time()
        self.create_flag(flag)
        logger.info(f"Feature Flag 업데이트: {name}, {kwargs}")

    def is_enabled(self, name: str, user_id: str | None = None) -> bool:
        """Feature Flag가 활성화되어 있는지 확인한다.

        Args:
            name: Flag 이름
            user_id: 사용자 ID (percentage 기반 분배에 사용)

        Returns:
            활성화 여부
        """
        flag = self.get_flag(name)
        if not flag or not flag.enabled:
            return False

        # percentage 기반 분배 (일관성 있는 해싱)
        if flag.percentage < 100 and user_id:
            hash_val = int(hashlib.md5(
                f"{name}:{user_id}".encode()
            ).hexdigest(), 16)
            bucket = hash_val % 100
            if bucket >= flag.percentage:
                return False

        return True

    def is_user_in_segment(self, name: str, user_segment: str) -> bool:
        """사용자 세그먼트가 Flag의 대상 세그먼트에 포함되는지 확인한다."""
        flag = self.get_flag(name)
        if not flag:
            return False
        if not flag.user_segments:
            return True  # 세그먼트 미지정이면 전체 대상
        return user_segment in flag.user_segments

    def kill_switch(self, name: str):
        """Feature Flag를 즉시 비활성화한다 (긴급 롤백용)."""
        self.update_flag(name, enabled=False, percentage=0.0)
        logger.warning(f"Kill Switch 실행: {name}")

    def gradual_rollout(self, name: str, target_percentage: float, step: float = 10.0):
        """점진적으로 배포 비율을 올린다.

        Args:
            name: Flag 이름
            target_percentage: 목표 비율 (%)
            step: 각 단계별 증가량 (%)
        """
        flag = self.get_flag(name)
        if not flag:
            raise ValueError(f"존재하지 않는 Flag: {name}")

        current = flag.percentage
        while current < target_percentage:
            current = min(current + step, target_percentage)
            self.update_flag(name, percentage=current, enabled=True)
            logger.info(f"점진적 배포: {name} → {current}%")
            # 실제로는 여기서 메트릭 수집 + 검증을 해야 함
            time.sleep(1)  # 프로덕션에서는 적절한 대기 시간 설정


class ModelRouter:
    """Feature Flag 기반 모델 라우팅.

    요청마다 Feature Flag를 확인하여 적절한 모델 버전으로 라우팅한다.
    """

    def __init__(self, feature_flag_service: FeatureFlagService):
        self.ff = feature_flag_service
        self.models: dict[str, Any] = {}  # {version: model_instance}
        self.default_version: str = ""
        self.fallback_version: str = ""

    def register_model(self, version: str, model: Any, is_default: bool = False):
        """모델 버전을 등록한다."""
        self.models[version] = model
        if is_default:
            self.default_version = version
        logger.info(f"모델 등록: v{version}, default={is_default}")

    def set_fallback(self, version: str):
        """Fallback 모델을 설정한다."""
        if version not in self.models:
            raise ValueError(f"등록되지 않은 모델 버전: {version}")
        self.fallback_version = version

    def route(self, user_id: str, request_data: dict) -> tuple[str, Any]:
        """요청을 적절한 모델로 라우팅한다.

        Args:
            user_id: 사용자 ID
            request_data: 추론 요청 데이터

        Returns:
            (버전, 추론 결과)
        """
        # 새 모델 버전 Feature Flag 확인
        for version in sorted(self.models.keys(), reverse=True):
            flag_name = f"model_{version.replace('.', '_')}_enabled"
            if self.ff.is_enabled(flag_name, user_id=user_id):
                try:
                    result = self._predict(version, request_data)
                    return version, result
                except Exception as e:
                    logger.error(f"모델 v{version} 추론 실패: {e}")
                    # Fallback으로 이동
                    break

        # 기본 모델 시도
        try:
            result = self._predict(self.default_version, request_data)
            return self.default_version, result
        except Exception as e:
            logger.error(f"기본 모델 추론 실패: {e}")

        # 최종 Fallback
        if self.fallback_version:
            try:
                result = self._predict(self.fallback_version, request_data)
                logger.warning(f"Fallback 모델 사용: v{self.fallback_version}")
                return self.fallback_version, result
            except Exception as e:
                logger.critical(f"Fallback 모델도 실패: {e}")

        raise RuntimeError("모든 모델 버전에서 추론 실패")

    def _predict(self, version: str, request_data: dict) -> Any:
        """특정 버전의 모델로 추론을 수행한다."""
        model = self.models.get(version)
        if model is None:
            raise ValueError(f"모델 버전 {version}이 로드되지 않음")
        # 실제 추론 로직 (모델 인터페이스에 따라 다름)
        return model(request_data)


# ── Feature Flag 기반 롤백 사용 예시 ─────────────────────────
#
# ff_service = FeatureFlagService(redis_url="redis://localhost:6379")
#
# # 새 모델 v1.3 Feature Flag 생성 (10%부터 시작)
# ff_service.create_flag(FeatureFlag(
#     name="model_1_3_0_enabled",
#     enabled=True,
#     percentage=10.0,
#     user_segments=["beta_testers"],
# ))
#
# # 문제 발생 시 즉시 롤백 (코드 배포 불필요!)
# ff_service.kill_switch("model_1_3_0_enabled")
#
# # 안정화 후 점진적 재배포
# ff_service.gradual_rollout("model_1_3_0_enabled", target_percentage=100, step=10)
```

---

## Circuit Breaker 패턴

연속 실패가 누적되면 요청 자체를 차단하여 장애 전파를 방지한다.

```python
import time
import logging
from enum import Enum
from threading import Lock

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"       # 정상: 모든 요청 통과
    OPEN = "open"           # 차단: 모든 요청 거부, fallback 반환
    HALF_OPEN = "half_open" # 테스트: 제한된 요청만 통과


class CircuitBreaker:
    """Circuit Breaker 패턴 구현.

    상태 전이: CLOSED → (실패 누적) → OPEN → (타임아웃) → HALF_OPEN → (성공) → CLOSED
                                                            └── (실패) → OPEN
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
    ):
        """
        Args:
            failure_threshold: OPEN 전환까지의 연속 실패 횟수
            recovery_timeout: OPEN → HALF_OPEN 전환까지의 대기 시간 (초)
            half_open_max_calls: HALF_OPEN 상태에서 허용하는 최대 테스트 요청 수
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = Lock()

    @property
    def state(self) -> CircuitState:
        """현재 상태 (OPEN → HALF_OPEN 자동 전환 포함)."""
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and time.time() - self._last_failure_time >= self.recovery_timeout
            ):
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info("Circuit Breaker: OPEN → HALF_OPEN")
            return self._state

    def allow_request(self) -> bool:
        """요청을 허용할지 판단한다."""
        current_state = self.state

        if current_state == CircuitState.CLOSED:
            return True
        elif current_state == CircuitState.OPEN:
            return False
        else:  # HALF_OPEN
            with self._lock:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

    def record_success(self):
        """성공을 기록한다."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Circuit Breaker: HALF_OPEN → CLOSED (복구)")
            else:
                self._failure_count = 0

    def record_failure(self):
        """실패를 기록한다."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._success_count = 0
                logger.warning("Circuit Breaker: HALF_OPEN → OPEN (재실패)")

            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit Breaker: CLOSED → OPEN "
                    f"(연속 {self._failure_count}회 실패)"
                )

    def call(self, func, *args, fallback=None, **kwargs):
        """Circuit Breaker를 통해 함수를 호출한다.

        Args:
            func: 호출할 함수
            fallback: Circuit OPEN 시 반환할 fallback 값 또는 함수
            *args, **kwargs: func에 전달할 인자

        Returns:
            func 또는 fallback의 반환값
        """
        if not self.allow_request():
            logger.warning("Circuit OPEN: fallback 반환")
            if callable(fallback):
                return fallback(*args, **kwargs)
            return fallback

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            logger.error(f"Circuit Breaker 에러 기록: {e}")
            if callable(fallback):
                return fallback(*args, **kwargs)
            return fallback


# ── 사용 예시 ──────────────────────────────────────────────────
# cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
#
# def predict(data):
#     return model.predict(data)
#
# def fallback_predict(data):
#     return lightweight_model.predict(data)
#
# result = cb.call(predict, input_data, fallback=fallback_predict)
```

---

## 롤백 전략 비교 요약

| 방법 | 롤백 속도 | 배포 필요 | 리소스 비용 | 자동화 난이도 | 적합 시나리오 |
|------|----------|----------|------------|-------------|-------------|
| **kubectl rollback** | 30~120초 | O (재배포) | 없음 | 낮음 | 간단한 버전 교체 |
| **Feature Flag** | **< 1초** | **X** | Redis 운영 | 중간 | 즉시 롤백 필요 |
| **Blue-Green 전환** | 5~10초 | X (전환만) | 2x | 낮음 | 전체 교체 |
| **Canary 트래픽 조정** | 5~30초 | X (비율 조정) | 1.1x | 중간 | 점진적 롤백 |
| **Circuit Breaker** | **즉시** | **X** | 없음 | 낮음 | 장애 격리 |

---

## 용어 체크리스트

학습을 마친 뒤 아래 항목을 스스로 점검해봐라.

- [ ] **Rollback Trigger**로 적합한 메트릭 5가지와 각각의 임계값 기준을 설명할 수 있는가?
- [ ] **Z-score 기반 이상 탐지**와 **EWMA 제어 한계**의 차이를 이해했는가?
- [ ] **Version Pinning**이 왜 프로덕션 안정성에 중요한지 설명할 수 있는가?
- [ ] **Feature Flag**로 롤백하면 왜 일반 배포 롤백보다 빠른지 이해했는가?
- [ ] **Circuit Breaker**의 CLOSED → OPEN → HALF_OPEN 상태 전이를 설명할 수 있는가?
- [ ] **Automated Rollback**에서 연속 위반 조건(consecutive_failures)이 왜 필요한지 아는가?
- [ ] **Graceful Degradation**과 단순 롤백의 차이를 설명할 수 있는가?
- [ ] **Blast Radius**를 줄이기 위한 배포 전략을 선택할 수 있는가?
- [ ] 수동 롤백 시 **포스트모템(Post-Mortem)** 작성의 중요성을 이해했는가?
- [ ] Feature Flag의 **일관성 있는 해싱**(consistent hashing)이 왜 percentage 배포에 필요한지 설명할 수 있는가?
