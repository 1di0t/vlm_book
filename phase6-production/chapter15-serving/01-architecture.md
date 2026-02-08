---
---

# 15.1 서빙 아키텍처 기본 구조

프로덕션 환경에서 OCR VLM 모델을 서빙하려면 단순히 모델을 로드해서 추론하는 것 이상의 아키텍처가 필요하다. 수천 명의 동시 사용자, 가변적인 트래픽, 장애 복구까지 고려한 시스템을 설계해야 한다.

---

> **핵심 용어 박스**
>
> | 용어 | 정의 |
> |------|------|
> | **API Gateway** | 클라이언트 요청의 단일 진입점. 인증, 속도 제한, 라우팅을 담당 |
> | **Load Balancer** | 다수의 백엔드 서버에 요청을 분산하는 컴포넌트 |
> | **Model Server** | 실제 모델 추론을 수행하는 워커 프로세스/컨테이너 |
> | **Request Queue** | 요청을 비동기로 버퍼링하여 백프레셔를 제어하는 큐 |
> | **SLA (Service Level Agreement)** | 서비스 품질 보장 계약. 응답 시간, 가용성 등을 수치로 정의 |
> | **Horizontal Scaling** | 서버 인스턴스 수를 늘려 처리량을 증가시키는 전략 |
> | **Vertical Scaling** | 단일 서버의 자원(GPU, RAM)을 증가시키는 전략 |

---

## 15.1.1 전체 시스템 아키텍처

### 요청 흐름

의료 문서 OCR 서빙 시스템의 전체 요청 흐름은 다음과 같다:

```
┌──────────┐     ┌─────────────┐     ┌───────────────┐     ┌──────────────────┐
│  Client  │────▶│ API Gateway │────▶│ Load Balancer │────▶│ Model Server Pool│
│ (Web/App)│     │  (Kong/     │     │  (Nginx/      │     │  (vLLM Workers)  │
│          │◀────│   Envoy)    │◀────│   HAProxy)    │◀────│                  │
└──────────┘     └─────────────┘     └───────────────┘     └──────────────────┘
                       │                                           │
                       ▼                                           ▼
                 ┌───────────┐                              ┌────────────┐
                 │ Auth/Rate │                              │  GPU Nodes │
                 │  Limiter  │                              │  (A100×N)  │
                 └───────────┘                              └────────────┘
                       │                                           │
                       ▼                                           ▼
                 ┌───────────┐                              ┌────────────┐
                 │  Request  │                              │  Response  │
                 │   Queue   │                              │   Cache    │
                 │ (Redis/   │                              │  (Redis)   │
                 │  RabbitMQ)│                              │            │
                 └───────────┘                              └────────────┘
```

### 계층별 책임

| 계층 | 구성 요소 | 핵심 책임 |
|------|-----------|-----------|
| Edge | API Gateway | TLS 종료, 인증, Rate Limiting, 요청 검증 |
| Distribution | Load Balancer | 헬스 체크 기반 트래픽 분산, 세션 어피니티 |
| Buffering | Request Queue | 비동기 요청 버퍼링, 우선순위 큐, 백프레셔 |
| Compute | Model Server | GPU 기반 모델 추론, 배치 처리 |
| Storage | Response Cache | 중복 요청 캐싱, 결과 저장 |

---

## 15.1.2 각 컴포넌트 상세

### API Gateway

API Gateway는 모든 외부 요청의 단일 진입점이다. 의료 문서 처리에서는 특히 보안이 중요하다.

```python
"""
API Gateway 핵심 기능 구현 예시
실제 프로덕션에서는 Kong, Envoy, AWS API Gateway 등을 사용하지만,
핵심 로직을 이해하기 위한 구현이다.
"""

import time
import hashlib
import hmac
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional


@dataclass
class RateLimitConfig:
    """Rate Limiting 설정"""
    requests_per_minute: int = 60
    burst_size: int = 10
    window_seconds: int = 60


class TokenBucketRateLimiter:
    """
    Token Bucket 알고리즘 기반 Rate Limiter.

    수학적 원리:
    - 버킷 용량: B (burst_size)
    - 토큰 충전 속도: r = requests_per_minute / 60 (tokens/sec)
    - 시간 t에서의 가용 토큰: min(B, tokens + r * elapsed_time)
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.buckets: dict[str, dict] = {}
        self.refill_rate = config.requests_per_minute / config.window_seconds

    def allow_request(self, client_id: str) -> bool:
        now = time.time()

        if client_id not in self.buckets:
            self.buckets[client_id] = {
                "tokens": self.config.burst_size,
                "last_refill": now
            }

        bucket = self.buckets[client_id]
        elapsed = now - bucket["last_refill"]

        # 토큰 충전: tokens = min(B, tokens + r * Δt)
        bucket["tokens"] = min(
            self.config.burst_size,
            bucket["tokens"] + self.refill_rate * elapsed
        )
        bucket["last_refill"] = now

        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True
        return False


class RequestValidator:
    """요청 검증기: 이미지 크기, 포맷, 인증 토큰 확인"""

    MAX_IMAGE_SIZE_MB = 20  # 의료 문서는 고해상도
    ALLOWED_FORMATS = {"image/png", "image/jpeg", "image/tiff", "application/pdf"}

    @staticmethod
    def validate_image(content_type: str, content_length: int) -> tuple[bool, str]:
        if content_type not in RequestValidator.ALLOWED_FORMATS:
            return False, f"지원하지 않는 포맷: {content_type}"

        max_bytes = RequestValidator.MAX_IMAGE_SIZE_MB * 1024 * 1024
        if content_length > max_bytes:
            return False, f"파일 크기 초과: {content_length / 1024 / 1024:.1f}MB > {RequestValidator.MAX_IMAGE_SIZE_MB}MB"

        return True, "OK"

    @staticmethod
    def verify_api_key(api_key: str, secret: str, timestamp: str, signature: str) -> bool:
        """HMAC 기반 API 키 검증"""
        expected = hmac.new(
            secret.encode(),
            f"{api_key}:{timestamp}".encode(),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)
```

### Load Balancer

요청을 여러 Model Server에 분배한다. GPU 서빙에서는 일반 웹 서버와 다른 분배 전략이 필요하다.

```python
"""
GPU-Aware Load Balancer 전략
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelServerNode:
    """모델 서버 노드 정보"""
    node_id: str
    host: str
    port: int
    gpu_count: int
    gpu_memory_total_gb: float
    gpu_memory_used_gb: float = 0.0
    active_requests: int = 0
    max_concurrent: int = 32
    is_healthy: bool = True

    @property
    def gpu_utilization(self) -> float:
        if self.gpu_memory_total_gb == 0:
            return 1.0
        return self.gpu_memory_used_gb / self.gpu_memory_total_gb

    @property
    def load_score(self) -> float:
        """
        종합 부하 점수 (낮을수록 여유).
        가중 합산:
          score = α × (active / max_concurrent) + β × gpu_utilization
        """
        alpha, beta = 0.4, 0.6
        request_ratio = self.active_requests / max(self.max_concurrent, 1)
        return alpha * request_ratio + beta * self.gpu_utilization


class LoadBalancerStrategy(ABC):
    """로드 밸런싱 전략 추상 클래스"""

    @abstractmethod
    def select_node(self, nodes: list[ModelServerNode]) -> Optional[ModelServerNode]:
        pass


class RoundRobinStrategy(LoadBalancerStrategy):
    """라운드 로빈: 단순하지만 GPU 부하를 고려하지 않음"""

    def __init__(self):
        self._index = 0

    def select_node(self, nodes: list[ModelServerNode]) -> Optional[ModelServerNode]:
        healthy = [n for n in nodes if n.is_healthy]
        if not healthy:
            return None
        node = healthy[self._index % len(healthy)]
        self._index += 1
        return node


class LeastLoadStrategy(LoadBalancerStrategy):
    """
    최소 부하 전략: GPU 메모리 + 활성 요청 수를 기반으로 선택.
    의료 문서 OCR처럼 요청별 처리 시간 편차가 큰 경우 이 전략이 유리하다.
    """

    def select_node(self, nodes: list[ModelServerNode]) -> Optional[ModelServerNode]:
        healthy = [n for n in nodes if n.is_healthy and n.active_requests < n.max_concurrent]
        if not healthy:
            return None
        return min(healthy, key=lambda n: n.load_score)


class WeightedRandomStrategy(LoadBalancerStrategy):
    """
    가중 랜덤: 노드 성능(GPU 수)에 비례하여 확률적 분배.
    이기종 GPU 클러스터에서 유용하다. (A100 vs V100 혼합)
    """

    def select_node(self, nodes: list[ModelServerNode]) -> Optional[ModelServerNode]:
        healthy = [n for n in nodes if n.is_healthy]
        if not healthy:
            return None
        weights = [n.gpu_count * (1 - n.gpu_utilization) for n in healthy]
        total = sum(weights)
        if total <= 0:
            return random.choice(healthy)
        return random.choices(healthy, weights=weights, k=1)[0]
```

---

## 15.1.3 스케일링 전략

### 수학적 원리: 용량 계획

서빙 시스템의 필요 인스턴스 수를 계산하는 공식:

$$
N_{\text{instances}} = \left\lceil \frac{\lambda \cdot \bar{t}}{C} \right\rceil
$$

여기서:
- $\lambda$: 초당 요청 수 (requests/sec)
- $\bar{t}$: 평균 처리 시간 (sec/request)
- $C$: 인스턴스당 동시 처리 가능 수

**예시: 의료 문서 OCR**

- 피크 시간 요청: $\lambda = 50$ req/s
- 평균 처리 시간: $\bar{t} = 2.0$s (고해상도 문서)
- 인스턴스당 동시 처리: $C = 8$ (배치 크기)

$$
N_{\text{instances}} = \left\lceil \frac{50 \times 2.0}{8} \right\rceil = \left\lceil 12.5 \right\rceil = 13
$$

### Horizontal vs Vertical Scaling

$$
\text{Throughput}_{\text{horizontal}} = N \times T_{\text{single}}
$$

$$
\text{Throughput}_{\text{vertical}} = T_{\text{single}} \times \frac{R_{\text{new}}}{R_{\text{old}}}
$$

여기서 $R$은 GPU 리소스(VRAM, FLOPS 등)를 의미한다.

| 전략 | 장점 | 단점 | 적합한 경우 |
|------|------|------|-------------|
| Horizontal | 무한 확장, 장애 격리 | 복잡한 관리, 네트워크 비용 | 트래픽 변동 큼 |
| Vertical | 단순한 구조, 낮은 지연 | 확장 한계, 단일 장애점 | 대형 모델 서빙 |
| Hybrid | 두 장점 결합 | 설계 복잡 | 프로덕션 환경 |

### Auto-Scaling 정책

```python
"""
GPU 기반 Auto-Scaling 정책
"""

from dataclasses import dataclass
from enum import Enum


class ScaleAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_CHANGE = "no_change"


@dataclass
class AutoScalePolicy:
    """
    GPU 서빙 전용 Auto-Scaling 정책.

    스케일 아웃 조건:
      avg_gpu_util > scale_up_threshold AND active_requests > queue_threshold

    스케일 인 조건:
      avg_gpu_util < scale_down_threshold FOR cooldown_period
    """
    scale_up_threshold: float = 0.75       # GPU 사용률 75% 이상이면 확장
    scale_down_threshold: float = 0.30     # GPU 사용률 30% 이하면 축소
    queue_threshold: int = 50              # 대기 큐 50건 이상이면 확장
    min_instances: int = 2                 # 최소 인스턴스 (고가용성)
    max_instances: int = 20                # 최대 인스턴스 (비용 제한)
    cooldown_seconds: int = 300            # 스케일링 쿨다운 (5분)
    scale_up_step: int = 2                 # 한 번에 2대씩 추가
    scale_down_step: int = 1               # 한 번에 1대씩 제거

    def evaluate(
        self,
        current_instances: int,
        avg_gpu_utilization: float,
        queue_length: int,
        seconds_since_last_scale: float
    ) -> ScaleAction:
        if seconds_since_last_scale < self.cooldown_seconds:
            return ScaleAction.NO_CHANGE

        # 스케일 아웃
        if (avg_gpu_utilization > self.scale_up_threshold
                or queue_length > self.queue_threshold):
            if current_instances < self.max_instances:
                return ScaleAction.SCALE_UP

        # 스케일 인
        if (avg_gpu_utilization < self.scale_down_threshold
                and queue_length == 0):
            if current_instances > self.min_instances:
                return ScaleAction.SCALE_DOWN

        return ScaleAction.NO_CHANGE

    def compute_target(self, action: ScaleAction, current: int) -> int:
        if action == ScaleAction.SCALE_UP:
            return min(current + self.scale_up_step, self.max_instances)
        elif action == ScaleAction.SCALE_DOWN:
            return max(current - self.scale_down_step, self.min_instances)
        return current
```

---

## 15.1.4 의료 문서 OCR 서빙 특수 요건

의료 문서 OCR은 일반 LLM 서빙과 다른 요구사항이 있다:

### 정확도 SLA

$$
\text{SLA}_{\text{accuracy}} = P(\text{CER} < \epsilon) \geq 1 - \delta
$$

- $\epsilon$: 허용 오차율 (의료 문서: $\epsilon = 0.01$, 1%)
- $\delta$: 위반 허용률 (일반적으로 $\delta = 0.001$, 0.1%)

### 응답 시간 SLA

| 문서 유형 | p50 목표 | p95 목표 | p99 목표 |
|-----------|---------|---------|---------|
| 처방전 (1-2페이지) | 1.0s | 2.0s | 3.0s |
| 진단서 (3-5페이지) | 2.5s | 5.0s | 8.0s |
| 병원 차트 (10+페이지) | 5.0s | 12.0s | 20.0s |

### 아키텍처 결정 포인트

```python
"""
의료 문서 OCR 서빙 설정
"""

@dataclass
class MedicalOCRServingConfig:
    """의료 문서 OCR 전용 서빙 설정"""

    # 모델 설정
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    tensor_parallel_size: int = 1
    max_model_len: int = 8192

    # 이미지 처리
    max_image_resolution: tuple[int, int] = (2048, 2048)
    supported_formats: tuple[str, ...] = ("png", "jpeg", "tiff", "pdf")
    max_pages_per_request: int = 20

    # 성능 설정
    max_concurrent_requests: int = 32
    request_timeout_seconds: int = 30
    gpu_memory_utilization: float = 0.90

    # 정확도 보장
    enable_confidence_scoring: bool = True
    min_confidence_threshold: float = 0.85
    enable_human_review_routing: bool = True  # 낮은 확신도 → 수동 검토

    # 보안 (HIPAA 준수)
    enable_request_logging: bool = True
    log_contains_phi: bool = False           # PHI(개인건강정보) 로그 제외
    enable_encryption_at_rest: bool = True
    audit_trail_enabled: bool = True
```

---

## 15.1.5 Kubernetes 배포 아키텍처

```yaml
# k8s-serving-architecture.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ocr-model-server
  namespace: medical-ocr
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0  # 무중단 배포
  selector:
    matchLabels:
      app: ocr-model-server
  template:
    metadata:
      labels:
        app: ocr-model-server
    spec:
      containers:
      - name: vllm-server
        image: medical-ocr/vllm-server:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "24Gi"
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ocr-model-hpa
  namespace: medical-ocr
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ocr-model-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "75"
  - type: Pods
    pods:
      metric:
        name: request_queue_length
      target:
        type: AverageValue
        averageValue: "20"
```

---

## 용어 체크리스트

학습 후 아래 질문에 답할 수 있어야 한다:

- [ ] API Gateway의 3가지 핵심 역할은?
- [ ] Token Bucket Rate Limiting의 두 파라미터(용량, 충전 속도)가 의미하는 바는?
- [ ] Least Load 전략이 Round Robin보다 GPU 서빙에 유리한 이유는?
- [ ] 필요 인스턴스 수 공식 $N = \lceil \lambda \bar{t} / C \rceil$에서 각 변수의 의미는?
- [ ] Horizontal Scaling과 Vertical Scaling의 트레이드오프는?
- [ ] 의료 문서 OCR에서 일반 LLM 서빙과 다른 SLA 요구사항 3가지는?
- [ ] Kubernetes HPA에서 GPU 사용률 기반 스케일링이 필요한 이유는?
- [ ] HIPAA 준수를 위해 서빙 아키텍처에서 반드시 고려해야 할 보안 요소는?
