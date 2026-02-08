# 17.2 배포 자동화

모델 학습이 끝나면 프로덕션에 배포해야 한다. 수동 배포는 실수를 유발하고, 롤백이 느리며, 확장이 불가능하다. CI/CD 파이프라인을 통해 모델 배포를 자동화하고, 배포 전략으로 리스크를 관리하는 방법을 다룬다.

> **핵심 용어**
>
> | 용어 | 정의 |
> |------|------|
> | **Blue-Green Deployment** | 동일한 두 환경(Blue/Green)을 운영하며, 새 버전을 비활성 환경에 배포 후 **트래픽을 한 번에 전환**하는 전략. 롤백이 즉시 가능하지만 리소스가 2배 필요하다. |
> | **Canary Deployment** | 새 버전에 전체 트래픽의 **일부(1~10%)**만 먼저 라우팅하고, 문제 없으면 점진적으로 비율을 높이는 전략. 리스크 최소화에 적합하다. |
> | **Rolling Update** | 기존 인스턴스를 하나씩 순차적으로 새 버전으로 교체하는 전략. 리소스 효율적이지만 **두 버전이 공존**하는 기간이 존재한다. |
> | **Shadow Mode** | 프로덕션 트래픽을 새 모델에도 **복제하여 전달**하되, 실제 응답은 기존 모델이 담당한다. 새 모델의 출력은 로깅만 하여 성능을 검증한다. |
> | **A/B Testing** | 사용자를 랜덤으로 그룹 분배하여 각 그룹에 다른 모델 버전을 서빙하고, **통계적으로 유의미한 차이**를 검증하는 방법. |
> | **CI/CD Pipeline** | 코드 변경 → 빌드 → 테스트 → 배포를 **자동화**하는 파이프라인. ML에서는 모델 학습/평가/등록/서빙까지 포함한다. |
> | **Health Check** | 배포된 서비스의 정상 동작 여부를 주기적으로 확인하는 엔드포인트. Liveness, Readiness, Startup 프로브로 나뉜다. |
> | **Feature Store** | ML 모델에 필요한 피처를 중앙에서 **관리/서빙**하는 시스템. 배포 시 학습-서빙 스큐를 방지한다. |

---

## 수학적 원리: Canary 통계적 유의성 검증

Canary 배포에서 새 모델(B)이 기존 모델(A)보다 정말 나은지 판단하려면 **통계적 가설 검정**이 필요하다. "그냥 좀 좋아 보이는데?" 수준으로는 안 된다.

### A/B Test Z-검정

두 모델의 평균 성능을 비교하는 **독립표본 Z-검정**이다.

**귀무가설** $H_0$: 두 모델의 성능 차이가 없다 ($\mu_A = \mu_B$)

**대립가설** $H_1$: 두 모델의 성능 차이가 있다 ($\mu_A \neq \mu_B$)

**검정 통계량:**

$$
z = \frac{\bar{x}_A - \bar{x}_B}{\sqrt{\dfrac{s_A^2}{n_A} + \dfrac{s_B^2}{n_B}}}
$$

여기서:
- $\bar{x}_A, \bar{x}_B$: 각 모델의 표본 평균 (예: accuracy)
- $s_A^2, s_B^2$: 각 모델의 표본 분산
- $n_A, n_B$: 각 그룹의 샘플 수

**판정:** $|z| > z_{\alpha/2}$이면 귀무가설을 기각한다. 유의수준 $\alpha = 0.05$일 때 $z_{0.025} = 1.96$이다.

### p-value 계산

$$
p\text{-value} = 2 \times (1 - \Phi(|z|))
$$

$\Phi$는 표준정규분포의 CDF다. $p < \alpha$이면 통계적으로 유의미한 차이가 있다고 판단한다.

### 필요 샘플 수 산출

실험 시작 전에 **필요한 최소 샘플 수**를 계산해야 한다. 그래야 "얼마나 트래픽을 Canary에 보내야 하는가"를 결정할 수 있다.

$$
n = \frac{(z_{\alpha/2} + z_\beta)^2 \times (\sigma_A^2 + \sigma_B^2)}{(\mu_A - \mu_B)^2}
$$

여기서:
- $z_{\alpha/2}$: 유의수준에 대응하는 Z-값 (양측 검정, $\alpha=0.05$이면 1.96)
- $z_\beta$: 검정력(power)에 대응하는 Z-값 (power=0.8이면 $z_\beta = 0.84$)
- $\sigma_A^2, \sigma_B^2$: 각 모델의 모분산 (사전 추정)
- $\mu_A - \mu_B$: 탐지하고 싶은 **최소 효과 크기** (Minimum Detectable Effect)

**예시:** 기존 모델 accuracy $\mu_A = 0.95$, $\sigma_A = 0.02$, 새 모델에서 $\mu_B = 0.96$을 기대하며, $\sigma_B = 0.02$라 가정하면:

$$
n = \frac{(1.96 + 0.84)^2 \times (0.02^2 + 0.02^2)}{(0.95 - 0.96)^2} = \frac{7.84 \times 0.0008}{0.0001} = \frac{0.006272}{0.0001} = 62.72
$$

각 그룹에 최소 **63개** 샘플이 필요하다. 실제로는 분산 추정 오차를 감안해 1.5~2배로 잡는다.

### 이항 비율 검정 (Proportion Test)

accuracy처럼 성공/실패 이진 변수의 경우 **비율 검정**이 더 적합하다.

$$
z = \frac{\hat{p}_A - \hat{p}_B}{\sqrt{\hat{p}(1 - \hat{p})\left(\dfrac{1}{n_A} + \dfrac{1}{n_B}\right)}}
$$

여기서 $\hat{p} = \dfrac{n_A \hat{p}_A + n_B \hat{p}_B}{n_A + n_B}$는 합동 비율(pooled proportion)이다.

---

## 배포 전략별 비교

| 전략 | 리스크 수준 | 롤백 속도 | 리소스 비용 | 사용자 영향 | 적합 시나리오 |
|------|------------|----------|------------|------------|-------------|
| **Blue-Green** | 낮음 | **즉시** (DNS/LB 전환) | **2x** (이중 환경) | 전체 사용자 일괄 전환 | 모델 아키텍처 변경, 대규모 업데이트 |
| **Canary** | **매우 낮음** | 빠름 (트래픽 비율 조정) | 1.1~1.5x | 소수만 영향 | 성능 불확실한 신규 모델 |
| **Rolling** | 중간 | 보통 (순차 롤백) | **1x** (동일 리소스) | 점진적 전환 | 마이너 업데이트, 패치 |
| **Shadow** | **없음** | 해당 없음 | 1.5~2x (복제 처리) | **없음** | 신규 모델 사전 검증 |

### 배포 전략 선택 플로우

```
새 모델 배포 결정
    │
    ├── 모델 아키텍처 변경? ──Yes──→ Shadow Mode로 사전 검증
    │                                    │
    │                                    └── 검증 통과 → Blue-Green 배포
    │
    └── 성능 소폭 개선? ──Yes──→ Canary 배포 (5% → 25% → 50% → 100%)
    │
    └── 버그 수정 / 핫픽스? ──Yes──→ Rolling Update
```

---

## 코드: 통계적 검정 구현

```python
import math
from dataclasses import dataclass
from scipy import stats
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """A/B 테스트 결과."""
    z_statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    effect_size: float
    sample_size_a: int
    sample_size_b: int
    mean_a: float
    mean_b: float
    winner: str  # "A", "B", "no_difference"


class CanaryStatisticalTester:
    """Canary 배포 시 통계적 유의성 검증을 수행하는 클래스.

    주요 기능:
    - Z-검정 기반 A/B 테스트
    - 필요 샘플 수 산출
    - 이항 비율 검정
    - Sequential Testing (조기 종료)
    """

    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        """
        Args:
            alpha: 유의수준 (Type I error rate). 기본 5%.
            power: 검정력 (1 - Type II error rate). 기본 80%.
        """
        if not (0 < alpha < 1) or not (0 < power < 1):
            raise ValueError("alpha와 power는 (0, 1) 범위여야 한다.")
        self.alpha = alpha
        self.power = power
        self.z_alpha = stats.norm.ppf(1 - alpha / 2)  # 양측 검정
        self.z_beta = stats.norm.ppf(power)

    def z_test(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
    ) -> ABTestResult:
        """독립표본 Z-검정을 수행한다.

        Args:
            data_a: 모델 A(기존)의 성능 데이터
            data_b: 모델 B(Canary)의 성능 데이터

        Returns:
            ABTestResult: 검정 결과
        """
        n_a, n_b = len(data_a), len(data_b)
        mean_a, mean_b = np.mean(data_a), np.mean(data_b)
        var_a, var_b = np.var(data_a, ddof=1), np.var(data_b, ddof=1)

        # Z-통계량 계산
        se = math.sqrt(var_a / n_a + var_b / n_b)
        if se == 0:
            logger.warning("표준오차가 0. 데이터에 분산이 없다.")
            z = 0.0
        else:
            z = (mean_a - mean_b) / se

        # p-value (양측 검정)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        is_significant = p_value < self.alpha
        effect_size = mean_b - mean_a

        if not is_significant:
            winner = "no_difference"
        elif mean_b > mean_a:
            winner = "B"
        else:
            winner = "A"

        result = ABTestResult(
            z_statistic=z,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=1 - self.alpha,
            effect_size=effect_size,
            sample_size_a=n_a,
            sample_size_b=n_b,
            mean_a=float(mean_a),
            mean_b=float(mean_b),
            winner=winner,
        )
        logger.info(
            f"Z-test: z={z:.4f}, p={p_value:.4f}, "
            f"significant={is_significant}, winner={winner}"
        )
        return result

    def proportion_test(
        self,
        successes_a: int,
        total_a: int,
        successes_b: int,
        total_b: int,
    ) -> ABTestResult:
        """이항 비율 검정을 수행한다.

        accuracy 같은 성공/실패 메트릭에 적합하다.
        """
        p_a = successes_a / total_a
        p_b = successes_b / total_b

        # 합동 비율
        p_pooled = (successes_a + successes_b) / (total_a + total_b)

        se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / total_a + 1 / total_b))
        if se == 0:
            z = 0.0
        else:
            z = (p_a - p_b) / se

        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        is_significant = p_value < self.alpha

        if not is_significant:
            winner = "no_difference"
        elif p_b > p_a:
            winner = "B"
        else:
            winner = "A"

        return ABTestResult(
            z_statistic=z,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=1 - self.alpha,
            effect_size=p_b - p_a,
            sample_size_a=total_a,
            sample_size_b=total_b,
            mean_a=p_a,
            mean_b=p_b,
            winner=winner,
        )

    def required_sample_size(
        self,
        sigma_a: float,
        sigma_b: float,
        min_effect: float,
    ) -> int:
        """필요 최소 샘플 수를 계산한다.

        Args:
            sigma_a: 모델 A의 성능 표준편차 (사전 추정)
            sigma_b: 모델 B의 성능 표준편차 (사전 추정)
            min_effect: 탐지하고 싶은 최소 효과 크기 (|μ_A - μ_B|)

        Returns:
            각 그룹에 필요한 최소 샘플 수
        """
        if min_effect <= 0:
            raise ValueError("min_effect는 양수여야 한다.")

        numerator = (self.z_alpha + self.z_beta) ** 2 * (sigma_a ** 2 + sigma_b ** 2)
        denominator = min_effect ** 2
        n = math.ceil(numerator / denominator)

        logger.info(
            f"필요 샘플 수: {n} (각 그룹), "
            f"alpha={self.alpha}, power={self.power}, "
            f"min_effect={min_effect}"
        )
        return n

    def sequential_test(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        max_samples: int,
        check_interval: int = 100,
    ) -> tuple[ABTestResult | None, bool]:
        """Sequential Testing: 데이터가 쌓이면서 조기 종료 판단.

        전체 샘플을 다 모을 때까지 기다리지 않고, 중간에 유의성이
        확보되면 조기 종료할 수 있다. O'Brien-Fleming 경계를 적용한다.

        Args:
            data_a: 모델 A 데이터 (누적)
            data_b: 모델 B 데이터 (누적)
            max_samples: 최대 샘플 수
            check_interval: 검정 수행 간격

        Returns:
            (ABTestResult or None, 조기종료 여부)
        """
        n = min(len(data_a), len(data_b))
        num_checks = max_samples // check_interval

        for i in range(1, num_checks + 1):
            current_n = min(i * check_interval, n)
            if current_n > n:
                break

            # O'Brien-Fleming 경계: 검정 횟수에 따라 유의수준 조정
            adjusted_alpha = self.alpha / math.sqrt(num_checks / i)

            result = self.z_test(data_a[:current_n], data_b[:current_n])

            if result.p_value < adjusted_alpha:
                logger.info(f"Sequential test 조기 종료: n={current_n}, p={result.p_value:.4f}")
                return result, True

        # 최종 검정
        if n > 0:
            return self.z_test(data_a[:n], data_b[:n]), False
        return None, False


# ── 사용 예시 ──────────────────────────────────────────────────
# tester = CanaryStatisticalTester(alpha=0.05, power=0.8)
#
# # 필요 샘플 수 계산
# n = tester.required_sample_size(sigma_a=0.02, sigma_b=0.02, min_effect=0.01)
# print(f"각 그룹 최소 {n}개 필요")
#
# # Canary 트래픽에서 수집된 데이터로 검정
# data_a = np.random.normal(0.95, 0.02, 200)  # 기존 모델
# data_b = np.random.normal(0.96, 0.02, 200)  # Canary 모델
# result = tester.z_test(data_a, data_b)
# print(f"유의미한 차이: {result.is_significant}, 승자: {result.winner}")
```

---

## GitHub Actions CI/CD 파이프라인

### 전체 파이프라인 구조

```
Push/PR → Lint & Test → Model Build → Model Evaluate → Registry → Deploy (Canary) → Validate → Full Deploy
```

### GitHub Actions Workflow YAML

```yaml
# .github/workflows/ml-cicd.yaml
name: ML Model CI/CD Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'configs/**'
      - 'data/**/*.dvc'
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: '3.11'
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
  AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
  AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
  MODEL_NAME: 'ocr-receipt'
  DOCKER_REGISTRY: ${{ secrets.DOCKER_REGISTRY }}

jobs:
  # ── Stage 1: 코드 품질 검사 ──────────────────────────────────
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install ruff pytest pytest-cov mypy

      - name: Lint (ruff)
        run: ruff check src/ --output-format=github

      - name: Type check (mypy)
        run: mypy src/ --ignore-missing-imports

      - name: Unit tests
        run: pytest tests/unit/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml

  # ── Stage 2: 모델 빌드 및 평가 ──────────────────────────────
  model-build:
    needs: lint-and-test
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Pull DVC data
        run: |
          pip install dvc[s3]
          dvc pull

      - name: Train model
        run: |
          python src/train.py \
            --config configs/train.yaml \
            --output models/checkpoints/

      - name: Evaluate model
        id: evaluate
        run: |
          python src/evaluate.py \
            --model models/checkpoints/best.pt \
            --data data/test/ \
            --output metrics/eval.json

          # 메트릭을 GitHub output으로 추출
          echo "accuracy=$(python -c "import json; print(json.load(open('metrics/eval.json'))['accuracy'])")" >> $GITHUB_OUTPUT
          echo "cer=$(python -c "import json; print(json.load(open('metrics/eval.json'))['cer'])")" >> $GITHUB_OUTPUT

      - name: Quality gate check
        run: |
          python scripts/quality_gate.py \
            --metrics metrics/eval.json \
            --min-accuracy 0.95 \
            --max-cer 0.05

      - name: Register model to MLflow
        if: github.ref == 'refs/heads/main'
        run: |
          python scripts/register_model.py \
            --model models/checkpoints/best.pt \
            --name ${{ env.MODEL_NAME }} \
            --metrics metrics/eval.json

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: model-artifacts
          path: |
            models/checkpoints/best.pt
            metrics/eval.json

  # ── Stage 3: Docker 이미지 빌드 ──────────────────────────────
  docker-build:
    needs: model-build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Download model artifacts
        uses: actions/download-artifact@v4
        with:
          name: model-artifacts

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.DOCKER_REGISTRY }}
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ${{ env.DOCKER_REGISTRY }}/${{ env.MODEL_NAME }}:${{ github.sha }}
            ${{ env.DOCKER_REGISTRY }}/${{ env.MODEL_NAME }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ── Stage 4: Canary 배포 ─────────────────────────────────────
  deploy-canary:
    needs: docker-build
    runs-on: ubuntu-latest
    environment: canary
    steps:
      - uses: actions/checkout@v4

      - name: Configure kubectl
        uses: azure/setup-kubectl@v3
        with:
          version: 'v1.28.0'

      - name: Set kubeconfig
        run: echo "${{ secrets.KUBECONFIG }}" | base64 -d > $HOME/.kube/config

      - name: Deploy canary (5% traffic)
        run: |
          kubectl set image deployment/${{ env.MODEL_NAME }}-canary \
            model=${{ env.DOCKER_REGISTRY }}/${{ env.MODEL_NAME }}:${{ github.sha }}

          kubectl patch virtualservice ${{ env.MODEL_NAME }} \
            --type=merge -p '{
              "spec": {
                "http": [{
                  "route": [
                    {"destination": {"host": "'"${{ env.MODEL_NAME }}"'-stable"}, "weight": 95},
                    {"destination": {"host": "'"${{ env.MODEL_NAME }}"'-canary"}, "weight": 5}
                  ]
                }]
              }
            }'

      - name: Wait and validate canary
        run: |
          echo "Canary 배포 후 10분 대기..."
          sleep 600

          python scripts/validate_canary.py \
            --canary-endpoint http://${{ env.MODEL_NAME }}-canary:8080/health \
            --stable-endpoint http://${{ env.MODEL_NAME }}-stable:8080/health \
            --metrics-endpoint http://prometheus:9090 \
            --min-accuracy 0.94 \
            --max-latency-p99 500 \
            --max-error-rate 0.01

  # ── Stage 5: 프로덕션 전체 배포 ──────────────────────────────
  deploy-production:
    needs: deploy-canary
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4

      - name: Configure kubectl
        uses: azure/setup-kubectl@v3
        with:
          version: 'v1.28.0'

      - name: Set kubeconfig
        run: echo "${{ secrets.KUBECONFIG }}" | base64 -d > $HOME/.kube/config

      - name: Promote to production (100% traffic)
        run: |
          # Stable 디플로이먼트 업데이트
          kubectl set image deployment/${{ env.MODEL_NAME }}-stable \
            model=${{ env.DOCKER_REGISTRY }}/${{ env.MODEL_NAME }}:${{ github.sha }}

          # 트래픽 100%로 전환
          kubectl patch virtualservice ${{ env.MODEL_NAME }} \
            --type=merge -p '{
              "spec": {
                "http": [{
                  "route": [
                    {"destination": {"host": "'"${{ env.MODEL_NAME }}"'-stable"}, "weight": 100},
                    {"destination": {"host": "'"${{ env.MODEL_NAME }}"'-canary"}, "weight": 0}
                  ]
                }]
              }
            }'

          kubectl rollout status deployment/${{ env.MODEL_NAME }}-stable --timeout=300s

      - name: Post-deploy health check
        run: |
          python scripts/health_check.py \
            --endpoint http://${{ env.MODEL_NAME }}-stable:8080 \
            --retries 5 \
            --interval 30

      - name: Notify
        if: always()
        uses: slackapi/slack-github-action@v1.25.0
        with:
          channel-id: '#ml-deployments'
          slack-message: |
            *배포 결과*: ${{ job.status }}
            *모델*: ${{ env.MODEL_NAME }}
            *커밋*: ${{ github.sha }}
            *배포자*: ${{ github.actor }}
        env:
          SLACK_BOT_TOKEN: ${{ secrets.SLACK_BOT_TOKEN }}
```

### GitLab CI 파이프라인 예시

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - evaluate
  - deploy-canary
  - deploy-production

variables:
  PYTHON_VERSION: "3.11"
  MODEL_NAME: "ocr-receipt"

.python-setup: &python-setup
  image: python:${PYTHON_VERSION}
  before_script:
    - pip install -r requirements.txt

test:
  <<: *python-setup
  stage: test
  script:
    - ruff check src/
    - pytest tests/unit/ -v --cov=src
  coverage: '/(?i)total.*? (100(?:\.0+)?\%|[1-9]?\d(?:\.\d+)?\%)$/'

build-model:
  <<: *python-setup
  stage: build
  tags: [gpu]
  script:
    - dvc pull
    - python src/train.py --config configs/train.yaml
    - python src/evaluate.py --model models/checkpoints/best.pt
  artifacts:
    paths:
      - models/checkpoints/best.pt
      - metrics/eval.json
    expire_in: 30 days

evaluate-quality:
  <<: *python-setup
  stage: evaluate
  needs: [build-model]
  script:
    - python scripts/quality_gate.py --metrics metrics/eval.json
  rules:
    - if: $CI_COMMIT_BRANCH == "main"

deploy-canary:
  stage: deploy-canary
  image: bitnami/kubectl:latest
  needs: [evaluate-quality]
  environment:
    name: canary
    url: https://canary.api.example.com
  script:
    - kubectl set image deployment/${MODEL_NAME}-canary model=${CI_REGISTRY_IMAGE}:${CI_COMMIT_SHA}
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
      when: manual

deploy-production:
  stage: deploy-production
  image: bitnami/kubectl:latest
  needs: [deploy-canary]
  environment:
    name: production
    url: https://api.example.com
  script:
    - kubectl set image deployment/${MODEL_NAME}-stable model=${CI_REGISTRY_IMAGE}:${CI_COMMIT_SHA}
    - kubectl rollout status deployment/${MODEL_NAME}-stable --timeout=300s
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
      when: manual
```

---

## 코드: 배포 스크립트 및 Health Check 자동화

### Deployment Manager

```python
import os
import time
import json
import logging
import subprocess
from enum import Enum
from dataclasses import dataclass
from typing import Callable

import requests

logger = logging.getLogger(__name__)


class DeployStrategy(Enum):
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    SHADOW = "shadow"


@dataclass
class DeployConfig:
    """배포 설정."""
    model_name: str
    image_tag: str
    strategy: DeployStrategy
    namespace: str = "default"
    canary_weight: int = 5           # Canary 초기 트래픽 비율 (%)
    canary_steps: list[int] = None   # Canary 단계별 비율
    rollout_timeout: int = 300       # 롤아웃 타임아웃 (초)
    health_check_retries: int = 5
    health_check_interval: int = 30  # 초

    def __post_init__(self):
        if self.canary_steps is None:
            self.canary_steps = [5, 25, 50, 75, 100]


class HealthChecker:
    """서비스 Health Check를 수행하는 클래스."""

    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_liveness(self) -> bool:
        """Liveness probe: 프로세스가 살아있는가?"""
        try:
            resp = requests.get(
                f"{self.base_url}/health/live",
                timeout=self.timeout,
            )
            return resp.status_code == 200
        except requests.RequestException as e:
            logger.warning(f"Liveness check 실패: {e}")
            return False

    def check_readiness(self) -> bool:
        """Readiness probe: 트래픽을 받을 준비가 됐는가?"""
        try:
            resp = requests.get(
                f"{self.base_url}/health/ready",
                timeout=self.timeout,
            )
            return resp.status_code == 200
        except requests.RequestException as e:
            logger.warning(f"Readiness check 실패: {e}")
            return False

    def check_model_loaded(self) -> bool:
        """모델이 정상 로드됐는지 확인."""
        try:
            resp = requests.get(
                f"{self.base_url}/health/model",
                timeout=self.timeout,
            )
            data = resp.json()
            return data.get("model_loaded", False)
        except requests.RequestException as e:
            logger.warning(f"Model health check 실패: {e}")
            return False

    def check_inference(self, sample_input: dict) -> bool:
        """실제 추론이 정상 동작하는지 smoke test."""
        try:
            resp = requests.post(
                f"{self.base_url}/predict",
                json=sample_input,
                timeout=self.timeout * 3,
            )
            return resp.status_code == 200 and "prediction" in resp.json()
        except requests.RequestException as e:
            logger.warning(f"Inference check 실패: {e}")
            return False

    def full_check(self, sample_input: dict | None = None) -> dict[str, bool]:
        """모든 Health Check를 수행한다."""
        results = {
            "liveness": self.check_liveness(),
            "readiness": self.check_readiness(),
            "model_loaded": self.check_model_loaded(),
        }
        if sample_input:
            results["inference"] = self.check_inference(sample_input)

        all_pass = all(results.values())
        results["overall"] = all_pass

        if not all_pass:
            logger.error(f"Health check 실패: {results}")
        else:
            logger.info("Health check 전체 통과")

        return results


class DeploymentManager:
    """모델 배포를 관리하는 클래스.

    kubectl 명령을 래핑하여 다양한 배포 전략을 구현한다.
    """

    def __init__(self, config: DeployConfig):
        self.config = config
        self.health_checker: HealthChecker | None = None

    def _kubectl(self, args: str) -> str:
        """kubectl 명령을 실행한다."""
        cmd = f"kubectl -n {self.config.namespace} {args}"
        try:
            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.error(f"kubectl 실패: {result.stderr}")
                raise RuntimeError(result.stderr)
            return result.stdout
        except subprocess.TimeoutExpired:
            raise RuntimeError("kubectl 타임아웃")

    def deploy(self) -> bool:
        """설정된 전략에 따라 배포를 실행한다."""
        strategy_map: dict[DeployStrategy, Callable] = {
            DeployStrategy.BLUE_GREEN: self._deploy_blue_green,
            DeployStrategy.CANARY: self._deploy_canary,
            DeployStrategy.ROLLING: self._deploy_rolling,
            DeployStrategy.SHADOW: self._deploy_shadow,
        }

        deploy_fn = strategy_map.get(self.config.strategy)
        if not deploy_fn:
            raise ValueError(f"지원하지 않는 전략: {self.config.strategy}")

        logger.info(f"배포 시작: {self.config.model_name} ({self.config.strategy.value})")
        return deploy_fn()

    def _deploy_blue_green(self) -> bool:
        """Blue-Green 배포."""
        image = f"{os.environ.get('DOCKER_REGISTRY', 'registry')}/{self.config.model_name}:{self.config.image_tag}"

        # 1. Green 환경에 새 버전 배포
        self._kubectl(
            f"set image deployment/{self.config.model_name}-green model={image}"
        )

        # 2. Green 롤아웃 완료 대기
        self._kubectl(
            f"rollout status deployment/{self.config.model_name}-green "
            f"--timeout={self.config.rollout_timeout}s"
        )

        # 3. Health Check
        if self.health_checker and not self.health_checker.check_readiness():
            logger.error("Green 환경 Health Check 실패. 배포 중단.")
            return False

        # 4. 트래픽 전환 (Service selector 변경)
        self._kubectl(
            f"patch service {self.config.model_name} "
            f"-p '{{\"spec\":{{\"selector\":{{\"version\":\"green\"}}}}}}'"
        )

        logger.info("Blue-Green 배포 완료: 트래픽이 Green으로 전환됨")
        return True

    def _deploy_canary(self) -> bool:
        """Canary 배포: 단계별 트래픽 증가."""
        image = f"{os.environ.get('DOCKER_REGISTRY', 'registry')}/{self.config.model_name}:{self.config.image_tag}"

        # Canary 디플로이먼트 업데이트
        self._kubectl(
            f"set image deployment/{self.config.model_name}-canary model={image}"
        )
        self._kubectl(
            f"rollout status deployment/{self.config.model_name}-canary "
            f"--timeout={self.config.rollout_timeout}s"
        )

        # 단계별 트래픽 증가
        for weight in self.config.canary_steps:
            stable_weight = 100 - weight
            logger.info(f"Canary 트래픽: {weight}%")

            self._kubectl(
                f"patch virtualservice {self.config.model_name} "
                f"--type=merge -p '{{\"spec\":{{\"http\":[{{\"route\":["
                f"{{\"destination\":{{\"host\":\"{self.config.model_name}-stable\"}},\"weight\":{stable_weight}}},"
                f"{{\"destination\":{{\"host\":\"{self.config.model_name}-canary\"}},\"weight\":{weight}}}"
                f"]}}]}}}}'"
            )

            if weight < 100:
                # 다음 단계 전 대기 및 검증
                logger.info(f"Canary {weight}% 검증 대기 (5분)...")
                time.sleep(300)

                if self.health_checker and not self.health_checker.check_readiness():
                    logger.error(f"Canary {weight}% 단계에서 Health Check 실패. 롤백 필요.")
                    return False

        logger.info("Canary 배포 완료: 100% 트래픽 전환")
        return True

    def _deploy_rolling(self) -> bool:
        """Rolling Update 배포."""
        image = f"{os.environ.get('DOCKER_REGISTRY', 'registry')}/{self.config.model_name}:{self.config.image_tag}"

        self._kubectl(
            f"set image deployment/{self.config.model_name} model={image}"
        )
        self._kubectl(
            f"rollout status deployment/{self.config.model_name} "
            f"--timeout={self.config.rollout_timeout}s"
        )

        logger.info("Rolling Update 배포 완료")
        return True

    def _deploy_shadow(self) -> bool:
        """Shadow Mode 배포: 트래픽 미러링."""
        image = f"{os.environ.get('DOCKER_REGISTRY', 'registry')}/{self.config.model_name}:{self.config.image_tag}"

        # Shadow 디플로이먼트 업데이트
        self._kubectl(
            f"set image deployment/{self.config.model_name}-shadow model={image}"
        )

        # Istio 트래픽 미러링 설정
        self._kubectl(
            f"patch virtualservice {self.config.model_name} "
            f"--type=merge -p '{{\"spec\":{{\"http\":[{{\"route\":["
            f"{{\"destination\":{{\"host\":\"{self.config.model_name}-stable\"}},\"weight\":100}}"
            f"],\"mirror\":{{\"host\":\"{self.config.model_name}-shadow\"}}"
            f"}}]}}}}'"
        )

        logger.info("Shadow Mode 배포 완료: 트래픽 미러링 활성화")
        return True


# ── 사용 예시 ──────────────────────────────────────────────────
# config = DeployConfig(
#     model_name="ocr-receipt",
#     image_tag="abc123",
#     strategy=DeployStrategy.CANARY,
#     canary_steps=[5, 25, 50, 100],
# )
# deployer = DeploymentManager(config)
# deployer.health_checker = HealthChecker("http://ocr-receipt-canary:8080")
# success = deployer.deploy()
```

---

## Dockerfile: 모델 서빙 이미지

```dockerfile
# Dockerfile
FROM python:3.11-slim AS base

WORKDIR /app

# 시스템 의존성
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# 소스 코드
COPY src/ src/
COPY configs/ configs/

# 모델 아티팩트
COPY models/checkpoints/best.pt models/

# Health check 엔드포인트용
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1

# 비root 사용자
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["python", "-m", "src.serve", "--port", "8080", "--model", "models/best.pt"]
```

---

## 용어 체크리스트

학습을 마친 뒤 아래 항목을 스스로 점검해봐라.

- [ ] **Blue-Green** 배포에서 롤백이 즉시 가능한 이유를 설명할 수 있는가?
- [ ] **Canary** 배포에서 통계적 유의성 검증이 왜 필요한지 이해했는가?
- [ ] A/B 테스트의 **Z-검정 공식**을 유도하고 p-value를 해석할 수 있는가?
- [ ] **필요 샘플 수** 공식에서 $z_{\alpha/2}$, $z_\beta$, $\sigma$, MDE 각각의 역할을 설명할 수 있는가?
- [ ] **Rolling Update**에서 두 버전이 공존하는 문제를 어떻게 처리하는지 아는가?
- [ ] **Shadow Mode**가 다른 배포 전략과 근본적으로 다른 점을 설명할 수 있는가?
- [ ] GitHub Actions YAML에서 `needs`, `environment`, `if` 조건의 역할을 이해했는가?
- [ ] **Health Check**의 Liveness/Readiness/Startup 프로브 차이를 설명할 수 있는가?
- [ ] Canary 배포에서 **Sequential Testing**이 왜 필요한지 (다중 검정 문제) 이해했는가?
- [ ] CI/CD 파이프라인에서 **Quality Gate**의 역할과 설정 기준을 정의할 수 있는가?
