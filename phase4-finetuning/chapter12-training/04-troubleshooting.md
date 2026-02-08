---
---

# 12.4 트러블슈팅

> 학습 중 발생하는 문제들을 **증상 → 원인 → 해결** 순서로 정리했다.
> 문제가 생기면 이 챕터를 레퍼런스처럼 펼쳐서 대응하라.

---

## 핵심 용어

| 용어 | 정의 | 왜 중요한가 |
|------|------|-------------|
| **Loss NaN** | Loss 값이 NaN(Not a Number)이 되는 현상 | 학습이 완전히 망가졌다는 신호. 즉시 중단 필요 |
| **OOM (Out of Memory)** | GPU 메모리가 부족해 학습이 중단되는 오류 | 가장 흔한 학습 중단 원인 |
| **Training Instability** | Loss가 급등(spike)하거나 진동하는 현상 | 수렴 실패로 이어질 수 있음 |
| **Divergence** | Loss가 감소하지 않고 계속 증가하는 현상 | 학습이 완전히 잘못된 방향으로 가고 있다는 신호 |
| **Gradient Clipping** | Gradient의 norm이 임계값을 초과하면 스케일링하여 제한하는 기법 | Gradient explosion 방지 |
| **Mixed Precision** | FP16/BF16과 FP32를 혼합 사용하는 학습 방식 | 메모리 절약 + 속도 향상, 수치 불안정 가능 |

---

## 문제 1: Loss NaN / Inf

### 증상

```
Step 100: loss = 2.3456
Step 200: loss = 1.8765
Step 300: loss = nan
```

학습 중 loss가 갑자기 NaN 또는 Inf로 변한다.

### 원인 & 해결 체크리스트

| # | 원인 | 확인 방법 | 해결 |
|---|------|-----------|------|
| 1 | **학습률 과대** | LR > 1e-3인지 확인 | LR을 1/10로 줄여보기 |
| 2 | **데이터 이상값** | 데이터에 NaN/Inf가 포함되었는지 확인 | 데이터 검증 (Ch 12.1) |
| 3 | **Mixed Precision 수치 불안정** | FP16 사용 중인지 확인 | BF16으로 변경, 또는 loss scaling 조정 |
| 4 | **Gradient Explosion** | grad_norm 로그 확인 | `max_grad_norm=1.0` 설정 |
| 5 | **0으로 나누기** | Loss 함수에 epsilon 누락 | `log(x + eps)` 형태로 수정 |
| 6 | **토큰화 오류** | `labels`에 -100 외 이상한 값 존재 | 토큰화 파이프라인 재검증 |

### 코드: NaN 탐지 & 진단

```python
import torch
import logging
from typing import Optional
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

logger = logging.getLogger(__name__)


class NaNDetector:
    """학습 중 NaN 발생을 탐지하고 원인을 진단한다."""

    @staticmethod
    def check_data(batch: dict) -> list[str]:
        """배치 데이터에 NaN/Inf가 있는지 확인한다."""
        issues = []
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                if torch.isnan(tensor).any():
                    count = torch.isnan(tensor).sum().item()
                    issues.append(f"{key}: {count}개의 NaN 발견")
                if torch.isinf(tensor).any():
                    count = torch.isinf(tensor).sum().item()
                    issues.append(f"{key}: {count}개의 Inf 발견")
        return issues

    @staticmethod
    def check_model_params(model: torch.nn.Module) -> list[str]:
        """모델 파라미터에 NaN/Inf가 있는지 확인한다."""
        issues = []
        for name, param in model.named_parameters():
            if torch.isnan(param.data).any():
                issues.append(f"[PARAM NaN] {name}")
            if torch.isinf(param.data).any():
                issues.append(f"[PARAM Inf] {name}")
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    issues.append(f"[GRAD NaN] {name}")
                if torch.isinf(param.grad).any():
                    issues.append(f"[GRAD Inf] {name}")
        return issues

    @staticmethod
    def check_loss(loss: torch.Tensor) -> bool:
        """Loss가 정상인지 확인한다."""
        if torch.isnan(loss):
            logger.error("Loss is NaN!")
            return False
        if torch.isinf(loss):
            logger.error("Loss is Inf!")
            return False
        if loss.item() > 1e6:
            logger.warning(f"Loss가 비정상적으로 큼: {loss.item():.2e}")
            return False
        return True

    @staticmethod
    def diagnose_nan(
        model: torch.nn.Module,
        batch: dict,
        loss: torch.Tensor,
    ) -> dict:
        """NaN 발생 시 종합 진단을 수행한다."""
        report = {
            "loss_ok": NaNDetector.check_loss(loss),
            "data_issues": NaNDetector.check_data(batch),
            "param_issues": NaNDetector.check_model_params(model),
        }

        if not report["loss_ok"]:
            logger.error("=== NaN 진단 보고서 ===")

            if report["data_issues"]:
                logger.error("[원인 1] 데이터에 NaN/Inf 존재:")
                for issue in report["data_issues"]:
                    logger.error(f"  - {issue}")

            if report["param_issues"]:
                logger.error("[원인 2] 모델 파라미터/Gradient에 NaN/Inf:")
                for issue in report["param_issues"]:
                    logger.error(f"  - {issue}")

            if not report["data_issues"] and not report["param_issues"]:
                logger.error("[원인 3] Loss 계산 과정에서 수치 오버플로우 추정")
                logger.error("  → 학습률 줄이기, BF16 사용, gradient clipping 확인")

        return report


class NaNStoppingCallback(TrainerCallback):
    """NaN 발생 시 학습을 즉시 중단하는 콜백."""

    def __init__(self, check_interval: int = 10):
        self.check_interval = check_interval

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return

        loss = logs.get("loss")
        if loss is not None and (loss != loss or loss == float("inf")):
            logger.error(f"[Step {state.global_step}] NaN/Inf loss 감지. 학습 중단.")
            control.should_training_stop = True
```

---

## 문제 2: OOM (Out of Memory)

### 증상

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
(GPU 0; 24.00 GiB total capacity; 21.50 GiB already allocated)
```

### 원인 & 해결 체크리스트

| # | 해결책 | 메모리 절감 | 속도 영향 | 우선순위 |
|---|--------|-------------|-----------|----------|
| 1 | **Batch Size 줄이기** | 비례 감소 | gradient accum으로 보완 | 가장 먼저 |
| 2 | **Gradient Checkpointing** | 30~50% | 20~30% 느림 | 두 번째 |
| 3 | **Max Sequence Length 줄이기** | 비례 감소 | 긴 텍스트 잘림 | 데이터 확인 후 |
| 4 | **BF16/FP16 사용** | ~50% | 거의 없음 | 기본 설정 |
| 5 | **LoRA Rank 줄이기** | 소폭 감소 | 표현력 감소 | 마지막 수단 |
| 6 | **Optimizer 변경** | 25~50% | 거의 없음 | 8-bit Adam 등 |
| 7 | **CPU Offloading** | 큰 폭 감소 | 크게 느림 | 극한 상황만 |

### 코드: OOM 방지 & 복구

```python
import gc


class OOMHandler:
    """OOM(Out of Memory) 문제를 진단하고 해결한다."""

    @staticmethod
    def estimate_memory_usage(
        model_params_billion: float,
        batch_size: int,
        seq_length: int,
        dtype_bytes: int = 2,  # bf16 = 2, fp32 = 4
        gradient_checkpointing: bool = False,
    ) -> dict:
        """
        학습 시 예상 GPU 메모리 사용량을 추정한다.

        근사 공식:
        - 모델 가중치: params * dtype_bytes
        - Gradient: params * dtype_bytes
        - Optimizer state (AdamW): params * 8 bytes (fp32 momentum + variance)
        - Activation: batch_size * seq_length * hidden_dim * num_layers * dtype_bytes

        Args:
            model_params_billion: 모델 파라미터 수 (십억 단위)
            batch_size: 배치 크기
            seq_length: 시퀀스 길이
            dtype_bytes: 데이터 타입 바이트 수
            gradient_checkpointing: gradient checkpointing 사용 여부

        Returns:
            dict: 메모리 사용량 추정 (GB)
        """
        params = model_params_billion * 1e9

        # 모델 가중치
        model_mem_gb = params * dtype_bytes / (1024 ** 3)

        # Gradient (학습 가능 파라미터만, LoRA면 전체의 ~1%)
        grad_mem_gb = model_mem_gb * 0.01  # LoRA 가정

        # Optimizer state (학습 가능 파라미터에 대해)
        optim_mem_gb = params * 0.01 * 8 / (1024 ** 3)  # LoRA 가정, AdamW fp32

        # Activation memory (근사치)
        activation_factor = 0.5 if gradient_checkpointing else 1.0
        activation_mem_gb = (
            batch_size * seq_length * 4096 * 32  # hidden_dim * num_layers 근사
            * dtype_bytes * activation_factor / (1024 ** 3)
        )

        total_gb = model_mem_gb + grad_mem_gb + optim_mem_gb + activation_mem_gb

        return {
            "model_gb": model_mem_gb,
            "gradient_gb": grad_mem_gb,
            "optimizer_gb": optim_mem_gb,
            "activation_gb": activation_mem_gb,
            "total_gb": total_gb,
            "recommendation": OOMHandler._recommend_gpu(total_gb),
        }

    @staticmethod
    def _recommend_gpu(total_gb: float) -> str:
        """필요 메모리에 따른 GPU 추천."""
        if total_gb <= 8:
            return "RTX 3070/4070 (8GB) 이상"
        elif total_gb <= 16:
            return "RTX 4080/A4000 (16GB) 이상"
        elif total_gb <= 24:
            return "RTX 3090/4090/A5000 (24GB) 이상"
        elif total_gb <= 48:
            return "A6000/A40 (48GB) 이상"
        elif total_gb <= 80:
            return "A100/H100 (80GB) 이상"
        else:
            return "멀티 GPU 또는 모델 병렬화 필요"

    @staticmethod
    def find_max_batch_size(
        model: torch.nn.Module,
        sample_batch: dict,
        max_batch_size: int = 64,
        device: str = "cuda",
    ) -> int:
        """
        OOM 없이 사용 가능한 최대 배치 크기를 탐색한다.
        Binary search로 효율적으로 탐색.

        Args:
            model: 모델
            sample_batch: 샘플 배치 (batch_size=1)
            max_batch_size: 탐색 상한
            device: 디바이스

        Returns:
            int: 최대 배치 크기
        """
        model.to(device)
        model.train()

        low, high = 1, max_batch_size
        best = 1

        while low <= high:
            mid = (low + high) // 2

            try:
                # 배치 크기 조정
                test_batch = {}
                for key, tensor in sample_batch.items():
                    if isinstance(tensor, torch.Tensor):
                        # 첫 번째 차원(batch)을 mid로 확장
                        repeat_dims = [mid] + [1] * (tensor.dim() - 1)
                        test_batch[key] = tensor.repeat(*repeat_dims).to(device)

                # Forward + Backward 테스트
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(**test_batch)
                    outputs.loss.backward()

                model.zero_grad()

                # 성공
                best = mid
                low = mid + 1
                logger.info(f"  Batch size {mid}: OK")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    high = mid - 1
                    logger.info(f"  Batch size {mid}: OOM")
                else:
                    raise
            finally:
                # 메모리 정리
                torch.cuda.empty_cache()
                gc.collect()

        logger.info(f"최대 Batch Size: {best}")
        return best

    @staticmethod
    def emergency_cleanup() -> None:
        """긴급 GPU 메모리 정리."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        logger.info("긴급 메모리 정리 완료")
```

---

## 문제 3: 학습 불안정 (Loss Spike)

### 증상

```
Step 1000: loss = 0.85
Step 1010: loss = 0.83
Step 1020: loss = 15.67  ← spike!
Step 1030: loss = 0.90
Step 1040: loss = 0.84
```

Loss가 갑자기 급등했다가 다시 돌아오는 현상.

### 원인 & 해결 체크리스트

| # | 원인 | 확인 방법 | 해결 |
|---|------|-----------|------|
| 1 | **Warmup 부족** | warmup_ratio 확인 (< 3%?) | 5~10%로 증가 |
| 2 | **Gradient Clipping 미설정** | `max_grad_norm` 확인 | 1.0으로 설정 |
| 3 | **배치 내 이상 데이터** | 해당 스텝의 데이터 확인 | 데이터 정제 |
| 4 | **학습률 과대** | 현재 LR 확인 | LR 줄이기 |
| 5 | **Mixed Precision 불안정** | FP16 사용 중인지 확인 | BF16으로 변경 |

### 코드: Loss Spike 모니터

```python
from collections import deque


class LossSpikeMonitor:
    """Loss spike를 실시간 감지하고 경고한다."""

    def __init__(
        self,
        window_size: int = 50,
        spike_threshold: float = 5.0,
        alert_callback=None,
    ):
        """
        Args:
            window_size: 이동 평균 윈도우 크기
            spike_threshold: spike 판정 배수 (평균 대비)
            alert_callback: spike 감지 시 호출할 함수
        """
        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self.alert_callback = alert_callback
        self.loss_history = deque(maxlen=10000)
        self.recent_losses = deque(maxlen=window_size)
        self.spike_events = []

    def update(self, step: int, loss: float) -> bool:
        """
        새 loss 값을 기록하고 spike 여부를 판단한다.

        Args:
            step: 현재 스텝
            loss: 현재 loss

        Returns:
            bool: spike 발생 여부
        """
        self.loss_history.append((step, loss))
        self.recent_losses.append(loss)

        if len(self.recent_losses) < self.window_size // 2:
            return False

        # 이동 평균 & 표준편차
        import numpy as np
        recent = list(self.recent_losses)
        mean_loss = np.mean(recent[:-1])  # 현재 값 제외
        std_loss = np.std(recent[:-1]) + 1e-8

        # Spike 판정: 현재 loss가 평균 + threshold * std 초과
        z_score = (loss - mean_loss) / std_loss
        is_spike = loss > mean_loss * self.spike_threshold or z_score > 5.0

        if is_spike:
            event = {
                "step": step,
                "loss": loss,
                "mean_loss": mean_loss,
                "z_score": z_score,
            }
            self.spike_events.append(event)

            logger.warning(
                f"[SPIKE] Step {step}: loss={loss:.4f} "
                f"(평균={mean_loss:.4f}, z={z_score:.1f})"
            )

            if self.alert_callback:
                self.alert_callback(event)

        return is_spike

    def get_summary(self) -> dict:
        """Spike 이벤트 요약을 반환한다."""
        return {
            "total_steps": len(self.loss_history),
            "num_spikes": len(self.spike_events),
            "spike_rate": len(self.spike_events) / max(len(self.loss_history), 1),
            "spike_events": self.spike_events[-10:],  # 최근 10개만
        }


class LossSpikeCallback(TrainerCallback):
    """Trainer에 연결하는 Loss Spike 감지 콜백."""

    def __init__(self, window_size: int = 50, spike_threshold: float = 5.0):
        self.monitor = LossSpikeMonitor(window_size, spike_threshold)

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.monitor.update(state.global_step, logs["loss"])

    def on_train_end(self, args, state, control, **kwargs):
        summary = self.monitor.get_summary()
        if summary["num_spikes"] > 0:
            logger.warning(
                f"학습 중 총 {summary['num_spikes']}회 Loss Spike 발생 "
                f"(spike rate: {summary['spike_rate']:.2%})"
            )
```

---

## 문제 4: 수렴하지 않음 (Loss가 안 줄어듦)

### 증상

```
Step 0:    loss = 3.45
Step 100:  loss = 3.44
Step 500:  loss = 3.42
Step 1000: loss = 3.41
```

Loss가 거의 변하지 않거나, 매우 느리게만 감소한다.

### 원인 & 해결 체크리스트

| # | 원인 | 확인 방법 | 해결 |
|---|------|-----------|------|
| 1 | **학습률이 너무 낮음** | 현재 LR 확인 (< 1e-6?) | LR Finder로 재탐색 |
| 2 | **학습률이 너무 높음** | Loss가 진동하는지 확인 | LR을 1/5로 줄이기 |
| 3 | **LoRA가 적용 안 됨** | `model.print_trainable_parameters()` | trainable > 0 확인 |
| 4 | **데이터 문제** | 라벨이 올바른지 확인 | Data Verification (Ch 12.1) |
| 5 | **Warmup이 너무 길음** | warmup 스텝 수 확인 | warmup_ratio 줄이기 |
| 6 | **Gradient Vanishing** | Gradient flow 확인 | 레이어별 gradient 분포 확인 |
| 7 | **Wrong Task Format** | 입력 형식이 모델에 맞는지 확인 | 프롬프트 템플릿 검증 |

### 코드: 수렴 진단

```python
class ConvergenceDiagnostic:
    """학습 수렴 여부를 진단한다."""

    @staticmethod
    def check_convergence(
        losses: list[float],
        window_size: int = 100,
        min_improvement: float = 0.001,
    ) -> dict:
        """
        Loss 이력에서 수렴 상태를 진단한다.

        Args:
            losses: loss 이력
            window_size: 비교 윈도우 크기
            min_improvement: 최소 개선률 (이 이상이면 학습 진행 중)

        Returns:
            dict: 진단 결과
        """
        import numpy as np

        if len(losses) < window_size * 2:
            return {"status": "insufficient_data", "message": "데이터 부족"}

        # 전반부 vs 후반부 비교
        first_half = np.mean(losses[:window_size])
        second_half = np.mean(losses[-window_size:])

        improvement = (first_half - second_half) / first_half

        # 최근 추세
        recent = losses[-window_size:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]

        result = {
            "first_half_mean": first_half,
            "second_half_mean": second_half,
            "improvement_pct": improvement * 100,
            "recent_trend": trend,  # 음수면 감소 (좋음), 양수면 증가 (나쁨)
        }

        if improvement < min_improvement:
            if trend > 0:
                result["status"] = "diverging"
                result["message"] = "Loss가 증가 추세 — 학습률 과대 또는 데이터 문제"
            else:
                result["status"] = "stagnant"
                result["message"] = (
                    f"Loss 개선이 {improvement*100:.2f}%로 미미 — "
                    "학습률 조정 또는 모델/데이터 점검 필요"
                )
        elif improvement > 0.5:
            result["status"] = "healthy"
            result["message"] = f"정상적으로 수렴 중 ({improvement*100:.1f}% 개선)"
        else:
            result["status"] = "slow"
            result["message"] = f"느리게 수렴 중 ({improvement*100:.1f}% 개선)"

        return result

    @staticmethod
    def check_trainable_params(model: torch.nn.Module) -> dict:
        """학습 가능 파라미터가 올바르게 설정되었는지 확인한다."""
        total = 0
        trainable = 0
        frozen = 0

        for name, param in model.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
            else:
                frozen += param.numel()

        result = {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "trainable_pct": trainable / total * 100 if total > 0 else 0,
        }

        if trainable == 0:
            logger.error("학습 가능 파라미터가 0개! LoRA 또는 unfreeze 설정 확인 필요")
            result["status"] = "error"
        elif result["trainable_pct"] > 50:
            logger.warning(
                f"학습 가능 파라미터가 전체의 {result['trainable_pct']:.1f}% — "
                "Full fine-tuning인지 확인"
            )
            result["status"] = "warning"
        else:
            logger.info(
                f"학습 파라미터: {trainable:,} / {total:,} "
                f"({result['trainable_pct']:.2f}%)"
            )
            result["status"] = "ok"

        return result
```

---

## 문제 5: 과적합 (Overfitting)

### 증상

```
Step 1000: train_loss=0.50, eval_loss=0.55
Step 2000: train_loss=0.20, eval_loss=0.58  ← eval 증가 시작
Step 3000: train_loss=0.05, eval_loss=0.72  ← 격차 확대
```

Train loss는 계속 줄어드는데, Eval loss는 오히려 증가한다.

### 원인 & 해결 체크리스트

| # | 해결책 | 효과 | 비용 |
|---|--------|------|------|
| 1 | **Early Stopping** | 과적합 지점에서 자동 종료 | 없음 |
| 2 | **LoRA Rank 줄이기** | 모델 용량 축소 = 과적합 억제 | 표현력 감소 |
| 3 | **Weight Decay 증가** | L2 정규화 강화 | 미세 조정 필요 |
| 4 | **Dropout 증가** | 랜덤 뉴런 비활성화로 정규화 | 학습 속도 저하 |
| 5 | **Data Augmentation** | 학습 데이터 다양성 증가 | 데이터 전처리 비용 |
| 6 | **학습 데이터 추가** | 근본 해결 | 데이터 수집 비용 |
| 7 | **Epoch 수 줄이기** | 학습 시간 축소 | 없음 |

### 코드: 과적합 모니터

```python
class OverfitMonitor:
    """Train/Eval loss 격차를 모니터링하여 과적합을 감지한다."""

    def __init__(self, threshold: float = 0.1, patience: int = 3):
        """
        Args:
            threshold: train_loss 대비 eval_loss 격차 허용 비율
            patience: 과적합 판정까지 연속 횟수
        """
        self.threshold = threshold
        self.patience = patience
        self.eval_history = []
        self.train_history = []
        self.overfit_count = 0
        self.best_eval_loss = float("inf")

    def update(self, train_loss: float, eval_loss: float, step: int) -> dict:
        """
        새 loss를 기록하고 과적합 여부를 판단한다.

        Returns:
            dict: {"overfitting": bool, "gap": float, "trend": str}
        """
        self.train_history.append((step, train_loss))
        self.eval_history.append((step, eval_loss))

        # 과적합 판정
        gap = eval_loss - train_loss
        gap_ratio = gap / max(train_loss, 1e-8)

        # Eval loss가 이전 best보다 나빠졌는지
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.overfit_count = 0
            trend = "improving"
        else:
            self.overfit_count += 1
            trend = "worsening"

        is_overfitting = self.overfit_count >= self.patience

        result = {
            "overfitting": is_overfitting,
            "gap": gap,
            "gap_ratio": gap_ratio,
            "trend": trend,
            "overfit_count": self.overfit_count,
            "best_eval_loss": self.best_eval_loss,
        }

        if is_overfitting:
            logger.warning(
                f"[OVERFIT] Step {step}: "
                f"train={train_loss:.4f}, eval={eval_loss:.4f} "
                f"(gap_ratio={gap_ratio:.2f}, count={self.overfit_count})"
            )

        return result
```

---

## 종합 디버깅 유틸리티

위의 모든 진단 도구를 통합한 디버깅 스위트:

```python
class TrainingDebugger:
    """학습 중 발생하는 모든 문제를 통합 진단한다."""

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.nan_detector = NaNDetector()
        self.spike_monitor = LossSpikeMonitor()
        self.overfit_monitor = OverfitMonitor()
        self.convergence = ConvergenceDiagnostic()
        self.loss_history = []
        self.eval_history = []

    def on_step(self, step: int, loss: float, batch: dict = None) -> dict:
        """
        매 학습 스텝마다 호출. 이상 징후를 탐지한다.

        Args:
            step: 현재 스텝
            loss: 현재 loss
            batch: 현재 배치 (선택)

        Returns:
            dict: 진단 결과
        """
        alerts = []
        self.loss_history.append(loss)

        # NaN 체크
        if loss != loss or loss == float("inf"):
            alerts.append({
                "type": "NaN",
                "severity": "critical",
                "message": f"Step {step}: Loss is NaN/Inf",
            })
            if batch:
                nan_report = self.nan_detector.diagnose_nan(
                    self.model, batch, torch.tensor(loss)
                )
                alerts[-1]["details"] = nan_report

        # Spike 체크
        if self.spike_monitor.update(step, loss):
            alerts.append({
                "type": "spike",
                "severity": "warning",
                "message": f"Step {step}: Loss spike detected ({loss:.4f})",
            })

        return {"alerts": alerts}

    def on_eval(self, step: int, train_loss: float, eval_loss: float) -> dict:
        """
        평가 시마다 호출. 과적합과 수렴 상태를 진단한다.

        Args:
            step: 현재 스텝
            train_loss: 학습 loss
            eval_loss: 평가 loss

        Returns:
            dict: 진단 결과
        """
        alerts = []
        self.eval_history.append((step, eval_loss))

        # 과적합 체크
        overfit_result = self.overfit_monitor.update(train_loss, eval_loss, step)
        if overfit_result["overfitting"]:
            alerts.append({
                "type": "overfitting",
                "severity": "warning",
                "message": (
                    f"과적합 감지: train={train_loss:.4f}, eval={eval_loss:.4f}, "
                    f"gap={overfit_result['gap']:.4f}"
                ),
            })

        # 수렴 체크 (충분한 데이터가 있을 때)
        if len(self.loss_history) > 200:
            conv_result = self.convergence.check_convergence(self.loss_history)
            if conv_result["status"] in ("stagnant", "diverging"):
                alerts.append({
                    "type": "convergence",
                    "severity": "warning",
                    "message": conv_result["message"],
                })

        return {"alerts": alerts, "overfit": overfit_result}

    def full_diagnosis(self) -> dict:
        """학습 전체에 대한 종합 진단 보고서를 생성한다."""
        report = {}

        # 1. 파라미터 확인
        report["params"] = self.convergence.check_trainable_params(self.model)

        # 2. Loss 추세
        if self.loss_history:
            report["convergence"] = self.convergence.check_convergence(
                self.loss_history
            )

        # 3. Spike 요약
        report["spikes"] = self.spike_monitor.get_summary()

        # 4. 메모리 상태
        if torch.cuda.is_available():
            report["memory"] = {
                "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
                "peak_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
            }

        # 5. Gradient 상태
        grad_norms = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norms.append(
                    (name, param.grad.norm().item())
                )

        if grad_norms:
            norms = [n for _, n in grad_norms]
            report["gradients"] = {
                "mean_norm": sum(norms) / len(norms),
                "max_norm": max(norms),
                "min_norm": min(norms),
                "num_zero": sum(1 for n in norms if n == 0),
            }

        logger.info("=== 종합 진단 보고서 ===")
        for section, data in report.items():
            logger.info(f"[{section}] {data}")

        return report


class DebugCallback(TrainerCallback):
    """TrainingDebugger를 Trainer에 연결하는 콜백."""

    def __init__(self, model: torch.nn.Module):
        self.debugger = TrainingDebugger(model)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            result = self.debugger.on_step(state.global_step, logs["loss"])
            for alert in result.get("alerts", []):
                if alert["severity"] == "critical":
                    control.should_training_stop = True

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            train_loss = metrics.get("train_loss", 0)
            eval_loss = metrics.get("eval_loss", 0)
            self.debugger.on_eval(state.global_step, train_loss, eval_loss)

    def on_train_end(self, args, state, control, **kwargs):
        self.debugger.full_diagnosis()
```

---

## 빠른 참조: 문제별 응급 대응

```
Loss NaN?
  → LR 1/10 줄이기 → BF16 확인 → 데이터 NaN 체크 → grad_norm=1.0

OOM?
  → batch_size 절반 → gradient_checkpointing=True → max_seq_len 줄이기

Loss Spike?
  → warmup_ratio 높이기 → max_grad_norm=1.0 → 해당 배치 데이터 확인

수렴 안 됨?
  → LR Finder 재실행 → trainable params 확인 → 데이터 라벨 확인

과적합?
  → Early Stopping → LoRA rank 줄이기 → weight_decay 높이기
```

---

## 용어 체크리스트

학습 전 아래 용어를 모두 설명할 수 있는지 확인하라:

- [ ] **Loss NaN**: Loss가 NaN(Not a Number)이 되는 현상. 학습률 과대, 데이터 이상, 수치 오버플로우가 주요 원인.
- [ ] **OOM (Out of Memory)**: GPU 메모리 부족 오류. Batch size 줄이기, Gradient Checkpointing이 1차 대응.
- [ ] **Training Instability**: Loss spike 등 학습이 불안정한 현상. Warmup 부족, Gradient Clipping 미설정이 주요 원인.
- [ ] **Divergence**: Loss가 지속적으로 증가하는 현상. 학습률 과대가 가장 흔한 원인.
- [ ] **Gradient Clipping**: Gradient norm이 임계값 초과 시 스케일링. `max_grad_norm=1.0`이 표준.
- [ ] **Mixed Precision**: FP16/BF16과 FP32 혼합 학습. BF16이 FP16보다 수치적으로 안정적.
- [ ] **Loss Spike**: Loss가 갑자기 급등하는 현상. 대부분 일시적이나 빈번하면 설정 점검 필요.
- [ ] **Overfitting**: Train loss만 감소하고 Eval loss가 증가하는 현상. 모델이 학습 데이터를 암기하는 것.
- [ ] **Z-score**: 데이터 포인트가 평균에서 표준편차 몇 배만큼 떨어져 있는지 나타내는 지표. 이상치 탐지에 사용.
- [ ] **Binary Search**: 탐색 범위를 반씩 줄여가는 효율적 탐색 알고리즘. 최대 batch size 탐색 등에 활용.
