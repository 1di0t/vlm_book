# 11.4 모니터링 설정

## 핵심 용어 박스

| 용어 | 정의 |
|------|------|
| **WandB (Weights & Biases)** | ML 실험 추적 플랫폼. loss, gradient, learning rate 등의 메트릭을 실시간으로 시각화하고, 하이퍼파라미터 비교와 팀 협업을 지원한다. |
| **TensorBoard** | Google이 만든 시각화 도구. 학습 메트릭을 로컬에서 확인할 수 있다. WandB보다 가볍지만 클라우드 기능이 제한적이다. |
| **Loss Curve** | 학습 진행에 따른 loss 값의 변화 그래프. 수렴 여부, 과적합, 학습 불안정성을 진단하는 가장 기본적인 지표. |
| **Gradient Norm** | 모든 파라미터 gradient의 L2 norm. $\|g\|_2 = \sqrt{\sum_i g_i^2}$. 학습 안정성의 핵심 지표이며, 급증하면 explosion, 0으로 수렴하면 vanishing이다. |
| **Learning Rate Schedule** | 학습 중 learning rate의 변화 패턴. 실제로 의도한 대로 감소/증가하는지 모니터링해야 한다. |
| **Gradient Explosion** | Gradient norm이 급격히 증가하여 파라미터 업데이트가 비정상적으로 커지는 현상. loss spike나 NaN의 원인. |
| **Gradient Vanishing** | Gradient norm이 0에 수렴하여 파라미터가 거의 업데이트되지 않는 현상. 학습이 멈춘 것처럼 보인다. |

---

## 개요

학습 모니터링은 "학습이 잘 되고 있는지"를 실시간으로 확인하는 과정이다. GPU를 수십 시간 돌려놓고 결과가 나쁘면 자원 낭비가 심각하다. 모니터링을 통해 **이상 징후를 조기에 감지**하고 학습을 중단하거나 하이퍼파라미터를 조정해야 한다.

모니터링할 핵심 메트릭:

| 메트릭 | 정상 범위 | 이상 징후 | 대응 |
|--------|-----------|-----------|------|
| **Train Loss** | 단조 감소 | 급증(spike), 정체(plateau) | LR 조정, 데이터 확인 |
| **Eval Loss** | Train loss와 유사하게 감소 | Train과 괴리 증가 | Early stopping, regularization |
| **Gradient Norm** | 0.1 ~ 10 (모델 의존) | > 100 또는 < 1e-6 | Grad clipping, LR 감소 |
| **Learning Rate** | 의도한 스케줄 따름 | 비정상 패턴 | Config 확인 |
| **GPU Memory** | 안정적 | 점진적 증가 (leak) | 코드 버그 확인 |
| **Throughput** | 안정적 (tokens/s) | 급격한 감소 | I/O 병목, 통신 문제 |

---

## 수학적 원리

### 1. 학습 이상 징후의 수학적 정의

**Gradient Explosion 감지:**

Gradient norm을 $g_t = \|\nabla_\theta \mathcal{L}(\theta_t)\|_2$로 정의할 때, 스텝 $t$에서 explosion은:

$$g_t > \alpha \cdot \text{EMA}(g, t)$$

여기서 $\text{EMA}(g, t)$는 gradient norm의 지수 이동 평균이고 $\alpha$는 임계 배수(보통 5~10).

$$\text{EMA}(g, t) = \beta \cdot \text{EMA}(g, t-1) + (1-\beta) \cdot g_t, \quad \beta = 0.99$$

**Gradient Vanishing 감지:**

$$g_t < \epsilon_{\text{vanish}}, \quad \epsilon_{\text{vanish}} \approx 10^{-7}$$

또는 일정 구간 동안 gradient norm이 지속적으로 감소:

$$\frac{g_t}{g_{t-K}} < \delta, \quad \delta = 0.01, \quad K = 100 \text{ steps}$$

**Loss Plateau 감지:**

최근 $W$ 스텝 동안 loss 변화율이 임계값 이하:

$$\frac{|\mathcal{L}_{t} - \mathcal{L}_{t-W}|}{|\mathcal{L}_{t-W}| + \epsilon} < \tau_{\text{plateau}}$$

여기서 $\tau_{\text{plateau}} \approx 0.001$ (0.1% 미만 변화).

**Loss Spike 감지:**

현재 loss가 최근 이동 평균 대비 급증:

$$\frac{\mathcal{L}_t}{\text{EMA}(\mathcal{L}, t)} > \gamma_{\text{spike}}$$

$\gamma_{\text{spike}} \approx 2.0$ (EMA 대비 2배 이상이면 spike).

### 2. Gradient Norm Clipping의 수학

Gradient clipping은 gradient 벡터의 norm이 임계값 $c$를 초과하면 스케일링한다:

$$\hat{g} = \begin{cases} g & \text{if } \|g\|_2 \leq c \\ c \cdot \frac{g}{\|g\|_2} & \text{if } \|g\|_2 > c \end{cases}$$

이 연산은 gradient의 **방향은 보존**하면서 **크기만 제한**한다. $c$는 보통 1.0으로 설정.

Clipping 전후의 gradient norm을 비교하면 clipping이 얼마나 자주 발동하는지 알 수 있다:

$$\text{clip\_ratio} = \frac{\text{count}(\|g\|_2 > c)}{\text{total steps}}$$

이 비율이 50%를 넘으면 learning rate가 너무 큰 것이다.

### 3. Exponential Moving Average (EMA)

메트릭의 단기 노이즈를 제거하고 추세를 파악하기 위해 EMA를 사용한다:

$$\text{EMA}_t = \beta \cdot \text{EMA}_{t-1} + (1 - \beta) \cdot x_t$$

편향 보정:

$$\widehat{\text{EMA}}_t = \frac{\text{EMA}_t}{1 - \beta^t}$$

$\beta = 0.99$이면 최근 약 100 스텝의 가중 평균, $\beta = 0.999$이면 약 1000 스텝이다.

---

## 코드: WandB 콜백 설정

### 기본 WandB 통합

```python
import os
import wandb
from transformers import TrainerCallback
from typing import Optional
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WandBSetup:
    """WandB 초기화 및 설정."""

    @staticmethod
    def init(
        project: str,
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        group: Optional[str] = None,
    ):
        """WandB 실험 초기화.

        환경변수 WANDB_API_KEY가 설정되어 있어야 한다.
        """
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            logger.warning(
                "WANDB_API_KEY 환경변수가 없다. "
                "wandb login 또는 환경변수를 설정해라."
            )
            wandb.init(mode="offline")
            return

        wandb.init(
            project=project,
            name=run_name,
            config=config,
            tags=tags or [],
            group=group,
            reinit=True,
        )
        logger.info(f"WandB 초기화: {project}/{run_name}")

    @staticmethod
    def finish():
        """WandB 실험 종료."""
        wandb.finish()
```

### 커스텀 모니터링 콜백

```python
class TrainingMonitorCallback(TrainerCallback):
    """학습 모니터링 콜백.

    이상 징후를 감지하고 WandB에 커스텀 메트릭을 로깅한다.
    """

    def __init__(
        self,
        grad_norm_threshold: float = 10.0,
        loss_spike_factor: float = 2.0,
        loss_plateau_window: int = 200,
        loss_plateau_threshold: float = 0.001,
        grad_vanish_threshold: float = 1e-7,
        ema_beta: float = 0.99,
        alert_webhook: Optional[str] = None,
    ):
        self.grad_norm_threshold = grad_norm_threshold
        self.loss_spike_factor = loss_spike_factor
        self.loss_plateau_window = loss_plateau_window
        self.loss_plateau_threshold = loss_plateau_threshold
        self.grad_vanish_threshold = grad_vanish_threshold
        self.ema_beta = ema_beta
        self.alert_webhook = alert_webhook

        # 내부 상태
        self.loss_history: list[float] = []
        self.grad_norm_history: list[float] = []
        self.loss_ema: Optional[float] = None
        self.grad_norm_ema: Optional[float] = None
        self.anomaly_count = 0
        self.consecutive_plateaus = 0

    def _update_ema(self, current: float, ema: Optional[float]) -> float:
        """EMA 업데이트."""
        if ema is None:
            return current
        return self.ema_beta * ema + (1 - self.ema_beta) * current

    def on_log(self, args, state, control, logs=None, **kwargs):
        """매 로깅 스텝마다 호출."""
        if logs is None:
            return

        step = state.global_step
        custom_metrics = {}

        # --- Loss 모니터링 ---
        loss = logs.get("loss")
        if loss is not None:
            self.loss_history.append(loss)
            self.loss_ema = self._update_ema(loss, self.loss_ema)

            custom_metrics["monitor/loss_ema"] = self.loss_ema
            custom_metrics["monitor/loss_raw"] = loss

            # Loss Spike 감지
            if self.loss_ema and loss > self.loss_spike_factor * self.loss_ema:
                self.anomaly_count += 1
                msg = (
                    f"[ALERT] Loss Spike at step {step}: "
                    f"{loss:.4f} (EMA: {self.loss_ema:.4f}, "
                    f"{loss / self.loss_ema:.1f}x)"
                )
                logger.warning(msg)
                custom_metrics["monitor/loss_spike"] = 1
                self._send_alert(msg)
            else:
                custom_metrics["monitor/loss_spike"] = 0

            # Loss Plateau 감지
            if len(self.loss_history) >= self.loss_plateau_window:
                old_loss = self.loss_history[-self.loss_plateau_window]
                change_rate = abs(loss - old_loss) / (abs(old_loss) + 1e-8)
                custom_metrics["monitor/loss_change_rate"] = change_rate

                if change_rate < self.loss_plateau_threshold:
                    self.consecutive_plateaus += 1
                    custom_metrics["monitor/plateau_count"] = self.consecutive_plateaus

                    if self.consecutive_plateaus >= 5:
                        msg = (
                            f"[ALERT] Loss Plateau at step {step}: "
                            f"변화율 {change_rate:.6f} "
                            f"(연속 {self.consecutive_plateaus}회)"
                        )
                        logger.warning(msg)
                        self._send_alert(msg)
                else:
                    self.consecutive_plateaus = 0

        # --- Gradient Norm 모니터링 ---
        grad_norm = logs.get("grad_norm")
        if grad_norm is not None:
            self.grad_norm_history.append(grad_norm)
            self.grad_norm_ema = self._update_ema(grad_norm, self.grad_norm_ema)

            custom_metrics["monitor/grad_norm_ema"] = self.grad_norm_ema
            custom_metrics["monitor/grad_norm_raw"] = grad_norm

            # Gradient Explosion 감지
            if grad_norm > self.grad_norm_threshold:
                self.anomaly_count += 1
                msg = (
                    f"[ALERT] Gradient Explosion at step {step}: "
                    f"grad_norm={grad_norm:.4f} "
                    f"(threshold: {self.grad_norm_threshold})"
                )
                logger.warning(msg)
                custom_metrics["monitor/grad_explosion"] = 1
                self._send_alert(msg)
            else:
                custom_metrics["monitor/grad_explosion"] = 0

            # Gradient Vanishing 감지
            if grad_norm < self.grad_vanish_threshold:
                self.anomaly_count += 1
                msg = (
                    f"[ALERT] Gradient Vanishing at step {step}: "
                    f"grad_norm={grad_norm:.2e} "
                    f"(threshold: {self.grad_vanish_threshold:.2e})"
                )
                logger.warning(msg)
                custom_metrics["monitor/grad_vanishing"] = 1
                self._send_alert(msg)
            else:
                custom_metrics["monitor/grad_vanishing"] = 0

        # --- Learning Rate 모니터링 ---
        lr = logs.get("learning_rate")
        if lr is not None:
            custom_metrics["monitor/learning_rate"] = lr

        # --- 총 이상 징후 수 ---
        custom_metrics["monitor/total_anomalies"] = self.anomaly_count

        # WandB 로깅
        if wandb.run is not None:
            wandb.log(custom_metrics, step=step)

    def _send_alert(self, message: str):
        """이상 징후 시 알림 전송."""
        if self.alert_webhook:
            try:
                import requests
                requests.post(
                    self.alert_webhook,
                    json={"text": message},
                    timeout=5,
                )
            except Exception as e:
                logger.error(f"알림 전송 실패: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        """학습 종료 시 요약."""
        logger.info(f"학습 완료. 총 이상 징후: {self.anomaly_count}회")
        if wandb.run is not None:
            wandb.summary["total_anomalies"] = self.anomaly_count
            wandb.summary["final_loss"] = (
                self.loss_history[-1] if self.loss_history else None
            )
```

---

## 코드: 커스텀 메트릭 로깅

### GPU 메모리 모니터

```python
class GPUMemoryMonitor:
    """GPU 메모리 사용량 모니터링."""

    @staticmethod
    def get_memory_stats(device_id: int = 0) -> dict:
        """현재 GPU 메모리 상태."""
        if not torch.cuda.is_available():
            return {}

        torch.cuda.synchronize(device_id)
        allocated = torch.cuda.memory_allocated(device_id) / 1e9
        reserved = torch.cuda.memory_reserved(device_id) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(device_id) / 1e9
        total = torch.cuda.get_device_properties(device_id).total_mem / 1e9

        return {
            "gpu/memory_allocated_gb": allocated,
            "gpu/memory_reserved_gb": reserved,
            "gpu/memory_peak_gb": max_allocated,
            "gpu/memory_total_gb": total,
            "gpu/memory_utilization": allocated / total * 100,
        }

    @staticmethod
    def log_memory(step: int, device_id: int = 0):
        """메모리 상태를 WandB에 로깅."""
        stats = GPUMemoryMonitor.get_memory_stats(device_id)
        if stats and wandb.run is not None:
            wandb.log(stats, step=step)
        return stats
```

### Layer별 Gradient 분석

```python
class LayerGradientAnalyzer:
    """레이어별 gradient 분석.

    어떤 레이어가 학습이 잘 되고 있고, 어디서 gradient가 소실/폭발하는지 진단한다.
    """

    def __init__(self, model, log_interval: int = 100):
        self.model = model
        self.log_interval = log_interval
        self.step_count = 0

    def analyze_and_log(self, step: int):
        """레이어별 gradient norm 계산 및 로깅."""
        self.step_count += 1
        if self.step_count % self.log_interval != 0:
            return

        layer_stats = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                param_norm = param.data.norm(2).item()

                # 업데이트 비율: gradient가 파라미터 대비 얼마나 큰지
                update_ratio = grad_norm / (param_norm + 1e-8)

                # 간결한 이름으로 변환
                short_name = self._shorten_name(name)
                layer_stats[f"layer_grad/{short_name}"] = grad_norm
                layer_stats[f"layer_ratio/{short_name}"] = update_ratio

        if wandb.run is not None:
            wandb.log(layer_stats, step=step)

        # 가장 큰/작은 gradient 출력
        if layer_stats:
            grad_items = {
                k: v for k, v in layer_stats.items()
                if k.startswith("layer_grad/")
            }
            if grad_items:
                max_layer = max(grad_items, key=grad_items.get)
                min_layer = min(grad_items, key=grad_items.get)
                logger.info(
                    f"[Step {step}] "
                    f"Max grad: {max_layer}={grad_items[max_layer]:.6f} | "
                    f"Min grad: {min_layer}={grad_items[min_layer]:.6f}"
                )

    @staticmethod
    def _shorten_name(name: str) -> str:
        """파라미터 이름을 간결하게."""
        # model.layers.15.self_attn.q_proj.weight → L15.attn.q
        parts = name.split(".")
        short_parts = []
        for p in parts:
            if p.isdigit():
                short_parts.append(f"L{p}")
            elif p in ("self_attn", "attention"):
                short_parts.append("attn")
            elif p in ("mlp", "feed_forward"):
                short_parts.append("mlp")
            elif "_proj" in p:
                short_parts.append(p.replace("_proj", ""))
            elif p in ("weight", "bias"):
                short_parts.append(p[0])
            elif p in ("model", "layers", "decoder"):
                continue
            else:
                short_parts.append(p[:4])
        return ".".join(short_parts)
```

### Throughput 모니터

```python
import time


class ThroughputMonitor:
    """학습 처리량 모니터링."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.total_tokens: int = 0
        self.total_samples: int = 0
        self.step_times: list[float] = []
        self._step_start: Optional[float] = None

    def start_step(self):
        """스텝 시작 시각 기록."""
        self._step_start = time.perf_counter()

    def end_step(self, num_tokens: int, batch_size: int):
        """스텝 종료. 처리량 계산."""
        if self._step_start is None:
            return

        elapsed = time.perf_counter() - self._step_start
        self.step_times.append(elapsed)
        self.total_tokens += num_tokens
        self.total_samples += batch_size

        if self.start_time is None:
            self.start_time = self._step_start

    def get_stats(self, step: int) -> dict:
        """현재까지의 처리량 통계."""
        if not self.step_times:
            return {}

        total_elapsed = time.perf_counter() - self.start_time
        recent_times = self.step_times[-100:]  # 최근 100스텝

        return {
            "throughput/tokens_per_sec": self.total_tokens / total_elapsed,
            "throughput/samples_per_sec": self.total_samples / total_elapsed,
            "throughput/step_time_ms": sum(recent_times) / len(recent_times) * 1000,
            "throughput/total_tokens": self.total_tokens,
            "throughput/total_time_min": total_elapsed / 60,
        }

    def log(self, step: int):
        """WandB에 로깅."""
        stats = self.get_stats(step)
        if stats and wandb.run is not None:
            wandb.log(stats, step=step)
        return stats
```

---

## 코드: 알림 설정

### Slack Webhook 알림

```python
import requests
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class AlertConfig:
    """알림 설정."""
    slack_webhook_url: Optional[str] = None
    discord_webhook_url: Optional[str] = None
    alert_on_loss_spike: bool = True
    alert_on_grad_explosion: bool = True
    alert_on_grad_vanishing: bool = True
    alert_on_training_complete: bool = True
    alert_on_nan: bool = True
    cooldown_seconds: int = 300  # 같은 유형 알림 최소 간격


class AlertManager:
    """학습 이상 징후 알림 관리자."""

    def __init__(self, config: AlertConfig):
        self.config = config
        self._last_alert_time: dict[str, float] = {}

    def _can_send(self, alert_type: str) -> bool:
        """쿨다운 체크."""
        import time
        now = time.time()
        last = self._last_alert_time.get(alert_type, 0)
        if now - last < self.config.cooldown_seconds:
            return False
        self._last_alert_time[alert_type] = now
        return True

    def send_slack(self, message: str, alert_type: str = "general"):
        """Slack 알림 전송."""
        if not self.config.slack_webhook_url:
            return
        if not self._can_send(alert_type):
            return

        try:
            payload = {
                "text": message,
                "username": "Training Monitor",
                "icon_emoji": ":robot_face:",
            }
            response = requests.post(
                self.config.slack_webhook_url,
                json=payload,
                timeout=10,
            )
            if response.status_code != 200:
                logger.error(f"Slack 알림 실패: {response.status_code}")
        except Exception as e:
            logger.error(f"Slack 알림 에러: {e}")

    def send_discord(self, message: str, alert_type: str = "general"):
        """Discord 알림 전송."""
        if not self.config.discord_webhook_url:
            return
        if not self._can_send(alert_type):
            return

        try:
            payload = {"content": message}
            response = requests.post(
                self.config.discord_webhook_url,
                json=payload,
                timeout=10,
            )
            if response.status_code not in (200, 204):
                logger.error(f"Discord 알림 실패: {response.status_code}")
        except Exception as e:
            logger.error(f"Discord 알림 에러: {e}")

    def alert(self, message: str, alert_type: str = "general"):
        """모든 설정된 채널로 알림."""
        self.send_slack(message, alert_type)
        self.send_discord(message, alert_type)


def create_alert_config_from_env() -> AlertConfig:
    """환경변수에서 알림 설정 로드."""
    return AlertConfig(
        slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL"),
    )
```

### 종합 모니터링 콜백 (전체 통합)

```python
class ComprehensiveMonitorCallback(TrainerCallback):
    """모든 모니터링을 통합한 콜백.

    사용법:
        trainer = Trainer(
            ...,
            callbacks=[
                ComprehensiveMonitorCallback(
                    project="ocr-vlm-finetune",
                    run_name="qwen25vl-lora-r32",
                    alert_config=create_alert_config_from_env(),
                )
            ],
        )
    """

    def __init__(
        self,
        project: str = "vlm-finetune",
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
        alert_config: Optional[AlertConfig] = None,
        grad_analysis_interval: int = 100,
    ):
        self.project = project
        self.run_name = run_name
        self.run_config = config

        # 하위 모니터
        self.training_monitor = TrainingMonitorCallback(
            alert_webhook=None,  # AlertManager가 대신 처리
        )
        self.gpu_monitor = GPUMemoryMonitor()
        self.throughput_monitor = ThroughputMonitor()
        self.grad_analyzer: Optional[LayerGradientAnalyzer] = None
        self.grad_analysis_interval = grad_analysis_interval

        # 알림
        self.alert_manager = (
            AlertManager(alert_config) if alert_config else None
        )

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """학습 시작 시 초기화."""
        WandBSetup.init(
            project=self.project,
            run_name=self.run_name,
            config=self.run_config,
        )

        if model is not None:
            self.grad_analyzer = LayerGradientAnalyzer(
                model, log_interval=self.grad_analysis_interval
            )

        if self.alert_manager:
            self.alert_manager.alert(
                f"학습 시작: {self.project}/{self.run_name}",
                alert_type="start",
            )

    def on_step_begin(self, args, state, control, **kwargs):
        """스텝 시작."""
        self.throughput_monitor.start_step()

    def on_step_end(self, args, state, control, **kwargs):
        """스텝 종료."""
        step = state.global_step

        # Throughput 기록 (토큰 수는 근사)
        self.throughput_monitor.end_step(
            num_tokens=args.per_device_train_batch_size * 2048,  # 근사값
            batch_size=args.per_device_train_batch_size,
        )

        # GPU 메모리 로깅 (100스텝마다)
        if step % 100 == 0:
            self.gpu_monitor.log_memory(step)
            self.throughput_monitor.log(step)

        # 레이어별 gradient 분석
        if self.grad_analyzer:
            self.grad_analyzer.analyze_and_log(step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """메트릭 로깅 시."""
        # 기본 모니터링 콜백 호출
        self.training_monitor.on_log(args, state, control, logs=logs, **kwargs)

        # 이상 징후 알림
        if self.alert_manager and logs:
            loss = logs.get("loss")
            grad_norm = logs.get("grad_norm")
            step = state.global_step

            if loss is not None and (
                torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss))
            ):
                self.alert_manager.alert(
                    f"[NaN/Inf Loss] Step {step}: loss={loss}",
                    alert_type="nan",
                )

            if grad_norm is not None and grad_norm > 100:
                self.alert_manager.alert(
                    f"[Gradient Explosion] Step {step}: grad_norm={grad_norm:.2f}",
                    alert_type="grad_explosion",
                )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """평가 시."""
        if metrics and wandb.run is not None:
            eval_metrics = {
                f"eval/{k}": v for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            wandb.log(eval_metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """학습 종료."""
        self.training_monitor.on_train_end(args, state, control, **kwargs)

        # 최종 요약
        summary = {
            "final_step": state.global_step,
            "total_anomalies": self.training_monitor.anomaly_count,
        }

        if self.throughput_monitor.start_time:
            stats = self.throughput_monitor.get_stats(state.global_step)
            summary.update(stats)

        if wandb.run is not None:
            for k, v in summary.items():
                wandb.summary[k] = v

        # 알림
        if self.alert_manager:
            msg = (
                f"학습 완료: {self.project}/{self.run_name}\n"
                f"총 스텝: {state.global_step}\n"
                f"이상 징후: {self.training_monitor.anomaly_count}회"
            )
            self.alert_manager.alert(msg, alert_type="complete")

        WandBSetup.finish()
```

---

## 코드: TensorBoard 설정

WandB 대신 TensorBoard를 사용하는 경우:

```python
from torch.utils.tensorboard import SummaryWriter


class TensorBoardMonitor:
    """TensorBoard 기반 모니터링."""

    def __init__(self, log_dir: str = "./tb_logs"):
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard 로그 디렉토리: {log_dir}")
        logger.info(f"실행: tensorboard --logdir {log_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_values: dict, step: int):
        self.writer.add_scalars(main_tag, tag_values, step)

    def log_histogram(self, tag: str, values, step: int):
        """파라미터 또는 gradient의 분포를 히스토그램으로."""
        self.writer.add_histogram(tag, values, step)

    def log_training_step(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        lr: float,
        **kwargs,
    ):
        """학습 스텝 메트릭 일괄 로깅."""
        self.log_scalar("train/loss", loss, step)
        self.log_scalar("train/grad_norm", grad_norm, step)
        self.log_scalar("train/learning_rate", lr, step)

        for key, value in kwargs.items():
            self.log_scalar(f"train/{key}", value, step)

    def log_model_gradients(self, model, step: int):
        """모델의 모든 gradient를 히스토그램으로 로깅."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.log_histogram(f"gradients/{name}", param.grad.cpu(), step)
                self.log_histogram(f"parameters/{name}", param.data.cpu(), step)

    def close(self):
        self.writer.close()
```

---

## 모니터링 대시보드 구성 가이드

### WandB 대시보드 권장 패널 구성

```
Row 1: 핵심 메트릭
├── Train Loss (line chart, smoothing=0.8)
├── Eval Loss (line chart)
└── Learning Rate (line chart)

Row 2: 안정성 지표
├── Gradient Norm (line chart, log scale)
├── Loss Spike 이벤트 (scatter)
└── Gradient Explosion/Vanishing (binary)

Row 3: 시스템 메트릭
├── GPU Memory Utilization (%)
├── Throughput (tokens/sec)
└── Step Time (ms)

Row 4: 레이어 분석 (필요 시)
├── Layer-wise Gradient Norm (heatmap)
└── Update Ratio by Layer (line chart)
```

### WandB Alert Rules (웹 UI에서 설정)

```python
# WandB 웹에서 Alerts 탭 → New Alert 클릭
# 또는 Python API로:

def setup_wandb_alerts():
    """WandB 자체 알림 규칙 설정 (참고용)."""
    # wandb.alert()는 run 내에서 직접 호출
    wandb.alert(
        title="Training Anomaly",
        text="Loss spike detected",
        level=wandb.AlertLevel.WARN,
        wait_duration=300,  # 5분 쿨다운
    )
```

---

## 실전 트러블슈팅

### 증상별 대응 가이드

| 증상 | 원인 | 확인 메트릭 | 대응 |
|------|------|------------|------|
| Loss가 NaN | Gradient explosion, FP16 overflow | `grad_norm`, `loss_scale` | LR 줄이기, BF16 전환, grad clipping 강화 |
| Loss가 감소 안 함 | LR 너무 작음, 데이터 문제 | `learning_rate`, 데이터 샘플 확인 | LR 키우기, 데이터 전처리 재점검 |
| Loss가 진동 | LR 너무 큼, batch size 너무 작음 | `loss` 변동폭 | LR 줄이기, batch size 키우기 |
| Eval loss만 증가 | 과적합 | `train_loss` vs `eval_loss` 갭 | Early stopping, dropout 증가, 데이터 증강 |
| GPU OOM | 배치/시퀀스 너무 큼 | `gpu/memory_peak_gb` | batch size 줄이기, gradient checkpointing |
| 학습 속도 저하 | I/O 병목, 통신 오버헤드 | `throughput/step_time_ms` | DataLoader workers 증가, 통신 최적화 |

### NaN 디버깅 코드

```python
def debug_nan_loss(model, batch, step: int):
    """NaN loss 발생 시 디버깅."""
    logger.error(f"NaN loss at step {step}. 디버깅 시작...")

    # 1. 입력 데이터 확인
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            has_nan = torch.isnan(value).any().item()
            has_inf = torch.isinf(value).any().item()
            logger.info(
                f"Input '{key}': shape={value.shape}, "
                f"dtype={value.dtype}, nan={has_nan}, inf={has_inf}"
            )

    # 2. 파라미터 확인
    nan_params = []
    inf_params = []
    for name, param in model.named_parameters():
        if torch.isnan(param.data).any():
            nan_params.append(name)
        if torch.isinf(param.data).any():
            inf_params.append(name)

    if nan_params:
        logger.error(f"NaN 파라미터: {nan_params[:10]}")
    if inf_params:
        logger.error(f"Inf 파라미터: {inf_params[:10]}")

    # 3. Gradient 확인
    if any(p.grad is not None for p in model.parameters()):
        nan_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                nan_grads.append(name)
        if nan_grads:
            logger.error(f"NaN gradient: {nan_grads[:10]}")

    # 4. Loss scale 확인 (FP16)
    logger.info("FP16 사용 시 Loss Scale을 확인해라. BF16 전환을 고려해라.")
```

---

## 용어 체크리스트

학습 후 아래 용어를 설명할 수 있는지 점검해봐라:

- [ ] **WandB**: 프로젝트, run, config, summary의 구조를 이해하고 설정할 수 있는가?
- [ ] **Loss Curve**: 정상적인 수렴 패턴과 과적합/발산 패턴을 구분할 수 있는가?
- [ ] **Gradient Norm**: L2 norm의 의미와 정상 범위를 아는가?
- [ ] **EMA**: 지수 이동 평균의 수식과 $\beta$ 값에 따른 smoothing 정도를 설명할 수 있는가?
- [ ] **Loss Spike**: 감지 조건(EMA 대비 배수)과 발생 원인을 설명할 수 있는가?
- [ ] **Loss Plateau**: 감지 조건(변화율 임계값)과 대응 방법을 아는가?
- [ ] **Gradient Explosion**: 발생 원인과 gradient clipping의 수학적 원리를 설명할 수 있는가?
- [ ] **Gradient Vanishing**: 감지 방법과 해결 전략(residual connection 등)을 아는가?
- [ ] **GPU Memory Monitoring**: allocated, reserved, peak의 차이를 아는가?
- [ ] **Throughput**: tokens/sec 계산법과 병목 원인(I/O, 통신, 연산)을 구분할 수 있는가?
