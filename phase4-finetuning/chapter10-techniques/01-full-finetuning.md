# 10.1 Full Fine-tuning

## 핵심 용어 박스

| 용어 | 정의 |
|------|------|
| **Full Fine-tuning** | 사전학습된 모델의 **모든 파라미터**를 타겟 태스크 데이터로 업데이트하는 전이학습 방법. 파라미터 수가 수십억 개일 경우 메모리와 연산 비용이 매우 크다. |
| **Catastrophic Forgetting** | 새로운 태스크를 학습하면서 기존에 학습했던 지식(이전 태스크의 loss landscape)이 **파괴적으로 손실**되는 현상. Fine-tuning의 근본적 한계. |
| **Learning Rate Warmup** | 학습 초기에 learning rate를 0에서 목표값까지 **점진적으로 증가**시키는 기법. Pre-trained weight를 급격히 변형하는 것을 방지한다. |
| **Weight Decay** | L2 정규화의 구현 형태. 파라미터 크기에 비례하는 페널티를 부과하여 과적합을 방지하고, 사전학습 가중치에서 과도하게 벗어나지 않도록 제약한다. |
| **Gradient Accumulation** | 미니배치를 여러 스텝에 걸쳐 누적한 뒤 한 번에 업데이트하는 기법. GPU 메모리가 부족할 때 effective batch size를 키울 수 있다. |

---

## 개요

Full Fine-tuning은 가장 직관적인 전이학습 전략이다. 사전학습된 모델의 모든 레이어, 모든 파라미터를 타겟 도메인 데이터로 업데이트한다. 파라미터 효율적 기법(LoRA, QLoRA 등)이 등장하기 전까지 사실상 유일한 fine-tuning 방법이었다.

**장점:**
- 모델의 전체 용량을 태스크에 최적화할 수 있어 이론적으로 가장 높은 성능 달성 가능
- 구현이 단순하고 이해하기 쉬움
- 도메인 간 gap이 클 때(예: 자연어 → 의료 영상) 효과적

**단점:**
- 모든 파라미터의 gradient를 저장해야 하므로 메모리 비용이 막대함
- 7B 모델 기준 학습 시 최소 ~56GB GPU 메모리 필요 (AdamW optimizer states 포함)
- Catastrophic Forgetting 위험
- 모델 체크포인트 크기가 원본과 동일

---

## 수학적 원리

### 1. 전체 파라미터 업데이트 (Vanilla SGD)

Full Fine-tuning의 핵심은 모든 파라미터 $\theta$에 대해 gradient descent를 수행하는 것이다.

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

여기서:
- $\theta_t$: 시점 $t$에서의 전체 모델 파라미터 벡터
- $\eta$: learning rate (학습률)
- $\nabla_\theta \mathcal{L}(\theta_t)$: 손실 함수의 파라미터에 대한 gradient

미니배치 $\mathcal{B}$를 사용하면:

$$\nabla_\theta \mathcal{L}(\theta_t) \approx \frac{1}{|\mathcal{B}|} \sum_{(x_i, y_i) \in \mathcal{B}} \nabla_\theta \ell(f_\theta(x_i), y_i)$$

### 2. AdamW Optimizer

실제 Fine-tuning에서는 거의 항상 AdamW를 사용한다. AdamW는 Adam에서 weight decay를 gradient가 아닌 파라미터에 직접 적용하는 변형이다.

**1차 모멘트 (평균):**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

**2차 모멘트 (비중심 분산):**

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

**편향 보정:**

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**파라미터 업데이트 (AdamW):**

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

여기서 $\lambda \theta_t$ 항이 weight decay다. 기존 Adam의 L2 regularization과 달리 adaptive learning rate의 영향을 받지 않는다.

> **AdamW의 메모리 비용**: 파라미터 $\theta$ 외에 $m_t$, $v_t$를 별도로 저장해야 하므로, optimizer states만으로 파라미터의 **2배** 메모리가 추가된다. FP32 기준 7B 모델이면 파라미터 28GB + optimizer states 56GB = 총 84GB.

### 3. Weight Decay (L2 Regularization)

Weight decay는 loss에 파라미터 크기의 L2 norm을 페널티로 추가한다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \frac{\lambda}{2} \|\theta\|^2$$

이를 미분하면:

$$\nabla_\theta \mathcal{L}_{\text{total}} = \nabla_\theta \mathcal{L}_{\text{task}} + \lambda \theta$$

결과적으로 파라미터 업데이트는:

$$\theta_{t+1} = \theta_t - \eta(\nabla_\theta \mathcal{L}_{\text{task}} + \lambda \theta_t) = (1 - \eta\lambda)\theta_t - \eta \nabla_\theta \mathcal{L}_{\text{task}}$$

$(1 - \eta\lambda)$ 항이 매 스텝 파라미터를 약간씩 줄이는(decay) 역할을 한다.

Fine-tuning에서 weight decay는 두 가지 역할을 한다:
- **과적합 방지**: 파라미터가 지나치게 커지는 것을 억제
- **사전학습 가중치 보존**: 원래 가중치에서 멀리 벗어나지 않도록 암묵적 제약

### 4. Catastrophic Forgetting의 수학적 이해

태스크 A에 대해 최적화된 파라미터 $\theta_A^*$가 있다고 하자. 이후 태스크 B를 학습하면:

$$\theta_B^* = \theta_A^* - \eta \sum_{t} \nabla_\theta \mathcal{L}_B(\theta_t)$$

문제는 $\theta_B^*$에서 태스크 A의 loss가 급격히 증가한다는 것이다:

$$\mathcal{L}_A(\theta_B^*) \gg \mathcal{L}_A(\theta_A^*)$$

이를 이해하기 위해 태스크 A의 loss를 $\theta_A^*$ 주변에서 2차 Taylor 전개하면:

$$\mathcal{L}_A(\theta_B^*) \approx \mathcal{L}_A(\theta_A^*) + (\theta_B^* - \theta_A^*)^T \nabla \mathcal{L}_A(\theta_A^*) + \frac{1}{2}(\theta_B^* - \theta_A^*)^T H_A (\theta_B^* - \theta_A^*)$$

$\theta_A^*$가 태스크 A의 최적점이므로 $\nabla \mathcal{L}_A(\theta_A^*) \approx 0$이고:

$$\mathcal{L}_A(\theta_B^*) \approx \mathcal{L}_A(\theta_A^*) + \frac{1}{2} \Delta\theta^T H_A \Delta\theta$$

여기서 $H_A$는 태스크 A loss의 Hessian 행렬이다. $H_A$의 **큰 고유값 방향**으로 $\Delta\theta$가 이동하면 loss가 급격히 증가한다. 이것이 Catastrophic Forgetting의 수학적 본질이다.

> **EWC (Elastic Weight Consolidation)**는 이 Hessian을 Fisher Information Matrix로 근사하여, 중요한 파라미터의 변화를 페널티로 부과하는 방법이다: $\mathcal{L}_{\text{EWC}} = \mathcal{L}_B(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_{A,i}^*)^2$

### 5. Learning Rate Schedule 비교

Fine-tuning에서 learning rate schedule은 성능에 큰 영향을 미친다.

**Step Decay:**

$$\eta_t = \eta_0 \times \gamma^{\lfloor t / T_{\text{step}} \rfloor}$$

$T_{\text{step}}$ 에폭마다 $\gamma$ (예: 0.1)를 곱해 learning rate를 계단식으로 감소.

**Cosine Annealing:**

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

주기적으로 learning rate가 감소했다 증가하는 패턴. 부드러운 감소가 특징이며, LLM fine-tuning에서 가장 널리 사용된다.

**Linear Decay with Warmup:**

$$\eta_t = \begin{cases} \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & \text{if } t < T_{\text{warmup}} \\ \eta_{\max} \cdot \frac{T - t}{T - T_{\text{warmup}}} & \text{if } t \geq T_{\text{warmup}} \end{cases}$$

Warmup 구간에서 learning rate를 선형으로 증가시키고, 이후 선형으로 감소시킨다.

| 스케줄 | 장점 | 단점 | 권장 상황 |
|--------|------|------|-----------|
| Step Decay | 구현 간단 | 비연속적 변화, 하이퍼파라미터 많음 | 전통적 CNN fine-tuning |
| Cosine Annealing | 부드러운 수렴, warm restart 가능 | 주기 설정 필요 | LLM/VLM fine-tuning |
| Linear + Warmup | 안정적, 직관적 | 최적점 부근에서 학습률 낭비 | 짧은 fine-tuning |

---

## VLM Full Fine-tuning 시 주의사항

Vision-Language Model을 Full Fine-tuning할 때는 단순히 모든 파라미터를 동일한 학습률로 업데이트하면 안 된다.

### Vision Encoder vs LLM 학습률 분리

VLM은 크게 세 구성요소로 나뉜다:
1. **Vision Encoder** (ViT 등): 이미지 특징 추출
2. **Projection/Connector**: 비전 임베딩을 언어 공간으로 매핑
3. **LLM Backbone**: 텍스트 생성

각 구성요소의 사전학습 정도와 태스크 적합성이 다르므로 학습률을 분리하는 것이 필수다:

| 구성요소 | 권장 학습률 | 이유 |
|----------|------------|------|
| Vision Encoder | $1 \times 10^{-6}$ ~ $5 \times 10^{-6}$ | 범용 시각 특징은 대부분 유지해야 함 |
| Connector | $1 \times 10^{-4}$ ~ $5 \times 10^{-4}$ | 도메인 매핑을 새로 학습해야 함 |
| LLM Backbone | $1 \times 10^{-5}$ ~ $5 \times 10^{-5}$ | 언어 능력 보존하면서 태스크 적응 |

### 메모리 요구량 (Full Fine-tuning)

| 모델 크기 | 파라미터 (FP16) | Optimizer States (FP32) | Gradient (FP16) | 활성화 메모리 | 총 추정 |
|-----------|----------------|------------------------|-----------------|-------------|---------|
| 7B | 14 GB | 56 GB | 14 GB | ~8 GB | ~92 GB |
| 13B | 26 GB | 104 GB | 26 GB | ~15 GB | ~171 GB |
| 70B | 140 GB | 560 GB | 140 GB | ~80 GB | ~920 GB |

> Full Fine-tuning으로 7B 모델을 학습하려면 A100 80GB GPU 2개 이상이 필요하다. 70B는 사실상 데이터 병렬 + 모델 병렬이 필수.

---

## 코드: Full Fine-tuning 루프 구현

### 기본 Full Fine-tuning 루프

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from dataclasses import dataclass
from typing import Optional
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FullFinetuneConfig:
    """Full Fine-tuning 설정."""
    model_name: str = "meta-llama/Llama-2-7b-hf"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    fp16: bool = True
    output_dir: str = "./checkpoints"
    # 스케줄러 선택: "cosine", "linear", "step"
    lr_scheduler_type: str = "cosine"


class FullFinetuner:
    """Full Fine-tuning 학습 루프 구현."""

    def __init__(self, config: FullFinetuneConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 로드 (모든 파라미터 학습 가능)
        logger.info(f"모델 로드: {config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.fp16 else torch.float32,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 학습 가능한 파라미터 수 확인
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            f"전체 파라미터: {total_params:,} | "
            f"학습 가능: {trainable_params:,} | "
            f"비율: {trainable_params / total_params * 100:.2f}%"
        )

        # Mixed Precision Scaler
        self.scaler = GradScaler(enabled=config.fp16)

    def _create_optimizer(self) -> AdamW:
        """Weight Decay를 적용할 파라미터와 제외할 파라미터를 분리."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # bias, LayerNorm weight에는 weight decay 미적용
            if "bias" in name or "layernorm" in name.lower() or "layer_norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        return optimizer

    def _create_scheduler(self, optimizer: AdamW, total_steps: int):
        """Learning Rate 스케줄러 생성."""
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        if self.config.lr_scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif self.config.lr_scheduler_type == "linear":
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            raise ValueError(f"지원하지 않는 스케줄러: {self.config.lr_scheduler_type}")

        return scheduler

    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """메인 학습 루프."""
        total_steps = (
            len(train_dataloader)
            // self.config.gradient_accumulation_steps
            * self.config.num_epochs
        )

        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer, total_steps)

        logger.info(f"총 학습 스텝: {total_steps}")
        logger.info(f"Warmup 스텝: {int(total_steps * self.config.warmup_ratio)}")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")

        global_step = 0
        best_eval_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_dataloader):
                # 데이터를 GPU로 이동
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Mixed Precision Forward
                with autocast(enabled=self.config.fp16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / self.config.gradient_accumulation_steps

                # Backward (gradient 누적)
                self.scaler.scale(loss).backward()
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps

                # Gradient Accumulation 완료 시점에 업데이트
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient Clipping
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    self.scaler.step(optimizer)
                    self.scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % 50 == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        avg_loss = epoch_loss / (step + 1)
                        logger.info(
                            f"[Epoch {epoch+1}/{self.config.num_epochs}] "
                            f"Step {global_step}/{total_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {current_lr:.2e}"
                        )

            # Epoch 종료 후 평가
            if eval_dataloader is not None:
                eval_loss = self._evaluate(eval_dataloader)
                logger.info(f"[Epoch {epoch+1}] Eval Loss: {eval_loss:.4f}")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    self._save_checkpoint(epoch, global_step, eval_loss)

        return self.model

    @torch.no_grad()
    def _evaluate(self, eval_dataloader: DataLoader) -> float:
        """평가 루프."""
        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with autocast(enabled=self.config.fp16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            total_loss += outputs.loss.item()
            total_steps += 1

        return total_loss / max(total_steps, 1)

    def _save_checkpoint(self, epoch: int, step: int, eval_loss: float):
        """체크포인트 저장."""
        import os
        save_path = os.path.join(self.config.output_dir, f"checkpoint-epoch{epoch+1}-step{step}")
        os.makedirs(save_path, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"체크포인트 저장: {save_path} (eval_loss: {eval_loss:.4f})")
```

### VLM Full Fine-tuning: 구성요소별 학습률 분리

```python
class VLMFullFinetuner:
    """VLM 구성요소별 학습률을 분리한 Full Fine-tuning."""

    def __init__(
        self,
        model,
        vision_lr: float = 2e-6,
        connector_lr: float = 2e-4,
        llm_lr: float = 2e-5,
        weight_decay: float = 0.01,
    ):
        self.model = model
        self.param_groups = self._build_param_groups(
            vision_lr, connector_lr, llm_lr, weight_decay
        )

    def _build_param_groups(
        self,
        vision_lr: float,
        connector_lr: float,
        llm_lr: float,
        weight_decay: float,
    ) -> list[dict]:
        """VLM 구성요소별 파라미터 그룹 생성."""
        vision_params = []
        connector_params = []
        llm_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # 모델 구조에 따라 이름 패턴이 다를 수 있음
            if "vision" in name or "visual" in name or "vit" in name:
                vision_params.append(param)
            elif "connector" in name or "projection" in name or "mm_projector" in name:
                connector_params.append(param)
            else:
                llm_params.append(param)

        param_groups = [
            {
                "params": vision_params,
                "lr": vision_lr,
                "weight_decay": weight_decay,
                "name": "vision_encoder",
            },
            {
                "params": connector_params,
                "lr": connector_lr,
                "weight_decay": weight_decay,
                "name": "connector",
            },
            {
                "params": llm_params,
                "lr": llm_lr,
                "weight_decay": weight_decay,
                "name": "llm_backbone",
            },
        ]

        for group in param_groups:
            num_params = sum(p.numel() for p in group["params"])
            logger.info(
                f"[{group['name']}] 파라미터: {num_params:,} | LR: {group['lr']:.2e}"
            )

        return param_groups

    def get_optimizer(self) -> AdamW:
        return AdamW(self.param_groups, betas=(0.9, 0.999), eps=1e-8)
```

### Gradient Accumulation 상세 구현

```python
def train_with_gradient_accumulation(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    accumulation_steps: int = 8,
    max_grad_norm: float = 1.0,
    fp16: bool = True,
):
    """
    Gradient Accumulation 상세 구현.

    Effective batch size = batch_size × accumulation_steps
    예: batch_size=4, accumulation_steps=8 → effective=32
    """
    scaler = GradScaler(enabled=fp16)
    model.train()
    optimizer.zero_grad()

    running_loss = 0.0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["labels"].cuda()

        # Forward pass with mixed precision
        with autocast(enabled=fp16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            # loss를 accumulation steps로 나눠야 평균이 맞다
            loss = outputs.loss / accumulation_steps

        # Backward (gradient가 누적됨)
        scaler.scale(loss).backward()
        running_loss += loss.item() * accumulation_steps

        # accumulation_steps마다 실제 업데이트
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            effective_step = (step + 1) // accumulation_steps
            avg_loss = running_loss / (step + 1)

            if effective_step % 10 == 0:
                logger.info(
                    f"Step {effective_step} | Loss: {avg_loss:.4f} | "
                    f"Grad Norm: {grad_norm:.4f}"
                )
```

### Learning Rate Schedule 시각화 유틸리티

```python
import matplotlib.pyplot as plt
import numpy as np


def visualize_lr_schedules(
    total_steps: int = 1000,
    warmup_steps: int = 100,
    max_lr: float = 2e-5,
    min_lr: float = 0.0,
):
    """세 가지 LR Schedule을 시각적으로 비교."""
    steps = np.arange(total_steps)

    # 1. Linear with Warmup
    linear_lr = np.where(
        steps < warmup_steps,
        max_lr * steps / warmup_steps,
        max_lr * (total_steps - steps) / (total_steps - warmup_steps),
    )

    # 2. Cosine with Warmup
    cosine_lr = np.where(
        steps < warmup_steps,
        max_lr * steps / warmup_steps,
        min_lr
        + 0.5
        * (max_lr - min_lr)
        * (1 + np.cos(np.pi * (steps - warmup_steps) / (total_steps - warmup_steps))),
    )

    # 3. Step Decay (매 300스텝마다 0.1 감소)
    step_decay_lr = np.where(
        steps < warmup_steps,
        max_lr * steps / warmup_steps,
        max_lr * (0.1 ** (steps // 300)),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, lr, title in zip(
        axes,
        [linear_lr, cosine_lr, step_decay_lr],
        ["Linear + Warmup", "Cosine + Warmup", "Step Decay"],
    ):
        ax.plot(steps, lr, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.axvline(x=warmup_steps, color="r", linestyle="--", alpha=0.5, label="Warmup End")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lr_schedules_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
```

---

## 실전 팁

### 1. Full Fine-tuning을 선택해야 하는 경우

| 상황 | Full FT 권장 여부 | 이유 |
|------|-------------------|------|
| 타겟 도메인이 사전학습 데이터와 매우 다름 | **O** | 전체 표현을 재조정해야 함 |
| 데이터가 충분 (10K+ 샘플) | **O** | 과적합 위험이 낮음 |
| 최고 성능이 필수 | **O** | 파라미터 제한 없이 최적화 |
| GPU 메모리 제한 (< 40GB) | **X** | LoRA/QLoRA 권장 |
| 빠른 실험 반복 필요 | **X** | 학습 시간이 너무 길다 |
| 모델 서빙 시 여러 태스크 | **X** | 태스크마다 전체 체크포인트 필요 |

### 2. Catastrophic Forgetting 완화 전략

1. **작은 Learning Rate 사용**: $10^{-5}$ ~ $10^{-6}$ 범위
2. **Warmup 필수**: 전체 스텝의 5~10%
3. **Early Stopping**: validation loss 모니터링
4. **Weight Decay**: 사전학습 가중치에서 벗어나는 것을 억제
5. **데이터 혼합**: 원본 태스크 데이터를 일부 섞어 학습

---

## 용어 체크리스트

학습 후 아래 용어를 설명할 수 있는지 점검해봐라:

- [ ] **Full Fine-tuning**: 전체 파라미터 업데이트의 의미와 메모리 비용을 설명할 수 있는가?
- [ ] **Catastrophic Forgetting**: 왜 발생하는지 Hessian 관점에서 설명할 수 있는가?
- [ ] **AdamW vs Adam**: Weight Decay가 왜 L2 regularization과 다른지 구분할 수 있는가?
- [ ] **Learning Rate Warmup**: 왜 필요한지, warmup 없이 학습하면 어떤 문제가 생기는지 아는가?
- [ ] **Cosine Annealing**: 수식을 작성하고, 왜 LLM fine-tuning에서 선호되는지 설명할 수 있는가?
- [ ] **Gradient Accumulation**: Effective batch size 계산법과 loss 스케일링을 이해하는가?
- [ ] **Weight Decay**: $(1 - \eta\lambda)$ 항의 의미를 설명할 수 있는가?
- [ ] **VLM 학습률 분리**: Vision Encoder, Connector, LLM에 왜 다른 학습률을 적용하는지 아는가?
- [ ] **Mixed Precision Training**: FP16 forward + FP32 optimizer의 이유를 설명할 수 있는가?
- [ ] **Gradient Clipping**: Max grad norm이 학습 안정성에 어떤 역할을 하는지 아는가?
