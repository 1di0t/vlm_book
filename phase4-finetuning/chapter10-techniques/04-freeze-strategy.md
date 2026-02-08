# 10.4 Freeze 전략

## 핵심 용어 박스

| 용어 | 정의 |
|------|------|
| **Layer Freeze** | 모델의 특정 레이어 파라미터를 동결(`requires_grad=False`)하여 학습 대상에서 제외하는 기법. 하위 레이어(범용 특징)를 보존하고 상위 레이어(태스크 특화)만 학습한다. |
| **Progressive Unfreezing** | 학습 초기에는 최상위 레이어만 학습하고, epoch가 진행될수록 **한 레이어씩 해동(unfreeze)**하며 학습 범위를 확장하는 기법. ULMFiT(Howard & Ruder, 2018)에서 제안. |
| **Discriminative Learning Rate** | 레이어마다 **서로 다른 학습률**을 적용하는 기법. 하위 레이어에는 작은 학습률, 상위 레이어에는 큰 학습률을 부여한다. |
| **Feature Extraction** | 사전학습 모델의 모든 레이어를 동결하고, 마지막 분류/생성 헤드만 학습하는 방식. 가장 보수적인 전이학습 전략. |
| **Connector Tuning** | VLM에서 Vision Encoder와 LLM을 모두 동결하고, 중간 Projection/Connector 레이어만 학습하는 전략. 모달리티 간 매핑만 새로 학습한다. |

---

## 개요

Freeze 전략은 "어떤 레이어를 학습하고 어떤 레이어를 동결할 것인가"를 결정하는 것이다. 단순해 보이지만, 이 선택이 성능, 학습 비용, 과적합 정도에 결정적 영향을 미친다.

**핵심 원리**: Neural network의 레이어는 계층적 특징을 학습한다.

| 레이어 위치 | 학습하는 특징 | 전이 가능성 |
|------------|-------------|------------|
| 하위 레이어 (입력에 가까움) | 범용 특징 (에지, 텍스처, 토큰 임베딩) | **높음** — 대부분의 태스크에서 재사용 가능 |
| 중간 레이어 | 중수준 특징 (패턴, 구문, 의미 구조) | 중간 |
| 상위 레이어 (출력에 가까움) | 태스크 특화 특징 (분류, 생성 패턴) | **낮음** — 태스크별로 달라짐 |

따라서:
- **하위 레이어는 동결**: 범용 특징을 보존하고 과적합 방지
- **상위 레이어는 학습**: 태스크에 맞게 조정
- **점진적으로 해동**: 학습이 안정화되면 더 깊은 레이어도 학습

---

## 수학적 원리

### 1. Discriminative Learning Rate

레이어별로 다른 학습률을 적용하는 공식:

$$\eta_l = \eta_{\text{base}} \times \xi^{L - l}$$

여기서:
- $\eta_l$: 레이어 $l$의 학습률
- $\eta_{\text{base}}$: 최상위 레이어의 학습률 (base learning rate)
- $\xi \in (0, 1)$: 감쇠 계수 (decay factor), 보통 $0.8 \sim 0.95$
- $L$: 총 레이어 수
- $l$: 현재 레이어 인덱스 (0부터 시작, 0이 최하위)

**예시 (10 레이어, $\eta_{\text{base}} = 10^{-4}$, $\xi = 0.9$):**

| 레이어 $l$ | $\xi^{L-l}$ | $\eta_l$ |
|-----------|------------|---------|
| 9 (최상위) | $0.9^1 = 0.90$ | $9.0 \times 10^{-5}$ |
| 8 | $0.9^2 = 0.81$ | $8.1 \times 10^{-5}$ |
| 7 | $0.9^3 = 0.729$ | $7.29 \times 10^{-5}$ |
| 5 | $0.9^5 = 0.590$ | $5.90 \times 10^{-5}$ |
| 2 | $0.9^8 = 0.430$ | $4.30 \times 10^{-5}$ |
| 0 (최하위) | $0.9^{10} = 0.349$ | $3.49 \times 10^{-5}$ |

하위 레이어로 갈수록 학습률이 지수적으로 감소한다.

**수학적 직관**: Gradient는 역전파 시 체인룰에 의해 곱해진다. 상위 레이어의 gradient가 크고 하위 레이어의 gradient가 작은 것이 자연스러운 구조다. Discriminative LR은 이 차이를 의도적으로 강화하여, 하위 레이어의 사전학습 지식을 더 강하게 보존한다.

### 2. Progressive Unfreezing

Progressive unfreezing은 시간에 따라 학습 가능한 레이어를 확장한다.

형식적으로, 에폭 $e$에서 학습 가능한 레이어 집합 $\mathcal{S}_e$:

$$\mathcal{S}_e = \{l : l \geq L - e\}$$

즉:
- 에폭 1: $\mathcal{S}_1 = \{L\}$ (마지막 레이어만)
- 에폭 2: $\mathcal{S}_2 = \{L-1, L\}$
- 에폭 3: $\mathcal{S}_3 = \{L-2, L-1, L\}$
- ...
- 에폭 $L$: $\mathcal{S}_L = \{1, 2, \ldots, L\}$ (전체)

파라미터 업데이트 규칙:

$$\theta_l^{(t+1)} = \begin{cases} \theta_l^{(t)} - \eta_l \nabla_{\theta_l} \mathcal{L}(\theta^{(t)}) & \text{if } l \in \mathcal{S}_e \\ \theta_l^{(t)} & \text{if } l \notin \mathcal{S}_e \end{cases}$$

**한 번에 여러 레이어를 해동하는 변형:**

$$\mathcal{S}_e = \{l : l \geq L - k \cdot e\}$$

여기서 $k$는 에폭당 해동하는 레이어 수 (기본 $k=1$).

### 3. Freeze가 Regularization으로 작용하는 이유

동결된 파라미터는 사전학습된 값에 고정되므로, 학습 가능한 파라미터 수가 줄어든다. 이는 모델의 **유효 용량(effective capacity)**을 제한하는 효과가 있다.

PAC-Bayes 관점에서, 학습 가능한 파라미터 수 $d_{\text{train}}$이 줄면 일반화 오차 상한이 감소한다:

$$\mathcal{L}_{\text{gen}} \leq \mathcal{L}_{\text{train}} + \sqrt{\frac{d_{\text{train}} \log(2n/\delta)}{2n}}$$

여기서 $n$은 학습 샘플 수, $\delta$는 신뢰도 파라미터다.

**데이터가 적을수록 더 많이 동결해야** 일반화 오차를 낮출 수 있다.

| 데이터 규모 | 권장 전략 |
|------------|---------|
| 100~1K 샘플 | Feature Extraction (헤드만 학습) |
| 1K~10K 샘플 | 상위 레이어만 학습 + Discriminative LR |
| 10K~100K 샘플 | Progressive Unfreezing |
| 100K+ 샘플 | Full Fine-tuning 또는 전체 LoRA |

### 4. VLM에서의 Freeze 전략

Vision-Language Model은 세 가지 구성요소로 이루어져 있고, 각각의 freeze 전략이 다르다.

**구성요소:**
1. **Vision Encoder** ($\theta_V$): ViT, SigLIP 등
2. **Connector/Projector** ($\theta_C$): MLP, Q-Former 등
3. **LLM Backbone** ($\theta_L$): LLaMA, Qwen 등

VLM의 학습 목표:

$$\min_{\theta_{\text{train}}} \mathcal{L}(\theta_V, \theta_C, \theta_L)$$

여기서 $\theta_{\text{train}} \subseteq \{\theta_V, \theta_C, \theta_L\}$은 학습 대상 파라미터.

**전략별 비교:**

| 전략 | $\theta_V$ | $\theta_C$ | $\theta_L$ | 학습 파라미터 비율 |
|------|-----------|-----------|-----------|----------------|
| A: Connector Only | Freeze | **Train** | Freeze | ~1% |
| B: Connector + LLM LoRA | Freeze | **Train** | **LoRA** | ~3% |
| C: Vision LoRA + Connector + LLM LoRA | **LoRA** | **Train** | **LoRA** | ~5% |
| D: Full (Vision freeze) | Freeze | **Train** | **Full** | ~85% |
| E: Full (전체) | **Full** | **Train** | **Full** | 100% |

**각 전략의 적합 상황:**

- **전략 A** (Connector Only): 가장 빠르고 경제적. Vision과 Language의 매핑만 조정하면 되는 경우. 예: 일반 VQA, 간단한 캡셔닝.
- **전략 B** (Connector + LLM LoRA): 가장 널리 사용. 언어 생성 능력도 태스크에 맞게 조정. 예: 도메인 특화 VQA, OCR fine-tuning.
- **전략 C** (Vision LoRA + Connector + LLM LoRA): 이미지 도메인이 사전학습과 다를 때. 예: 의료 영상, 위성 이미지, 문서 OCR.
- **전략 D/E** (Full): 최고 성능 필요, 자원 충분할 때. 대규모 데이터셋 필수.

---

## 코드: Progressive Unfreezing 구현

### 기본 Progressive Unfreezing

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from typing import Optional
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProgressiveUnfreezingConfig:
    """Progressive Unfreezing 설정."""
    model_name: str = "meta-llama/Llama-2-7b-hf"
    num_epochs: int = 10
    layers_per_epoch: int = 4         # 에폭당 해동할 레이어 수
    initial_frozen_layers: int = -1   # -1이면 전체 동결 후 시작
    learning_rate: float = 2e-5
    decay_factor: float = 0.9        # 하위 레이어 학습률 감쇠
    warmup_ratio: float = 0.1
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0


class ProgressiveUnfreezer:
    """
    Progressive Unfreezing 학습 관리자.

    에폭마다 레이어를 점진적으로 해동하며,
    새로 해동된 레이어에는 discriminative learning rate를 적용한다.
    """

    def __init__(self, config: ProgressiveUnfreezingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델의 transformer 레이어 목록 추출
        self.layers = self._get_transformer_layers()
        self.total_layers = len(self.layers)

        if config.initial_frozen_layers == -1:
            config.initial_frozen_layers = self.total_layers

        logger.info(f"총 Transformer 레이어: {self.total_layers}")

        # 초기 상태: 모든 레이어 동결, LM head만 학습
        self._freeze_all_layers()
        self._unfreeze_lm_head()

    def _get_transformer_layers(self) -> list:
        """모델에서 Transformer 레이어 목록 추출."""
        # LLaMA 구조 기준
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        # GPT-2 / GPT-Neo 구조
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        else:
            raise ValueError("지원하지 않는 모델 구조. _get_transformer_layers를 수정해라.")

    def _freeze_all_layers(self):
        """모든 파라미터 동결."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("모든 파라미터 동결 완료")

    def _unfreeze_lm_head(self):
        """LM head (출력 레이어) 해동."""
        if hasattr(self.model, "lm_head"):
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
            num_params = sum(p.numel() for p in self.model.lm_head.parameters())
            logger.info(f"LM head 해동: {num_params:,} 파라미터")

    def unfreeze_layers(self, num_layers_from_top: int):
        """
        상위 N개 레이어를 해동.

        Args:
            num_layers_from_top: 최상위부터 해동할 레이어 수
        """
        layers_to_unfreeze = min(num_layers_from_top, self.total_layers)

        for i in range(self.total_layers):
            layer = self.layers[self.total_layers - 1 - i]
            if i < layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                for param in layer.parameters():
                    param.requires_grad = False

        # 학습 가능 파라미터 통계
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"해동 레이어: 상위 {layers_to_unfreeze}/{self.total_layers} | "
            f"학습 파라미터: {trainable:,} ({trainable/total*100:.2f}%)"
        )

    def _build_discriminative_optimizer(
        self, unfrozen_layers: int
    ) -> AdamW:
        """Discriminative Learning Rate가 적용된 optimizer 생성."""
        param_groups = []

        # 1. LM head
        if hasattr(self.model, "lm_head"):
            head_params = [p for p in self.model.lm_head.parameters() if p.requires_grad]
            if head_params:
                param_groups.append({
                    "params": head_params,
                    "lr": self.config.learning_rate,
                    "name": "lm_head",
                })

        # 2. Transformer 레이어 (상위부터 하위로)
        for i in range(unfrozen_layers):
            layer_idx = self.total_layers - 1 - i
            layer = self.layers[layer_idx]

            layer_params = [p for p in layer.parameters() if p.requires_grad]
            if not layer_params:
                continue

            # Discriminative LR: 하위 레이어일수록 작은 학습률
            layer_lr = self.config.learning_rate * (self.config.decay_factor ** (i + 1))

            param_groups.append({
                "params": layer_params,
                "lr": layer_lr,
                "name": f"layer_{layer_idx}",
            })

        # 학습률 정보 로깅
        for group in param_groups:
            num_p = sum(p.numel() for p in group["params"])
            logger.info(f"  [{group['name']}] LR: {group['lr']:.2e}, Params: {num_p:,}")

        return AdamW(param_groups, weight_decay=0.01)

    def train(self, train_dataloader, eval_dataloader=None):
        """Progressive Unfreezing 학습 루프."""

        for epoch in range(self.config.num_epochs):
            # 에폭에 따라 레이어 해동
            unfrozen_layers = min(
                (epoch + 1) * self.config.layers_per_epoch,
                self.total_layers,
            )
            self.unfreeze_layers(unfrozen_layers)

            # Discriminative LR로 optimizer 재생성
            optimizer = self._build_discriminative_optimizer(unfrozen_layers)

            # 에폭별 scheduler
            steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
            warmup_steps = int(steps_per_epoch * self.config.warmup_ratio)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=steps_per_epoch,
            )

            # 학습 루프
            self.model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / self.config.gradient_accumulation_steps

                loss.backward()
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_loss = epoch_loss / len(train_dataloader)
            logger.info(
                f"[Epoch {epoch+1}/{self.config.num_epochs}] "
                f"해동 레이어: {unfrozen_layers} | "
                f"Loss: {avg_loss:.4f}"
            )

            # 평가
            if eval_dataloader is not None:
                eval_loss = self._evaluate(eval_dataloader)
                logger.info(f"  Eval Loss: {eval_loss:.4f}")

        return self.model

    @torch.no_grad()
    def _evaluate(self, eval_dataloader) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            total_loss += outputs.loss.item()
            count += 1
        return total_loss / max(count, 1)
```

---

## 코드: VLM 레이어별 학습률 설정

### VLM Freeze 전략 관리자

```python
from peft import LoraConfig, get_peft_model, TaskType
from dataclasses import dataclass
from enum import Enum


class VLMFreezeStrategy(Enum):
    """VLM Freeze 전략 열거형."""
    CONNECTOR_ONLY = "connector_only"           # 전략 A
    CONNECTOR_LLM_LORA = "connector_llm_lora"   # 전략 B
    ALL_LORA = "all_lora"                       # 전략 C
    VISION_FREEZE_LLM_FULL = "vision_freeze_llm_full"  # 전략 D


@dataclass
class VLMFreezeConfig:
    """VLM Freeze 전략 설정."""
    strategy: VLMFreezeStrategy = VLMFreezeStrategy.CONNECTOR_LLM_LORA
    # Discriminative LR
    vision_lr: float = 1e-6
    connector_lr: float = 1e-4
    llm_lr: float = 2e-5
    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = None


class VLMFreezeManager:
    """
    VLM의 Freeze 전략을 관리하는 클래스.

    Vision Encoder, Connector, LLM 각각에 대해
    freeze/unfreeze/LoRA를 설정한다.
    """

    def __init__(self, model, config: VLMFreezeConfig):
        self.model = model
        self.config = config

        # 모델 구성요소 분류
        self.vision_params = []
        self.connector_params = []
        self.llm_params = []
        self._classify_parameters()

    def _classify_parameters(self):
        """모델 파라미터를 Vision/Connector/LLM으로 분류."""
        for name, param in self.model.named_parameters():
            if any(key in name for key in ["vision", "visual", "vit", "image_encoder"]):
                self.vision_params.append((name, param))
            elif any(key in name for key in ["connector", "projection", "mm_projector", "bridge"]):
                self.connector_params.append((name, param))
            else:
                self.llm_params.append((name, param))

        logger.info(
            f"파라미터 분류 결과:\n"
            f"  Vision:    {sum(p.numel() for _, p in self.vision_params):>12,}\n"
            f"  Connector: {sum(p.numel() for _, p in self.connector_params):>12,}\n"
            f"  LLM:       {sum(p.numel() for _, p in self.llm_params):>12,}"
        )

    def apply_strategy(self):
        """설정된 전략을 모델에 적용."""
        strategy = self.config.strategy

        if strategy == VLMFreezeStrategy.CONNECTOR_ONLY:
            return self._apply_connector_only()
        elif strategy == VLMFreezeStrategy.CONNECTOR_LLM_LORA:
            return self._apply_connector_llm_lora()
        elif strategy == VLMFreezeStrategy.ALL_LORA:
            return self._apply_all_lora()
        elif strategy == VLMFreezeStrategy.VISION_FREEZE_LLM_FULL:
            return self._apply_vision_freeze_llm_full()
        else:
            raise ValueError(f"알 수 없는 전략: {strategy}")

    def _apply_connector_only(self):
        """전략 A: Connector만 학습."""
        # 전체 동결
        for param in self.model.parameters():
            param.requires_grad = False

        # Connector만 해동
        for name, param in self.connector_params:
            param.requires_grad = True

        self._log_trainable_stats("Connector Only")
        return self.model

    def _apply_connector_llm_lora(self):
        """전략 B: Connector full + LLM LoRA."""
        # 전체 동결
        for param in self.model.parameters():
            param.requires_grad = False

        # Connector 해동
        for name, param in self.connector_params:
            param.requires_grad = True

        # LLM에 LoRA 적용
        target_modules = self.config.lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
        ]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        self.model = get_peft_model(self.model, lora_config)
        self._log_trainable_stats("Connector + LLM LoRA")
        return self.model

    def _apply_all_lora(self):
        """전략 C: Vision LoRA + Connector full + LLM LoRA."""
        # 전체 동결
        for param in self.model.parameters():
            param.requires_grad = False

        # Connector 해동
        for name, param in self.connector_params:
            param.requires_grad = True

        # Vision + LLM에 LoRA 적용 (target_modules로 제어)
        target_modules = self.config.lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            # Vision encoder의 attention도 포함
            modules_to_save=["connector", "mm_projector"],
        )

        self.model = get_peft_model(self.model, lora_config)
        self._log_trainable_stats("All LoRA + Connector")
        return self.model

    def _apply_vision_freeze_llm_full(self):
        """전략 D: Vision 동결 + Connector + LLM 전체 학습."""
        # Vision 동결
        for name, param in self.vision_params:
            param.requires_grad = False

        # Connector + LLM 학습
        for name, param in self.connector_params:
            param.requires_grad = True
        for name, param in self.llm_params:
            param.requires_grad = True

        self._log_trainable_stats("Vision Freeze + LLM Full")
        return self.model

    def _log_trainable_stats(self, strategy_name: str):
        """학습 가능 파라미터 통계 출력."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"[{strategy_name}] "
            f"학습: {trainable:,} ({trainable/total*100:.2f}%) | "
            f"동결: {total-trainable:,} ({(total-trainable)/total*100:.2f}%)"
        )

    def get_optimizer_groups(self) -> list[dict]:
        """구성요소별 discriminative learning rate를 적용한 optimizer groups."""
        groups = []

        # Vision (학습하는 경우)
        vision_trainable = [
            p for _, p in self.vision_params if p.requires_grad
        ]
        if vision_trainable:
            groups.append({
                "params": vision_trainable,
                "lr": self.config.vision_lr,
                "weight_decay": 0.01,
                "name": "vision",
            })

        # Connector
        connector_trainable = [
            p for _, p in self.connector_params if p.requires_grad
        ]
        if connector_trainable:
            groups.append({
                "params": connector_trainable,
                "lr": self.config.connector_lr,
                "weight_decay": 0.01,
                "name": "connector",
            })

        # LLM (LoRA 파라미터 포함)
        llm_trainable = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad
            and not any(key in n for key in ["vision", "visual", "vit", "connector", "projection", "mm_projector"])
        ]
        if llm_trainable:
            groups.append({
                "params": llm_trainable,
                "lr": self.config.llm_lr,
                "weight_decay": 0.01,
                "name": "llm",
            })

        for g in groups:
            num_p = sum(p.numel() for p in g["params"])
            logger.info(f"  [{g['name']}] LR: {g['lr']:.2e} | Params: {num_p:,}")

        return groups
```

### VLM Freeze 전략 비교 유틸리티

```python
def compare_vlm_strategies(model_class, model_name: str):
    """
    다양한 VLM Freeze 전략의 학습 파라미터 수를 비교.
    실제 모델을 로드하여 정확한 수치를 출력한다.
    """
    strategies = [
        VLMFreezeStrategy.CONNECTOR_ONLY,
        VLMFreezeStrategy.CONNECTOR_LLM_LORA,
        VLMFreezeStrategy.ALL_LORA,
        VLMFreezeStrategy.VISION_FREEZE_LLM_FULL,
    ]

    results = []

    for strategy in strategies:
        # 매번 모델을 새로 로드 (전략 적용 후 상태가 변하므로)
        model = model_class.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
        )

        config = VLMFreezeConfig(strategy=strategy)
        manager = VLMFreezeManager(model, config)
        model = manager.apply_strategy()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        results.append({
            "strategy": strategy.value,
            "trainable": trainable,
            "total": total,
            "ratio": trainable / total * 100,
        })

        del model
        torch.cuda.empty_cache()

    # 결과 출력
    print(f"\n{'Strategy':<35} {'Trainable':>15} {'Total':>15} {'Ratio':>10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['strategy']:<35} "
            f"{r['trainable']:>15,} "
            f"{r['total']:>15,} "
            f"{r['ratio']:>9.2f}%"
        )

    return results
```

### Discriminative Learning Rate 단독 사용

```python
def create_discriminative_lr_optimizer(
    model: nn.Module,
    base_lr: float = 2e-5,
    decay_factor: float = 0.9,
    weight_decay: float = 0.01,
) -> AdamW:
    """
    레이어별 discriminative learning rate를 적용한 optimizer.

    최상위 레이어: base_lr
    그 아래: base_lr * decay_factor
    그 아래: base_lr * decay_factor^2
    ...
    """
    # Transformer 레이어 추출 (LLaMA 구조 기준)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = list(model.model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = list(model.transformer.h)
    else:
        # 단순 fallback: 전체를 하나의 그룹으로
        return AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    num_layers = len(layers)
    param_groups = []

    # Embedding 레이어 (가장 작은 학습률)
    embed_params = []
    for name, param in model.named_parameters():
        if "embed" in name and param.requires_grad:
            embed_params.append(param)

    if embed_params:
        embed_lr = base_lr * (decay_factor ** num_layers)
        param_groups.append({
            "params": embed_params,
            "lr": embed_lr,
            "weight_decay": weight_decay,
        })

    # Transformer 레이어 (하위 → 상위)
    for i, layer in enumerate(layers):
        layer_params = [p for p in layer.parameters() if p.requires_grad]
        if not layer_params:
            continue

        # 상위 레이어일수록 큰 학습률
        distance_from_top = num_layers - 1 - i
        layer_lr = base_lr * (decay_factor ** distance_from_top)

        param_groups.append({
            "params": layer_params,
            "lr": layer_lr,
            "weight_decay": weight_decay,
        })

    # LM head (base_lr 그대로)
    if hasattr(model, "lm_head"):
        head_params = [p for p in model.lm_head.parameters() if p.requires_grad]
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": base_lr,
                "weight_decay": 0.0,  # head에는 weight decay 미적용
            })

    # 학습률 분포 출력
    lrs = [g["lr"] for g in param_groups]
    print(f"Discriminative LR 범위: {min(lrs):.2e} ~ {max(lrs):.2e}")
    print(f"감쇠 비율: {min(lrs)/max(lrs):.4f}")

    return AdamW(param_groups)
```

### Layer-wise Freeze 분석 도구

```python
def analyze_layer_importance(
    model: nn.Module,
    eval_dataloader,
    device: torch.device,
    num_samples: int = 100,
) -> dict:
    """
    각 레이어를 개별적으로 동결했을 때의 성능 변화를 측정.
    어떤 레이어가 태스크에 중요한지 파악하는 데 사용.

    주의: 시간이 오래 걸린다. 레이어 수 × 평가 시간.
    """
    model.eval()

    # 기준 loss (전체 해동 상태)
    base_loss = _quick_eval(model, eval_dataloader, device, num_samples)
    print(f"기준 Loss (전체 해동): {base_loss:.4f}")

    # Transformer 레이어 추출
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = list(model.model.layers)
    else:
        return {}

    results = {}

    for i, layer in enumerate(layers):
        # 레이어 i를 동결 (나머지는 해동)
        original_states = {}
        for name, param in layer.named_parameters():
            original_states[name] = param.requires_grad
            param.requires_grad = False

        # 평가
        frozen_loss = _quick_eval(model, eval_dataloader, device, num_samples)
        importance = frozen_loss - base_loss  # 양수면 해당 레이어가 중요

        results[f"layer_{i}"] = {
            "frozen_loss": frozen_loss,
            "importance": importance,
        }

        # 복원
        for name, param in layer.named_parameters():
            param.requires_grad = original_states[name]

        print(f"Layer {i:2d} | Frozen Loss: {frozen_loss:.4f} | Importance: {importance:+.4f}")

    return results


@torch.no_grad()
def _quick_eval(model, dataloader, device, num_samples):
    """빠른 평가 (제한된 샘플 수)."""
    model.eval()
    total_loss = 0.0
    count = 0

    for batch in dataloader:
        if count >= num_samples:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()
        count += input_ids.size(0)

    return total_loss / max(count // input_ids.size(0), 1)
```

---

## VLM Fine-tuning 전략별 비교: 성능 vs 비용 트레이드오프

### 정량적 비교 (7B VLM 기준 추정)

| 전략 | 학습 파라미터 | GPU 메모리 | 학습 시간 (상대) | 성능 (상대) |
|------|-------------|-----------|----------------|------------|
| A: Connector Only | ~20M (0.3%) | ~8 GB | 1x | 70~80% |
| B: Connector + LLM LoRA | ~80M (1.1%) | ~12 GB | 2x | 85~95% |
| C: All LoRA | ~120M (1.7%) | ~16 GB | 3x | 90~97% |
| D: Vision Freeze + LLM Full | ~6.5B (93%) | ~50 GB | 8x | 95~99% |
| E: Full Fine-tuning | ~7B (100%) | ~90 GB | 10x | 100% |

### 상황별 권장 전략

```
                        데이터 양
                   적음 (<1K)    많음 (>100K)
              ┌──────────────┬──────────────┐
    유사      │  전략 A/B     │  전략 B/C     │
  도메인 gap  │ Connector    │ LoRA         │
              ├──────────────┼──────────────┤
    큰        │  전략 B/C     │  전략 D/E     │
  도메인 gap  │ LoRA         │ Full FT      │
              └──────────────┴──────────────┘
```

**예시 시나리오:**
- 일반 VQA 챗봇 → **전략 B** (Connector + LLM LoRA)
- 의료 X-ray 리포트 생성 → **전략 C** (Vision LoRA 포함, 의료 이미지 도메인 gap 큼)
- 한국어 문서 OCR → **전략 B 또는 C** (텍스트 인식은 vision 특징 조정 필요)
- 대규모 멀티모달 사전학습 → **전략 E** (전체 Fine-tuning)

---

## 실전 팁

### 1. Freeze 전략 디버깅

```python
def debug_freeze_state(model: nn.Module):
    """모델의 freeze 상태를 모듈별로 출력."""
    module_stats = {}

    for name, param in model.named_parameters():
        # 최상위 모듈 이름 추출
        top_module = name.split(".")[0]
        if top_module not in module_stats:
            module_stats[top_module] = {"frozen": 0, "trainable": 0}

        if param.requires_grad:
            module_stats[top_module]["trainable"] += param.numel()
        else:
            module_stats[top_module]["frozen"] += param.numel()

    print(f"{'Module':<30} {'Frozen':>12} {'Trainable':>12} {'Status':>10}")
    print("-" * 70)
    for module, stats in module_stats.items():
        total = stats["frozen"] + stats["trainable"]
        if stats["trainable"] == 0:
            status = "FROZEN"
        elif stats["frozen"] == 0:
            status = "TRAIN"
        else:
            status = "PARTIAL"

        print(
            f"{module:<30} "
            f"{stats['frozen']:>12,} "
            f"{stats['trainable']:>12,} "
            f"{status:>10}"
        )
```

### 2. 흔한 실수

| 실수 | 증상 | 해결 |
|------|------|------|
| Embedding 동결 잊음 | 토큰 임베딩이 변형, 기존 어휘 오작동 | `model.embed_tokens.weight.requires_grad = False` |
| LayerNorm 동결 | 학습 불안정, loss 발산 | LayerNorm은 항상 학습 가능하게 유지 |
| lm_head 동결 | 출력 분포가 고정, 학습 안 됨 | 태스크 학습 시 lm_head는 해동 |
| 전체 동결 후 LoRA 미적용 | 학습 파라미터 0, loss 변화 없음 | `model.print_trainable_parameters()`로 확인 |
| Discriminative LR 방향 반대 | 하위 레이어에 큰 LR → 사전학습 파괴 | decay_factor가 0~1 사이인지 확인 |

---

## 용어 체크리스트

학습 후 아래 용어를 설명할 수 있는지 점검해봐라:

- [ ] **Layer Freeze**: 어떤 레이어를 동결해야 하는지, 판단 기준을 설명할 수 있는가?
- [ ] **Progressive Unfreezing**: 에폭 $e$에서 학습 가능한 레이어 집합 $\mathcal{S}_e$를 수식으로 표현할 수 있는가?
- [ ] **Discriminative Learning Rate**: $\eta_l = \eta_{\text{base}} \times \xi^{L-l}$ 수식의 각 변수를 설명할 수 있는가?
- [ ] **Feature Extraction vs Fine-tuning**: 두 접근법의 차이와 각각의 적합 상황을 아는가?
- [ ] **VLM 구성요소**: Vision Encoder, Connector, LLM의 역할과 각각의 freeze 전략을 설명할 수 있는가?
- [ ] **전략 A~E 비교**: 5가지 VLM freeze 전략의 장단점과 적합 상황을 구분할 수 있는가?
- [ ] **Regularization 효과**: freeze가 왜 정규화 역할을 하는지 PAC-Bayes 관점에서 설명할 수 있는가?
- [ ] **데이터 규모별 전략**: 데이터가 100개 vs 10만개일 때 어떤 전략을 택해야 하는지 아는가?
- [ ] **Discriminative LR 구현**: PyTorch에서 레이어별 학습률을 설정하는 코드를 작성할 수 있는가?
- [ ] **Connector Tuning**: VLM에서 connector만 학습하는 것이 왜 효과적인지 설명할 수 있는가?
