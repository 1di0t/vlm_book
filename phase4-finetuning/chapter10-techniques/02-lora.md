# 10.2 LoRA (Low-Rank Adaptation)

## 핵심 용어 박스

| 용어 | 정의 |
|------|------|
| **LoRA (Low-Rank Adaptation)** | 사전학습된 가중치 행렬을 동결하고, **저랭크 행렬 쌍 (B, A)**을 추가 학습하여 파라미터 효율적 fine-tuning을 수행하는 기법. Microsoft Research에서 2021년 발표. |
| **Rank (r)** | LoRA에서 분해된 행렬의 내부 차원. $r \ll \min(d, k)$로 설정하며, 값이 클수록 표현력이 높지만 파라미터 수도 증가한다. 일반적으로 4~64 범위 사용. |
| **Alpha ($\alpha$)** | LoRA의 스케일링 하이퍼파라미터. $\Delta W = (\alpha / r) B A$에서 학습률 스케일링 역할을 한다. 보통 $\alpha = 2r$ 또는 $\alpha = r$로 설정. |
| **Target Modules** | LoRA를 적용할 대상 레이어. Transformer의 Q, K, V, O projection과 FFN 중 선택한다. 어디에 적용하느냐에 따라 성능과 파라미터 수가 달라진다. |
| **Merge** | 학습 완료 후 $W' = W_0 + (\alpha/r) B A$로 원본 가중치에 LoRA를 병합하는 것. 추론 시 추가 연산 없이 원래 모델과 동일한 구조로 서빙 가능. |
| **SVD (Singular Value Decomposition)** | 행렬을 $W = U\Sigma V^T$로 분해하는 방법. LoRA가 유효한 이유의 수학적 근거를 제공한다. |

---

## 개요

LoRA는 파라미터 효율적 fine-tuning (PEFT)의 대표적 방법이다. 핵심 아이디어는 간단하다:

> **사전학습된 가중치 행렬의 변화량 $\Delta W$는 저랭크(low-rank)로 충분히 근사할 수 있다.**

Full fine-tuning에서 $d \times k$ 크기의 가중치 행렬을 업데이트하려면 $d \times k$개의 파라미터 gradient를 계산해야 한다. LoRA는 이 $\Delta W$를 $B \times A$ ($B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$)로 분해하여, $r \times (d + k)$개의 파라미터만 학습한다.

**예시 (LLaMA-7B, hidden_dim=4096):**
- Full: $4096 \times 4096 = 16,777,216$ 파라미터/레이어
- LoRA (r=16): $16 \times (4096 + 4096) = 131,072$ 파라미터/레이어
- **절약률: 99.2%**

---

## 수학적 원리

### 1. 기본 수식

사전학습된 가중치 행렬 $W_0 \in \mathbb{R}^{d \times k}$에 대해, LoRA는 다음과 같이 정의한다:

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

여기서:
- $W_0 \in \mathbb{R}^{d \times k}$: 동결된 사전학습 가중치 (gradient 계산 안 함)
- $B \in \mathbb{R}^{d \times r}$: 저랭크 행렬 (학습 대상)
- $A \in \mathbb{R}^{r \times k}$: 저랭크 행렬 (학습 대상)
- $r \ll \min(d, k)$: rank (차원 병목)
- $x \in \mathbb{R}^{k}$: 입력 벡터

스케일링 팩터를 포함하면:

$$h = W_0 x + \frac{\alpha}{r} B A x$$

### 2. 파라미터 절약 분석

Full fine-tuning에서 $W_0$를 업데이트하면:

$$|\Delta W|_{\text{full}} = d \times k$$

LoRA에서:

$$|\Delta W|_{\text{LoRA}} = d \times r + r \times k = r(d + k)$$

**절약 비율:**

$$\text{Ratio} = \frac{r(d + k)}{d \times k}$$

$d = k$인 정사각 행렬의 경우:

$$\text{Ratio} = \frac{2r}{d}$$

| $d$ | $r$ | Full 파라미터 | LoRA 파라미터 | 절약률 |
|-----|-----|-------------|-------------|--------|
| 4096 | 4 | 16.7M | 32.8K | 99.8% |
| 4096 | 16 | 16.7M | 131K | 99.2% |
| 4096 | 64 | 16.7M | 524K | 96.9% |
| 4096 | 256 | 16.7M | 2.1M | 87.5% |

### 3. 스케일링 팩터: $\alpha / r$

LoRA의 출력에 $\alpha / r$ 스케일링을 곱하는 이유가 무엇인가?

$$\Delta W = \frac{\alpha}{r} B A$$

**핵심 인사이트**: $r$을 바꿔도 학습 동역학(learning dynamics)이 크게 변하지 않도록 하기 위해서다.

$A$를 Gaussian($\sigma = 1/\sqrt{r}$)으로, $B$를 0으로 초기화하면, $BA$의 출력 스케일은 $r$에 따라 달라진다. $\alpha/r$을 곱하면 이 스케일을 정규화하여, $r$을 변경할 때 learning rate를 재조정할 필요가 없어진다.

실전에서의 $\alpha$ 설정:
- $\alpha = r$: 스케일링이 1이 되어 가장 직관적
- $\alpha = 2r$: HuggingFace PEFT 기본값. 약간 더 큰 업데이트
- $\alpha$를 고정하고 $r$만 조절: $r$이 커지면 개별 업데이트 크기가 작아짐

### 4. SVD와의 관계: 왜 Low-Rank가 유효한가?

어떤 행렬 $W \in \mathbb{R}^{d \times k}$이든 Singular Value Decomposition으로 분해할 수 있다:

$$W = U \Sigma V^T = \sum_{i=1}^{\min(d,k)} \sigma_i u_i v_i^T$$

여기서:
- $U \in \mathbb{R}^{d \times d}$: 좌특이벡터 (orthogonal)
- $\Sigma$: 특이값 대각행렬, $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $V \in \mathbb{R}^{k \times k}$: 우특이벡터 (orthogonal)

**상위 $r$개 특이값만 보존하면:**

$$W_r = \sum_{i=1}^{r} \sigma_i u_i v_i^T = U_r \Sigma_r V_r^T$$

이것이 Frobenius norm 관점에서 $W$의 최적 rank-$r$ 근사(Eckart-Young-Mirsky 정리):

$$\|W - W_r\|_F^2 = \sum_{i=r+1}^{\min(d,k)} \sigma_i^2$$

### 5. Singular Value Decay: 수학적 근거

사전학습된 Transformer 가중치 행렬의 특이값은 **급격히 감소(decay)**하는 패턴을 보인다.

특이값이 빠르게 감소한다는 것은:

$$\frac{\sum_{i=r+1}^{d} \sigma_i^2}{\sum_{i=1}^{d} \sigma_i^2} \ll 1 \quad \text{(작은 } r \text{에서도)}$$

즉, 상위 소수의 특이값이 행렬 정보의 대부분을 포함하고 있다. Fine-tuning에서의 **변화량** $\Delta W = W_{\text{finetuned}} - W_{\text{pretrained}}$도 마찬가지로 저랭크 구조를 가진다.

**실험적 관찰 (Hu et al., 2021):**
- GPT-3의 attention weight 변화량 $\Delta W$의 유효 rank는 매우 낮음 (1~4 정도)
- $r=4$로도 full fine-tuning 성능의 95% 이상 달성
- $r$을 64 이상으로 올려도 성능 향상이 미미함

이는 fine-tuning이 사전학습된 표현의 **저차원 부분공간(low-dimensional subspace)**만 조정하면 충분하다는 것을 의미한다.

### 6. LoRA Gradient 계산

Loss $\mathcal{L}$에 대한 $A$와 $B$의 gradient를 유도하자.

Forward:

$$h = W_0 x + \frac{\alpha}{r} B A x$$

$A$에 대한 gradient:

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} B^T \frac{\partial \mathcal{L}}{\partial h} x^T$$

$B$에 대한 gradient:

$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \frac{\partial \mathcal{L}}{\partial h} (Ax)^T$$

여기서 $\frac{\partial \mathcal{L}}{\partial h}$는 출력 $h$에 대한 loss의 gradient다.

**메모리 관점**: $W_0$에 대한 gradient를 저장할 필요가 없으므로 ($W_0$는 동결), optimizer states도 $A$와 $B$에 대해서만 유지하면 된다. 이것이 LoRA의 메모리 절약의 핵심이다.

### 7. 초기화 전략

LoRA의 초기화는 학습 시작 시 $\Delta W = 0$이 되도록 설계된다:

- **$A$**: Kaiming uniform 또는 Gaussian 초기화
  $$A_{ij} \sim \mathcal{N}(0, \sigma^2), \quad \sigma = \frac{1}{\sqrt{r}}$$
- **$B$**: **영행렬(zero matrix)**로 초기화
  $$B = \mathbf{0}_{d \times r}$$

따라서 학습 시작 시:

$$\Delta W = \frac{\alpha}{r} B A = \frac{\alpha}{r} \cdot \mathbf{0} \cdot A = \mathbf{0}$$

이로써 학습 초기에 모델 출력이 사전학습된 상태와 동일하게 유지된다. 학습이 진행되면서 $B$가 0에서 벗어나며 점진적으로 $\Delta W$가 형성된다.

> **왜 $B = 0$이고 $A$는 random인가?** 반대로 하면($A = 0$, $B$ random) gradient가 $\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} B^T \frac{\partial \mathcal{L}}{\partial h} x^T$로 동일하게 비영(non-zero)이 되지만, 관례적으로 "출력 쪽을 0으로" 하는 것이 표준이다. 어느 쪽이든 수학적으로는 동등하다.

### 8. Rank 선택 가이드

| 기준 | 낮은 $r$ (4~8) | 중간 $r$ (16~32) | 높은 $r$ (64~128) |
|------|----------------|------------------|-------------------|
| **파라미터 수** | 매우 적음 | 적당 | 상대적으로 많음 |
| **학습 속도** | 빠름 | 적당 | 느림 |
| **표현력** | 단순 태스크에 충분 | 대부분의 태스크 | 복잡한 도메인 적응 |
| **과적합 위험** | 낮음 | 중간 | 높음 (데이터 적을 때) |
| **권장 상황** | 유사 도메인, 데이터 적음 | 범용 | 도메인 gap 큼, 데이터 충분 |

**경험적 가이드라인:**
- 대부분의 NLP 태스크: $r = 8 \sim 16$이면 충분
- OCR/VLM 도메인 적응: $r = 16 \sim 64$ 권장
- 다국어 적응 등 큰 변화: $r = 64 \sim 128$

### 9. Target Module 선택

Transformer 레이어에서 LoRA를 적용할 수 있는 위치:

```
Multi-Head Attention:
  Q = W_q · x    ← LoRA 적용 가능
  K = W_k · x    ← LoRA 적용 가능
  V = W_v · x    ← LoRA 적용 가능
  O = W_o · h    ← LoRA 적용 가능 (attention output projection)

Feed-Forward Network:
  FFN_up   = W_up · x      ← LoRA 적용 가능
  FFN_gate = W_gate · x    ← LoRA 적용 가능 (SwiGLU 등)
  FFN_down = W_down · h    ← LoRA 적용 가능
```

**어디에 적용하는 것이 효과적인가?**

| Target | 파라미터 증가 | 효과 | 권장도 |
|--------|-------------|------|--------|
| Q만 | 최소 | 약함 | 낮음 |
| Q + V | 적음 | 적당 | **높음** (기본값) |
| Q + K + V + O | 중간 | 좋음 | **높음** |
| Q + K + V + O + FFN | 많음 | 가장 좋음 | 예산 충분할 때 |

원 논문(Hu et al.)에서는 Q와 V에만 적용해도 full fine-tuning의 성능에 근접한다는 것을 보였다. 하지만 최근 연구들에서는 **모든 linear layer에 적용**하는 것이 일관적으로 더 좋은 결과를 보인다.

---

## 코드: LoRA Layer 직접 구현

### 방법 1: LoRA Layer from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LoRALinear(nn.Module):
    """
    LoRA가 적용된 Linear 레이어.

    원본 가중치 W_0는 동결하고,
    저랭크 행렬 B @ A를 학습하여 ΔW를 근사한다.

    h = W_0 @ x + (alpha/r) * B @ A @ x
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        self.merged = False

        # 원본 가중치 (동결)
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # LoRA 행렬
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout (정규화 용도)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # 초기화
        self._init_lora_weights()

    def _init_lora_weights(self):
        """A는 Kaiming uniform, B는 0으로 초기화."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            # 이미 merge된 경우 원본 linear만 사용
            return self.linear(x)

        # h = W_0 @ x + (alpha/r) * B @ A @ x
        base_output = self.linear(x)
        lora_output = (
            F.linear(
                F.linear(self.lora_dropout(x), self.lora_A),
                self.lora_B,
            )
            * self.scaling
        )
        return base_output + lora_output

    def merge(self):
        """LoRA 가중치를 원본에 병합. 추론 시 추가 연산 제거."""
        if not self.merged:
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def unmerge(self):
        """병합 해제. 추가 학습 시 사용."""
        if self.merged:
            self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, "
            f"scaling={self.scaling:.4f}, merged={self.merged}"
        )
```

### LoRA를 기존 모델에 주입하는 유틸리티

```python
def inject_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> nn.Module:
    """
    기존 모델의 특정 Linear 레이어를 LoRA Linear로 교체.

    Args:
        model: 원본 모델
        target_modules: LoRA를 적용할 모듈 이름 목록
                       예: ["q_proj", "v_proj", "k_proj", "o_proj"]
        rank: LoRA rank
        alpha: 스케일링 팩터
        dropout: LoRA dropout rate

    Returns:
        LoRA가 주입된 모델
    """
    lora_params = 0
    frozen_params = 0

    for name, module in model.named_modules():
        # target module인지 확인
        if not any(target in name for target in target_modules):
            continue

        if not isinstance(module, nn.Linear):
            continue

        # LoRA Linear로 교체
        lora_layer = LoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        # 원본 가중치 복사
        lora_layer.linear.weight.data = module.weight.data.clone()
        if module.bias is not None:
            lora_layer.linear.bias.data = module.bias.data.clone()

        # 모델에서 교체
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, lora_layer)

        layer_lora_params = rank * (module.in_features + module.out_features)
        lora_params += layer_lora_params

    # 원본 파라미터 동결
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
            frozen_params += param.numel()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"LoRA 주입 완료:")
    print(f"  학습 가능 파라미터: {trainable:,} ({trainable/total*100:.2f}%)")
    print(f"  동결 파라미터:     {frozen_params:,} ({frozen_params/total*100:.2f}%)")
    print(f"  Rank: {rank}, Alpha: {alpha}, Scaling: {alpha/rank:.4f}")

    return model
```

### LoRA 모델 학습 루프

```python
def train_lora_model(
    model: nn.Module,
    train_dataloader,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
):
    """LoRA 파라미터만 학습하는 루프."""
    # LoRA 파라미터만 optimizer에 전달
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    print(f"Optimizer에 전달된 파라미터 그룹: {len(lora_params)}")

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(
                    f"[Epoch {epoch+1}] Step {step+1} | "
                    f"Loss: {epoch_loss/(step+1):.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

    return model
```

### LoRA 가중치 저장/로드

```python
def save_lora_weights(model: nn.Module, save_path: str):
    """LoRA 파라미터만 저장. 용량이 매우 작다."""
    import os
    os.makedirs(save_path, exist_ok=True)

    lora_state_dict = {
        name: param.data
        for name, param in model.named_parameters()
        if "lora_" in name
    }

    torch.save(lora_state_dict, os.path.join(save_path, "lora_weights.pt"))

    total_size = sum(p.numel() * p.element_size() for p in lora_state_dict.values())
    print(f"LoRA 가중치 저장: {save_path}")
    print(f"  파라미터 수: {sum(p.numel() for p in lora_state_dict.values()):,}")
    print(f"  파일 크기: {total_size / 1024 / 1024:.2f} MB")


def load_lora_weights(model: nn.Module, load_path: str) -> nn.Module:
    """저장된 LoRA 가중치를 모델에 로드."""
    import os
    lora_state_dict = torch.load(
        os.path.join(load_path, "lora_weights.pt"),
        map_location="cpu",
    )

    model_state = model.state_dict()
    for name, param in lora_state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)
        else:
            print(f"Warning: {name}이 모델에 없음. 건너뜀.")

    print(f"LoRA 가중치 로드 완료: {load_path}")
    return model
```

---

## 코드: HuggingFace PEFT 사용법

### 방법 2: PEFT 라이브러리 (실전 권장)

```python
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch


def setup_lora_with_peft(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] = None,
):
    """PEFT 라이브러리를 사용한 LoRA 설정."""

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Target modules 기본값
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    # LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",  # bias는 학습하지 않음
        # modules_to_save=["lm_head"],  # 필요시 특정 모듈을 full로 학습
    )

    # LoRA 적용
    model = get_peft_model(model, lora_config)

    # 학습 가능 파라미터 출력
    model.print_trainable_parameters()
    # 출력 예시: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622

    return model, tokenizer, lora_config


def train_with_peft(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    output_dir: str = "./lora-output",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 4,
):
    """HuggingFace Trainer를 사용한 LoRA 학습."""

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        report_to="none",
        gradient_checkpointing=True,
        # DeepSpeed 사용 시
        # deepspeed="ds_config.json",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # 학습 시작
    trainer.train()

    # LoRA 어댑터만 저장 (매우 작은 용량)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model
```

### LoRA 어댑터 로드 및 Merge

```python
def load_and_merge_lora(
    base_model_name: str,
    lora_adapter_path: str,
    merge: bool = True,
):
    """
    저장된 LoRA 어댑터를 로드하고 선택적으로 병합.

    merge=True: W' = W_0 + (alpha/r)*BA 로 병합하여 추론 최적화
    merge=False: 어댑터를 별도로 유지 (추가 학습 가능)
    """
    # 베이스 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    if merge:
        # 병합: 추론 시 추가 연산 없음
        model = model.merge_and_unload()
        print("LoRA 가중치가 베이스 모델에 병합됨.")
        print("추론 시 원래 모델과 동일한 구조로 동작.")
    else:
        print("LoRA 어댑터가 별도로 유지됨.")

    return model


def compare_lora_configs():
    """다양한 LoRA 설정의 파라미터 수 비교."""
    configs = [
        {"r": 4, "alpha": 8, "targets": ["q_proj", "v_proj"]},
        {"r": 8, "alpha": 16, "targets": ["q_proj", "v_proj"]},
        {"r": 16, "alpha": 32, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        {"r": 16, "alpha": 32, "targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                                              "gate_proj", "up_proj", "down_proj"]},
        {"r": 64, "alpha": 128, "targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"]},
    ]

    # LLaMA-7B 기준 차원
    hidden_dim = 4096
    intermediate_dim = 11008  # FFN intermediate size
    num_layers = 32

    print(f"{'Config':<50} {'Params':>12} {'% of 7B':>10}")
    print("-" * 75)

    total_model_params = 7_000_000_000  # 대략적 7B

    for cfg in configs:
        params = 0
        for target in cfg["targets"]:
            if target in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                d, k = hidden_dim, hidden_dim
            elif target in ["gate_proj", "up_proj"]:
                d, k = intermediate_dim, hidden_dim
            elif target == "down_proj":
                d, k = hidden_dim, intermediate_dim
            else:
                continue

            # LoRA 파라미터: r * (d + k) per layer
            params += cfg["r"] * (d + k) * num_layers

        target_str = "+".join(t.replace("_proj", "") for t in cfg["targets"])
        config_str = f"r={cfg['r']}, α={cfg['alpha']}, [{target_str}]"
        pct = params / total_model_params * 100

        print(f"{config_str:<50} {params:>12,} {pct:>9.4f}%")
```

---

## LoRA 변형들

### DoRA (Weight-Decomposed Low-Rank Adaptation)

가중치를 magnitude와 direction으로 분해한 뒤, direction에만 LoRA를 적용:

$$W' = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}$$

여기서 $m$은 학습 가능한 magnitude 벡터, $\|\cdot\|_c$는 column-wise norm이다.

### LoRA+

$A$와 $B$에 서로 다른 학습률을 적용:

$$\eta_B = \eta_A \times \lambda, \quad \lambda > 1$$

$B$에 더 큰 학습률을 적용하면 수렴 속도가 향상된다는 관찰에 기반한다.

### rsLoRA (Rank-Stabilized LoRA)

스케일링 팩터를 $\alpha / r$ 대신 $\alpha / \sqrt{r}$로 변경:

$$\Delta W = \frac{\alpha}{\sqrt{r}} BA$$

높은 rank에서 학습 안정성이 개선된다.

---

## 용어 체크리스트

학습 후 아래 용어를 설명할 수 있는지 점검해봐라:

- [ ] **LoRA**: 왜 $\Delta W$를 $BA$로 분해하는지, 그 수학적 동기를 설명할 수 있는가?
- [ ] **Rank**: $r$의 선택이 성능과 메모리에 어떤 영향을 미치는지 정량적으로 설명할 수 있는가?
- [ ] **Alpha**: $\alpha/r$ 스케일링의 목적과, $\alpha$를 바꾸면 어떤 효과가 있는지 아는가?
- [ ] **Target Modules**: Q, K, V, O, FFN 중 어디에 LoRA를 적용할지 판단 기준을 설명할 수 있는가?
- [ ] **Merge**: $W' = W_0 + (\alpha/r)BA$의 의미와, merge 후 추론 성능이 변하지 않는 이유를 아는가?
- [ ] **SVD**: 특이값 분해와 LoRA의 관계, Eckart-Young-Mirsky 정리를 설명할 수 있는가?
- [ ] **Singular Value Decay**: 사전학습 가중치의 특이값이 급격히 감소하는 현상이 왜 LoRA를 가능하게 하는지 아는가?
- [ ] **초기화**: $B=0$, $A \sim \mathcal{N}(0, 1/\sqrt{r})$인 이유를 설명할 수 있는가?
- [ ] **Gradient 계산**: $\partial \mathcal{L} / \partial A$와 $\partial \mathcal{L} / \partial B$를 직접 유도할 수 있는가?
- [ ] **파라미터 절약**: $d=4096$, $r=16$일 때 절약률을 즉시 계산할 수 있는가?
- [ ] **DoRA, LoRA+, rsLoRA**: 각 변형의 핵심 차이점을 한 문장으로 설명할 수 있는가?
- [ ] **PEFT 라이브러리**: `LoraConfig`, `get_peft_model`, `merge_and_unload` 사용법을 아는가?
