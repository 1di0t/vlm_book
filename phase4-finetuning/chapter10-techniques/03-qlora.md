# 10.3 QLoRA (Quantized LoRA)

## 핵심 용어 박스

| 용어 | 정의 |
|------|------|
| **QLoRA** | 사전학습 가중치를 **4-bit로 양자화**한 상태에서 LoRA를 적용하는 기법. 단일 48GB GPU로 65B 모델 fine-tuning이 가능하다. Dettmers et al. (2023) 발표. |
| **NormalFloat4 (NF4)** | 정규분포를 가정하여 양자화 수준(quantization level)을 분위수(quantile) 위치에 배치하는 4-bit 데이터 타입. 일반 INT4보다 정보 손실이 적다. |
| **Double Quantization** | 양자화 상수(quantization constant) 자체를 **FP8로 재양자화**하여 메모리를 추가 절약하는 기법. 블록당 상수 저장 비용을 32bit → 8bit로 줄인다. |
| **Paged Optimizer** | GPU 메모리가 부족할 때 optimizer states를 **CPU RAM으로 자동 스왑**하는 메커니즘. NVIDIA unified memory를 활용한다. |
| **Block-wise Quantization** | 가중치 텐서를 일정 크기 블록(예: 64개)으로 나누어 블록별로 독립적 양자화를 수행. 블록마다 별도의 scale/zero-point를 가진다. |
| **Dequantization** | 양자화된 가중치를 연산 시점에 원래 정밀도(FP16/BF16)로 복원하는 과정. Forward pass마다 수행되며, 추가 연산 비용 발생. |

---

## 개요

QLoRA는 세 가지 핵심 기술을 결합한다:

1. **NormalFloat4 (NF4) 양자화**: 사전학습 가중치를 4-bit로 압축
2. **Double Quantization**: 양자화 상수의 메모리 비용도 줄임
3. **Paged Optimizers**: GPU-CPU 메모리 스왑으로 OOM 방지

이 조합으로 **16-bit LoRA와 동일한 성능**을 유지하면서 **메모리를 3~4배 절약**한다.

**왜 QLoRA가 중요한가?**
- 7B 모델: 단일 RTX 3090 (24GB)에서 fine-tuning 가능
- 13B 모델: 단일 A100 40GB에서 fine-tuning 가능
- 70B 모델: 단일 A100 80GB에서 fine-tuning 가능

---

## 수학적 원리

### 1. 양자화 기본: Uniform Quantization

$b$-bit 양자화의 기본 공식:

$$q = \text{round}\left(\frac{x - x_{\min}}{x_{\max} - x_{\min}} \times (2^b - 1)\right)$$

역양자화(dequantization):

$$\hat{x} = q \times \frac{x_{\max} - x_{\min}}{2^b - 1} + x_{\min}$$

또는 scale-zero point 형식으로:

$$\hat{x} = s \cdot q + z$$

여기서:
- $s = \frac{x_{\max} - x_{\min}}{2^b - 1}$: scale factor
- $z = x_{\min}$: zero point

**4-bit Uniform 양자화의 문제**: 양자화 레벨이 $2^4 = 16$개밖에 없어서, 값의 분포가 불균등하면 정보 손실이 크다.

### 2. NormalFloat4 (NF4): 정규분포 최적 양자화

핵심 관찰: **사전학습된 neural network 가중치는 대략 정규분포를 따른다.**

$$w \sim \mathcal{N}(0, \sigma^2)$$

NF4는 이 가정 하에 양자화 레벨을 **정규분포의 분위수(quantile)**에 배치한다.

**Step 1: 분위수 계산**

표준정규분포 $\mathcal{N}(0, 1)$의 역 CDF(percent point function)를 사용:

$$q_i = \Phi^{-1}\left(\frac{i + 0.5}{2^b}\right), \quad i = 0, 1, \ldots, 2^b - 1$$

$b = 4$일 때 $2^4 = 16$개의 양자화 레벨:

$$q_i = \Phi^{-1}\left(\frac{i + 0.5}{16}\right)$$

| $i$ | $(i+0.5)/16$ | $q_i = \Phi^{-1}(\cdot)$ |
|-----|---------------|--------------------------|
| 0 | 0.03125 | -1.863 |
| 1 | 0.09375 | -1.318 |
| 2 | 0.15625 | -1.010 |
| 3 | 0.21875 | -0.776 |
| 4 | 0.28125 | -0.577 |
| 5 | 0.34375 | -0.403 |
| 6 | 0.40625 | -0.237 |
| 7 | 0.46875 | -0.078 |
| 8 | 0.53125 | 0.078 |
| 9 | 0.59375 | 0.237 |
| 10 | 0.65625 | 0.403 |
| 11 | 0.71875 | 0.577 |
| 12 | 0.78125 | 0.776 |
| 13 | 0.84375 | 1.010 |
| 14 | 0.90625 | 1.318 |
| 15 | 0.96875 | 1.863 |

**Step 2: 정규화 및 양자화**

가중치를 absmax로 정규화하고 가장 가까운 $q_i$에 매핑:

$$w_{\text{norm}} = \frac{w}{\max(|w|)}$$

$$\text{quantize}(w_{\text{norm}}) = \arg\min_{q_i} |w_{\text{norm}} - q_i|$$

**왜 NF4가 Uniform INT4보다 좋은가?**

정규분포에서 대부분의 값은 0 근처에 밀집되어 있다. Uniform 양자화는 이 밀집 구간에 불충분한 레벨을 할당하고, 드문 극값에 불필요하게 많은 레벨을 낭비한다. NF4는 **정보 이론적으로 최적(information-theoretically optimal)**인 양자화로, 각 레벨이 동일한 확률 질량(probability mass)을 가진다.

정보 이론적으로, 양자화 오차의 기대값을 최소화하는 것은:

$$\min_{q_0, \ldots, q_{2^b-1}} \mathbb{E}_{w \sim \mathcal{N}(0,1)} \left[ (w - \text{quantize}(w))^2 \right]$$

NF4의 분위수 배치가 이 문제의 (근사적) 최적해다.

### 3. Block-wise Quantization

텐서 전체를 하나의 scale로 양자화하면 값 범위가 다른 영역에서 정밀도가 떨어진다. Block-wise 양자화는 이를 해결한다:

텐서 $W$를 크기 $B$의 블록으로 분할:

$$W = [W^{(1)}, W^{(2)}, \ldots, W^{(n)}], \quad |W^{(i)}| = B$$

각 블록별 독립 양자화:

$$s^{(i)} = \max(|W^{(i)}|)$$

$$W^{(i)}_{\text{quant}} = \text{NF4}\left(\frac{W^{(i)}}{s^{(i)}}\right)$$

역양자화:

$$\hat{W}^{(i)} = s^{(i)} \cdot \text{dequant}(W^{(i)}_{\text{quant}})$$

일반적으로 block size $B = 64$를 사용한다.

**메모리 계산 (블록 양자화만):**

$$\text{bits/param} = 4 + \frac{32}{B} = 4 + \frac{32}{64} = 4.5 \text{ bits}$$

4-bit 양자화 자체에 4bit, 블록당 FP32 scale 상수에 $32/64 = 0.5$bit 추가.

### 4. Double Quantization: 상수의 양자화

Block-wise 양자화의 scale 상수 $s^{(i)}$가 FP32 (32bit)이면, 블록이 많을수록 상수 저장 비용이 무시할 수 없다.

**Double Quantization**은 이 scale 상수들을 **FP8로 재양자화**한다:

$$s^{(i)}_{\text{quant}} = \text{FP8}(s^{(i)})$$

구체적으로:
1. $k$개의 블록 scale을 하나의 "슈퍼블록"으로 묶음 (기본 $k = 256$)
2. 슈퍼블록 내 scale 값들을 FP8로 양자화
3. 슈퍼블록의 scale 상수 1개만 FP32로 저장

**메모리 계산 (Double Quantization):**

기존 블록 양자화:

$$\text{bits/param} = 4 + \frac{32}{B}$$

Double Quantization 적용 후:

$$\text{bits/param} = 4 + \frac{8}{B} + \frac{32}{B \times k}$$

$B = 64$, $k = 256$ 대입:

$$= 4 + \frac{8}{64} + \frac{32}{64 \times 256} = 4 + 0.125 + 0.00195 \approx 4.127 \text{ bits}$$

| 양자화 방식 | bits/param | 7B 모델 메모리 |
|------------|-----------|--------------|
| FP16 | 16.0 | 14.0 GB |
| INT8 | 8.0 | 7.0 GB |
| NF4 (블록) | 4.5 | 3.9 GB |
| NF4 + Double Quant | 4.127 | 3.6 GB |

### 5. QLoRA Forward Pass

QLoRA의 forward pass는 다음과 같이 동작한다:

$$h = \text{dequant}(W_{\text{NF4}}) \cdot x + \frac{\alpha}{r} B A x$$

단계별:
1. NF4 가중치를 BF16으로 역양자화: $\hat{W} = \text{dequant}(W_{\text{NF4}})$
2. 역양자화된 가중치로 base output 계산: $h_{\text{base}} = \hat{W} \cdot x$
3. LoRA output 계산 (BF16): $h_{\text{lora}} = \frac{\alpha}{r} B A x$
4. 합산: $h = h_{\text{base}} + h_{\text{lora}}$

**Backward pass에서:**
- $W_{\text{NF4}}$에 대한 gradient는 계산하지 않음 (동결)
- $B$, $A$에 대한 gradient만 BF16으로 계산
- Optimizer states ($m$, $v$)도 LoRA 파라미터에 대해서만 유지

### 6. Paged Optimizer

GPU 메모리가 부족할 때 optimizer states를 CPU로 자동 스왑한다.

**NVIDIA Unified Memory 활용:**
- GPU 메모리 부족 시 자동으로 CPU RAM으로 페이지 전환
- 필요할 때 다시 GPU로 가져옴
- `bitsandbytes` 라이브러리의 `PagedAdamW` 구현

$$\text{GPU 메모리} = W_{\text{NF4}} + B_{\text{BF16}} + A_{\text{BF16}} + \text{activations}$$

$$\text{CPU 메모리} = m_{\text{FP32}} + v_{\text{FP32}} \quad (\text{필요시 스왑})$$

일반적 상황에서는 GPU 메모리가 충분하므로 paged optimizer가 개입하지 않지만, 긴 시퀀스나 큰 배치에서 일시적으로 메모리가 부족할 때 OOM 대신 graceful degradation을 제공한다.

---

## 메모리 비교표

### Full FT vs LoRA vs QLoRA (실측 기반 추정)

**7B 모델 (LLaMA-2-7B 기준)**

| 항목 | Full FT (FP16) | LoRA (FP16) | QLoRA (NF4) |
|------|---------------|-------------|-------------|
| 모델 가중치 | 14.0 GB | 14.0 GB | 3.6 GB |
| LoRA 파라미터 | - | 0.08 GB | 0.08 GB |
| Optimizer States | 56.0 GB | 0.32 GB | 0.32 GB |
| Gradient | 14.0 GB | 0.08 GB | 0.08 GB |
| 활성화 메모리 | ~8 GB | ~8 GB | ~6 GB |
| **총 추정** | **~92 GB** | **~22 GB** | **~10 GB** |
| 최소 GPU | A100 80GB × 2 | A100 40GB × 1 | **RTX 3090 24GB** |

**13B 모델**

| 항목 | Full FT (FP16) | LoRA (FP16) | QLoRA (NF4) |
|------|---------------|-------------|-------------|
| 모델 가중치 | 26.0 GB | 26.0 GB | 6.7 GB |
| Optimizer States | 104.0 GB | 0.52 GB | 0.52 GB |
| **총 추정** | **~170 GB** | **~42 GB** | **~18 GB** |
| 최소 GPU | A100 80GB × 4 | A100 80GB × 1 | **A100 40GB** |

**70B 모델**

| 항목 | Full FT (FP16) | LoRA (FP16) | QLoRA (NF4) |
|------|---------------|-------------|-------------|
| 모델 가중치 | 140.0 GB | 140.0 GB | 36.2 GB |
| Optimizer States | 560.0 GB | 2.1 GB | 2.1 GB |
| **총 추정** | **~920 GB** | **~180 GB** | **~48 GB** |
| 최소 GPU | A100 80GB × 16 | A100 80GB × 4 | **A100 80GB × 1** |

> QLoRA 덕분에 70B 모델도 **단일 A100 80GB**에서 fine-tuning이 가능하다. Full FT로는 16장이 필요한 것과 비교하면 혁신적인 차이다.

---

## 성능 비교: QLoRA vs Full Fine-tuning

Dettmers et al. (2023) 논문 결과 요약:

| 벤치마크 | Full FT (16-bit) | LoRA (16-bit) | QLoRA (4-bit) |
|----------|-----------------|---------------|---------------|
| MMLU (5-shot) | 63.2 | 62.8 | **63.0** |
| ARC (25-shot) | 57.1 | 56.5 | 56.8 |
| HellaSwag | 82.4 | 82.1 | 82.0 |
| 평균 | 67.6 | 67.1 | 67.3 |

4-bit 양자화임에도 **16-bit Full FT와 거의 동일한 성능**을 보인다.

---

## 코드: BitsAndBytes 설정 + QLoRA 학습

### QLoRA 설정 및 모델 로드

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QLoRAConfig:
    """QLoRA 학습 설정."""
    model_name: str = "meta-llama/Llama-2-7b-hf"
    # 양자화 설정
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"       # "nf4" 또는 "fp4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    use_double_quant: bool = True           # Double Quantization
    # LoRA 설정
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = None
    # 학습 설정
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    output_dir: str = "./qlora-output"
    # Paged Optimizer
    optim: str = "paged_adamw_32bit"


def setup_qlora(config: QLoRAConfig):
    """QLoRA 모델 설정."""

    # 1. BitsAndBytes 양자화 설정
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.use_double_quant,
    )

    logger.info(f"양자화 설정: {config.bnb_4bit_quant_type}, "
                f"compute_dtype={config.bnb_4bit_compute_dtype}, "
                f"double_quant={config.use_double_quant}")

    # 2. 모델 로드 (4-bit 양자화 적용)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. k-bit 학습 준비
    # gradient checkpointing 활성화, 입력 임베딩 gradient 활성화
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # 4. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. LoRA 설정
    if config.target_modules is None:
        config.target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )

    # 6. LoRA 적용
    model = get_peft_model(model, lora_config)

    # 메모리 사용량 출력
    model.print_trainable_parameters()
    _print_memory_usage(model)

    return model, tokenizer, lora_config


def _print_memory_usage(model):
    """모델 메모리 사용량 출력."""
    total_mem = 0
    for param in model.parameters():
        mem = param.numel() * param.element_size()
        total_mem += mem

    # 4-bit 파라미터는 element_size()가 1 (uint8 컨테이너)이므로 보정
    logger.info(f"모델 메모리 (추정): {total_mem / 1024**3:.2f} GB")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU 메모리 - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
```

### QLoRA 학습 루프 (SFTTrainer 사용)

```python
def train_qlora(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    config: QLoRAConfig = None,
):
    """SFTTrainer를 사용한 QLoRA 학습."""
    if config is None:
        config = QLoRAConfig()

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        # Mixed Precision: 4-bit 모델이므로 bf16 사용
        bf16=True,
        fp16=False,
        # Paged Optimizer
        optim=config.optim,
        # 메모리 최적화
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # 로깅
        logging_steps=10,
        logging_first_step=True,
        # 평가 및 저장
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        # 기타
        report_to="none",
        max_grad_norm=1.0,
        # 데이터 로더
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        packing=True,  # 짧은 시퀀스를 묶어서 효율적 배치 구성
    )

    # 학습 전 메모리 상태
    if torch.cuda.is_available():
        logger.info(f"학습 전 GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # 학습
    train_result = trainer.train()

    # 학습 후 메모리 상태
    if torch.cuda.is_available():
        logger.info(f"학습 후 GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # 결과 저장
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    metrics = train_result.metrics
    logger.info(f"학습 완료: {metrics}")

    return trainer, metrics
```

### QLoRA 수동 학습 루프 (저수준 제어)

```python
def train_qlora_manual(
    model,
    train_dataloader,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 4,
    use_paged_optimizer: bool = True,
):
    """
    수동 QLoRA 학습 루프.
    SFTTrainer 없이 직접 제어하고 싶을 때 사용.
    """
    import bitsandbytes as bnb
    from transformers import get_cosine_schedule_with_warmup

    # LoRA 파라미터만 수집
    lora_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
    ]
    logger.info(f"학습 대상 파라미터: {sum(p.numel() for p in lora_params):,}")

    # Paged Optimizer 설정
    if use_paged_optimizer:
        optimizer = bnb.optim.PagedAdamW32bit(
            lora_params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        logger.info("Paged AdamW 32bit optimizer 사용")
    else:
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=learning_rate,
            weight_decay=0.01,
        )

    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            # Forward (4-bit 가중치는 자동으로 dequantize → compute → quantize)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / gradient_accumulation_steps

            loss.backward()
            epoch_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 20 == 0:
                    avg_loss = epoch_loss / (step + 1)
                    lr = scheduler.get_last_lr()[0]
                    gpu_mem = torch.cuda.memory_allocated() / 1024**3
                    logger.info(
                        f"[Epoch {epoch+1}] Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                        f"GPU: {gpu_mem:.1f}GB"
                    )

        logger.info(f"[Epoch {epoch+1}] 평균 Loss: {epoch_loss/len(train_dataloader):.4f}")

    return model
```

### NF4 양자화 시뮬레이션 (교육용)

```python
import numpy as np
from scipy.stats import norm


def simulate_nf4_quantization(weights: np.ndarray, block_size: int = 64):
    """
    NF4 양자화 시뮬레이션.
    실제 bitsandbytes 구현을 간소화한 교육용 코드.
    """
    # NF4 양자화 레벨 생성 (정규분포 분위수)
    num_levels = 16  # 4-bit → 2^4 = 16
    nf4_levels = np.array([
        norm.ppf((i + 0.5) / num_levels)
        for i in range(num_levels)
    ])

    # 0을 정확히 표현하기 위해 대칭 조정
    # 실제 NF4는 [-1, 1] 범위로 정규화 후 0을 포함하도록 비대칭 처리
    nf4_levels = nf4_levels / np.max(np.abs(nf4_levels))

    print(f"NF4 양자화 레벨 ({num_levels}개):")
    for i, level in enumerate(nf4_levels):
        print(f"  [{i:2d}] = {level:+.4f}")

    # 블록 단위 양자화
    n = len(weights)
    num_blocks = (n + block_size - 1) // block_size

    quantized = np.zeros_like(weights)
    indices = np.zeros(n, dtype=np.int8)
    scales = np.zeros(num_blocks)

    total_error = 0.0

    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        block = weights[start:end]

        # Absmax 정규화
        scale = np.max(np.abs(block))
        scales[b] = scale

        if scale == 0:
            continue

        normalized = block / scale

        # 가장 가까운 NF4 레벨 찾기
        for i, val in enumerate(normalized):
            idx = np.argmin(np.abs(nf4_levels - val))
            indices[start + i] = idx
            quantized[start + i] = nf4_levels[idx] * scale

        # 블록 양자화 오차
        block_error = np.mean((block - quantized[start:end]) ** 2)
        total_error += block_error

    rmse = np.sqrt(total_error / num_blocks)
    max_error = np.max(np.abs(weights - quantized))

    print(f"\n양자화 결과:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Max Error: {max_error:.6f}")
    print(f"  블록 수: {num_blocks}, 블록 크기: {block_size}")

    # 메모리 계산
    original_bits = n * 32  # FP32
    quant_bits = n * 4 + num_blocks * 32  # 4-bit indices + FP32 scales
    print(f"  압축률: {original_bits / quant_bits:.2f}x")
    print(f"  bits/param: {quant_bits / n:.2f}")

    return quantized, indices, scales


def compare_quantization_methods(num_samples: int = 10000):
    """Uniform INT4 vs NF4 양자화 비교."""
    # 정규분포 가중치 생성
    np.random.seed(42)
    weights = np.random.randn(num_samples).astype(np.float32) * 0.02

    print("=" * 60)
    print("Uniform INT4 양자화")
    print("=" * 60)

    # Uniform INT4
    w_min, w_max = weights.min(), weights.max()
    scale_uniform = (w_max - w_min) / 15
    q_uniform = np.round((weights - w_min) / scale_uniform).clip(0, 15)
    dq_uniform = q_uniform * scale_uniform + w_min
    rmse_uniform = np.sqrt(np.mean((weights - dq_uniform) ** 2))
    print(f"RMSE: {rmse_uniform:.6f}")

    print("\n" + "=" * 60)
    print("NF4 양자화")
    print("=" * 60)

    quantized_nf4, _, _ = simulate_nf4_quantization(weights)
    rmse_nf4 = np.sqrt(np.mean((weights - quantized_nf4) ** 2))

    print(f"\n{'='*60}")
    print(f"비교 결과:")
    print(f"  Uniform INT4 RMSE: {rmse_uniform:.6f}")
    print(f"  NF4 RMSE:          {rmse_nf4:.6f}")
    print(f"  NF4 개선율: {(1 - rmse_nf4/rmse_uniform)*100:.1f}%")
```

### Double Quantization 시뮬레이션

```python
def simulate_double_quantization(
    weights: np.ndarray,
    block_size: int = 64,
    super_block_size: int = 256,
):
    """
    Double Quantization 시뮬레이션.
    1차: 가중치를 NF4로 양자화 (scale은 FP32)
    2차: scale 상수들을 FP8로 재양자화
    """
    n = len(weights)
    num_blocks = (n + block_size - 1) // block_size

    # 1차 양자화: 블록별 absmax scale 계산
    first_scales = np.zeros(num_blocks)
    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        first_scales[b] = np.max(np.abs(weights[start:end]))

    # 2차 양자화: scale을 FP8로 양자화 (시뮬레이션)
    num_super_blocks = (num_blocks + super_block_size - 1) // super_block_size

    second_scales = np.zeros(num_super_blocks)
    quantized_first_scales = np.zeros(num_blocks)

    for sb in range(num_super_blocks):
        start = sb * super_block_size
        end = min(start + super_block_size, num_blocks)
        sub_scales = first_scales[start:end]

        # 슈퍼블록의 absmax
        second_scale = np.max(np.abs(sub_scales))
        second_scales[sb] = second_scale

        if second_scale == 0:
            continue

        # FP8 양자화 시뮬레이션 (E4M3: 지수4비트, 가수3비트)
        # 간소화: 256 레벨로 uniform 양자화
        normalized = sub_scales / second_scale
        quantized = np.round(normalized * 127) / 127  # 7-bit 정밀도 근사
        quantized_first_scales[start:end] = quantized * second_scale

    # 메모리 계산
    # 원본: n × 32bit
    # 1차만: n × 4bit + num_blocks × 32bit
    # Double: n × 4bit + num_blocks × 8bit + num_super_blocks × 32bit
    original_bits = n * 32
    single_quant_bits = n * 4 + num_blocks * 32
    double_quant_bits = n * 4 + num_blocks * 8 + num_super_blocks * 32

    print(f"Double Quantization 결과:")
    print(f"  가중치 수: {n:,}")
    print(f"  1차 블록 수: {num_blocks:,} (크기: {block_size})")
    print(f"  2차 슈퍼블록 수: {num_super_blocks:,} (크기: {super_block_size})")
    print(f"\n메모리:")
    print(f"  원본 FP32:       {original_bits/8/1024:.2f} KB ({original_bits/n:.1f} bits/param)")
    print(f"  1차 양자화 (NF4): {single_quant_bits/8/1024:.2f} KB ({single_quant_bits/n:.2f} bits/param)")
    print(f"  Double Quant:     {double_quant_bits/8/1024:.2f} KB ({double_quant_bits/n:.3f} bits/param)")
    print(f"\n  추가 절약: {(single_quant_bits - double_quant_bits)/8/1024:.2f} KB "
          f"({(1 - double_quant_bits/single_quant_bits)*100:.1f}%)")

    return double_quant_bits / n
```

---

## 실전 QLoRA 레시피

### 모델 크기별 추천 설정

| 설정 | 7B | 13B | 70B |
|------|-----|------|------|
| GPU | RTX 3090/4090 (24GB) | A100 40GB | A100 80GB |
| Quant Type | NF4 | NF4 | NF4 |
| Double Quant | True | True | True |
| Compute Dtype | BF16 | BF16 | BF16 |
| LoRA Rank | 16 | 16 | 16~32 |
| LoRA Alpha | 32 | 32 | 32~64 |
| Batch Size | 4 | 2 | 1 |
| Grad Accum | 4 | 8 | 16 |
| LR | 2e-4 | 2e-4 | 1e-4 |
| Optimizer | Paged AdamW | Paged AdamW | Paged AdamW |
| Max Seq Len | 2048 | 2048 | 2048 |
| Gradient Ckpt | True | True | True |

### QLoRA의 한계

1. **추론 속도 저하**: 매 forward pass마다 dequantization 필요 → FP16 대비 ~10% 느림
2. **양자화 오차 누적**: 학습 중 base weight의 미세한 오차가 LoRA 학습에 영향
3. **Merge 후 정밀도**: 4-bit에서 merge하면 정보 손실. FP16으로 dequant → merge → 재양자화 권장
4. **특수 하드웨어 의존**: bitsandbytes가 CUDA만 지원 (AMD, Apple Silicon은 제한적)

---

## 용어 체크리스트

학습 후 아래 용어를 설명할 수 있는지 점검해봐라:

- [ ] **QLoRA**: LoRA와의 차이점, 세 가지 핵심 기술을 나열할 수 있는가?
- [ ] **NF4**: Uniform INT4 대비 왜 더 좋은지, 분위수 배치의 수학적 원리를 설명할 수 있는가?
- [ ] **역 CDF ($\Phi^{-1}$)**: NF4 양자화 레벨 생성 수식을 직접 작성할 수 있는가?
- [ ] **Block-wise Quantization**: 왜 블록 단위로 양자화하는지, block size의 영향을 아는가?
- [ ] **Double Quantization**: 메모리 절약량을 수식으로 계산할 수 있는가? (4.5bit → 4.127bit)
- [ ] **Paged Optimizer**: GPU-CPU 메모리 스왑이 언제 발생하는지, 성능 영향을 아는가?
- [ ] **Dequantization**: Forward pass에서 어떤 순서로 연산이 일어나는지 설명할 수 있는가?
- [ ] **BitsAndBytesConfig**: `load_in_4bit`, `bnb_4bit_quant_type`, `bnb_4bit_compute_dtype`, `bnb_4bit_use_double_quant` 각 옵션의 역할을 아는가?
- [ ] **prepare_model_for_kbit_training**: 이 함수가 내부적으로 무엇을 하는지 아는가?
- [ ] **메모리 비교**: 7B 모델 기준 Full FT, LoRA, QLoRA의 메모리 차이를 즉시 설명할 수 있는가?
