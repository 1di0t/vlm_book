# Chapter 4.1: Fine-tuning 기법

## 개요

VLM Fine-tuning의 핵심 기법인 Full Fine-tuning, LoRA, QLoRA를 다룬다.

---

## 1. Fine-tuning 방법론 비교

| 방법 | 학습 파라미터 | GPU 메모리 | 성능 | 적합한 상황 |
|------|-------------|-----------|------|------------|
| **Full Fine-tuning** | 전체 | 매우 높음 | 최고 | 대규모 데이터, 충분한 리소스 |
| **LoRA** | 1-10% | 낮음 | 우수 | 제한된 리소스, 도메인 적응 |
| **QLoRA** | 1-10% | 매우 낮음 | 우수 | 매우 제한된 리소스 |
| **Adapter** | < 5% | 낮음 | 양호 | 다중 태스크 |

---

## 2. LoRA (Low-Rank Adaptation)

> **논문**: Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
> - arXiv: https://arxiv.org/abs/2106.09685
> - ICLR 2022

### 2.1 핵심 아이디어

기존 가중치 W를 고정하고, 저차원 행렬 분해로 업데이트:

```
W' = W + ΔW = W + BA

W: (d × k) - 원래 가중치 (frozen)
B: (d × r) - 학습 가능
A: (r × k) - 학습 가능
r << min(d, k) - rank (보통 4, 8, 16, 32)
```

**왜 작동하는가?**
- 사전학습된 모델의 가중치 변화는 low intrinsic rank를 가짐
- 적은 파라미터로도 효과적인 적응 가능

### 2.2 구현

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """LoRA 레이어 구현"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0
    ):
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 원래 Linear 레이어 (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False

        # LoRA 파라미터
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 초기화
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 원래 출력 + LoRA 출력
        original = self.linear(x)
        lora = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return original + lora

    def merge_weights(self):
        """LoRA 가중치를 원래 가중치에 병합 (추론 최적화)"""
        self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()


class LoRAModel(nn.Module):
    """LoRA 적용 모델 래퍼"""

    def __init__(self, base_model, target_modules: list, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.base_model = base_model

        # Target modules에 LoRA 적용
        for name, module in base_model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    parent = self._get_parent(base_model, name)
                    attr_name = name.split('.')[-1]

                    lora_layer = LoRALayer(
                        module.in_features,
                        module.out_features,
                        rank=rank,
                        alpha=alpha
                    )
                    lora_layer.linear.weight = module.weight

                    setattr(parent, attr_name, lora_layer)

    def _get_parent(self, model, name):
        """부모 모듈 찾기"""
        parts = name.split('.')[:-1]
        parent = model
        for part in parts:
            parent = getattr(parent, part)
        return parent

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def save_lora_weights(self, path: str):
        """LoRA 가중치만 저장"""
        lora_state = {}
        for name, module in self.base_model.named_modules():
            if isinstance(module, LoRALayer):
                lora_state[f"{name}.lora_A"] = module.lora_A.data
                lora_state[f"{name}.lora_B"] = module.lora_B.data
        torch.save(lora_state, path)

    def load_lora_weights(self, path: str):
        """LoRA 가중치 로드"""
        lora_state = torch.load(path)
        for name, module in self.base_model.named_modules():
            if isinstance(module, LoRALayer):
                module.lora_A.data = lora_state[f"{name}.lora_A"]
                module.lora_B.data = lora_state[f"{name}.lora_B"]
```

### 2.3 Target Modules 선택

| 모델 | 권장 Target Modules |
|------|-------------------|
| LLaMA/Qwen | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj |
| GPT | c_attn, c_proj |
| BERT | query, value |
| ViT | qkv, proj |

**VLM에서의 선택:**
```python
# LLaVA 스타일 VLM
target_modules = [
    # LLM Attention
    "q_proj", "v_proj", "k_proj", "o_proj",
    # LLM MLP
    "gate_proj", "up_proj", "down_proj",
    # Projector (선택적)
    "mm_projector"
]

# Vision Encoder는 보통 제외 (또는 매우 낮은 rank)
```

### 2.4 Rank 선택 가이드

| Rank | 파라미터 | 용도 |
|------|---------|------|
| 4 | 최소 | 가벼운 적응 |
| 8 | 표준 | 대부분의 태스크 |
| 16 | 많음 | 복잡한 도메인 |
| 32+ | 최대 | Full fine-tuning에 가까움 |

---

## 3. QLoRA (Quantized LoRA)

> **논문**: Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs"
> - arXiv: https://arxiv.org/abs/2305.14314
> - NeurIPS 2023

### 3.1 핵심 기술

1. **4-bit NormalFloat (NF4)**: 정규분포 가중치에 최적화된 양자화
2. **Double Quantization**: 양자화 상수도 양자화
3. **Paged Optimizers**: 메모리 스파이크 관리

### 3.2 메모리 비교

| 방법 | 7B 모델 메모리 | 13B 모델 메모리 |
|------|---------------|-----------------|
| Full FP16 | ~14GB | ~26GB |
| LoRA FP16 | ~14GB (+ LoRA) | ~26GB (+ LoRA) |
| QLoRA 4-bit | ~4GB (+ LoRA) | ~8GB (+ LoRA) |

### 3.3 HuggingFace PEFT 사용

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# QLoRA 준비
model = prepare_model_for_kbit_training(model)

# LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA 적용
model = get_peft_model(model, lora_config)

# 학습 가능 파라미터 확인
model.print_trainable_parameters()
# trainable params: 39,976,960 || all params: 3,540,389,888 || trainable%: 1.13%
```

---

## 4. VLM Fine-tuning 전략

### 4.1 컴포넌트별 학습 전략

| 컴포넌트 | 전략 | 이유 |
|---------|------|------|
| Vision Encoder | Freeze 또는 매우 낮은 lr | 이미 강력한 visual representation |
| Projector | 학습 | Vision-Language 연결 필수 |
| LLM | LoRA | 효율적 적응 |

```python
class VLMFineTuner:
    """VLM Fine-tuning 설정"""

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def setup_training(self):
        """컴포넌트별 학습 설정"""

        # 1. Vision Encoder: Freeze (또는 매우 낮은 lr)
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = self.config.get("train_vision", False)

        # 2. Projector: 학습
        for param in self.model.projector.parameters():
            param.requires_grad = True

        # 3. LLM: LoRA 적용
        if self.config.get("use_lora", True):
            self._apply_lora_to_llm()
        else:
            for param in self.model.llm.parameters():
                param.requires_grad = True

    def _apply_lora_to_llm(self):
        """LLM에 LoRA 적용"""
        from peft import get_peft_model, LoraConfig

        lora_config = LoraConfig(
            r=self.config.get("lora_rank", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            target_modules=self.config.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=self.config.get("lora_dropout", 0.05)
        )

        self.model.llm = get_peft_model(self.model.llm, lora_config)

    def get_optimizer_groups(self):
        """컴포넌트별 learning rate 설정"""

        param_groups = []

        # Projector: 기본 lr
        param_groups.append({
            "params": self.model.projector.parameters(),
            "lr": self.config["learning_rate"]
        })

        # LLM LoRA: 기본 lr
        lora_params = [p for n, p in self.model.llm.named_parameters() if "lora" in n]
        param_groups.append({
            "params": lora_params,
            "lr": self.config["learning_rate"]
        })

        # Vision Encoder (학습 시): 매우 낮은 lr
        if self.config.get("train_vision", False):
            param_groups.append({
                "params": self.model.vision_encoder.parameters(),
                "lr": self.config["learning_rate"] * 0.1
            })

        return param_groups


# 사용 예시
config = {
    "learning_rate": 2e-4,
    "use_lora": True,
    "lora_rank": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "train_vision": False
}

finetuner = VLMFineTuner(vlm_model, config)
finetuner.setup_training()
```

### 4.2 데이터 규모별 권장 설정

| 데이터 규모 | Vision Encoder | Projector | LLM | 권장 방법 |
|------------|---------------|-----------|-----|----------|
| < 1K | Freeze | 학습 | LoRA (r=4) | QLoRA |
| 1K - 10K | Freeze | 학습 | LoRA (r=8-16) | LoRA |
| 10K - 100K | 낮은 lr | 학습 | LoRA (r=16-32) | LoRA/Full |
| > 100K | 학습 가능 | 학습 | Full 또는 LoRA | Full |

---

## 5. 학습 설정 예시

### 5.1 Qwen2-VL LoRA Fine-tuning

```yaml
# config.yaml
model:
  name: "Qwen/Qwen2-VL-7B-Instruct"
  vision_encoder_lr_scale: 0.0  # Freeze
  projector_lr_scale: 1.0

lora:
  enabled: true
  r: 64
  alpha: 128
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  dropout: 0.05
  bias: "none"

training:
  output_dir: "./outputs"
  num_train_epochs: 3
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2e-4
  weight_decay: 0.01
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  logging_steps: 10
  save_steps: 500
  eval_steps: 500
  bf16: true
  gradient_checkpointing: true
  dataloader_num_workers: 4

data:
  train_file: "train.json"
  eval_file: "eval.json"
  max_seq_length: 2048
  image_resolution: "dynamic"
```

### 5.2 학습 스크립트

```python
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
import yaml

def main():
    # Config 로드
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # 모델 로드
    model = load_vlm_model(config["model"]["name"])

    # LoRA 적용
    if config["lora"]["enabled"]:
        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            target_modules=config["lora"]["target_modules"],
            lora_dropout=config["lora"]["dropout"],
            bias=config["lora"]["bias"]
        )
        model = get_peft_model(model, lora_config)

    # 데이터 로드
    train_dataset = load_vlm_dataset(config["data"]["train_file"])
    eval_dataset = load_vlm_dataset(config["data"]["eval_file"])

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        warmup_ratio=config["training"]["warmup_ratio"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        bf16=config["training"]["bf16"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        eval_steps=config["training"]["eval_steps"],
        evaluation_strategy="steps",
        save_total_limit=3,
        report_to="wandb"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=vlm_data_collator
    )

    # 학습
    trainer.train()

    # 저장
    trainer.save_model()

if __name__ == "__main__":
    main()
```

---

## 핵심 참고 자료

### 논문
- **LoRA** (Hu et al., 2021)
  - https://arxiv.org/abs/2106.09685
  - GitHub: https://github.com/microsoft/LoRA

- **QLoRA** (Dettmers et al., 2023)
  - https://arxiv.org/abs/2305.14314
  - GitHub: https://github.com/artidoro/qlora

### 라이브러리
- **HuggingFace PEFT**: https://github.com/huggingface/peft
- **bitsandbytes**: https://github.com/TimDettmers/bitsandbytes

---

## 핵심 요약

| 설정 | LoRA | QLoRA | Full FT |
|------|------|-------|---------|
| 메모리 | 낮음 | 매우 낮음 | 높음 |
| 학습 속도 | 빠름 | 보통 | 느림 |
| 성능 | 우수 | 우수 | 최고 |
| 권장 상황 | 일반적 | 리소스 제한 | 대규모 데이터 |
