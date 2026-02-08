---
---

# 11.3 Config 예시

## 핵심 용어 박스

| 용어 | 정의 |
|------|------|
| **LoRA (Low-Rank Adaptation)** | 사전학습된 가중치를 동결하고, 저랭크 행렬 $A$, $B$를 추가하여 $\Delta W = BA$로 적응하는 파라미터 효율적 기법. 학습 파라미터가 전체의 0.1~1%에 불과하다. |
| **Learning Rate** | 파라미터 업데이트 크기를 결정하는 스칼라. Full fine-tuning은 $10^{-5}$, LoRA는 $10^{-4}$ 대역이 일반적이다. |
| **Gradient Accumulation** | 여러 미니배치의 gradient를 누적한 뒤 한 번에 업데이트. GPU 메모리 부족 시 effective batch size를 키우는 방법. |
| **LoRA Rank ($r$)** | 저랭크 근사의 차원. $r$이 클수록 표현력이 높지만 파라미터가 증가한다. OCR VLM에서 보통 $r=8 \sim 64$. |
| **LoRA Alpha ($\alpha$)** | LoRA의 스케일링 계수. 실제 스케일은 $\alpha / r$로 적용된다. 보통 $\alpha = 2r$로 설정. |
| **Target Modules** | LoRA를 적용할 모델 레이어. 일반적으로 `q_proj`, `v_proj`에 적용하고, 성능이 부족하면 `k_proj`, `o_proj`, `gate_proj` 등을 추가한다. |
| **Warmup Ratio** | 전체 학습 스텝 중 learning rate를 0에서 목표값까지 점진적으로 올리는 비율. 보통 3~10%. |

---

## 개요

이 챕터에서는 **Qwen2.5-VL + LoRA + DeepSpeed**를 기반으로 한 실전 학습 설정을 다룬다. 각 하이퍼파라미터의 의미와 선택 기준, 그리고 실제 config 파일 작성법을 제시한다.

핵심은 세 가지 설정 파일이다:

| 파일 | 용도 | 형식 |
|------|------|------|
| `training_config.yaml` | 학습 전반 설정 (모델, 데이터, 하이퍼파라미터) | YAML |
| `lora_config.yaml` | LoRA 설정 (rank, alpha, target_modules) | YAML |
| `ds_config.json` | DeepSpeed 최적화 설정 (ZeRO, precision) | JSON |

---

## 수학적 원리

### 1. Learning Rate 선택의 근거

Optimizer의 파라미터 업데이트:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

$\eta$ (learning rate)가 너무 크면:
- $\|\theta_{t+1} - \theta_t\|$가 커져서 사전학습 지식을 파괴 (catastrophic forgetting)
- loss landscape의 sharp minima를 건너뛰어 발산

$\eta$가 너무 작으면:
- 수렴이 느림
- local minima에 빠질 가능성 증가

**Full Fine-tuning vs LoRA의 learning rate 차이:**

Full fine-tuning에서 업데이트되는 파라미터 수: $|\theta| = N$ (전체)
LoRA에서 업데이트되는 파라미터 수: $|\theta_{\text{LoRA}}| = 2 \times r \times d \times L_{\text{target}}$

LoRA 파라미터는 0에서 시작하므로 (A는 random init, B는 zero init) 상대적으로 큰 업데이트가 필요하다. 따라서 learning rate가 Full의 5~30배 높다:

| 기법 | 권장 LR 범위 | 이유 |
|------|-------------|------|
| Full Fine-tuning | $1 \times 10^{-5}$ ~ $5 \times 10^{-5}$ | 이미 학습된 가중치를 미세 조정 |
| LoRA | $1 \times 10^{-4}$ ~ $3 \times 10^{-4}$ | 새로운 저랭크 행렬을 빠르게 학습 |
| QLoRA | $1 \times 10^{-4}$ ~ $3 \times 10^{-4}$ | LoRA와 동일 (양자화는 base model만) |

### 2. Batch Size와 Learning Rate의 관계

Linear Scaling Rule:

$$\eta_{\text{new}} = \eta_{\text{base}} \times \frac{B_{\text{new}}}{B_{\text{base}}}$$

Effective batch size $B$를 $k$배 키우면, learning rate도 $k$배 키워야 동일한 학습 역학을 유지한다.

단, 너무 큰 batch size는 generalization gap을 키울 수 있다:

$$\text{Generalization Error} \propto \sqrt{\frac{B}{N_{\text{data}}}}$$

따라서 실전에서는 linear scaling 후에도 약간의 튜닝이 필요하다.

**Gradient Accumulation과의 관계:**

$$B_{\text{eff}} = B_{\text{micro}} \times N_{\text{gpu}} \times N_{\text{accum}}$$

예: micro batch 2, GPU 4, accumulation 4 → effective batch size = 32

### 3. LoRA Rank와 표현력

LoRA에서 원래 가중치 $W_0 \in \mathbb{R}^{d \times k}$에 대해:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A$$

여기서 $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$.

$\Delta W$의 rank는 최대 $r$이므로, $r$이 클수록 더 복잡한 adaptation이 가능하다.

**파라미터 수:**

원래 가중치: $d \times k$ 파라미터
LoRA: $r \times (d + k)$ 파라미터

비율: $\frac{r(d+k)}{dk} = r \left(\frac{1}{k} + \frac{1}{d}\right)$

Qwen2.5-VL-7B 기준 ($d = k = 4096$):
- $r=8$: $8 \times 8192 = 65536$ (원래 $16M$의 0.39%)
- $r=32$: $32 \times 8192 = 262144$ (원래의 1.56%)
- $r=64$: $64 \times 8192 = 524288$ (원래의 3.13%)

### 4. LoRA Alpha의 스케일링 효과

실제 LoRA 출력:

$$h = W_0 x + \frac{\alpha}{r} B A x$$

$\alpha / r$이 스케일링 팩터다. 관례적으로 $\alpha = 2r$로 설정하면:

$$\frac{\alpha}{r} = 2$$

이 경우 LoRA의 기여가 원래 가중치의 출력과 비슷한 스케일이 된다.

| $r$ | $\alpha$ | 스케일($\alpha/r$) | 특성 |
|-----|----------|-------------------|------|
| 8 | 16 | 2.0 | 표준 |
| 16 | 32 | 2.0 | 표준 |
| 32 | 32 | 1.0 | 보수적 (기존 지식 보존) |
| 64 | 128 | 2.0 | 표준 |
| 8 | 32 | 4.0 | 공격적 (빠른 적응) |

---

## Config 파일: YAML 기반 학습 설정

### `training_config.yaml`

```yaml
# =============================================================
# Qwen2.5-VL + LoRA Fine-tuning Configuration
# OCR Book Detection / Text Recognition 태스크
# =============================================================

# --- 모델 설정 ---
model:
  name: "Qwen/Qwen2.5-VL-7B-Instruct"
  revision: "main"
  torch_dtype: "bfloat16"
  attn_implementation: "flash_attention_2"  # flash_attention_2 또는 sdpa
  trust_remote_code: true

# --- 데이터 설정 ---
data:
  train_path: "./data/train.jsonl"
  eval_path: "./data/eval.jsonl"
  max_length: 2048                # 최대 시퀀스 길이
  max_image_size: 1280            # 이미지 최대 해상도 (장변)
  min_image_size: 256             # 이미지 최소 해상도
  image_factor: 28                # Qwen2.5-VL의 이미지 패치 크기
  num_workers: 4
  preprocessing:
    shuffle: true
    seed: 42

# --- 학습 하이퍼파라미터 ---
training:
  num_epochs: 3
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 8  # effective batch = 2 × 4gpu × 8 = 64
  learning_rate: 2.0e-4           # LoRA 학습률
  weight_decay: 0.01
  max_grad_norm: 1.0
  warmup_ratio: 0.05              # 전체 스텝의 5%
  lr_scheduler_type: "cosine"
  logging_steps: 10
  eval_steps: 200
  save_steps: 500
  save_total_limit: 3             # 최대 체크포인트 수
  metric_for_best_model: "eval_loss"
  greater_is_better: false

# --- 출력 설정 ---
output:
  dir: "./outputs/qwen25vl-lora-ocr"
  logging_dir: "./logs/qwen25vl-lora-ocr"
  report_to: "wandb"              # wandb, tensorboard, none

# --- 환경 설정 ---
environment:
  seed: 42
  dataloader_pin_memory: true
  bf16: true
  fp16: false
  gradient_checkpointing: true    # activation 메모리 절약
  deepspeed: "./configs/ds_config_zero2.json"
```

### `lora_config.yaml`

```yaml
# =============================================================
# LoRA Configuration for Qwen2.5-VL
# =============================================================

lora:
  r: 32                           # LoRA rank
  lora_alpha: 64                  # 스케일링 계수 (alpha/r = 2.0)
  lora_dropout: 0.05              # dropout 비율
  bias: "none"                    # none, all, lora_only

  # 적용 대상 모듈
  target_modules:
    - "q_proj"                    # Query projection
    - "k_proj"                    # Key projection
    - "v_proj"                    # Value projection
    - "o_proj"                    # Output projection
    - "gate_proj"                 # MLP gate
    - "up_proj"                   # MLP up
    - "down_proj"                 # MLP down

  # LoRA 적용 범위
  modules_to_save: null           # LoRA 외에 full fine-tune할 모듈 (예: embed_tokens)
  task_type: "CAUSAL_LM"

  # VLM 특수 설정
  vision_lora:
    enabled: false                # Vision Encoder에도 LoRA 적용 여부
    vision_target_modules:
      - "qkv"
    vision_r: 8                   # Vision LoRA는 작은 rank로 충분
    vision_alpha: 16

# --- LoRA Rank 선택 가이드 ---
# r=8:   가장 가벼움. 간단한 태스크 (분류, 단순 추출)
# r=16:  범용적. 대부분의 태스크에 적합
# r=32:  OCR처럼 세밀한 텍스트 인식이 필요한 경우 권장
# r=64:  표현력 최대. 도메인 gap이 클 때 (의료, 법률 등)
# r=128: Full fine-tuning에 근접. 메모리 이점이 줄어듦

# --- Target Modules 선택 가이드 ---
# 최소: q_proj, v_proj              → 빠른 실험용
# 표준: q_proj, k_proj, v_proj, o_proj → 일반적 권장
# 최대: + gate_proj, up_proj, down_proj → 최고 성능 추구
```

---

## Config 파일: DeepSpeed JSON

### `ds_config_zero2.json` (권장 기본값)

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "steps_per_print": 100,
    "wall_clock_breakdown": false,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    }
}
```

### `ds_config_zero3_offload.json` (메모리 극한 절약)

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "sub_group_size": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    }
}
```

---

## 코드: 설정 로더

### Config 로더 구현

```python
import yaml
import json
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    revision: str = "main"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    trust_remote_code: bool = True


@dataclass
class DataConfig:
    train_path: str = "./data/train.jsonl"
    eval_path: str = "./data/eval.jsonl"
    max_length: int = 2048
    max_image_size: int = 1280
    min_image_size: int = 256
    num_workers: int = 4
    seed: int = 42


@dataclass
class TrainingConfig:
    num_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 500
    save_total_limit: int = 3


@dataclass
class LoRAConfig:
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    task_type: str = "CAUSAL_LM"
    vision_lora_enabled: bool = False
    vision_r: int = 8
    vision_alpha: int = 16


@dataclass
class OutputConfig:
    dir: str = "./outputs"
    logging_dir: str = "./logs"
    report_to: str = "wandb"


@dataclass
class FullConfig:
    """전체 학습 설정을 하나로 통합."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    deepspeed_config: Optional[str] = None
    seed: int = 42


class ConfigLoader:
    """YAML/JSON 기반 설정 로더."""

    @staticmethod
    def load_yaml(path: str) -> dict:
        """YAML 파일 로드."""
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"YAML 설정 로드: {path}")
        return config

    @staticmethod
    def load_json(path: str) -> dict:
        """JSON 파일 로드."""
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        logger.info(f"JSON 설정 로드: {path}")
        return config

    @classmethod
    def load_full_config(
        cls,
        training_config_path: str,
        lora_config_path: Optional[str] = None,
        deepspeed_config_path: Optional[str] = None,
    ) -> FullConfig:
        """전체 설정 로드 및 병합."""

        # 기본 설정
        full_config = FullConfig()

        # Training config 로드
        if os.path.exists(training_config_path):
            raw = cls.load_yaml(training_config_path)
            cls._update_dataclass(full_config.model, raw.get("model", {}))
            cls._update_dataclass(full_config.data, raw.get("data", {}))
            cls._update_dataclass(full_config.training, raw.get("training", {}))
            cls._update_dataclass(full_config.output, raw.get("output", {}))

            env = raw.get("environment", {})
            full_config.seed = env.get("seed", 42)
            full_config.deepspeed_config = env.get("deepspeed", deepspeed_config_path)

        # LoRA config 로드
        if lora_config_path and os.path.exists(lora_config_path):
            raw = cls.load_yaml(lora_config_path)
            lora_raw = raw.get("lora", {})
            cls._update_dataclass(full_config.lora, lora_raw)

            # Vision LoRA 설정
            vision_lora = lora_raw.get("vision_lora", {})
            if vision_lora.get("enabled", False):
                full_config.lora.vision_lora_enabled = True
                full_config.lora.vision_r = vision_lora.get("vision_r", 8)
                full_config.lora.vision_alpha = vision_lora.get("vision_alpha", 16)

        # DeepSpeed config 경로 확인
        if deepspeed_config_path:
            full_config.deepspeed_config = deepspeed_config_path

        cls._validate_config(full_config)
        return full_config

    @staticmethod
    def _update_dataclass(dc, updates: dict):
        """Dataclass의 필드를 딕셔너리로 업데이트."""
        for key, value in updates.items():
            if hasattr(dc, key):
                setattr(dc, key, value)

    @staticmethod
    def _validate_config(config: FullConfig):
        """설정 유효성 검증."""
        errors = []

        # Learning rate 범위 확인
        lr = config.training.learning_rate
        if lr > 1e-2:
            errors.append(f"Learning rate {lr}이 너무 크다. 1e-2 이하로 설정해라.")
        if lr < 1e-7:
            errors.append(f"Learning rate {lr}이 너무 작다. 1e-7 이상으로 설정해라.")

        # LoRA rank 확인
        if config.lora.r < 1 or config.lora.r > 256:
            errors.append(f"LoRA rank {config.lora.r}가 비정상 범위다. 1~256 사이로 설정해라.")

        # Batch size 확인
        if config.training.per_device_train_batch_size < 1:
            errors.append("Batch size는 1 이상이어야 한다.")

        # Gradient accumulation 확인
        if config.training.gradient_accumulation_steps < 1:
            errors.append("Gradient accumulation steps는 1 이상이어야 한다.")

        # Warmup ratio 범위
        if not 0 <= config.training.warmup_ratio <= 0.5:
            errors.append(f"Warmup ratio {config.training.warmup_ratio}는 0~0.5 범위여야 한다.")

        if errors:
            for e in errors:
                logger.error(f"Config 검증 실패: {e}")
            raise ValueError(f"Config 검증 실패: {'; '.join(errors)}")

        logger.info("Config 검증 통과")


def print_config_summary(config: FullConfig):
    """설정 요약 출력."""
    import torch

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch = (
        config.training.per_device_train_batch_size
        * num_gpus
        * config.training.gradient_accumulation_steps
    )

    logger.info("=" * 60)
    logger.info("학습 설정 요약")
    logger.info("=" * 60)
    logger.info(f"모델: {config.model.name}")
    logger.info(f"정밀도: {config.model.torch_dtype}")
    logger.info(f"LoRA rank: {config.lora.r} (alpha: {config.lora.lora_alpha})")
    logger.info(f"LoRA scale: {config.lora.lora_alpha / config.lora.r:.1f}")
    logger.info(f"Target modules: {config.lora.target_modules}")
    logger.info(f"Learning rate: {config.training.learning_rate:.1e}")
    logger.info(f"GPU 수: {num_gpus}")
    logger.info(f"Per-GPU batch: {config.training.per_device_train_batch_size}")
    logger.info(f"Gradient accum: {config.training.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {effective_batch}")
    logger.info(f"Epochs: {config.training.num_epochs}")
    logger.info(f"DeepSpeed: {config.deepspeed_config or 'None'}")
    logger.info("=" * 60)
```

---

## 코드: 학습 스크립트 엔트리포인트

### `train.py` (메인 엔트리포인트)

```python
"""
Qwen2.5-VL + LoRA Fine-tuning 엔트리포인트.

사용법:
    # 단일 GPU
    python train.py --config configs/training_config.yaml --lora_config configs/lora_config.yaml

    # 멀티 GPU (DeepSpeed)
    deepspeed --num_gpus=4 train.py \
        --config configs/training_config.yaml \
        --lora_config configs/lora_config.yaml \
        --deepspeed configs/ds_config_zero2.json
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL LoRA Fine-tuning")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="학습 설정 YAML 경로",
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        default=None,
        help="LoRA 설정 YAML 경로",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="DeepSpeed config JSON 경로",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="분산 학습 local rank (DeepSpeed 자동 설정)",
    )
    return parser.parse_args()


def setup_seed(seed: int):
    """재현성을 위한 시드 설정."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic 모드 (성능 약간 감소)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_and_processor(config):
    """모델과 프로세서 로드."""
    logger.info(f"모델 로드: {config.model.name}")

    # dtype 매핑
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.model.torch_dtype, torch.bfloat16)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model.name,
        torch_dtype=torch_dtype,
        attn_implementation=config.model.attn_implementation,
        trust_remote_code=config.model.trust_remote_code,
    )

    processor = AutoProcessor.from_pretrained(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
    )

    return model, processor


def apply_lora(model, lora_config):
    """LoRA 적용."""
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        target_modules=lora_config.target_modules,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)

    # 학습 가능 파라미터 통계
    model.print_trainable_parameters()

    return model


def create_training_arguments(config) -> TrainingArguments:
    """HuggingFace TrainingArguments 생성."""
    args = TrainingArguments(
        output_dir=config.output.dir,
        logging_dir=config.output.logging_dir,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        eval_strategy="steps",
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=config.data.num_workers,
        report_to=config.output.report_to,
        seed=config.seed,
        deepspeed=config.deepspeed_config,
        remove_unused_columns=False,
    )
    return args


def load_datasets(config, processor):
    """데이터셋 로드 및 전처리."""
    logger.info("데이터셋 로드 중...")

    train_dataset = load_dataset(
        "json",
        data_files=config.data.train_path,
        split="train",
    )

    eval_dataset = None
    if config.data.eval_path and os.path.exists(config.data.eval_path):
        eval_dataset = load_dataset(
            "json",
            data_files=config.data.eval_path,
            split="train",
        )

    logger.info(f"Train: {len(train_dataset)} samples")
    if eval_dataset:
        logger.info(f"Eval: {len(eval_dataset)} samples")

    return train_dataset, eval_dataset


def main():
    args = parse_args()

    # 설정 로드
    config = ConfigLoader.load_full_config(
        training_config_path=args.config,
        lora_config_path=args.lora_config,
        deepspeed_config_path=args.deepspeed,
    )

    # 시드 설정
    setup_seed(config.seed)

    # 설정 요약 출력
    print_config_summary(config)

    # 모델 + 프로세서 로드
    model, processor = load_model_and_processor(config)

    # LoRA 적용
    model = apply_lora(model, config.lora)

    # 데이터셋 로드
    train_dataset, eval_dataset = load_datasets(config, processor)

    # TrainingArguments 생성
    training_args = create_training_arguments(config)

    # Trainer 생성 및 실행
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("학습 시작")
    train_result = trainer.train()

    # 최종 모델 저장
    trainer.save_model(os.path.join(config.output.dir, "final"))
    logger.info(f"최종 모델 저장: {config.output.dir}/final")

    # 학습 결과 출력
    metrics = train_result.metrics
    logger.info(f"학습 완료: {metrics}")

    return metrics


if __name__ == "__main__":
    main()
```

---

## 하이퍼파라미터 튜닝 가이드

### 권장 시작점

| 파라미터 | OCR (문서) | OCR (수식) | 일반 VQA |
|----------|-----------|-----------|----------|
| LoRA rank | 32 | 64 | 16 |
| LoRA alpha | 64 | 128 | 32 |
| Learning rate | 2e-4 | 1e-4 | 2e-4 |
| Batch size (eff) | 32~64 | 16~32 | 64~128 |
| Epochs | 3~5 | 5~10 | 1~3 |
| Warmup ratio | 0.05 | 0.1 | 0.03 |
| Max seq length | 2048 | 4096 | 1024 |

### 튜닝 우선순위

1. **Learning rate** (가장 영향력이 큼)
   - 너무 높으면: loss spike, NaN
   - 너무 낮으면: 수렴 느림, underfitting
   - 탐색: {5e-5, 1e-4, 2e-4, 3e-4, 5e-4}

2. **LoRA rank** (성능-효율 트레이드오프)
   - 작으면: 빠르지만 표현력 제한
   - 크면: 느리지만 성능 향상 (수렴)
   - 탐색: {8, 16, 32, 64}

3. **Effective batch size** (안정성에 영향)
   - 작으면: 노이즈 크지만 generalization 좋을 수 있음
   - 크면: 안정적이지만 linear scaling rule 적용 필요
   - 탐색: {16, 32, 64, 128}

4. **Epochs** (과적합 vs 학습 부족)
   - 데이터가 적으면: 1~2 epoch (과적합 위험)
   - 데이터가 많으면: 3~5 epoch
   - Early stopping으로 자동 결정 권장

---

## 용어 체크리스트

학습 후 아래 용어를 설명할 수 있는지 점검해봐라:

- [ ] **LoRA rank**: $r$ 값에 따른 파라미터 수 변화와 표현력의 관계를 계산할 수 있는가?
- [ ] **LoRA alpha**: $\alpha / r$ 스케일링이 왜 필요한지, $\alpha = 2r$이 관례인 이유를 아는가?
- [ ] **Target modules**: `q_proj`, `v_proj`만 적용할 때와 전체 적용할 때의 차이를 설명할 수 있는가?
- [ ] **Learning rate 범위**: Full fine-tuning과 LoRA에서 LR이 다른 이유를 수학적으로 설명할 수 있는가?
- [ ] **Linear Scaling Rule**: batch size를 키울 때 LR도 키워야 하는 이유를 아는가?
- [ ] **Gradient Accumulation**: effective batch size 계산법과 loss 나누기의 의미를 이해하는가?
- [ ] **Warmup ratio**: 왜 필요하고, 너무 길거나 짧으면 어떤 문제가 생기는지 아는가?
- [ ] **DeepSpeed ZeRO Stage**: YAML의 `deepspeed` 필드와 JSON config의 관계를 이해하는가?
- [ ] **Cosine scheduler**: 왜 LLM fine-tuning에서 step decay보다 선호되는지 설명할 수 있는가?
- [ ] **Config 검증**: learning rate, rank, batch size의 유효 범위를 알고, 비정상 값을 감지할 수 있는가?
