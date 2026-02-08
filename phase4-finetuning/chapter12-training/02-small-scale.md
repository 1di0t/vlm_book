---
---

# 12.2 Small Scale 학습

> 전체 데이터로 학습하기 전에, 데이터의 1~5%만으로 빠르게 실험해서 최적의 하이퍼파라미터를 찾는 단계다.
> GPU 시간을 효율적으로 쓰려면 이 단계를 절대 건너뛰지 마라.

---

## 핵심 용어

| 용어 | 정의 | 왜 중요한가 |
|------|------|-------------|
| **Data Subset** | 전체 데이터의 일부(1~5%)를 추출한 소규모 데이터셋 | 빠른 실험 반복으로 최적 설정을 탐색할 수 있음 |
| **Hyperparameter Search** | 학습률, 배치 크기, LoRA rank 등 학습 설정을 체계적으로 탐색하는 과정 | 최적 하이퍼파라미터는 모델/데이터 조합마다 다름 |
| **Ablation Study** | 한 번에 하나의 변수만 바꿔가며 각 요소의 기여도를 측정하는 실험 설계 | 어떤 설정이 성능에 실제로 기여하는지 입증 |
| **Learning Rate Schedule** | 학습 과정에서 학습률을 동적으로 조절하는 전략 | 고정 학습률보다 높은 성능과 안정적인 수렴 달성 |
| **Warmup** | 학습 초반에 학습률을 0에서 목표값까지 점진적으로 올리는 기법 | 초기 불안정한 gradient로 인한 학습 발산 방지 |

---

## 12.2.1 Data Subset 구성

### 원칙

소규모 실험에서도 **데이터 분포가 전체 데이터를 대표**해야 한다.
단순히 앞에서 N개를 자르면 특정 패턴에 편향될 수 있다.

### 코드: Stratified Subset 생성

```python
import random
from collections import Counter
from datasets import Dataset, DatasetDict
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def create_representative_subset(
    dataset: Dataset,
    fraction: float = 0.05,
    stratify_column: Optional[str] = None,
    seed: int = 42,
) -> Dataset:
    """
    전체 데이터셋에서 분포를 유지하며 subset을 생성한다.

    Args:
        dataset: 전체 데이터셋
        fraction: 추출 비율 (0.01 ~ 0.1 권장)
        stratify_column: 층화 추출 기준 컬럼 (None이면 랜덤)
        seed: 랜덤 시드

    Returns:
        Dataset: 서브셋
    """
    total = len(dataset)
    target_size = max(int(total * fraction), 1)

    if stratify_column and stratify_column in dataset.column_names:
        # 층화 추출: 각 클래스 비율 유지
        labels = dataset[stratify_column]
        label_counts = Counter(labels)

        selected_indices = []
        random.seed(seed)

        for label, count in label_counts.items():
            label_indices = [i for i, l in enumerate(labels) if l == label]
            n_select = max(int(len(label_indices) * fraction), 1)
            selected = random.sample(label_indices, min(n_select, len(label_indices)))
            selected_indices.extend(selected)

        random.shuffle(selected_indices)
        subset = dataset.select(selected_indices[:target_size])
    else:
        # 단순 랜덤 추출
        subset = dataset.shuffle(seed=seed).select(range(target_size))

    logger.info(
        f"Subset 생성: {total} -> {len(subset)} ({len(subset)/total*100:.1f}%)"
    )

    return subset


def create_experiment_subsets(
    dataset: Dataset,
    fractions: list[float] = [0.01, 0.02, 0.05, 0.10],
    seed: int = 42,
) -> dict[float, Dataset]:
    """다양한 비율의 서브셋을 한 번에 생성한다."""
    subsets = {}
    for frac in fractions:
        subsets[frac] = create_representative_subset(dataset, fraction=frac, seed=seed)
        logger.info(f"  {frac*100:.0f}% subset: {len(subsets[frac])} samples")
    return subsets
```

---

## 수학적 원리

### Learning Rate Schedule

고정 학습률($\eta = \text{const}$)보다 동적 스케줄이 거의 항상 우수하다.
Fine-tuning에서 가장 널리 쓰이는 조합은 **Linear Warmup + Cosine Decay**다.

---

### Cosine Annealing

학습률을 코사인 함수를 따라 서서히 감소시킨다:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t\pi}{T}\right)\right)$$

- $\eta_t$: 스텝 $t$에서의 학습률
- $\eta_{\max}$: 최대 학습률 (warmup 종료 시점의 값)
- $\eta_{\min}$: 최소 학습률 (보통 0 또는 $\eta_{\max}$의 1~10%)
- $T$: 전체 학습 스텝 수
- $t$: 현재 스텝

**왜 Cosine인가?**
- 학습 초기: 큰 학습률로 빠르게 탐색
- 학습 중기: 서서히 줄이며 수렴
- 학습 말기: 매우 작은 학습률로 미세 조정

---

### Linear Warmup

학습 초기에 학습률을 0에서 $\eta_{\max}$까지 선형으로 증가시킨다:

$$\eta_t = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}$$

- $T_{\text{warmup}}$: warmup 스텝 수

**왜 Warmup이 필요한가?**

학습 초반에는 모델의 가중치가 랜덤(또는 pre-trained)이기 때문에 gradient 방향이 불안정하다.
큰 학습률을 바로 적용하면 가중치가 크게 요동치면서 학습이 발산할 수 있다.

---

### Linear Warmup + Cosine Decay 조합

실전에서 가장 많이 쓰이는 스케줄:

$$\eta_t = \begin{cases}
\eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & \text{if } t < T_{\text{warmup}} \\
\eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{(t - T_{\text{warmup}})\pi}{T - T_{\text{warmup}}}\right)\right) & \text{if } t \geq T_{\text{warmup}}
\end{cases}$$

**Warmup ratio 선택 기준:**
- 전체 스텝의 **3~10%**가 표준
- 데이터가 적으면: 3~5% (빠르게 학습 시작)
- 데이터가 많으면: 5~10% (안정적 출발)
- LoRA fine-tuning: 3~5%면 충분 (pre-trained weight가 이미 안정적)

### 코드: Learning Rate Schedule 시각화

```python
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    eta_min_ratio: float = 0.0,
):
    """
    Linear Warmup + Cosine Decay 스케줄러.

    Args:
        optimizer: 옵티마이저
        num_warmup_steps: warmup 스텝 수
        num_training_steps: 전체 학습 스텝 수
        eta_min_ratio: 최소 학습률 비율 (eta_max 대비)

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup 구간
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine Decay 구간
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(eta_min_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


def plot_lr_schedules(
    total_steps: int = 1000,
    max_lr: float = 2e-4,
    warmup_ratios: list[float] = [0.03, 0.05, 0.10],
    save_path: Optional[str] = None,
) -> None:
    """다양한 warmup ratio에 따른 LR Schedule을 비교 시각화한다."""
    import torch

    fig, ax = plt.subplots(figsize=(12, 6))

    for ratio in warmup_ratios:
        warmup_steps = int(total_steps * ratio)

        # 더미 모델/옵티마이저
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([dummy_param], lr=max_lr)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        lrs = []
        for step in range(total_steps):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        ax.plot(lrs, label=f"Warmup {ratio*100:.0f}% ({warmup_steps} steps)")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Linear Warmup + Cosine Decay Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
```

---

## 12.2.2 Hyperparameter Search

### 탐색 대상

Fine-tuning에서 탐색해야 할 주요 하이퍼파라미터:

| 하이퍼파라미터 | 일반적 범위 | 설명 |
|----------------|-------------|------|
| Learning Rate | 1e-5 ~ 5e-4 | 가장 중요. LR Finder 결과 참고 |
| Batch Size | 4 ~ 32 (effective) | 메모리 허용 범위 내 최대 |
| LoRA Rank (r) | 8 ~ 64 | 높을수록 표현력 증가, 파라미터 수 비례 증가 |
| LoRA Alpha | r ~ 2r | 보통 r과 동일하게 설정 |
| Warmup Ratio | 0.03 ~ 0.10 | 전체 스텝의 3~10% |
| Weight Decay | 0.0 ~ 0.1 | 과적합 방지. LoRA에선 0.0~0.01 |
| Epochs | 1 ~ 5 | 데이터 양에 반비례 |
| Max Seq Length | 512 ~ 2048 | 데이터 텍스트 길이에 맞춤 |

### 탐색 전략

**Grid Search**: 모든 조합을 시도 — 비용이 비싸지만 확실.
**Random Search**: 랜덤 조합 N개 — Grid보다 효율적 (Bergstra & Bengio, 2012).
**Bayesian Optimization**: 이전 결과를 기반으로 다음 조합 선택 — 가장 효율적.

Small scale에선 **Random Search**가 가장 실용적:

```python
import itertools
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """하이퍼파라미터 실험 설정."""
    name: str
    learning_rate: float
    batch_size: int
    lora_rank: int
    lora_alpha: int
    warmup_ratio: float
    weight_decay: float
    num_epochs: int
    data_fraction: float = 0.05


def generate_random_configs(
    num_experiments: int = 10,
    seed: int = 42,
) -> list[ExperimentConfig]:
    """
    랜덤 하이퍼파라미터 조합을 생성한다.

    Args:
        num_experiments: 실험 횟수
        seed: 랜덤 시드

    Returns:
        list: ExperimentConfig 리스트
    """
    random.seed(seed)
    configs = []

    # 탐색 범위 정의
    lr_range = (1e-5, 5e-4)
    batch_sizes = [4, 8, 16]
    lora_ranks = [8, 16, 32, 64]
    warmup_ratios = [0.03, 0.05, 0.10]
    weight_decays = [0.0, 0.01, 0.05, 0.1]

    for i in range(num_experiments):
        lr = 10 ** random.uniform(
            np.log10(lr_range[0]), np.log10(lr_range[1])
        )
        batch_size = random.choice(batch_sizes)
        rank = random.choice(lora_ranks)
        warmup = random.choice(warmup_ratios)
        wd = random.choice(weight_decays)

        config = ExperimentConfig(
            name=f"exp_{i:03d}",
            learning_rate=lr,
            batch_size=batch_size,
            lora_rank=rank,
            lora_alpha=rank,  # alpha = rank이 기본
            warmup_ratio=warmup,
            weight_decay=wd,
            num_epochs=3,
            data_fraction=0.05,
        )
        configs.append(config)

    return configs
```

---

## 12.2.3 Small Scale 실험 실행

```python
import torch
import json
import os
from datetime import datetime
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType


def run_small_scale_experiment(
    config: ExperimentConfig,
    full_dataset: Dataset,
    eval_dataset: Dataset,
    model_name: str,
    output_base_dir: str = "./experiments",
    device: str = "cuda",
) -> dict:
    """
    단일 하이퍼파라미터 조합으로 Small Scale 학습을 실행한다.

    Args:
        config: 실험 설정
        full_dataset: 전체 학습 데이터셋
        eval_dataset: 평가 데이터셋
        model_name: 베이스 모델 이름
        output_base_dir: 결과 저장 기본 디렉토리
        device: 디바이스

    Returns:
        dict: 실험 결과
    """
    output_dir = os.path.join(output_base_dir, config.name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"실험 시작: {config.name}")
    logger.info(f"  LR={config.learning_rate:.2e}, BS={config.batch_size}, "
                f"LoRA_r={config.lora_rank}, Warmup={config.warmup_ratio}")

    start_time = datetime.now()

    try:
        # 데이터 서브셋 생성
        train_subset = create_representative_subset(
            full_dataset, fraction=config.data_fraction
        )

        # 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # LoRA 적용
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logger.info(f"  학습 파라미터 수: {trainable_params:,}")

        # 학습 설정
        total_steps = (
            len(train_subset) // config.batch_size * config.num_epochs
        )
        warmup_steps = int(total_steps * config.warmup_ratio)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=config.weight_decay,
            lr_scheduler_type="cosine",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=True,
            report_to="none",
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )

        # Trainer 생성 및 학습
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=eval_dataset,
        )

        train_result = trainer.train()

        # 평가
        eval_result = trainer.evaluate()

        elapsed = (datetime.now() - start_time).total_seconds()

        result = {
            "name": config.name,
            "config": {
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "lora_rank": config.lora_rank,
                "warmup_ratio": config.warmup_ratio,
                "weight_decay": config.weight_decay,
                "data_fraction": config.data_fraction,
            },
            "train_loss": train_result.training_loss,
            "eval_loss": eval_result["eval_loss"],
            "trainable_params": trainable_params,
            "elapsed_seconds": elapsed,
            "total_steps": train_result.global_step,
            "status": "success",
        }

    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"실험 {config.name} 실패: {e}")
        result = {
            "name": config.name,
            "config": vars(config),
            "status": "failed",
            "error": str(e),
            "elapsed_seconds": elapsed,
        }

    # 결과 저장
    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # GPU 메모리 해제
    if "model" in dir():
        del model
    torch.cuda.empty_cache()

    return result


def run_experiment_sweep(
    configs: list[ExperimentConfig],
    full_dataset: Dataset,
    eval_dataset: Dataset,
    model_name: str,
    output_base_dir: str = "./experiments",
) -> list[dict]:
    """
    여러 실험을 순차 실행하고 결과를 비교한다.

    Args:
        configs: 실험 설정 리스트
        full_dataset: 전체 학습 데이터셋
        eval_dataset: 평가 데이터셋
        model_name: 모델 이름
        output_base_dir: 결과 저장 디렉토리

    Returns:
        list: 전체 실험 결과
    """
    all_results = []

    for i, config in enumerate(configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"실험 [{i+1}/{len(configs)}]: {config.name}")
        logger.info(f"{'='*60}")

        result = run_small_scale_experiment(
            config=config,
            full_dataset=full_dataset,
            eval_dataset=eval_dataset,
            model_name=model_name,
            output_base_dir=output_base_dir,
        )
        all_results.append(result)

    # 결과 정렬 및 출력
    print_experiment_comparison(all_results)

    # 전체 결과 저장
    with open(os.path.join(output_base_dir, "sweep_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results
```

---

## 12.2.4 결과 비교 & 분석

```python
from tabulate import tabulate


def print_experiment_comparison(results: list[dict]) -> None:
    """실험 결과를 테이블 형태로 비교 출력한다."""

    # 성공한 실험만 필터
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    if failed:
        logger.warning(f"{len(failed)}개 실험 실패: {[r['name'] for r in failed]}")

    if not successful:
        logger.error("성공한 실험이 없음")
        return

    # eval_loss 기준 정렬
    successful.sort(key=lambda x: x["eval_loss"])

    headers = ["Rank", "Name", "LR", "BS", "LoRA_r", "Warmup",
               "Train Loss", "Eval Loss", "Time(s)"]

    rows = []
    for i, r in enumerate(successful):
        cfg = r["config"]
        rows.append([
            i + 1,
            r["name"],
            f"{cfg['learning_rate']:.2e}",
            cfg["batch_size"],
            cfg["lora_rank"],
            f"{cfg['warmup_ratio']:.0%}",
            f"{r['train_loss']:.4f}",
            f"{r['eval_loss']:.4f}",
            f"{r['elapsed_seconds']:.0f}",
        ])

    print("\n" + "=" * 80)
    print("실험 결과 비교 (Eval Loss 기준 정렬)")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Best config
    best = successful[0]
    print(f"\n최적 설정: {best['name']}")
    print(f"  Eval Loss: {best['eval_loss']:.4f}")
    print(f"  Config: {json.dumps(best['config'], indent=2)}")


def plot_experiment_results(
    results: list[dict],
    save_path: Optional[str] = None,
) -> None:
    """실험 결과를 시각화한다."""
    successful = [r for r in results if r["status"] == "success"]

    if not successful:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. LR vs Eval Loss
    lrs = [r["config"]["learning_rate"] for r in successful]
    eval_losses = [r["eval_loss"] for r in successful]

    axes[0].scatter(lrs, eval_losses, s=80, alpha=0.7)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Learning Rate")
    axes[0].set_ylabel("Eval Loss")
    axes[0].set_title("Learning Rate vs Eval Loss")
    axes[0].grid(True, alpha=0.3)

    # 2. LoRA Rank vs Eval Loss
    ranks = [r["config"]["lora_rank"] for r in successful]
    axes[1].scatter(ranks, eval_losses, s=80, alpha=0.7, color="coral")
    axes[1].set_xlabel("LoRA Rank")
    axes[1].set_ylabel("Eval Loss")
    axes[1].set_title("LoRA Rank vs Eval Loss")
    axes[1].grid(True, alpha=0.3)

    # 3. Trainable Params vs Eval Loss
    params = [r["trainable_params"] for r in successful]
    axes[2].scatter(params, eval_losses, s=80, alpha=0.7, color="green")
    axes[2].set_xlabel("Trainable Parameters")
    axes[2].set_ylabel("Eval Loss")
    axes[2].set_title("Model Size vs Eval Loss")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
```

---

## 12.2.5 Ablation Study 설계

Ablation Study는 "이 설정이 정말 필요한가?"를 검증하는 실험이다.
한 번에 하나의 변수만 바꿔서 각 요소의 기여도를 측정한다.

### Ablation 설계 예시

기준(baseline) 설정을 먼저 정하고, 각 변수를 하나씩 변경:

```python
def generate_ablation_configs(
    baseline: ExperimentConfig,
) -> list[ExperimentConfig]:
    """
    Baseline 기준으로 Ablation Study 설정을 생성한다.

    각 변수를 하나만 바꿔서 기여도를 측정한다.

    Args:
        baseline: 기준 설정

    Returns:
        list: Ablation 실험 설정 리스트
    """
    configs = [baseline]  # baseline 포함

    # --- Ablation 1: LoRA Rank ---
    for rank in [8, 16, 32, 64]:
        if rank == baseline.lora_rank:
            continue
        cfg = ExperimentConfig(
            name=f"ablation_rank_{rank}",
            learning_rate=baseline.learning_rate,
            batch_size=baseline.batch_size,
            lora_rank=rank,
            lora_alpha=rank,
            warmup_ratio=baseline.warmup_ratio,
            weight_decay=baseline.weight_decay,
            num_epochs=baseline.num_epochs,
            data_fraction=baseline.data_fraction,
        )
        configs.append(cfg)

    # --- Ablation 2: Target Modules ---
    # 이건 ExperimentConfig에 필드를 추가하거나 별도 처리 필요
    # 여기서는 개념만 제시
    target_module_sets = {
        "qv_only": ["q_proj", "v_proj"],
        "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    }

    # --- Ablation 3: Data Fraction ---
    for frac in [0.01, 0.02, 0.05, 0.10]:
        if frac == baseline.data_fraction:
            continue
        cfg = ExperimentConfig(
            name=f"ablation_data_{int(frac*100)}pct",
            learning_rate=baseline.learning_rate,
            batch_size=baseline.batch_size,
            lora_rank=baseline.lora_rank,
            lora_alpha=baseline.lora_alpha,
            warmup_ratio=baseline.warmup_ratio,
            weight_decay=baseline.weight_decay,
            num_epochs=baseline.num_epochs,
            data_fraction=frac,
        )
        configs.append(cfg)

    # --- Ablation 4: Learning Rate ---
    for lr_mult in [0.2, 0.5, 2.0, 5.0]:
        lr = baseline.learning_rate * lr_mult
        cfg = ExperimentConfig(
            name=f"ablation_lr_{lr:.0e}",
            learning_rate=lr,
            batch_size=baseline.batch_size,
            lora_rank=baseline.lora_rank,
            lora_alpha=baseline.lora_alpha,
            warmup_ratio=baseline.warmup_ratio,
            weight_decay=baseline.weight_decay,
            num_epochs=baseline.num_epochs,
            data_fraction=baseline.data_fraction,
        )
        configs.append(cfg)

    logger.info(f"Ablation 실험 {len(configs)}개 설정 생성 (baseline 포함)")
    return configs
```

### Ablation 결과 해석 기준

| 변수 | 기대 패턴 | 주의 |
|------|-----------|------|
| LoRA Rank 증가 | 성능 향상 후 수렴 (rank 32 이후 효과 감소) | Rank가 크면 과적합 위험 |
| Target Modules 추가 | 더 많은 모듈 = 더 높은 표현력 | 학습 시간/메모리 비용 증가 |
| Data Fraction 증가 | 성능 개선, 수확 체감 | 데이터 양 대비 비용-효과 분석 |
| Learning Rate | 최적점 존재, 너무 크면 발산 | LR Finder 결과와 교차 검증 |

---

## 12.2.6 실전 워크플로우

```
1. Data Subset 생성 (5%)
       ↓
2. Random Search (10~20개 조합)
       ↓
3. Top-3 설정 선택
       ↓
4. Ablation Study (각 변수별 상세 탐색)
       ↓
5. 최종 설정 확정
       ↓
6. Full Scale 학습 (Chapter 12.3)
```

이 과정에서 쓰는 GPU 시간은 전체의 10% 미만이지만,
Full Scale에서의 성공 확률을 크게 높여준다.

---

## 용어 체크리스트

학습 전 아래 용어를 모두 설명할 수 있는지 확인하라:

- [ ] **Data Subset**: 전체 데이터의 일부를 추출한 소규모 데이터셋. 빠른 실험 반복에 사용.
- [ ] **Stratified Sampling**: 클래스 비율을 유지하며 데이터를 추출하는 방법.
- [ ] **Hyperparameter Search**: 학습 설정을 체계적으로 탐색하는 과정. Grid, Random, Bayesian 방식.
- [ ] **Random Search**: 하이퍼파라미터 공간에서 무작위 조합을 선택하는 탐색 방법. Grid보다 효율적.
- [ ] **Ablation Study**: 한 번에 하나의 변수만 변경하여 각 요소의 기여도를 측정하는 실험 방법.
- [ ] **Cosine Annealing**: 학습률을 코사인 함수 형태로 서서히 감소시키는 스케줄. $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(t\pi/T))$.
- [ ] **Linear Warmup**: 학습 초반에 학습률을 0에서 목표값까지 선형으로 증가시키는 기법. 초기 발산 방지.
- [ ] **Warmup Ratio**: 전체 학습 스텝 중 warmup에 할당하는 비율. 보통 3~10%.
- [ ] **Weight Decay**: L2 정규화의 일종. 가중치가 커지는 것을 억제하여 과적합 방지.
- [ ] **Effective Batch Size**: `per_device_batch_size * num_devices * gradient_accumulation_steps`. 실제 한 번의 업데이트에 사용되는 샘플 수.
