---
---

# 12.3 Full Scale 학습

> Small Scale 실험에서 확정된 최적 하이퍼파라미터로, 전체 데이터를 사용해 본격적으로 학습하는 단계다.
> 체크포인트 관리, Early Stopping, 메모리 최적화가 핵심이다.

---

## 핵심 용어

| 용어 | 정의 | 왜 중요한가 |
|------|------|-------------|
| **Checkpoint** | 학습 중간 상태(모델 가중치, 옵티마이저 상태, 스텝 수 등)를 디스크에 저장한 것 | 학습 중단 시 처음부터 다시 시작하지 않아도 됨 |
| **Resume Training** | 저장된 체크포인트에서 학습을 이어서 재개하는 것 | GPU 장애, OOM 등 예기치 않은 중단 시 필수 |
| **Early Stopping** | Validation loss가 더 이상 개선되지 않으면 학습을 조기 종료하는 기법 | 과적합 방지, GPU 시간 절약 |
| **Evaluation Loop** | 학습 중 주기적으로 검증 데이터로 모델 성능을 측정하는 과정 | 과적합 탐지, 최적 모델 선택 기준 |
| **Gradient Accumulation** | 여러 미니배치의 gradient를 누적한 후 한 번에 가중치 업데이트하는 기법 | 메모리 부족 시 effective batch size를 키울 수 있음 |
| **Gradient Checkpointing** | Forward pass의 중간 결과를 저장하지 않고, backward 시 재계산하는 메모리 절약 기법 | VRAM 사용량을 30~50% 절감 (속도 20~30% 감소 트레이드오프) |

---

## 12.3.1 전체 학습 설정

Small Scale 실험 결과를 바탕으로 최종 설정을 확정한다.

### 코드: 전체 학습 TrainingArguments

```python
import os
import json
import torch
import logging
from datetime import datetime
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset

logger = logging.getLogger(__name__)


def create_full_scale_training_args(
    output_dir: str,
    # === Small Scale에서 확정된 값 ===
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 8,
    num_epochs: int = 3,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    # === Full Scale 전용 설정 ===
    gradient_accumulation_steps: int = 4,
    gradient_checkpointing: bool = True,
    save_steps: int = 500,
    eval_steps: int = 500,
    save_total_limit: int = 3,
    logging_steps: int = 10,
    max_grad_norm: float = 1.0,
    bf16: bool = True,
    dataloader_num_workers: int = 4,
    resume_from_checkpoint: str = None,
) -> TrainingArguments:
    """
    Full Scale 학습용 TrainingArguments를 생성한다.

    Args:
        output_dir: 체크포인트 및 로그 저장 디렉토리
        learning_rate: 학습률
        per_device_batch_size: 디바이스당 배치 크기
        num_epochs: 전체 에폭 수
        warmup_ratio: warmup 비율
        weight_decay: 가중치 감쇄
        gradient_accumulation_steps: gradient 누적 스텝
        gradient_checkpointing: gradient checkpointing 사용 여부
        save_steps: 체크포인트 저장 간격 (스텝)
        eval_steps: 평가 간격 (스텝)
        save_total_limit: 최대 체크포인트 보존 수
        logging_steps: 로깅 간격
        max_grad_norm: gradient clipping 최대 norm
        bf16: bfloat16 사용 여부
        dataloader_num_workers: 데이터로더 워커 수
        resume_from_checkpoint: 이어서 학습할 체크포인트 경로

    Returns:
        TrainingArguments
    """
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps
    logger.info(f"Effective Batch Size: {effective_batch_size}")
    logger.info(f"Gradient Checkpointing: {gradient_checkpointing}")

    args = TrainingArguments(
        output_dir=output_dir,

        # 학습 하이퍼파라미터
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        max_grad_norm=max_grad_norm,

        # 메모리 최적화
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        bf16=bf16,
        tf32=True,

        # 체크포인트 & 평가
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # 로깅
        logging_steps=logging_steps,
        logging_first_step=True,
        report_to=["tensorboard"],

        # 기타
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,

        # Resume
        resume_from_checkpoint=resume_from_checkpoint,
    )

    return args
```

---

## 12.3.2 체크포인트 전략

### 왜 체크포인트가 중요한가

Full Scale 학습은 수 시간~수 일이 걸린다.
학습 중 GPU 장애, OOM, 전원 문제 등이 발생하면 체크포인트 없이는 처음부터 다시 해야 한다.

### 체크포인트 관리 정책

| 정책 | 설정 | 설명 |
|------|------|------|
| **주기적 저장** | `save_steps=500` | 매 500 스텝마다 저장 |
| **Best Model 보존** | `load_best_model_at_end=True` | eval_loss 기준 최선의 모델 유지 |
| **디스크 관리** | `save_total_limit=3` | 최근 3개만 보존, 이전 것은 자동 삭제 |
| **Adapter만 저장** | LoRA 사용 시 자동 | Base model 제외, LoRA 가중치만 저장 |

### 코드: 체크포인트 관리 유틸리티

```python
import shutil
from pathlib import Path
from typing import Optional


class CheckpointManager:
    """학습 체크포인트를 관리하는 유틸리티."""

    def __init__(self, output_dir: str, save_total_limit: int = 3):
        self.output_dir = Path(output_dir)
        self.save_total_limit = save_total_limit
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoints(self) -> list[Path]:
        """저장된 체크포인트 목록을 반환한다 (오래된 순)."""
        checkpoints = sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        return checkpoints

    def get_latest_checkpoint(self) -> Optional[str]:
        """가장 최근 체크포인트 경로를 반환한다."""
        checkpoints = self.get_checkpoints()
        if checkpoints:
            return str(checkpoints[-1])
        return None

    def get_best_checkpoint(self) -> Optional[str]:
        """
        Best model 체크포인트를 찾는다.
        trainer_state.json에서 best_model_checkpoint를 읽는다.
        """
        state_file = self.output_dir / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)
            best = state.get("best_model_checkpoint")
            if best and Path(best).exists():
                return best
        return self.get_latest_checkpoint()

    def cleanup_old_checkpoints(self) -> None:
        """save_total_limit을 초과하는 오래된 체크포인트를 삭제한다."""
        checkpoints = self.get_checkpoints()

        if len(checkpoints) <= self.save_total_limit:
            return

        to_delete = checkpoints[: len(checkpoints) - self.save_total_limit]
        for ckpt in to_delete:
            logger.info(f"오래된 체크포인트 삭제: {ckpt}")
            shutil.rmtree(ckpt)

    def get_disk_usage(self) -> dict:
        """체크포인트가 차지하는 디스크 용량을 확인한다."""
        checkpoints = self.get_checkpoints()
        total_bytes = 0
        details = []

        for ckpt in checkpoints:
            size = sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file())
            total_bytes += size
            details.append({
                "name": ckpt.name,
                "size_mb": size / (1024 * 1024),
            })

        return {
            "total_mb": total_bytes / (1024 * 1024),
            "total_gb": total_bytes / (1024 ** 3),
            "num_checkpoints": len(checkpoints),
            "details": details,
        }

    def export_best_model(self, export_dir: str) -> str:
        """Best model을 별도 디렉토리로 내보낸다."""
        best_ckpt = self.get_best_checkpoint()
        if best_ckpt is None:
            raise ValueError("저장된 체크포인트가 없음")

        export_path = Path(export_dir)
        if export_path.exists():
            shutil.rmtree(export_path)

        shutil.copytree(best_ckpt, export_path)
        logger.info(f"Best model 내보내기 완료: {export_path}")
        return str(export_path)
```

---

## 12.3.3 Early Stopping

### 개념

Validation loss가 일정 횟수(patience) 연속으로 개선되지 않으면 학습을 조기 종료한다.
과적합을 방지하고, 불필요한 GPU 시간을 절약한다.

### 코드: Early Stopping Callback

```python
class EarlyStoppingCallback(TrainerCallback):
    """
    Validation loss 기반 Early Stopping.

    Args:
        patience: 성능 개선 없이 허용하는 평가 횟수
        min_delta: 개선으로 인정하는 최소 loss 감소량
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_step = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ):
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        if eval_loss < self.best_loss - self.min_delta:
            # 개선됨
            self.best_loss = eval_loss
            self.counter = 0
            self.best_step = state.global_step
            logger.info(
                f"[EarlyStopping] 개선: eval_loss={eval_loss:.4f} "
                f"(step {state.global_step})"
            )
        else:
            # 개선 안 됨
            self.counter += 1
            logger.info(
                f"[EarlyStopping] 개선 없음 ({self.counter}/{self.patience}): "
                f"eval_loss={eval_loss:.4f}, best={self.best_loss:.4f}"
            )

            if self.counter >= self.patience:
                logger.info(
                    f"[EarlyStopping] 학습 조기 종료. "
                    f"Best: step {self.best_step}, loss={self.best_loss:.4f}"
                )
                control.should_training_stop = True


class DetailedLoggingCallback(TrainerCallback):
    """학습 중 상세 정보를 로깅하는 콜백."""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        logger.info(f"학습 시작: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"총 스텝: {state.max_steps}")
        logger.info(f"에폭 수: {args.num_train_epochs}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.log_interval != 0:
            return

        if logs is None:
            return

        elapsed = (datetime.now() - self.start_time).total_seconds()
        steps_per_sec = state.global_step / max(elapsed, 1)
        remaining_steps = state.max_steps - state.global_step
        eta_seconds = remaining_steps / max(steps_per_sec, 1e-6)
        eta_hours = eta_seconds / 3600

        # GPU 메모리 사용량
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / (1024 ** 3)
            mem_total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            mem_pct = mem_used / mem_total * 100
        else:
            mem_used = mem_total = mem_pct = 0

        logger.info(
            f"[Step {state.global_step}/{state.max_steps}] "
            f"Loss: {logs.get('loss', 'N/A'):.4f} | "
            f"LR: {logs.get('learning_rate', 0):.2e} | "
            f"Speed: {steps_per_sec:.2f} steps/s | "
            f"GPU: {mem_used:.1f}/{mem_total:.1f}GB ({mem_pct:.0f}%) | "
            f"ETA: {eta_hours:.1f}h"
        )

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"학습 완료. 총 소요 시간: {elapsed/3600:.2f}시간")
```

---

## 12.3.4 메모리 최적화

### Gradient Accumulation

메모리가 부족해서 원하는 batch size를 쓸 수 없을 때,
여러 미니배치의 gradient를 누적한 후 한 번에 업데이트한다.

**Effective Batch Size** 계산:

$$\text{Effective BS} = \text{per\_device\_bs} \times \text{num\_devices} \times \text{grad\_accum\_steps}$$

예시: `per_device_bs=4`, `num_devices=1`, `grad_accum_steps=8` → Effective BS = 32

### Gradient Checkpointing

Forward pass에서 중간 활성화값(activation)을 메모리에 저장하지 않고,
backward pass에서 필요할 때 다시 계산한다.

| 항목 | 일반 | Gradient Checkpointing |
|------|------|------------------------|
| VRAM 사용 | 높음 (모든 activation 저장) | 30~50% 절감 |
| 학습 속도 | 기준 | 20~30% 느림 |
| 수치 정확도 | 기준 | 동일 (재계산이므로) |

### 코드: 메모리 최적화 설정

```python
def setup_memory_optimization(
    model: torch.nn.Module,
    use_gradient_checkpointing: bool = True,
    use_cpu_offload: bool = False,
) -> torch.nn.Module:
    """
    모델에 메모리 최적화를 적용한다.

    Args:
        model: 대상 모델
        use_gradient_checkpointing: gradient checkpointing 사용
        use_cpu_offload: CPU offloading 사용 (극한 상황)

    Returns:
        최적화된 모델
    """
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Gradient Checkpointing 활성화")

    if use_cpu_offload:
        # DeepSpeed ZeRO-Offload 또는 FSDP CPU offload
        logger.info("CPU Offloading 활성화 — 속도가 크게 저하될 수 있음")

    # 불필요한 캐시 비활성화
    if hasattr(model, "config"):
        model.config.use_cache = False

    return model


def log_memory_stats(prefix: str = "") -> dict:
    """현재 GPU 메모리 상태를 로깅한다."""
    if not torch.cuda.is_available():
        return {}

    stats = {
        "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
        "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "free_gb": (
            torch.cuda.get_device_properties(0).total_mem
            - torch.cuda.memory_allocated()
        ) / (1024 ** 3),
    }

    logger.info(
        f"[Memory {prefix}] "
        f"Allocated: {stats['allocated_gb']:.2f}GB | "
        f"Reserved: {stats['reserved_gb']:.2f}GB | "
        f"Peak: {stats['max_allocated_gb']:.2f}GB | "
        f"Free: {stats['free_gb']:.2f}GB"
    )

    return stats
```

---

## 12.3.5 전체 학습 루프

모든 구성 요소를 통합한 Full Scale 학습 함수:

```python
def run_full_scale_training(
    model_name: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: str,
    # === 최적 하이퍼파라미터 (Small Scale에서 확정) ===
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 8,
    num_epochs: int = 3,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_target_modules: list[str] = None,
    # === 메모리 최적화 ===
    gradient_accumulation_steps: int = 4,
    gradient_checkpointing: bool = True,
    # === 체크포인트 & 평가 ===
    save_steps: int = 500,
    eval_steps: int = 500,
    save_total_limit: int = 3,
    # === Early Stopping ===
    early_stopping_patience: int = 3,
    early_stopping_min_delta: float = 0.001,
    # === Resume ===
    resume_from_checkpoint: str = None,
) -> dict:
    """
    Full Scale 학습을 실행한다.

    Args:
        model_name: 베이스 모델 이름/경로
        train_dataset: 학습 데이터셋
        eval_dataset: 평가 데이터셋
        output_dir: 출력 디렉토리
        ...: 하이퍼파라미터들

    Returns:
        dict: 학습 결과
    """
    os.makedirs(output_dir, exist_ok=True)

    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    logger.info("=" * 60)
    logger.info("Full Scale 학습 시작")
    logger.info("=" * 60)
    logger.info(f"모델: {model_name}")
    logger.info(f"학습 데이터: {len(train_dataset)} samples")
    logger.info(f"평가 데이터: {len(eval_dataset)} samples")

    # === 1. 모델 로드 ===
    log_memory_stats("모델 로드 전")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Flash Attention 사용
    )

    log_memory_stats("모델 로드 후")

    # === 2. LoRA 적용 ===
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # === 3. 메모리 최적화 ===
    model = setup_memory_optimization(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
    )

    log_memory_stats("LoRA + 최적화 적용 후")

    # === 4. Resume 확인 ===
    ckpt_manager = CheckpointManager(output_dir, save_total_limit)

    if resume_from_checkpoint is None:
        # 자동으로 최신 체크포인트 탐색
        latest = ckpt_manager.get_latest_checkpoint()
        if latest:
            logger.info(f"자동 감지된 체크포인트에서 재개: {latest}")
            resume_from_checkpoint = latest

    # === 5. Training Arguments ===
    training_args = create_full_scale_training_args(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_batch_size=per_device_batch_size,
        num_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    # === 6. Callbacks ===
    callbacks = [
        EarlyStoppingCallback(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
        ),
        DetailedLoggingCallback(log_interval=50),
    ]

    # === 7. Trainer 생성 ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    # === 8. 학습 실행 ===
    start_time = datetime.now()

    try:
        train_result = trainer.train(
            resume_from_checkpoint=resume_from_checkpoint
        )
    except KeyboardInterrupt:
        logger.warning("사용자가 학습을 중단함. 현재 상태 저장 중...")
        trainer.save_model(os.path.join(output_dir, "interrupted_checkpoint"))
        raise
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        # 오류 발생 시에도 현재 상태 저장 시도
        try:
            trainer.save_model(os.path.join(output_dir, "error_checkpoint"))
        except Exception:
            pass
        raise

    elapsed = (datetime.now() - start_time).total_seconds()

    # === 9. 최종 평가 ===
    eval_result = trainer.evaluate()

    # === 10. Best Model 저장 ===
    best_model_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_dir)

    # 토크나이저도 함께 저장
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(best_model_dir)

    # === 11. 결과 정리 ===
    result = {
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "total_steps": train_result.global_step,
        "elapsed_hours": elapsed / 3600,
        "best_model_path": best_model_dir,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available()
        else 0,
        "config": {
            "learning_rate": learning_rate,
            "batch_size": per_device_batch_size,
            "effective_batch_size": per_device_batch_size * gradient_accumulation_steps,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "num_epochs": num_epochs,
            "warmup_ratio": warmup_ratio,
        },
    }

    # 결과 저장
    with open(os.path.join(output_dir, "training_result.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("학습 완료")
    logger.info(f"  Train Loss: {result['train_loss']:.4f}")
    logger.info(f"  Eval Loss: {result['eval_loss']:.4f}")
    logger.info(f"  총 소요 시간: {result['elapsed_hours']:.2f}시간")
    logger.info(f"  Peak GPU Memory: {result['peak_memory_gb']:.2f}GB")
    logger.info(f"  Best Model: {best_model_dir}")
    logger.info("=" * 60)

    # GPU 메모리 정리
    del model, trainer
    torch.cuda.empty_cache()

    return result
```

---

## 12.3.6 Resume Training (학습 재개)

학습이 중간에 중단되었을 때 이어서 하는 방법:

```python
def resume_training(
    output_dir: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    model_name: str,
    checkpoint_path: str = None,
) -> dict:
    """
    중단된 학습을 이어서 실행한다.

    Args:
        output_dir: 기존 학습 출력 디렉토리
        train_dataset: 학습 데이터셋
        eval_dataset: 평가 데이터셋
        model_name: 베이스 모델 이름
        checkpoint_path: 특정 체크포인트 경로 (None이면 최신)

    Returns:
        dict: 학습 결과
    """
    ckpt_manager = CheckpointManager(output_dir)

    if checkpoint_path is None:
        checkpoint_path = ckpt_manager.get_latest_checkpoint()

    if checkpoint_path is None:
        raise ValueError(f"{output_dir}에 체크포인트가 없음")

    logger.info(f"체크포인트에서 학습 재개: {checkpoint_path}")

    # trainer_state.json에서 이전 설정 로드
    state_file = Path(checkpoint_path) / "trainer_state.json"
    if state_file.exists():
        with open(state_file, "r") as f:
            trainer_state = json.load(f)
        logger.info(f"  이전 스텝: {trainer_state.get('global_step', 'N/A')}")
        logger.info(f"  이전 에폭: {trainer_state.get('epoch', 'N/A')}")

    # training_result.json에서 이전 config 로드 (있으면)
    config_file = Path(output_dir) / "training_result.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            prev_config = json.load(f).get("config", {})
    else:
        prev_config = {}

    # 이전 설정으로 재개
    return run_full_scale_training(
        model_name=model_name,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        learning_rate=prev_config.get("learning_rate", 2e-4),
        per_device_batch_size=prev_config.get("batch_size", 8),
        num_epochs=prev_config.get("num_epochs", 3),
        warmup_ratio=prev_config.get("warmup_ratio", 0.05),
        lora_rank=prev_config.get("lora_rank", 32),
        lora_alpha=prev_config.get("lora_alpha", 32),
        resume_from_checkpoint=checkpoint_path,
    )
```

---

## 12.3.7 학습 후 모델 로드

```python
def load_trained_model(
    base_model_name: str,
    adapter_path: str,
    device: str = "cuda",
    merge_adapter: bool = False,
) -> tuple:
    """
    학습 완료된 LoRA 모델을 로드한다.

    Args:
        base_model_name: 베이스 모델 이름
        adapter_path: LoRA adapter 경로
        device: 디바이스
        merge_adapter: adapter를 base model에 병합할지 여부

    Returns:
        tuple: (model, tokenizer)
    """
    # 베이스 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # LoRA adapter 로드
    model = PeftModel.from_pretrained(model, adapter_path)

    if merge_adapter:
        # Adapter를 베이스 모델에 병합 (추론 속도 향상)
        model = model.merge_and_unload()
        logger.info("LoRA adapter가 base model에 병합됨")

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    logger.info(f"모델 로드 완료: {adapter_path}")
    return model, tokenizer
```

---

## 12.3.8 학습 모니터링 대시보드

TensorBoard를 활용한 모니터링:

```bash
# TensorBoard 실행
tensorboard --logdir ./output_dir/runs --port 6006

# 원격 서버라면 SSH 터널링
ssh -L 6006:localhost:6006 user@server
```

모니터링해야 할 주요 지표:

| 지표 | 정상 패턴 | 이상 징후 |
|------|-----------|-----------|
| train/loss | 꾸준히 감소 | 정체, 급등, NaN |
| eval/loss | 감소 후 수렴 | train_loss만 감소 (과적합) |
| train/learning_rate | Warmup → Cosine Decay | 갑자기 0 (스케줄러 오류) |
| train/grad_norm | 일정 범위 유지 | 급증 (gradient explosion) |
| system/gpu_memory | 일정 | 점진 증가 (메모리 누수) |

---

## 용어 체크리스트

학습 전 아래 용어를 모두 설명할 수 있는지 확인하라:

- [ ] **Checkpoint**: 학습 중간 상태를 디스크에 저장한 것. 모델 가중치, 옵티마이저 상태, 스케줄러 상태 포함.
- [ ] **Resume Training**: 체크포인트에서 학습을 이어서 재개하는 것. 스텝 수, 학습률 상태 등이 복원됨.
- [ ] **Early Stopping**: Validation loss 개선이 없으면 학습을 조기 종료. `patience`로 허용 횟수 설정.
- [ ] **Evaluation Loop**: 학습 중 주기적으로 검증 데이터로 성능을 측정하는 과정.
- [ ] **Gradient Accumulation**: 여러 미니배치의 gradient를 누적 후 한 번에 업데이트. Effective BS = per_device_bs * num_devices * accum_steps.
- [ ] **Gradient Checkpointing**: 중간 activation을 저장하지 않고 backward 시 재계산. VRAM 30~50% 절감, 속도 20~30% 감소.
- [ ] **save_total_limit**: 최대 보존 체크포인트 수. 초과 시 오래된 것부터 삭제.
- [ ] **load_best_model_at_end**: 학습 종료 시 eval_loss 기준 best checkpoint를 자동 로드.
- [ ] **Flash Attention**: Attention 연산의 IO-aware 최적화. 메모리 절감 + 속도 향상.
- [ ] **merge_and_unload**: LoRA adapter를 base model에 병합하여 추론 속도를 향상시키는 기법.
