---
---

# 12.1 Sanity Check

> 학습을 본격적으로 시작하기 전, 모델/데이터/파이프라인이 정상 동작하는지 검증하는 단계다.
> Sanity Check 없이 Full Scale 학습에 들어가면, 수 시간~수 일의 GPU 시간을 날릴 수 있다.

---

## 핵심 용어

| 용어 | 정의 | 왜 중요한가 |
|------|------|-------------|
| **Overfit Test** | 단일 배치(1 batch)에 모델을 과적합시켜 loss가 0에 수렴하는지 확인하는 테스트 | 모델 구조, loss 함수, 데이터 파이프라인이 정상인지 한 번에 검증 |
| **Gradient Flow** | 역전파 시 각 레이어를 통과하는 gradient의 크기와 분포 | Vanishing/Exploding gradient 조기 탐지 |
| **Learning Rate Finder** | 학습률을 로그 스케일로 점진 증가시키며 최적 구간을 탐색하는 기법 | 잘못된 학습률은 학습 실패의 가장 흔한 원인 |
| **Data Verification** | 이미지-레이블 매칭, 토큰화 정확성, 전처리 파이프라인 검증 | "Garbage in, garbage out" — 데이터가 틀리면 모든 게 틀림 |

---

## 12.1.1 1 Batch Overfit 테스트

### 개념

전체 데이터셋 대신 **단 1개의 배치(batch)**만 반복 학습시킨다.
정상적인 모델이라면 이 배치에 대해 loss가 거의 0까지 떨어져야 한다.

만약 1 batch overfit이 안 되면:
- 모델 구조에 버그가 있거나
- Loss 함수 설정이 잘못되었거나
- 데이터 파이프라인에 문제가 있다는 뜻

### 코드: 1 Batch Overfit Test

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def overfit_single_batch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_steps: int = 100,
    target_loss: float = 0.01,
    device: str = "cuda",
) -> dict:
    """
    단일 배치에 대해 모델을 과적합시키는 Sanity Check.

    Args:
        model: 학습 대상 모델
        dataloader: 데이터로더 (첫 번째 배치만 사용)
        optimizer: 옵티마이저
        num_steps: 최대 반복 횟수
        target_loss: 목표 loss (이 이하면 성공)
        device: 디바이스

    Returns:
        dict: {"success": bool, "final_loss": float, "losses": list}
    """
    model.train()
    model.to(device)

    # 첫 번째 배치만 추출
    batch = next(iter(dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}

    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"[Step {step}] Loss가 NaN/Inf — 모델 또는 데이터에 심각한 문제")
            return {
                "success": False,
                "final_loss": float("nan"),
                "losses": losses,
                "error": "NaN/Inf loss detected",
            }

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        losses.append(current_loss)

        if step % 10 == 0:
            logger.info(f"[Step {step:3d}] Loss: {current_loss:.6f}")

        if current_loss < target_loss:
            logger.info(f"[SUCCESS] Step {step}에서 target loss {target_loss} 달성")
            return {
                "success": True,
                "final_loss": current_loss,
                "losses": losses,
            }

    success = losses[-1] < target_loss
    status = "SUCCESS" if success else "FAIL"
    logger.info(f"[{status}] 최종 Loss: {losses[-1]:.6f}")

    return {
        "success": success,
        "final_loss": losses[-1],
        "losses": losses,
    }
```

### 결과 해석

| 결과 | 진단 | 조치 |
|------|------|------|
| Loss가 0 근처로 수렴 | 모델, loss, 데이터 파이프라인 정상 | 다음 단계 진행 |
| Loss가 줄지만 0에 도달 못함 | 학습률이 너무 낮거나 모델 용량 부족 | 학습률 상향, 스텝 수 증가 |
| Loss가 전혀 안 줄어듦 | Loss 함수 설정 오류 또는 gradient 단절 | Gradient flow 확인 |
| Loss가 NaN | 수치 불안정, 학습률 과대, 데이터 이상값 | 아래 트러블슈팅 참고 |

---

## 12.1.2 Gradient Flow 확인

### 개념

역전파(backpropagation) 시 gradient가 모든 레이어에 제대로 전파되는지 확인한다.

**Vanishing Gradient**: gradient가 0에 가까우면 해당 레이어는 학습이 안 된다.
**Exploding Gradient**: gradient가 극단적으로 크면 학습이 불안정해진다.

### 코드: Gradient Flow 시각화

```python
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict


def plot_gradient_flow(
    named_parameters,
    save_path: Optional[str] = None,
    figsize: tuple = (20, 8),
) -> dict:
    """
    모델의 각 레이어별 gradient 분포를 시각화한다.

    Args:
        named_parameters: model.named_parameters()
        save_path: 저장 경로 (None이면 plt.show())
        figsize: 그래프 크기

    Returns:
        dict: 레이어별 gradient 통계
    """
    ave_grads = []
    max_grads = []
    layers = []
    grad_stats = OrderedDict()

    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None:
            grad = param.grad.detach().cpu()

            ave_grads.append(grad.abs().mean().item())
            max_grads.append(grad.abs().max().item())
            layers.append(name)

            grad_stats[name] = {
                "mean": grad.abs().mean().item(),
                "max": grad.abs().max().item(),
                "std": grad.std().item(),
                "zeros_pct": (grad == 0).float().mean().item() * 100,
            }

    if not layers:
        logger.warning("Gradient가 있는 파라미터가 없음 — backward()를 먼저 호출했는지 확인")
        return {}

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 평균 gradient 크기
    axes[0].barh(range(len(layers)), ave_grads, color="steelblue", alpha=0.7)
    axes[0].set_yticks(range(len(layers)))
    axes[0].set_yticklabels(layers, fontsize=6)
    axes[0].set_xlabel("Average Gradient Magnitude")
    axes[0].set_title("Average Gradient per Layer")
    axes[0].set_xscale("log")

    # 최대 gradient 크기
    axes[1].barh(range(len(layers)), max_grads, color="coral", alpha=0.7)
    axes[1].set_yticks(range(len(layers)))
    axes[1].set_yticklabels(layers, fontsize=6)
    axes[1].set_xlabel("Max Gradient Magnitude")
    axes[1].set_title("Max Gradient per Layer")
    axes[1].set_xscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Gradient flow 그래프 저장: {save_path}")
    else:
        plt.show()

    plt.close()

    # 이상 탐지
    _detect_gradient_anomalies(grad_stats)

    return grad_stats


def _detect_gradient_anomalies(grad_stats: dict) -> None:
    """Gradient 이상 징후를 자동으로 탐지해서 경고한다."""
    for name, stats in grad_stats.items():
        if stats["mean"] < 1e-7:
            logger.warning(
                f"[VANISHING] {name}: mean gradient = {stats['mean']:.2e}"
            )
        if stats["max"] > 1e3:
            logger.warning(
                f"[EXPLODING] {name}: max gradient = {stats['max']:.2e}"
            )
        if stats["zeros_pct"] > 90:
            logger.warning(
                f"[DEAD] {name}: {stats['zeros_pct']:.1f}%의 gradient가 0"
            )
```

### Gradient Flow 진단 기준

| 증상 | 원인 | 해결 |
|------|------|------|
| 모든 gradient가 극히 작음 ($< 10^{-7}$) | Vanishing gradient | Residual connection 확인, 활성함수 변경 |
| 특정 레이어만 gradient 0 | 해당 레이어 freeze 상태 또는 연결 끊김 | `requires_grad` 설정 확인 |
| Gradient가 극단적으로 큼 ($> 10^{3}$) | Exploding gradient | Gradient clipping 적용 |
| LoRA 레이어에만 gradient 존재 | 정상 (LoRA fine-tuning 시) | 의도된 동작 확인 |

---

## 12.1.3 Learning Rate Finder

### 개념

**Learning Rate**는 학습에서 가장 중요한 하이퍼파라미터다.
LR Finder는 학습률을 매우 작은 값에서 시작해 로그 스케일로 점진 증가시키면서,
각 학습률에서의 loss를 기록한다.

최적 학습률은 보통 **loss가 가장 가파르게 감소하는 구간**에 있다.

### 수학적 원리

로그 스케일 스윕:

$$\eta_i = \eta_{\min} \cdot \left(\frac{\eta_{\max}}{\eta_{\min}}\right)^{i/N}$$

여기서:
- $\eta_{\min}$: 시작 학습률 (보통 $10^{-7}$)
- $\eta_{\max}$: 종료 학습률 (보통 $10^{0}$)
- $N$: 총 스텝 수
- $i$: 현재 스텝

### 코드: Learning Rate Finder

```python
import math
import copy


class LRFinder:
    """
    Learning Rate Finder.
    로그 스케일로 학습률을 증가시키며 최적 구간을 탐색한다.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        # 원본 state 백업 (탐색 후 복원용)
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())

    def find(
        self,
        dataloader: DataLoader,
        lr_min: float = 1e-7,
        lr_max: float = 1.0,
        num_steps: int = 200,
        smooth_factor: float = 0.05,
        diverge_threshold: float = 4.0,
    ) -> dict:
        """
        LR Range Test를 실행한다.

        Args:
            dataloader: 학습 데이터로더
            lr_min: 시작 학습률
            lr_max: 종료 학습률
            num_steps: 총 탐색 스텝 수
            smooth_factor: Loss smoothing 계수
            diverge_threshold: 발산 판정 배수 (best_loss 대비)

        Returns:
            dict: {"lrs": list, "losses": list, "best_lr": float}
        """
        self.model.train()
        self.model.to(self.device)

        # 로그 스케일 학습률 배열 생성
        log_lrs = torch.linspace(
            math.log10(lr_min), math.log10(lr_max), num_steps
        )
        lrs_schedule = [10 ** lr for lr in log_lrs]

        lrs = []
        losses = []
        best_loss = float("inf")
        smoothed_loss = None

        data_iter = iter(dataloader)

        for step, lr in enumerate(lrs_schedule):
            # 학습률 설정
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # 배치 가져오기 (데이터 소진 시 재시작)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward + Backward
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                logger.info(f"[LR Finder] Step {step}, LR={lr:.2e}: 발산 (NaN/Inf)")
                break

            loss.backward()
            self.optimizer.step()

            current_loss = loss.item()

            # Exponential smoothing
            if smoothed_loss is None:
                smoothed_loss = current_loss
            else:
                smoothed_loss = (
                    smooth_factor * current_loss
                    + (1 - smooth_factor) * smoothed_loss
                )

            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            lrs.append(lr)
            losses.append(smoothed_loss)

            # 발산 감지
            if smoothed_loss > diverge_threshold * best_loss:
                logger.info(f"[LR Finder] Step {step}, LR={lr:.2e}: 발산 감지")
                break

        # 최적 학습률: loss가 가장 가파르게 감소하는 지점
        best_lr = self._find_steepest_descent(lrs, losses)

        # 원본 state 복원
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        logger.info(f"[LR Finder] 추천 학습률: {best_lr:.2e}")

        return {"lrs": lrs, "losses": losses, "best_lr": best_lr}

    def _find_steepest_descent(self, lrs: list, losses: list) -> float:
        """Loss의 기울기가 가장 가파른 지점의 학습률을 찾는다."""
        if len(losses) < 3:
            return lrs[0]

        gradients = []
        for i in range(1, len(losses)):
            grad = (losses[i] - losses[i - 1]) / (
                math.log10(lrs[i]) - math.log10(lrs[i - 1])
            )
            gradients.append(grad)

        # 가장 가파른 감소 지점 (가장 큰 음수 기울기)
        min_idx = np.argmin(gradients)
        return lrs[min_idx]

    def plot(self, results: dict, save_path: Optional[str] = None) -> None:
        """LR Finder 결과를 시각화한다."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(results["lrs"], results["losses"], linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate (log scale)")
        ax.set_ylabel("Loss (smoothed)")
        ax.set_title("Learning Rate Finder")

        # 추천 학습률 표시
        best_lr = results["best_lr"]
        ax.axvline(x=best_lr, color="red", linestyle="--", label=f"Best LR: {best_lr:.2e}")
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

## 12.1.4 데이터 검증

### 핵심 체크 항목

데이터 문제는 학습 실패의 가장 흔한 원인이다. 다음을 반드시 검증:

1. **이미지-레이블 매칭**: 이미지와 OCR 텍스트가 실제로 일치하는가
2. **토큰화 정확성**: 토큰화 후 디코딩 결과가 원본과 동일한가
3. **특수 토큰**: BOS, EOS, PAD 토큰이 올바르게 삽입되었는가
4. **데이터 분포**: 클래스 불균형, 텍스트 길이 분포 등

### 코드: 데이터 검증 유틸리티

```python
from PIL import Image
from transformers import AutoProcessor
import random


class DataVerifier:
    """학습 데이터의 정합성을 검증하는 유틸리티."""

    def __init__(self, processor, tokenizer):
        self.processor = processor
        self.tokenizer = tokenizer

    def verify_sample(self, sample: dict, verbose: bool = True) -> dict:
        """
        단일 샘플의 정합성을 검증한다.

        Args:
            sample: {"image": PIL.Image, "text": str, ...}
            verbose: 상세 출력 여부

        Returns:
            dict: 검증 결과
        """
        issues = []

        # 1. 이미지 검증
        image = sample.get("image")
        if image is None:
            issues.append("이미지가 None")
        elif isinstance(image, Image.Image):
            if image.size[0] == 0 or image.size[1] == 0:
                issues.append(f"이미지 크기 이상: {image.size}")
            if image.mode not in ("RGB", "L", "RGBA"):
                issues.append(f"이미지 모드 이상: {image.mode}")

        # 2. 텍스트 검증
        text = sample.get("text", "")
        if not text or len(text.strip()) == 0:
            issues.append("텍스트가 비어있음")

        # 3. 토큰화 왕복 검증 (roundtrip)
        if text:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True)

            # 공백 정규화 후 비교
            original_normalized = " ".join(text.split())
            decoded_normalized = " ".join(decoded.split())

            if original_normalized != decoded_normalized:
                issues.append(
                    f"토큰화 왕복 불일치:\n"
                    f"  원본: {original_normalized[:100]}\n"
                    f"  복원: {decoded_normalized[:100]}"
                )

            # 토큰 길이 확인
            if len(token_ids) > self.tokenizer.model_max_length:
                issues.append(
                    f"토큰 길이 초과: {len(token_ids)} > "
                    f"{self.tokenizer.model_max_length}"
                )

        # 4. 특수 토큰 확인
        if hasattr(self.tokenizer, "bos_token_id"):
            if self.tokenizer.bos_token_id is None:
                issues.append("BOS 토큰이 설정되지 않음")
        if hasattr(self.tokenizer, "eos_token_id"):
            if self.tokenizer.eos_token_id is None:
                issues.append("EOS 토큰이 설정되지 않음")

        result = {
            "valid": len(issues) == 0,
            "issues": issues,
            "token_count": len(token_ids) if text else 0,
        }

        if verbose and issues:
            for issue in issues:
                logger.warning(f"[DataVerifier] {issue}")

        return result

    def verify_dataset(
        self,
        dataset,
        num_samples: int = 100,
        random_seed: int = 42,
    ) -> dict:
        """
        데이터셋에서 무작위 샘플을 추출해 검증한다.

        Args:
            dataset: HuggingFace Dataset 또는 리스트
            num_samples: 검증할 샘플 수
            random_seed: 랜덤 시드

        Returns:
            dict: 검증 통계
        """
        random.seed(random_seed)
        total = len(dataset)
        indices = random.sample(range(total), min(num_samples, total))

        valid_count = 0
        issue_counts = {}
        token_lengths = []

        for idx in indices:
            sample = dataset[idx]
            result = self.verify_sample(sample, verbose=False)

            if result["valid"]:
                valid_count += 1

            token_lengths.append(result["token_count"])

            for issue in result["issues"]:
                key = issue.split(":")[0].split("\n")[0]
                issue_counts[key] = issue_counts.get(key, 0) + 1

        stats = {
            "total_checked": len(indices),
            "valid_count": valid_count,
            "valid_pct": valid_count / len(indices) * 100,
            "issue_summary": issue_counts,
            "token_length_stats": {
                "mean": np.mean(token_lengths),
                "median": np.median(token_lengths),
                "max": max(token_lengths),
                "min": min(token_lengths),
                "std": np.std(token_lengths),
            },
        }

        logger.info(
            f"[DataVerifier] 검증 완료: {valid_count}/{len(indices)} 정상 "
            f"({stats['valid_pct']:.1f}%)"
        )

        if issue_counts:
            logger.warning(f"[DataVerifier] 발견된 문제: {issue_counts}")

        return stats

    def visualize_samples(
        self,
        dataset,
        num_samples: int = 4,
        figsize: tuple = (16, 8),
    ) -> None:
        """이미지와 텍스트를 나란히 시각화해서 매칭을 육안으로 확인한다."""
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

        fig, axes = plt.subplots(1, num_samples, figsize=figsize)
        if num_samples == 1:
            axes = [axes]

        for ax, idx in zip(axes, indices):
            sample = dataset[idx]
            image = sample.get("image")
            text = sample.get("text", "N/A")

            if image:
                ax.imshow(image)
            ax.set_title(f"[{idx}] {text[:50]}...", fontsize=8)
            ax.axis("off")

        plt.tight_layout()
        plt.show()
        plt.close()
```

---

## 12.1.5 통합 Sanity Check 실행

모든 검증을 한 번에 실행하는 통합 함수:

```python
def run_sanity_check(
    model: torch.nn.Module,
    dataloader: DataLoader,
    dataset,
    processor,
    tokenizer,
    lr: float = 1e-4,
    device: str = "cuda",
    output_dir: str = "./sanity_check_results",
) -> dict:
    """
    전체 Sanity Check 파이프라인을 실행한다.

    순서:
    1. 데이터 검증
    2. 1 Batch Overfit 테스트
    3. Gradient Flow 확인
    4. Learning Rate Finder

    Args:
        model: 모델
        dataloader: 데이터로더
        dataset: 원본 데이터셋 (검증용)
        processor: 프로세서
        tokenizer: 토크나이저
        lr: 초기 학습률
        device: 디바이스
        output_dir: 결과 저장 디렉토리

    Returns:
        dict: 전체 검증 결과
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # === Step 1: 데이터 검증 ===
    logger.info("=" * 60)
    logger.info("[1/4] 데이터 검증 시작")
    logger.info("=" * 60)

    verifier = DataVerifier(processor, tokenizer)
    data_result = verifier.verify_dataset(dataset, num_samples=100)
    results["data_verification"] = data_result

    if data_result["valid_pct"] < 95.0:
        logger.error(
            f"데이터 정합성 {data_result['valid_pct']:.1f}% — "
            "95% 미만이면 데이터 정제 필요"
        )

    # === Step 2: 1 Batch Overfit ===
    logger.info("=" * 60)
    logger.info("[2/4] 1 Batch Overfit 테스트 시작")
    logger.info("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    overfit_result = overfit_single_batch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        num_steps=100,
        device=device,
    )
    results["overfit_test"] = overfit_result

    if not overfit_result["success"]:
        logger.error("1 Batch Overfit 실패 — 모델/Loss/데이터 점검 필요")

    # === Step 3: Gradient Flow ===
    logger.info("=" * 60)
    logger.info("[3/4] Gradient Flow 확인")
    logger.info("=" * 60)

    # 한 번 forward-backward 수행 후 gradient 확인
    batch = next(iter(dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}

    optimizer.zero_grad()
    outputs = model(**batch)
    outputs.loss.backward()

    grad_stats = plot_gradient_flow(
        model.named_parameters(),
        save_path=os.path.join(output_dir, "gradient_flow.png"),
    )
    results["gradient_flow"] = grad_stats

    # === Step 4: Learning Rate Finder ===
    logger.info("=" * 60)
    logger.info("[4/4] Learning Rate Finder 실행")
    logger.info("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)
    lr_finder = LRFinder(model, optimizer, device=device)
    lr_result = lr_finder.find(dataloader, num_steps=200)
    lr_finder.plot(
        lr_result,
        save_path=os.path.join(output_dir, "lr_finder.png"),
    )
    results["lr_finder"] = {
        "best_lr": lr_result["best_lr"],
        "num_steps": len(lr_result["lrs"]),
    }

    # === 종합 결과 ===
    logger.info("=" * 60)
    logger.info("Sanity Check 종합 결과")
    logger.info("=" * 60)
    logger.info(f"  데이터 정합성: {data_result['valid_pct']:.1f}%")
    logger.info(f"  1 Batch Overfit: {'PASS' if overfit_result['success'] else 'FAIL'}")
    logger.info(f"  추천 학습률: {lr_result['best_lr']:.2e}")

    return results
```

---

## 용어 체크리스트

학습 전 아래 용어를 모두 설명할 수 있는지 확인하라:

- [ ] **Overfit Test**: 단일 배치로 loss가 0에 도달하는지 검증하는 테스트. 모델/loss/데이터 파이프라인 정상 동작 확인 목적.
- [ ] **Gradient Flow**: 역전파 시 각 레이어를 통과하는 gradient의 크기와 분포. Vanishing/Exploding 문제 조기 탐지.
- [ ] **Learning Rate Finder**: 학습률을 로그 스케일로 점진 증가시키며 loss 변화를 관찰하는 최적 학습률 탐색 기법.
- [ ] **Data Verification**: 이미지-레이블 매칭, 토큰화 왕복 검증, 특수 토큰 확인 등 데이터 정합성 검증.
- [ ] **Vanishing Gradient**: Gradient가 0에 가까워져 학습이 진행되지 않는 현상.
- [ ] **Exploding Gradient**: Gradient가 극단적으로 커져 학습이 불안정해지는 현상.
- [ ] **Exponential Smoothing**: 노이즈가 많은 신호를 부드럽게 만드는 기법. LR Finder에서 loss smoothing에 사용.
- [ ] **Roundtrip Test**: 인코딩-디코딩 왕복 후 원본과 동일한지 확인하는 검증 방법.
