---
---

# 11.2 Mixed Precision Training

## 핵심 용어 박스

| 용어 | 정의 |
|------|------|
| **FP32 (Single Precision)** | 32비트 부동소수점. sign(1) + exponent(8) + mantissa(23). 범위 $\pm 3.4 \times 10^{38}$. 학습의 기본 정밀도이며 수치적으로 가장 안정적이다. |
| **FP16 (Half Precision)** | 16비트 부동소수점. sign(1) + exponent(5) + mantissa(10). 범위 $\pm 65504$. 메모리 절반, 연산 2배 빠르지만 **overflow/underflow** 위험이 크다. |
| **BF16 (Brain Floating Point)** | Google Brain이 설계한 16비트 형식. sign(1) + exponent(8) + mantissa(7). FP32와 동일한 범위를 가지되 정밀도가 낮다. Loss Scaling이 불필요하다. |
| **Loss Scaling** | FP16의 underflow를 방지하기 위해 loss에 스케일 팩터 $s$를 곱해 gradient 크기를 키우는 기법. backward 후 $s$로 나눠 원래 크기로 복원한다. |
| **AMP (Automatic Mixed Precision)** | PyTorch의 `torch.cuda.amp` 모듈. forward는 FP16/BF16으로, backward의 gradient 누적과 optimizer는 FP32로 자동 관리한다. |
| **GradScaler** | PyTorch AMP에서 Dynamic Loss Scaling을 수행하는 클래스. overflow를 감지하면 scale factor를 줄이고, 안정적이면 키운다. |
| **Master Weights** | Optimizer가 보유하는 FP32 정밀도의 파라미터 사본. FP16 연산 후 FP32에서 정확한 업데이트를 수행하고, 다시 FP16으로 캐스팅한다. |

---

## 개요

Mixed Precision Training은 학습 과정에서 **두 가지 이상의 수치 정밀도**를 혼합 사용하는 기법이다. 핵심 아이디어는 간단하다:

- **Forward/Backward**: FP16 또는 BF16으로 수행 (메모리 절약 + 연산 가속)
- **Parameter Update**: FP32로 수행 (수치 안정성 보장)

이 조합으로 메모리 사용량을 **절반**으로 줄이고, Tensor Core를 활용해 연산 속도를 **2~3배** 높일 수 있다.

| 항목 | FP32 Only | Mixed Precision |
|------|-----------|-----------------|
| 파라미터 메모리 (7B) | 28 GB | 14 GB (FP16) + 28 GB (FP32 master) |
| Activation 메모리 | 기준 | ~50% 감소 |
| 연산 속도 (A100) | 19.5 TFLOPS | 312 TFLOPS (FP16 Tensor Core) |
| 수치 안정성 | 최고 | Loss Scaling 필요 (FP16) |

---

## 수학적 원리

### 1. 부동소수점 표현

IEEE 754 부동소수점 수의 일반적 표현:

$$x = (-1)^s \times 2^{e - \text{bias}} \times (1 + \sum_{i=1}^{p} b_i \cdot 2^{-i})$$

여기서 $s$는 부호비트, $e$는 exponent(지수), $b_i$는 mantissa(가수)의 각 비트, $p$는 mantissa 비트 수다.

### 2. FP16 상세

**비트 구성:** sign(1) + exponent(5) + mantissa(10) = 16 bits

$$\text{bias} = 2^{5-1} - 1 = 15$$

**표현 가능 범위:**

- 최솟값 (subnormal): $2^{-24} \approx 5.96 \times 10^{-8}$
- 최솟값 (normal): $2^{-14} \approx 6.10 \times 10^{-5}$
- 최댓값: $(2 - 2^{-10}) \times 2^{15} = 65504$

**정밀도:**

$$\epsilon_{\text{FP16}} = 2^{-10} \approx 9.77 \times 10^{-4}$$

인접한 두 FP16 숫자의 상대 간격이 약 0.1%다. 즉, 1024와 1025를 구분할 수 있지만 1024.5는 표현 불가.

### 3. BF16 상세

**비트 구성:** sign(1) + exponent(8) + mantissa(7) = 16 bits

$$\text{bias} = 2^{8-1} - 1 = 127$$

**표현 가능 범위:**

- 최솟값 (normal): $2^{-126} \approx 1.18 \times 10^{-38}$
- 최댓값: $(2 - 2^{-7}) \times 2^{127} \approx 3.39 \times 10^{38}$

**정밀도:**

$$\epsilon_{\text{BF16}} = 2^{-7} \approx 7.81 \times 10^{-3}$$

인접 숫자 간 상대 간격이 약 0.78%. FP16보다 정밀도는 8배 낮지만, FP32와 동일한 지수 범위를 가진다.

### 4. FP16 vs BF16 비교

| 특성 | FP16 | BF16 | FP32 |
|------|------|------|------|
| 총 비트 | 16 | 16 | 32 |
| Exponent 비트 | 5 | 8 | 8 |
| Mantissa 비트 | 10 | 7 | 23 |
| 최댓값 | 65,504 | $\sim 3.4 \times 10^{38}$ | $\sim 3.4 \times 10^{38}$ |
| 최소 정규값 | $\sim 6.1 \times 10^{-5}$ | $\sim 1.2 \times 10^{-38}$ | $\sim 1.2 \times 10^{-38}$ |
| 정밀도 ($\epsilon$) | $9.77 \times 10^{-4}$ | $7.81 \times 10^{-3}$ | $1.19 \times 10^{-7}$ |
| Loss Scaling 필요 | **O** | **X** | - |
| Tensor Core 지원 | A100, H100, 4090 등 | A100+, H100, 4090 등 | - |

### 5. Loss Scaling의 수학

FP16 학습에서 gradient의 절대값이 매우 작으면($< 2^{-24}$), FP16의 표현 범위 밖으로 떨어져 **underflow**(0이 됨)가 발생한다.

**Static Loss Scaling:**

Forward에서 loss에 스케일 팩터 $s$를 곱한다:

$$\text{scaled\_loss} = \mathcal{L} \times s$$

Chain rule에 의해 모든 gradient도 $s$배가 된다:

$$\nabla_\theta \text{scaled\_loss} = s \cdot \nabla_\theta \mathcal{L}$$

Optimizer 업데이트 전에 $s$로 나눠 원래 gradient를 복원한다:

$$\nabla_\theta \mathcal{L} = \frac{\nabla_\theta \text{scaled\_loss}}{s}$$

이 과정에서 원래 underflow되던 작은 gradient가 $s$배 커져서 FP16 표현 범위 안에 들어오게 된다.

**Dynamic Loss Scaling:**

Static scaling은 $s$를 미리 정해야 하므로 범용적이지 않다. Dynamic scaling은 다음 알고리즘으로 $s$를 자동 조절한다:

$$s_{t+1} = \begin{cases} s_t / 2 & \text{if overflow detected (inf/nan in gradient)} \\ s_t \times 2 & \text{if no overflow for } K \text{ consecutive steps} \\ s_t & \text{otherwise} \end{cases}$$

여기서 $K$는 보통 2000 스텝 (PyTorch 기본값). 초기 $s_0$은 보통 $2^{16} = 65536$.

**Overflow 감지 시 해당 스텝은 건너뛴다** (파라미터 업데이트 없음). 이는 학습 초기에 빈번할 수 있지만, $s$가 적절한 값으로 수렴하면 안정된다.

### 6. Mixed Precision 학습 흐름 (수학적)

전체 학습 한 스텝의 흐름:

**Step 1: FP32 → FP16 캐스팅**

$$\theta_{\text{fp16}} = \text{cast}_{\text{fp16}}(\theta_{\text{fp32}})$$

**Step 2: Forward (FP16)**

$$\mathcal{L}_{\text{fp16}} = f(\theta_{\text{fp16}}, x)$$

**Step 3: Loss Scaling**

$$\mathcal{L}_{\text{scaled}} = s \cdot \mathcal{L}_{\text{fp16}}$$

**Step 4: Backward (FP16 gradient)**

$$g_{\text{fp16}} = \nabla_\theta \mathcal{L}_{\text{scaled}}$$

**Step 5: Unscale + FP32 변환**

$$g_{\text{fp32}} = \text{cast}_{\text{fp32}}(g_{\text{fp16}}) / s$$

**Step 6: Gradient Clipping (FP32)**

$$g_{\text{clipped}} = \text{clip}(g_{\text{fp32}}, \text{max\_norm})$$

**Step 7: Optimizer Update (FP32)**

$$\theta_{\text{fp32}} \leftarrow \theta_{\text{fp32}} - \eta \cdot \text{Adam}(g_{\text{clipped}})$$

**Step 8: 다음 스텝의 FP16 파라미터 갱신**

$$\theta_{\text{fp16}} \leftarrow \text{cast}_{\text{fp16}}(\theta_{\text{fp32}})$$

Master weights ($\theta_{\text{fp32}}$)를 유지하는 이유: FP16에서 직접 업데이트하면 $\eta \cdot g$가 $\theta$에 비해 너무 작아서 더해져도 값이 변하지 않는 **stagnation** 문제가 발생한다. FP32에서는 정밀도가 충분하므로 미세한 업데이트도 반영된다.

---

## FP16 vs BF16 선택 기준

### BF16을 선택해야 하는 경우 (권장)

1. **GPU 지원**: A100, H100, RTX 4090 등 BF16 Tensor Core가 있는 GPU
2. **LLM/VLM 학습**: gradient 범위가 넓어 FP16 overflow 가능성이 높은 경우
3. **간편한 설정**: Loss Scaling이 불필요하므로 디버깅이 쉬움
4. **대부분의 최신 모델**: Llama, Qwen, Mistral 등은 BF16 학습을 기본 권장

### FP16을 선택해야 하는 경우

1. **GPU 제한**: V100 등 BF16을 지원하지 않는 GPU
2. **높은 정밀도 필요**: 수치 연산이 민감한 특수 태스크 (mantissa 10 > 7)
3. **추론 최적화**: TensorRT 등에서 FP16이 더 널리 최적화되어 있는 경우

### 선택 의사결정 트리

```
GPU가 BF16 지원?
├─ Yes → BF16 사용 (Loss Scaling 불필요, 범위 넓음)
└─ No
   ├─ gradient가 안정적? → FP16 + Static Loss Scaling
   └─ gradient 불안정? → FP16 + Dynamic Loss Scaling (GradScaler)
```

---

## 코드: PyTorch AMP (BF16)

### 기본 AMP 학습 루프

```python
import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MixedPrecisionConfig:
    """Mixed Precision 학습 설정."""
    precision: str = "bf16"           # "fp16", "bf16", "fp32"
    initial_loss_scale: float = 2**16  # FP16 Dynamic Scaling 초기값
    growth_interval: int = 2000        # 스케일 증가 간격 (스텝)
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 4


class MixedPrecisionTrainer:
    """Mixed Precision Training 구현."""

    def __init__(self, model, config: MixedPrecisionConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        # 정밀도별 설정
        if config.precision == "bf16":
            self.amp_dtype = torch.bfloat16
            self.use_scaler = False  # BF16은 GradScaler 불필요
        elif config.precision == "fp16":
            self.amp_dtype = torch.float16
            self.use_scaler = True   # FP16은 GradScaler 필수
        else:
            self.amp_dtype = torch.float32
            self.use_scaler = False

        # GradScaler (FP16만 사용)
        self.scaler = GradScaler(
            enabled=self.use_scaler,
            init_scale=config.initial_loss_scale,
            growth_interval=config.growth_interval,
        )

        logger.info(
            f"Mixed Precision: {config.precision} | "
            f"GradScaler: {'ON' if self.use_scaler else 'OFF'}"
        )

    def train_step(
        self,
        batch: dict,
        optimizer: AdamW,
        scheduler=None,
        step: int = 0,
    ) -> dict:
        """단일 학습 스텝."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward (Mixed Precision)
        with autocast(device_type="cuda", dtype=self.amp_dtype):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / self.config.gradient_accumulation_steps

        # Backward
        if self.use_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        metrics = {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
        }

        # Gradient Accumulation 완료 시 업데이트
        if (step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.use_scaler:
                # FP16: unscale → clip → step → update scaler
                self.scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.scaler.step(optimizer)
                self.scaler.update()

                metrics["loss_scale"] = self.scaler.get_scale()
            else:
                # BF16/FP32: 바로 clip → step
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                optimizer.step()

            metrics["grad_norm"] = grad_norm.item()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

        return metrics
```

### GradScaler 상세 설정 (FP16)

```python
def create_fp16_grad_scaler(
    init_scale: float = 2**16,
    growth_factor: float = 2.0,
    backoff_factor: float = 0.5,
    growth_interval: int = 2000,
) -> GradScaler:
    """FP16 Dynamic Loss Scaling 설정.

    Args:
        init_scale: 초기 스케일 팩터 (기본 65536)
        growth_factor: overflow 없이 growth_interval 지나면 스케일에 곱할 값
        backoff_factor: overflow 감지 시 스케일에 곱할 값
        growth_interval: 스케일 증가를 시도하는 간격 (스텝 수)
    """
    scaler = GradScaler(
        init_scale=init_scale,
        growth_factor=growth_factor,
        backoff_factor=backoff_factor,
        growth_interval=growth_interval,
        enabled=True,
    )
    return scaler


def check_for_overflow(model) -> bool:
    """Gradient에 inf/nan이 있는지 확인."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                logger.warning(f"Overflow detected in {name}")
                return True
    return False
```

### BF16 vs FP16 벤치마크 유틸리티

```python
import time
import torch


def benchmark_precision(
    model,
    dummy_input: dict,
    num_iterations: int = 100,
    warmup: int = 10,
):
    """FP32, FP16, BF16 성능 비교."""
    results = {}

    for precision, dtype in [
        ("fp32", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ]:
        model_copy = model.to(dtype)
        input_ids = dummy_input["input_ids"].cuda()

        # Warmup
        for _ in range(warmup):
            with autocast(device_type="cuda", dtype=dtype):
                _ = model_copy(input_ids)

        torch.cuda.synchronize()

        # 벤치마크
        start = time.perf_counter()
        for _ in range(num_iterations):
            with autocast(device_type="cuda", dtype=dtype):
                _ = model_copy(input_ids)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # 메모리 측정
        torch.cuda.reset_peak_memory_stats()
        with autocast(device_type="cuda", dtype=dtype):
            _ = model_copy(input_ids)
        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        results[precision] = {
            "avg_time_ms": elapsed / num_iterations * 1000,
            "throughput": num_iterations / elapsed,
            "peak_memory_gb": peak_memory,
        }

        logger.info(
            f"[{precision}] "
            f"Avg: {results[precision]['avg_time_ms']:.2f}ms | "
            f"Throughput: {results[precision]['throughput']:.1f} iter/s | "
            f"Peak Memory: {peak_memory:.2f} GB"
        )

    return results
```

### 전체 학습 스크립트 (AMP 통합)

```python
def train_with_mixed_precision(
    model_name: str,
    train_dataset,
    precision: str = "bf16",
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
):
    """Mixed Precision 학습 전체 흐름."""

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if precision == "bf16" else torch.float32,
    ).cuda()

    # Trainer 설정
    mp_config = MixedPrecisionConfig(
        precision=precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    trainer = MixedPrecisionTrainer(model, mp_config)

    # Optimizer (FP32 master weights 자동 관리)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        fused=True,  # PyTorch 2.0+ fused optimizer (더 빠름)
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 학습 루프
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            metrics = trainer.train_step(batch, optimizer, step=step)
            epoch_loss += metrics["loss"]
            global_step += 1

            if global_step % 100 == 0:
                avg_loss = epoch_loss / (step + 1)
                log_msg = f"[Epoch {epoch+1}] Step {global_step} | Loss: {avg_loss:.4f}"

                if "grad_norm" in metrics:
                    log_msg += f" | Grad Norm: {metrics['grad_norm']:.4f}"
                if "loss_scale" in metrics:
                    log_msg += f" | Loss Scale: {metrics['loss_scale']:.0f}"

                logger.info(log_msg)

    return model
```

---

## Mixed Precision과 메모리 절약 계산

7B 모델 기준 메모리 분석:

| 항목 | FP32 Only | Mixed Precision (BF16 + FP32) |
|------|-----------|-------------------------------|
| 모델 파라미터 | 28 GB (FP32) | 14 GB (BF16) + 28 GB (FP32 master) = 42 GB |
| Gradient | 28 GB (FP32) | 14 GB (BF16) |
| Optimizer State ($m$, $v$) | 56 GB (FP32) | 56 GB (FP32) |
| Activation | ~16 GB | ~8 GB |
| **합계** | **~128 GB** | **~120 GB** |

> "잠깐, Mixed Precision이 오히려 메모리를 더 쓰는 거 아냐?"

**파라미터 자체**는 FP16 + FP32 master를 유지하므로 약간 더 쓴다. 하지만 핵심 절약은 **activation 메모리**에서 발생한다. Forward의 중간 결과(activation)가 FP16이면 절반으로 줄어든다. Batch size가 클수록 activation 비중이 커지므로 절약 효과가 극대화된다.

또한 ZeRO와 결합하면:

| 구성 | GPU당 메모리 (4 GPU) |
|------|---------------------|
| FP32 + DDP | ~128 GB |
| BF16 Mixed + DDP | ~120 GB |
| BF16 Mixed + ZeRO Stage 2 | ~42 GB |
| BF16 Mixed + ZeRO Stage 3 | ~30 GB |

---

## 흔한 문제와 해결

### 1. FP16 Overflow (Loss = NaN)

```python
def handle_nan_loss(loss, scaler, optimizer, model):
    """NaN loss 발생 시 처리."""
    if torch.isnan(loss) or torch.isinf(loss):
        logger.warning("NaN/Inf loss detected! Skipping step.")
        optimizer.zero_grad()

        # GradScaler가 자동으로 scale을 줄임
        # 수동으로도 가능:
        if scaler.is_enabled():
            current_scale = scaler.get_scale()
            logger.warning(f"Current loss scale: {current_scale}")

        return True  # skip this step
    return False
```

### 2. BF16 정밀도 문제

BF16은 mantissa가 7비트뿐이라 작은 값의 차이를 구분하지 못한다:

```python
# BF16 정밀도 한계 데모
a = torch.tensor(1.0, dtype=torch.bfloat16)
b = torch.tensor(1.0 + 1e-3, dtype=torch.bfloat16)
print(f"BF16: {a == b}")  # True (구분 불가)

c = torch.tensor(1.0, dtype=torch.float16)
d = torch.tensor(1.0 + 1e-3, dtype=torch.float16)
print(f"FP16: {c == d}")  # False (구분 가능)
```

이 때문에 **softmax, layer norm, loss 계산**은 FP32로 수행하는 것이 권장된다. PyTorch AMP는 이를 자동으로 처리한다.

### 3. autocast와 호환되지 않는 연산

```python
# autocast 내에서 FP32가 강제되는 연산 목록
# - torch.nn.functional.cross_entropy
# - torch.nn.functional.binary_cross_entropy_with_logits
# - torch.nn.functional.layer_norm
# - torch.nn.functional.softmax (큰 텐서)

# 커스텀 연산에서 명시적 캐스팅이 필요한 경우:
with autocast(device_type="cuda", dtype=torch.bfloat16):
    # 일부 연산을 FP32로 강제
    with autocast(device_type="cuda", enabled=False):
        # 이 블록 내에서는 FP32로 실행
        precise_result = some_numerically_sensitive_op(x.float())
```

---

## Tensor Core 활용 최적화

Tensor Core는 특정 차원이 8(FP16) 또는 16(BF16)의 배수일 때 최고 성능을 발휘한다.

```python
def optimize_for_tensor_core(config: dict) -> dict:
    """Tensor Core 최적화를 위한 설정 조정."""

    # Vocabulary size를 64의 배수로 패딩
    if config.get("vocab_size") and config["vocab_size"] % 64 != 0:
        original = config["vocab_size"]
        config["vocab_size"] = ((original + 63) // 64) * 64
        logger.info(f"Vocab size: {original} → {config['vocab_size']} (64배수 패딩)")

    # Hidden size가 8의 배수인지 확인
    if config.get("hidden_size") and config["hidden_size"] % 8 != 0:
        logger.warning(
            f"hidden_size={config['hidden_size']}는 8의 배수가 아님. "
            "Tensor Core 활용이 제한될 수 있음."
        )

    # Batch size를 8의 배수로 설정
    if config.get("batch_size") and config["batch_size"] % 8 != 0:
        original = config["batch_size"]
        config["batch_size"] = max(8, ((original + 7) // 8) * 8)
        logger.info(f"Batch size: {original} → {config['batch_size']} (8배수)")

    return config
```

---

## 용어 체크리스트

학습 후 아래 용어를 설명할 수 있는지 점검해봐라:

- [ ] **FP16**: exponent(5) + mantissa(10) 구조와 최대 표현 범위 65504의 의미를 아는가?
- [ ] **BF16**: FP32와 동일한 exponent 범위를 갖는 이유와 Loss Scaling이 불필요한 이유를 설명할 수 있는가?
- [ ] **Loss Scaling**: underflow 방지 원리를 chain rule로 설명할 수 있는가?
- [ ] **Dynamic Loss Scaling**: overflow 감지 시 scale factor를 어떻게 조절하는지 알고리즘을 설명할 수 있는가?
- [ ] **Master Weights**: FP32 파라미터 사본을 유지하는 이유(stagnation 방지)를 이해하는가?
- [ ] **GradScaler**: PyTorch에서 GradScaler의 역할과 `scaler.scale()`, `scaler.unscale_()`, `scaler.step()`, `scaler.update()` 각 메서드의 기능을 아는가?
- [ ] **autocast**: 어떤 연산이 자동으로 FP16/BF16으로 변환되고, 어떤 연산이 FP32로 유지되는지 아는가?
- [ ] **Tensor Core**: 차원이 8/16의 배수여야 최적 성능인 이유를 설명할 수 있는가?
- [ ] **Activation 메모리**: Mixed Precision에서 activation 메모리 절감이 파라미터 메모리보다 큰 이유를 아는가?
- [ ] **정밀도 vs 범위 트레이드오프**: mantissa 비트가 정밀도를, exponent 비트가 범위를 결정하는 관계를 설명할 수 있는가?
