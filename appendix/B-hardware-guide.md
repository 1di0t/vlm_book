# Appendix B: 하드웨어 가이드

LLM/VLM 학습과 추론에 필요한 하드웨어 스펙, VRAM 계산 공식, GPU 비교, 클라우드 비용까지 총정리한다.
GPU 선택을 잘못하면 돈만 날리거나, 학습 중간에 OOM으로 터진다. 이 가이드를 기반으로 합리적인 판단을 내려라.

---

## B.1 GPU 비교표

### B.1.1 주요 GPU 스펙 비교

| GPU | 아키텍처 | VRAM | 메모리 타입 | 대역폭 | FP16 TFLOPS | BF16 TFLOPS | FP8 TFLOPS | TDP | 대략적 가격 |
|-----|---------|------|-----------|--------|-------------|-------------|------------|-----|-----------|
| RTX 4090 | Ada Lovelace | 24 GB | GDDR6X | 1,008 GB/s | 82.6 | 82.6 | 165.2 | 450W | ~$1,600 |
| RTX 6000 Ada | Ada Lovelace | 48 GB | GDDR6 (ECC) | 960 GB/s | 91.1 | 91.1 | 182.2 | 300W | ~$6,800 |
| L40S | Ada Lovelace | 48 GB | GDDR6 (ECC) | 864 GB/s | 91.6 | 91.6 | 183.2 | 350W | 클라우드 전용 |
| A100 40GB | Ampere | 40 GB | HBM2e | 1,555 GB/s | 77.9 | 312* | - | 400W | ~$10,000 |
| A100 80GB | Ampere | 80 GB | HBM2e | 2,039 GB/s | 77.9 | 312* | - | 400W | ~$15,000 |
| H100 SXM | Hopper | 80 GB | HBM3 | 3,350 GB/s | 267 | 989* | 1,979* | 700W | ~$30,000 |
| H200 SXM | Hopper | 141 GB | HBM3e | 4,800 GB/s | 267 | 989* | 1,979* | 700W | ~$35,000+ |

> \* Tensor Core 사용 시 수치 (Sparsity 포함 시 2배). 일반 CUDA Core FP16은 이보다 낮다.

### B.1.2 GPU 세대별 핵심 특징

| 세대 | Compute Capability | 핵심 기능 |
|------|-------------------|-----------|
| Turing (RTX 20xx) | SM 7.5 | 첫 Tensor Core, FP16 학습 지원 |
| Ampere (A100, RTX 30xx) | SM 8.0 / 8.6 | BF16, TF32, Sparsity, MIG |
| Ada Lovelace (RTX 40xx, L40S) | SM 8.9 | FP8, 향상된 Tensor Core |
| Hopper (H100, H200) | SM 9.0 | Transformer Engine, FP8, NVLink 4 |

### B.1.3 학습 vs 추론 GPU 선택 가이드

| 용도 | 추천 GPU | 이유 |
|------|---------|------|
| 개인 실험 (7B LoRA) | RTX 4090 (24GB) | 가성비 최고, QLoRA로 7B 충분히 학습 가능 |
| 연구실 (7B~13B Full FT) | A100 80GB x2~4 | VRAM 넉넉, NVLink 지원 |
| 프로덕션 학습 (34B+) | H100 x4~8 | HBM3 대역폭, Transformer Engine |
| 추론 서빙 (7B~13B) | L40S / RTX 4090 | 가격 대비 추론 성능 우수 |
| 추론 서빙 (70B+) | H100 / A100 80GB x2+ | 대형 모델은 VRAM이 곧 정의 |

---

## B.2 VRAM 필요량 계산 공식

모델 학습과 추론에 필요한 GPU 메모리를 정확히 계산하는 건 가장 중요한 사전 작업이다.

### B.2.1 기본 메모리 구성 요소

LLM/VLM 학습 시 GPU 메모리는 다음 요소로 구성된다:

$$
\text{Total Memory} = M_{\text{params}} + M_{\text{gradients}} + M_{\text{optimizer}} + M_{\text{activations}} + M_{\text{KV cache}} + M_{\text{overhead}}
$$

각 항목의 의미:

| 구성 요소 | 설명 | 학습 시 | 추론 시 |
|-----------|------|---------|---------|
| $M_{\text{params}}$ | 모델 파라미터 가중치 | O | O |
| $M_{\text{gradients}}$ | 역전파 그래디언트 | O | X |
| $M_{\text{optimizer}}$ | 옵티마이저 상태 (Adam: m, v) | O | X |
| $M_{\text{activations}}$ | 순전파 중간 활성값 | O | 일부 |
| $M_{\text{KV cache}}$ | Key-Value 캐시 (Attention) | X | O |
| $M_{\text{overhead}}$ | CUDA 커널, 임시 버퍼 등 | ~500MB | ~500MB |

### B.2.2 학습 시 메모리 공식

파라미터 수 $P$ (단위: 개수)에 대해:

#### Full Fine-tuning (FP32)

$$
M_{\text{Full-FP32}} = P \times (\underbrace{4}_{\text{param}} + \underbrace{4}_{\text{grad}} + \underbrace{12}_{\text{Adam states}}) = 20P \text{ bytes}
$$

Adam 옵티마이저 상태:
- $m$ (first moment): $4P$ bytes (FP32)
- $v$ (second moment): $4P$ bytes (FP32)
- Master weights: $4P$ bytes (FP32)

#### Full Fine-tuning (Mixed Precision, BF16/FP16)

$$
M_{\text{Full-Mixed}} = P \times (\underbrace{2}_{\text{param (BF16)}} + \underbrace{2}_{\text{grad (BF16)}} + \underbrace{4+4+4}_{\text{Adam (FP32)}}) = 18P \text{ bytes}
$$

> Mixed Precision에서는 forward/backward는 BF16으로 하지만, Adam 옵티마이저 상태와 master weights는 FP32로 유지한다.

#### LoRA Fine-tuning (BF16 base + LoRA params)

$$
M_{\text{LoRA}} = \underbrace{2P}_{\text{frozen base (BF16)}} + \underbrace{18 \times P_{\text{LoRA}}}_{\text{LoRA trainable params}}
$$

LoRA 파라미터 수 계산:

$$
P_{\text{LoRA}} = 2 \times r \times d \times L_{\text{target}}
$$

여기서:
- $r$: LoRA rank (보통 8~128)
- $d$: hidden dimension
- $L_{\text{target}}$: LoRA가 적용되는 레이어 수 (Q, K, V, O 등)

예를 들어 7B 모델, rank=64, hidden_dim=4096, target=4 (Q,K,V,O), 32 layers:

$$
P_{\text{LoRA}} = 2 \times 64 \times 4096 \times 4 \times 32 = 67,108,864 \approx 67M
$$

#### QLoRA Fine-tuning (4-bit base + LoRA)

$$
M_{\text{QLoRA}} = \underbrace{0.5P}_{\text{4-bit quantized base}} + \underbrace{18 \times P_{\text{LoRA}}}_{\text{LoRA trainable}} + \underbrace{\alpha}_{\text{quantization overhead}}
$$

> 4-bit 양자화 시 파라미터당 0.5 bytes. 실제로는 blockwise quantization의 scale/zero-point 오버헤드($\alpha$)로 약 0.55~0.6P bytes 정도 된다.

### B.2.3 추론 시 메모리 공식

#### 모델 가중치

$$
M_{\text{params}} = P \times b_{\text{precision}}
$$

| Precision | $b_{\text{precision}}$ | 7B 모델 기준 |
|-----------|----------------------|-------------|
| FP32 | 4 bytes | 28 GB |
| BF16/FP16 | 2 bytes | 14 GB |
| INT8 | 1 byte | 7 GB |
| INT4 (GPTQ/AWQ) | 0.5 bytes | 3.5 GB |

#### KV Cache

Transformer의 각 레이어에서 Key와 Value를 캐싱한다:

$$
M_{\text{KV}} = 2 \times L \times B \times S \times d_{\text{head}} \times n_{\text{heads}} \times b_{\text{precision}}
$$

여기서:
- $L$: 레이어 수
- $B$: batch size
- $S$: sequence length (최대 컨텍스트)
- $d_{\text{head}}$: head dimension (보통 128)
- $n_{\text{heads}}$: KV head 수 (GQA 사용 시 query head보다 적음)
- $b_{\text{precision}}$: 바이트 수 (BF16 = 2)

예시: Qwen2-VL-7B (32 layers, 4 KV heads, head_dim=128, BF16):

$$
M_{\text{KV}} = 2 \times 32 \times B \times S \times 128 \times 4 \times 2 = 65,536 \times B \times S \text{ bytes}
$$

batch=1, seq_len=4096일 때:

$$
M_{\text{KV}} = 65{,}536 \times 1 \times 4{,}096 = 268{,}435{,}456 \text{ bytes} \approx 256 \text{ MB}
$$

### B.2.4 Activation Memory (학습 시)

Transformer 레이어당 activation 메모리 근사치:

$$
M_{\text{act}} \approx L \times B \times S \times d_{\text{model}} \times (34 + 5 \times \frac{n_{\text{heads}} \times S}{d_{\text{model}}}) \times b_{\text{precision}}
$$

Gradient Checkpointing 사용 시 $\sqrt{L}$ 비율로 줄어든다:

$$
M_{\text{act,ckpt}} \approx \frac{M_{\text{act}}}{\sqrt{L}} \quad (\text{대략적 근사})
$$

> 실제로는 재계산(recomputation)으로 학습 속도가 약 30% 느려지지만, VRAM을 크게 절약한다.

---

## B.3 모델 크기별 GPU 필요량

### B.3.1 학습 시 필요 VRAM 추정

아래 표는 batch_size=1, seq_len=2048 기준이다. Activation + 오버헤드 포함.

| 모델 | Full FT (FP32) | Full FT (BF16 Mixed) | LoRA (r=64, BF16) | QLoRA (4-bit, r=64) |
|------|---------------|---------------------|-------------------|---------------------|
| **7B** | ~140 GB (4×A100 40G) | ~130 GB (2×A100 80G) | ~18 GB (1×RTX 4090) | ~8 GB (1×RTX 3090) |
| **13B** | ~260 GB (4×A100 80G) | ~240 GB (3×A100 80G) | ~30 GB (1×A100 40G) | ~12 GB (1×RTX 4090) |
| **34B** | ~680 GB (8×H100) | ~612 GB (8×A100 80G) | ~72 GB (1×A100 80G) | ~22 GB (1×RTX 4090) |
| **72B** | ~1.4 TB | ~1.3 TB | ~148 GB (2×A100 80G) | ~42 GB (1×A100 80G) |

> gradient checkpointing + gradient accumulation으로 실제 필요량을 더 줄일 수 있다.

### B.3.2 추론 시 필요 VRAM 추정

batch_size=1, seq_len=4096 기준.

| 모델 | FP16/BF16 | INT8 | INT4 (GPTQ/AWQ) |
|------|-----------|------|-----------------|
| **7B** | ~14.5 GB | ~7.5 GB | ~4.2 GB |
| **13B** | ~26.5 GB | ~13.5 GB | ~7.5 GB |
| **34B** | ~68.5 GB | ~34.5 GB | ~18 GB |
| **72B** | ~144.5 GB | ~72.5 GB | ~37 GB |

---

## B.4 멀티 GPU 통신 토폴로지

### B.4.1 GPU 간 연결 방식

| 연결 방식 | 대역폭 (양방향) | 지연 | 비고 |
|-----------|---------------|------|------|
| PCIe Gen4 x16 | 64 GB/s | 높음 | 소비자 GPU 기본 |
| PCIe Gen5 x16 | 128 GB/s | 중간 | 서버 GPU |
| NVLink 3 (A100) | 600 GB/s | 낮음 | GPU 쌍당 |
| NVLink 4 (H100) | 900 GB/s | 매우 낮음 | GPU 쌍당 |
| NVSwitch (DGX) | All-to-All | 최저 | DGX 전용 |

### B.4.2 Tensor Parallelism vs Data Parallelism

$$
\text{Throughput}_{\text{TP}} \propto \frac{1}{\text{communication overhead}} \quad \Rightarrow \text{NVLink 필수}
$$

$$
\text{Throughput}_{\text{DP}} \propto N_{\text{GPU}} \times \left(1 - \frac{t_{\text{comm}}}{t_{\text{compute}}}\right)
$$

| 전략 | GPU 간 통신 | PCIe에서 효율 | NVLink에서 효율 | 적합한 상황 |
|------|-----------|-------------|---------------|------------|
| Data Parallel (DDP) | Gradient AllReduce | 양호 | 우수 | 모델이 1 GPU에 들어갈 때 |
| ZeRO Stage 2 | Gradient + Optimizer 분산 | 양호 | 우수 | 옵티마이저 메모리 절약 |
| ZeRO Stage 3 | 전부 분산 | 보통 | 양호 | 대형 모델, 메모리 극한 |
| Tensor Parallel | 레이어 내부 분할 | 비효율 | 우수 | NVLink 있는 서버 |
| Pipeline Parallel | 레이어 간 분할 | 양호 | 우수 | 초대형 모델 (100B+) |

---

## B.5 클라우드 GPU 비용 비교

### B.5.1 주요 클라우드 GPU 인스턴스 (2024~2025 기준)

| 클라우드 | 인스턴스 타입 | GPU | GPU 수 | VRAM | 시간당 비용 (on-demand) |
|---------|-------------|-----|--------|------|----------------------|
| **AWS** | p4d.24xlarge | A100 40GB | 8 | 320 GB | ~$32.77/hr |
| **AWS** | p5.48xlarge | H100 80GB | 8 | 640 GB | ~$98.32/hr |
| **AWS** | g5.xlarge | A10G 24GB | 1 | 24 GB | ~$1.01/hr |
| **AWS** | g6.xlarge | L4 24GB | 1 | 24 GB | ~$0.80/hr |
| **GCP** | a2-highgpu-1g | A100 40GB | 1 | 40 GB | ~$3.67/hr |
| **GCP** | a2-ultragpu-8g | A100 80GB | 8 | 640 GB | ~$40.97/hr |
| **GCP** | a3-highgpu-8g | H100 80GB | 8 | 640 GB | ~$98.32/hr |
| **RunPod** | Community | A100 80GB | 1 | 80 GB | ~$1.64/hr |
| **RunPod** | Community | H100 SXM | 1 | 80 GB | ~$3.29/hr |
| **RunPod** | Community | RTX 4090 | 1 | 24 GB | ~$0.44/hr |
| **Lambda Labs** | gpu_1x_a100_sxm4 | A100 SXM 80GB | 1 | 80 GB | ~$1.29/hr |
| **Lambda Labs** | gpu_1x_h100_sxm5 | H100 SXM | 1 | 80 GB | ~$2.49/hr |
| **Vast.ai** | Community | RTX 4090 | 1 | 24 GB | ~$0.30/hr |
| **Vast.ai** | Community | A100 80GB | 1 | 80 GB | ~$1.10/hr |

> **참고**: 가격은 변동이 크다. Spot/Preemptible 인스턴스 사용 시 50~70% 할인 가능하지만 언제든 중단될 수 있다.

### B.5.2 학습 시나리오별 비용 추정

7B VLM 모델을 10K 스텝 학습한다고 가정 (약 3~5시간 소요):

| 시나리오 | GPU 선택 | 예상 시간 | 비용 |
|---------|---------|----------|------|
| QLoRA (r=64) | 1× RTX 4090 (RunPod) | ~4시간 | ~$1.76 |
| QLoRA (r=64) | 1× A100 80GB (Lambda) | ~2.5시간 | ~$3.23 |
| LoRA (r=64, BF16) | 1× A100 80GB (Lambda) | ~3시간 | ~$3.87 |
| Full FT (BF16) | 4× A100 80GB (AWS spot) | ~2시간 | ~$26 |
| Full FT (BF16) | 8× H100 (GCP) | ~0.7시간 | ~$69 |

> **가성비 최강**: RunPod RTX 4090 + QLoRA 조합. 대부분의 실험은 이걸로 충분하다.

---

## B.6 VRAM 계산기 코드

아래 Python 코드로 원하는 설정의 VRAM 필요량을 계산할 수 있다.

```python
#!/usr/bin/env python3
"""
VRAM 계산기 (vram_calculator.py)

사용법:
    python vram_calculator.py
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class ModelConfig:
    """모델 아키텍처 설정"""
    name: str
    params_billion: float       # 파라미터 수 (B 단위)
    num_layers: int             # Transformer 레이어 수
    hidden_dim: int             # Hidden dimension
    num_attention_heads: int    # Query head 수
    num_kv_heads: int           # KV head 수 (GQA)
    head_dim: int = 128         # Head dimension
    vocab_size: int = 152064    # 어휘 크기
    intermediate_size: int = 0  # FFN intermediate (0이면 자동 계산)


@dataclass
class TrainingConfig:
    """학습 설정"""
    method: str                     # "full_fp32", "full_mixed", "lora", "qlora"
    batch_size: int = 1
    seq_len: int = 2048
    gradient_checkpointing: bool = True
    lora_rank: int = 64
    lora_target_modules: int = 4    # Q, K, V, O 등
    optimizer: str = "adamw"        # "adamw", "sgd", "adam_8bit"


# === 사전 정의된 모델 설정 ===
MODELS = {
    "qwen2-vl-7b": ModelConfig(
        name="Qwen2-VL-7B",
        params_billion=7.07,
        num_layers=28,
        hidden_dim=3584,
        num_attention_heads=28,
        num_kv_heads=4,
        head_dim=128,
    ),
    "qwen2-vl-72b": ModelConfig(
        name="Qwen2-VL-72B",
        params_billion=72.7,
        num_layers=80,
        hidden_dim=8192,
        num_attention_heads=64,
        num_kv_heads=8,
        head_dim=128,
    ),
    "llava-1.5-7b": ModelConfig(
        name="LLaVA-1.5-7B",
        params_billion=7.06,
        num_layers=32,
        hidden_dim=4096,
        num_attention_heads=32,
        num_kv_heads=32,
        head_dim=128,
    ),
    "llava-1.5-13b": ModelConfig(
        name="LLaVA-1.5-13B",
        params_billion=13.0,
        num_layers=40,
        hidden_dim=5120,
        num_attention_heads=40,
        num_kv_heads=40,
        head_dim=128,
    ),
    "internvl2-26b": ModelConfig(
        name="InternVL2-26B",
        params_billion=25.5,
        num_layers=48,
        hidden_dim=6144,
        num_attention_heads=48,
        num_kv_heads=48,
        head_dim=128,
    ),
}


def bytes_to_gb(b: float) -> float:
    """바이트를 GB로 변환"""
    return b / (1024 ** 3)


def calc_param_memory(model: ModelConfig, precision_bytes: float) -> float:
    """모델 파라미터 메모리 (bytes)"""
    return model.params_billion * 1e9 * precision_bytes


def calc_gradient_memory(model: ModelConfig, precision_bytes: float) -> float:
    """그래디언트 메모리 (bytes)"""
    return model.params_billion * 1e9 * precision_bytes


def calc_optimizer_memory(
    model: ModelConfig,
    trainable_params: float,
    optimizer: str = "adamw"
) -> float:
    """옵티마이저 상태 메모리 (bytes)"""
    if optimizer == "adamw":
        # Adam: master weights (FP32) + m (FP32) + v (FP32) = 12 bytes/param
        return trainable_params * 12
    elif optimizer == "sgd":
        # SGD with momentum: momentum (FP32) = 4 bytes/param
        return trainable_params * 4
    elif optimizer == "adam_8bit":
        # 8-bit Adam: m (INT8) + v (INT8) + master (FP32) = 6 bytes/param
        return trainable_params * 6
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")


def calc_lora_params(model: ModelConfig, config: TrainingConfig) -> float:
    """LoRA 학습 가능 파라미터 수"""
    # 각 target module에 대해: 2 * rank * hidden_dim * num_layers
    lora_params = (
        2 * config.lora_rank * model.hidden_dim
        * config.lora_target_modules * model.num_layers
    )
    return lora_params


def calc_kv_cache(model: ModelConfig, batch_size: int, seq_len: int,
                  precision_bytes: float = 2) -> float:
    """KV Cache 메모리 (bytes)"""
    # 2 (K+V) * layers * batch * seq * kv_heads * head_dim * precision
    return (
        2 * model.num_layers * batch_size * seq_len
        * model.num_kv_heads * model.head_dim * precision_bytes
    )


def calc_activation_memory(
    model: ModelConfig,
    batch_size: int,
    seq_len: int,
    precision_bytes: float = 2,
    gradient_checkpointing: bool = False,
) -> float:
    """Activation 메모리 근사치 (bytes)"""
    # 근사 공식: L * B * S * d * 34 * precision (간략화)
    act_per_layer = batch_size * seq_len * model.hidden_dim * 34 * precision_bytes
    total = act_per_layer * model.num_layers

    if gradient_checkpointing:
        # Gradient checkpointing: sqrt(L) 비율로 감소
        total = total / math.sqrt(model.num_layers)

    return total


def calc_training_vram(model: ModelConfig, config: TrainingConfig) -> dict:
    """학습 시 총 VRAM 계산"""
    result = {}

    if config.method == "full_fp32":
        param_bytes = 4
        grad_bytes = 4
        trainable_params = model.params_billion * 1e9
        param_mem = calc_param_memory(model, param_bytes)
        grad_mem = calc_gradient_memory(model, grad_bytes)
        opt_mem = calc_optimizer_memory(model, trainable_params, config.optimizer)

    elif config.method == "full_mixed":
        param_bytes = 2  # BF16
        grad_bytes = 2   # BF16
        trainable_params = model.params_billion * 1e9
        param_mem = calc_param_memory(model, param_bytes)
        grad_mem = calc_gradient_memory(model, grad_bytes)
        opt_mem = calc_optimizer_memory(model, trainable_params, config.optimizer)

    elif config.method == "lora":
        param_bytes = 2  # BF16 (frozen)
        lora_params = calc_lora_params(model, config)
        param_mem = calc_param_memory(model, param_bytes)
        grad_mem = lora_params * 2  # BF16 gradients for LoRA only
        opt_mem = calc_optimizer_memory(model, lora_params, config.optimizer)
        trainable_params = lora_params
        result["lora_params_million"] = round(lora_params / 1e6, 1)

    elif config.method == "qlora":
        param_bytes = 0.55  # ~4-bit + quantization overhead
        lora_params = calc_lora_params(model, config)
        param_mem = calc_param_memory(model, param_bytes)
        grad_mem = lora_params * 2
        opt_mem = calc_optimizer_memory(model, lora_params, config.optimizer)
        trainable_params = lora_params
        result["lora_params_million"] = round(lora_params / 1e6, 1)

    else:
        raise ValueError(f"Unknown method: {config.method}")

    act_mem = calc_activation_memory(
        model, config.batch_size, config.seq_len,
        precision_bytes=2, gradient_checkpointing=config.gradient_checkpointing
    )

    overhead = 500 * 1024 * 1024  # ~500MB CUDA overhead

    total = param_mem + grad_mem + opt_mem + act_mem + overhead

    result.update({
        "model": model.name,
        "method": config.method,
        "params_memory_gb": round(bytes_to_gb(param_mem), 2),
        "gradient_memory_gb": round(bytes_to_gb(grad_mem), 2),
        "optimizer_memory_gb": round(bytes_to_gb(opt_mem), 2),
        "activation_memory_gb": round(bytes_to_gb(act_mem), 2),
        "overhead_gb": round(bytes_to_gb(overhead), 2),
        "total_vram_gb": round(bytes_to_gb(total), 2),
        "trainable_params_billion": round(trainable_params / 1e9, 4),
    })

    return result


def calc_inference_vram(
    model: ModelConfig,
    precision: str = "bf16",
    batch_size: int = 1,
    seq_len: int = 4096,
) -> dict:
    """추론 시 VRAM 계산"""
    precision_map = {
        "fp32": 4, "bf16": 2, "fp16": 2,
        "int8": 1, "int4": 0.5, "fp8": 1,
    }
    p_bytes = precision_map[precision]

    param_mem = calc_param_memory(model, p_bytes)
    kv_mem = calc_kv_cache(model, batch_size, seq_len, precision_bytes=2)
    overhead = 500 * 1024 * 1024

    total = param_mem + kv_mem + overhead

    return {
        "model": model.name,
        "precision": precision,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "params_memory_gb": round(bytes_to_gb(param_mem), 2),
        "kv_cache_gb": round(bytes_to_gb(kv_mem), 2),
        "overhead_gb": round(bytes_to_gb(overhead), 2),
        "total_vram_gb": round(bytes_to_gb(total), 2),
    }


def recommend_gpu(vram_gb: float) -> list[str]:
    """필요 VRAM에 따른 GPU 추천"""
    gpus = [
        ("RTX 4090", 24),
        ("RTX 6000 Ada / L40S", 48),
        ("A100 40GB", 40),
        ("A100 80GB", 80),
        ("H100 80GB", 80),
        ("2× A100 80GB", 160),
        ("4× A100 80GB", 320),
        ("8× A100 80GB", 640),
        ("8× H100 80GB", 640),
    ]
    recommendations = []
    for name, vram in gpus:
        if vram >= vram_gb * 1.1:  # 10% 마진
            recommendations.append(f"{name} ({vram}GB)")
        if len(recommendations) >= 3:
            break
    return recommendations if recommendations else ["8× H100 이상 필요"]


def main():
    print("=" * 70)
    print("  VRAM Calculator for LLM/VLM Training & Inference")
    print("=" * 70)

    # === 학습 시나리오 ===
    scenarios = [
        TrainingConfig(method="full_fp32", batch_size=1, seq_len=2048),
        TrainingConfig(method="full_mixed", batch_size=1, seq_len=2048),
        TrainingConfig(method="lora", batch_size=4, seq_len=2048, lora_rank=64),
        TrainingConfig(method="qlora", batch_size=4, seq_len=2048, lora_rank=64),
    ]

    for model_key in ["qwen2-vl-7b", "qwen2-vl-72b"]:
        model = MODELS[model_key]
        print(f"\n{'─' * 70}")
        print(f"  모델: {model.name} ({model.params_billion}B params)")
        print(f"{'─' * 70}")

        for cfg in scenarios:
            result = calc_training_vram(model, cfg)
            gpus = recommend_gpu(result["total_vram_gb"])
            print(f"\n  [{cfg.method}] batch={cfg.batch_size}, seq={cfg.seq_len}")
            print(f"    파라미터:    {result['params_memory_gb']:>8.2f} GB")
            print(f"    그래디언트:  {result['gradient_memory_gb']:>8.2f} GB")
            print(f"    옵티마이저:  {result['optimizer_memory_gb']:>8.2f} GB")
            print(f"    Activation:  {result['activation_memory_gb']:>8.2f} GB")
            print(f"    오버헤드:    {result['overhead_gb']:>8.2f} GB")
            print(f"    ────────────────────────────")
            print(f"    총 VRAM:     {result['total_vram_gb']:>8.2f} GB")
            print(f"    추천 GPU:    {', '.join(gpus)}")

    # === 추론 시나리오 ===
    print(f"\n{'=' * 70}")
    print("  추론 VRAM 추정 (batch=1, seq=4096)")
    print(f"{'=' * 70}")

    for model_key in ["qwen2-vl-7b", "qwen2-vl-72b"]:
        model = MODELS[model_key]
        print(f"\n  모델: {model.name}")
        for prec in ["bf16", "int8", "int4"]:
            result = calc_inference_vram(model, precision=prec)
            print(f"    [{prec:>4s}] 파라미터={result['params_memory_gb']:.1f}GB "
                  f"+ KV={result['kv_cache_gb']:.2f}GB "
                  f"= 총 {result['total_vram_gb']:.1f}GB")


if __name__ == "__main__":
    main()
```

---

## B.7 스토리지 가이드

### B.7.1 학습 데이터 + 모델 저장 공간

| 항목 | 크기 | 비고 |
|------|------|------|
| 7B 모델 가중치 (BF16) | ~14 GB | safetensors 포맷 |
| 72B 모델 가중치 (BF16) | ~145 GB | |
| 7B 모델 가중치 (GPTQ 4-bit) | ~4 GB | |
| LoRA 어댑터 (r=64) | 50~200 MB | 모델 크기 대비 매우 작음 |
| OCR 학습 데이터 (10K images) | ~5~20 GB | 해상도에 따라 다름 |
| HuggingFace 캐시 | 50~200 GB | 여러 모델 다운로드 시 |
| wandb 로그 | ~1~10 GB | 실험 수에 비례 |
| 체크포인트 (Full FT, 7B) | ~28 GB/개 | BF16 기준 |
| 체크포인트 (LoRA, 7B) | ~200 MB/개 | 어댑터만 저장 |

### B.7.2 스토리지 권장 사항

- **NVMe SSD**: 학습 데이터 + 모델 로딩용. HDD 사용 시 DataLoader에서 병목 발생.
- **최소 1TB**: 모델 2~3개 + 데이터셋 + 체크포인트 저장.
- **네트워크 스토리지**: 멀티 노드 학습 시 NFS 또는 S3/GCS 마운트 (FUSE).

---

## B.8 CPU / RAM 가이드

| 항목 | 최소 | 권장 |
|------|------|------|
| CPU 코어 | 8 | 16~32 |
| RAM | 32 GB | 64~128 GB |
| RAM/GPU 비율 | GPU VRAM × 2 | GPU VRAM × 4 |

> DeepSpeed ZeRO-Offload 사용 시 CPU RAM이 핵심이다. GPU에서 옵티마이저 상태를 CPU로 내리기 때문에 RAM이 넉넉해야 한다.

DataLoader `num_workers` 계산:

$$
\text{num\_workers} \approx \min\left(\text{CPU cores} - 2, \quad 4 \times N_{\text{GPU}}\right)
$$

---

## B.9 전력 / 냉각 고려사항

고성능 GPU는 전력 소비가 크다. 서버룸이나 가정 환경에서 사전에 확인해야 할 사항:

| GPU 구성 | 예상 시스템 전력 | 필요 PSU | 권장 회로 |
|---------|---------------|---------|----------|
| 1× RTX 4090 | ~650W | 850W+ | 일반 가정용 OK |
| 2× RTX 4090 | ~1100W | 1600W+ | 20A 전용 회로 |
| 4× A100 (PCIe) | ~1800W | 2400W+ | 서버용 PDU |
| 8× H100 (SXM) | ~6500W | DGX 전용 | 3상 전원 |

> RTX 4090 2장 이상을 가정에서 돌리면 차단기가 내려갈 수 있다. 전력 사전 확인 필수.
