# 14.1 Quantization (양자화)

> **모델 가중치와 활성값의 비트 수를 줄여 메모리 사용량과 추론 속도를 최적화하는 기법**

---

## 핵심 용어

| 용어 | 정의 |
|------|------|
| **PTQ** (Post-Training Quantization) | 학습 완료된 모델에 바로 양자화를 적용하는 방식. 재학습 불필요 |
| **QAT** (Quantization-Aware Training) | 학습 중 양자화 시뮬레이션을 삽입해 양자화 오차를 최소화하는 방식 |
| **AWQ** (Activation-Aware Weight Quantization) | 활성값(activation) 분포를 관찰해 중요 채널을 보호하며 가중치를 양자화 |
| **GPTQ** (Generative Pre-Trained Transformer Quantization) | Hessian 기반 역행렬을 활용한 layer-wise 최적 양자화 기법 |
| **W4A16** | 가중치 4bit, 활성값 16bit로 양자화하는 설정 |
| **W8A8** | 가중치 8bit, 활성값 8bit로 양자화하는 설정 |
| **Scale (s)** | 실수 범위를 정수 범위로 매핑하는 스케일 팩터 |
| **Zero-point (z)** | 양자화된 정수 공간에서 실수 0에 대응하는 오프셋 값 |
| **Calibration** | PTQ에서 대표 데이터를 통과시켜 활성값 범위를 수집하는 과정 |
| **Per-channel / Per-tensor** | 양자화 파라미터를 채널별 또는 텐서 전체에 적용하는 단위 |

---

## 수학적 원리

### 1. 균일 양자화 (Uniform Quantization)

실수값 $x$를 $b$-bit 정수 $q$로 변환하는 기본 공식:

$$
q = \text{clamp}\left(\text{round}\left(\frac{x}{s}\right) + z, \; 0, \; 2^b - 1\right)
$$

역양자화(Dequantization)로 근사 복원:

$$
\hat{x} = s \cdot (q - z)
$$

여기서 $\hat{x} \approx x$ 이고, 양자화 오차 $\epsilon = x - \hat{x}$가 발생한다.

### 2. Scale과 Zero-point 계산

**Asymmetric Quantization** (비대칭):

$$
s = \frac{x_{\max} - x_{\min}}{2^b - 1}
$$

$$
z = -\text{round}\left(\frac{x_{\min}}{s}\right)
$$

**Symmetric Quantization** (대칭):

$$
s = \frac{\max(|x_{\max}|, |x_{\min}|)}{2^{b-1} - 1}
$$

$$
z = 0
$$

대칭 양자화는 zero-point가 0이라서 연산이 더 빠르지만, 비대칭 분포에서는 범위 낭비가 생긴다.

### 3. AWQ: Activation-Aware Weight Quantization

AWQ의 핵심 아이디어는 **모든 가중치가 동등하게 중요하지 않다**는 것이다. 활성값이 큰 채널의 가중치는 양자화 오차에 더 민감하다.

**목표 함수**:

$$
\min_{\hat{W}} \| WX - \hat{W}X \|^2
$$

여기서 $W$는 원본 가중치, $\hat{W}$는 양자화된 가중치, $X$는 calibration 활성값이다.

**채널별 Saliency 계산**:

$$
\text{saliency}_c = \| W_c \|_2 \cdot \| X_c \|_2
$$

saliency가 높은 채널에 대해 스케일링 팩터 $\alpha_c$를 적용:

$$
W'_c = W_c \cdot \alpha_c, \quad X'_c = X_c / \alpha_c
$$

이렇게 하면 $W'_c X'_c = W_c X_c$로 수학적 등가를 유지하면서, $W'_c$의 값 범위가 확대되어 양자화 오차가 줄어든다. 최적 스케일 $\alpha_c$는 그리드 서치로 결정한다.

### 4. GPTQ: Hessian 기반 Layer-wise 양자화

OBS(Optimal Brain Surgeon)에서 파생된 기법으로, 각 가중치를 순차적으로 양자화하면서 나머지 가중치를 보정한다.

**양자화 오차 보정 공식**:

$$
\delta_q = \frac{w_q - \hat{w}_q}{[H^{-1}]_{qq}}
$$

여기서:
- $w_q$: 원본 가중치
- $\hat{w}_q$: 양자화된 가중치
- $H$: Hessian 행렬 ($H = 2X^TX$로 근사)
- $[H^{-1}]_{qq}$: Hessian 역행렬의 $(q,q)$ 대각 원소

나머지 미양자화 가중치에 대한 업데이트:

$$
w_j \leftarrow w_j - \frac{w_q - \hat{w}_q}{[H^{-1}]_{qq}} \cdot [H^{-1}]_{qj}, \quad \forall j > q
$$

GPTQ는 열(column) 단위로 처리하며, 128열씩 그룹으로 묶어 Hessian 역행렬을 Cholesky 분해로 효율적으로 계산한다.

### 5. 양자화 오차 분석

**Mean Squared Error**:

$$
\text{MSE} = \mathbb{E}[(x - \hat{x})^2] = \frac{s^2}{12}
$$

균일 분포 가정 시, 양자화 오차는 scale의 제곱에 비례한다. bit 수를 1 늘리면 scale이 절반이 되므로 MSE는 1/4로 감소한다.

**Signal-to-Quantization-Noise Ratio**:

$$
\text{SQNR} = 10 \log_{10}\left(\frac{\sigma_x^2}{\text{MSE}}\right) \approx 6.02b + C \;\text{(dB)}
$$

bit당 약 6dB의 SQNR 개선이 있다.

---

## PTQ vs QAT vs AWQ vs GPTQ 비교

| 항목 | PTQ | QAT | AWQ | GPTQ |
|------|-----|-----|-----|------|
| **재학습 필요** | 불필요 | 필요 (전체 학습) | 불필요 | 불필요 |
| **Calibration 데이터** | 소량 (수백 샘플) | 전체 학습 데이터 | 소량 | 소량 |
| **양자화 시간** | 수 분 | 수 시간~수 일 | 수십 분 | 수십 분 |
| **정확도 손실 (W4)** | 큼 (1~5%) | 최소 (<0.5%) | 작음 (0.5~1%) | 작음 (0.5~1%) |
| **지원 bit** | 8/4 bit | 8/4 bit | 4/3 bit | 4/3/2 bit |
| **GPU 메모리** | 원본 모델 크기 | 원본 + 그래디언트 | 원본 모델 크기 | 원본 모델 크기 |
| **LLM 적합성** | 낮음 | 비현실적 (비용) | 높음 | 높음 |
| **구현 난이도** | 쉬움 | 어려움 | 보통 | 보통 |
| **대표 라이브러리** | PyTorch native | PyTorch native | AutoAWQ | AutoGPTQ |

### 의료 문서 OCR에서의 양자화 전략

의료 문서 OCR 모델은 **정확도가 최우선**이다. 약물 용량, 환자 정보 등 오류 허용 범위가 극도로 좁기 때문에:

1. **W8A8 PTQ**: 가장 안전한 선택. 정확도 손실 거의 없음
2. **W4A16 AWQ**: 메모리 절약이 필요할 때. calibration에 의료 문서 샘플 반드시 포함
3. **GPTQ W4**: AWQ와 유사 성능이지만 특정 모델에서 더 나을 수 있음
4. **QAT**: LLM 규모에서는 비현실적. 작은 OCR 전용 모델이면 고려 가능

---

## 코드

### 1. 균일 양자화 기본 구현

```python
"""
균일 양자화(Uniform Quantization) 기본 구현
- Symmetric / Asymmetric 양자화 모두 지원
- Per-tensor / Per-channel 단위 지원
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QuantConfig:
    """양자화 설정"""
    bits: int = 8
    mode: Literal["symmetric", "asymmetric"] = "symmetric"
    granularity: Literal["per_tensor", "per_channel"] = "per_tensor"
    channel_axis: int = 0


class UniformQuantizer:
    """균일 양자화 구현체"""

    def __init__(self, config: QuantConfig):
        self.config = config
        self.scale: Optional[torch.Tensor] = None
        self.zero_point: Optional[torch.Tensor] = None

    def _compute_qparams_symmetric(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """대칭 양자화 파라미터 계산"""
        qmax = 2 ** (self.config.bits - 1) - 1

        if self.config.granularity == "per_channel":
            # 채널 축을 제외한 나머지 축에서 max 계산
            dims = [i for i in range(x.ndim) if i != self.config.channel_axis]
            x_absmax = x.abs().amax(dim=dims, keepdim=True)
        else:
            x_absmax = x.abs().max()

        scale = x_absmax / qmax
        scale = torch.clamp(scale, min=1e-8)  # 0 방지
        zero_point = torch.zeros_like(scale, dtype=torch.int64)

        return scale, zero_point

    def _compute_qparams_asymmetric(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """비대칭 양자화 파라미터 계산"""
        qmin = 0
        qmax = 2 ** self.config.bits - 1

        if self.config.granularity == "per_channel":
            dims = [i for i in range(x.ndim) if i != self.config.channel_axis]
            x_min = x.amin(dim=dims, keepdim=True)
            x_max = x.amax(dim=dims, keepdim=True)
        else:
            x_min = x.min()
            x_max = x.max()

        scale = (x_max - x_min) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-8)
        zero_point = torch.round(-x_min / scale).to(torch.int64)
        zero_point = torch.clamp(zero_point, qmin, qmax)

        return scale, zero_point

    def calibrate(self, x: torch.Tensor) -> None:
        """Calibration: scale과 zero_point 계산"""
        if self.config.mode == "symmetric":
            self.scale, self.zero_point = self._compute_qparams_symmetric(x)
        else:
            self.scale, self.zero_point = self._compute_qparams_asymmetric(x)

        logger.info(
            f"Calibrated: bits={self.config.bits}, mode={self.config.mode}, "
            f"scale_range=[{self.scale.min():.6f}, {self.scale.max():.6f}]"
        )

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """실수 텐서를 양자화된 정수 텐서로 변환"""
        if self.scale is None:
            raise RuntimeError("calibrate()를 먼저 호출해야 한다")

        if self.config.mode == "symmetric":
            qmin = -(2 ** (self.config.bits - 1))
            qmax = 2 ** (self.config.bits - 1) - 1
        else:
            qmin = 0
            qmax = 2 ** self.config.bits - 1

        q = torch.round(x / self.scale) + self.zero_point
        q = torch.clamp(q, qmin, qmax).to(torch.int8 if self.config.bits <= 8 else torch.int16)

        return q

    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        """양자화된 정수 텐서를 실수 텐서로 복원"""
        if self.scale is None:
            raise RuntimeError("calibrate()를 먼저 호출해야 한다")

        return self.scale * (q.float() - self.zero_point.float())

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """양자화 후 역양자화 (Fake Quantization)"""
        q = self.quantize(x)
        return self.dequantize(q)


def compute_quantization_error(
    original: torch.Tensor,
    quantized: torch.Tensor
) -> dict:
    """양자화 오차 지표 계산"""
    diff = original - quantized
    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    max_err = diff.abs().max().item()

    signal_power = (original ** 2).mean().item()
    sqnr = 10 * np.log10(signal_power / mse) if mse > 0 else float("inf")

    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_err,
        "sqnr_db": sqnr,
    }


# 사용 예시
if __name__ == "__main__":
    torch.manual_seed(42)
    weight = torch.randn(256, 512)  # 예시 가중치

    for bits in [8, 4]:
        for mode in ["symmetric", "asymmetric"]:
            config = QuantConfig(bits=bits, mode=mode)
            quantizer = UniformQuantizer(config)
            quantizer.calibrate(weight)

            w_hat = quantizer.quantize_dequantize(weight)
            errors = compute_quantization_error(weight, w_hat)

            print(f"\n[{bits}bit {mode}]")
            print(f"  MSE:      {errors['mse']:.6f}")
            print(f"  MAE:      {errors['mae']:.6f}")
            print(f"  Max Err:  {errors['max_error']:.6f}")
            print(f"  SQNR:     {errors['sqnr_db']:.2f} dB")

            # 메모리 절약 계산
            original_size = weight.nelement() * 4  # FP32 = 4 bytes
            quantized_size = weight.nelement() * (bits / 8)
            compression = original_size / quantized_size
            print(f"  압축비:   {compression:.1f}x")
```

### 2. AutoAWQ 사용법

```python
"""
AutoAWQ를 사용한 LLM 양자화 파이프라인
- 의료 문서 OCR 모델 양자화에 최적화
"""

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch
import json
import logging
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AWQQuantizer:
    """AutoAWQ 기반 모델 양자화 래퍼"""

    def __init__(
        self,
        model_id: str,
        quant_config: Optional[dict] = None,
        output_dir: str = "./quantized_model",
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.quant_config = quant_config or {
            "zero_point": True,        # Asymmetric quantization
            "q_group_size": 128,       # 그룹 크기 (작을수록 정밀, 메모리 증가)
            "w_bit": 4,                # 가중치 bit 수
            "version": "GEMM",         # GEMM 또는 GEMV
        }

        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """원본 모델 로드"""
        logger.info(f"모델 로드 중: {self.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self.model = AutoAWQForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            safetensors=True,
        )

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"파라미터 수: {param_count / 1e9:.2f}B")

    def prepare_calibration_data(
        self,
        dataset: Optional[List[str]] = None,
        n_samples: int = 128,
        seq_len: int = 512,
    ) -> List[str]:
        """Calibration 데이터 준비 (의료 문서 샘플 포함 권장)"""
        if dataset is not None:
            logger.info(f"커스텀 calibration 데이터 사용: {len(dataset)}개 샘플")
            return dataset[:n_samples]

        # 기본 calibration 데이터 (실제로는 의료 문서 텍스트 사용해야 함)
        logger.warning(
            "기본 calibration 데이터 사용 중. "
            "의료 문서 OCR 모델이라면 도메인 데이터를 넣어야 정확도가 보장된다."
        )
        calib_data = [
            "환자명: 홍길동, 생년월일: 1990-01-15, 진료과: 내과",
            "처방전: Metformin 500mg 1일 2회, 식후 30분",
            "검사결과: HbA1c 7.2%, 공복혈당 142mg/dL",
            "진단명: 제2형 당뇨병 (E11.9)",
        ] * (n_samples // 4)

        return calib_data[:n_samples]

    def quantize(
        self,
        calibration_data: Optional[List[str]] = None,
    ) -> None:
        """AWQ 양자화 실행"""
        if self.model is None:
            self.load_model()

        calib_data = self.prepare_calibration_data(calibration_data)

        logger.info(f"양자화 시작: {self.quant_config}")
        self.model.quantize(
            self.tokenizer,
            quant_config=self.quant_config,
            calib_data=calib_data,
        )
        logger.info("양자화 완료")

    def save(self) -> None:
        """양자화된 모델 저장"""
        logger.info(f"저장 경로: {self.output_dir}")

        self.model.save_quantized(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))

        # 양자화 설정 저장
        config_path = self.output_dir / "quant_config.json"
        with open(config_path, "w") as f:
            json.dump(self.quant_config, f, indent=2)

        logger.info("모델 저장 완료")

    def validate(self, test_prompts: List[str]) -> List[str]:
        """양자화된 모델 검증"""
        results = []
        for prompt in test_prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(
                self.model.device
            )
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                )
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(result)
            logger.info(f"Input:  {prompt[:50]}...")
            logger.info(f"Output: {result[:100]}...")

        return results


# 사용 예시
if __name__ == "__main__":
    quantizer = AWQQuantizer(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        quant_config={
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM",
        },
        output_dir="./models/qwen2.5-7b-awq-w4",
    )

    # 의료 문서 calibration 데이터
    medical_calib = [
        "처방전 내용: Aspirin 100mg, 1일 1회",
        "혈액검사 결과: WBC 6.5, RBC 4.8, Hb 14.2",
        # ... 실제로는 128개 이상의 도메인 샘플 필요
    ]

    quantizer.quantize(calibration_data=medical_calib)
    quantizer.save()
```

### 3. AutoGPTQ 사용법

```python
"""
AutoGPTQ를 사용한 LLM 양자화
- GPTQ 알고리즘 기반 4bit 양자화
"""

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
import torch
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quantize_with_gptq(
    model_id: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    calibration_texts: List[str] = None,
    desc_act: bool = True,
) -> None:
    """GPTQ 양자화 실행

    Args:
        model_id: HuggingFace 모델 ID
        output_dir: 저장 경로
        bits: 양자화 bit 수 (2, 3, 4, 8)
        group_size: 양자화 그룹 크기
        calibration_texts: Calibration용 텍스트 리스트
        desc_act: Activation order로 열 정렬 (정확도 ↑, 속도 ↓)
    """

    # 1. 양자화 설정
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=True,                # Symmetric quantization
        true_sequential=True,    # 레이어 순차 처리 (메모리 절약)
    )

    # 2. 모델 & 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoGPTQForCausalLM.from_pretrained(
        model_id,
        quantize_config=quantize_config,
        trust_remote_code=True,
    )

    # 3. Calibration 데이터 준비
    if calibration_texts is None:
        calibration_texts = [
            "의료 기록: 환자 ID 2024-001, 진단명 급성 위염",
        ] * 128

    calibration_dataset = [
        tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        for text in calibration_texts
    ]

    # 4. 양자화 실행
    logger.info(f"GPTQ 양자화 시작 ({bits}bit, group_size={group_size})")
    model.quantize(calibration_dataset)

    # 5. 저장
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"양자화 모델 저장 완료: {output_dir}")


def compare_model_sizes(original_path: str, quantized_path: str) -> Dict[str, float]:
    """원본 vs 양자화 모델 크기 비교"""
    import os

    def get_dir_size(path: str) -> int:
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return total

    orig_size = get_dir_size(original_path)
    quant_size = get_dir_size(quantized_path)

    return {
        "original_gb": orig_size / (1024 ** 3),
        "quantized_gb": quant_size / (1024 ** 3),
        "compression_ratio": orig_size / quant_size,
        "size_reduction_pct": (1 - quant_size / orig_size) * 100,
    }


if __name__ == "__main__":
    quantize_with_gptq(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        output_dir="./models/qwen2.5-7b-gptq-w4",
        bits=4,
        group_size=128,
    )
```

---

## 양자화 포맷과 메모리 계산

### 모델 크기 추정 공식

$$
\text{Size (GB)} = \frac{N_{\text{params}} \times b}{8 \times 10^9}
$$

| 모델 크기 | FP32 | FP16 | INT8 | INT4 |
|-----------|------|------|------|------|
| 7B | 28 GB | 14 GB | 7 GB | 3.5 GB |
| 13B | 52 GB | 26 GB | 13 GB | 6.5 GB |
| 70B | 280 GB | 140 GB | 70 GB | 35 GB |

### 추론 시 GPU 메모리 요구량 (근사)

$$
\text{GPU Memory} \approx \text{Model Size} + \text{KV Cache} + \text{Activation Memory} + \text{Overhead}
$$

4bit 양자화 7B 모델의 경우:
- 모델: ~3.5 GB
- KV Cache (2K context): ~0.5 GB
- Activation + Overhead: ~1 GB
- **총 ~5 GB** → RTX 3060 (12GB)에서 여유롭게 구동 가능

---

## 실전 팁

### 양자화 선택 가이드

```
GPU VRAM 충분 (>= 2x 모델 크기)?
├── Yes → FP16 그대로 사용
└── No → 양자화 필요
    ├── 정확도 최우선 → W8A8 PTQ
    ├── 메모리 절약 필요 → W4A16
    │   ├── 빠른 양자화 → AWQ (GEMM)
    │   └── 최고 정확도 → GPTQ (desc_act=True)
    └── 극한 경량화 → W3 또는 W2 (정확도 크게 하락 가능)
```

### 의료 OCR에서의 주의사항

1. **Calibration 데이터에 반드시 의료 문서 포함**: 약물명, 용량, 단위 등 도메인 특화 토큰이 정확히 표현되어야 한다
2. **양자화 후 반드시 평가**: CER/WER뿐 아니라 약물 용량 정확도, 진단 코드 정확도를 별도로 측정
3. **Per-channel 양자화 선호**: 의료 텍스트는 분포가 불균일하므로 per-tensor보다 per-channel이 안전하다
4. **Group size 128 이하 권장**: 그룹이 작을수록 세밀한 양자화가 가능하다 (메모리 약간 증가)

---

## 용어 체크리스트

| 용어 | 이해 여부 | 핵심 포인트 |
|------|:---------:|-------------|
| PTQ | [ ] | 학습 후 바로 양자화. 빠르지만 정확도 손실 가능 |
| QAT | [ ] | 학습 중 양자화 시뮬레이션. 최고 정확도지만 비용 큼 |
| AWQ | [ ] | 활성값 기반 중요 채널 보호. LLM에 최적 |
| GPTQ | [ ] | Hessian 기반 가중치 보정. 높은 압축률 가능 |
| W4A16 | [ ] | 가중치 4bit, 활성값 16bit. 메모리 75% 절약 |
| W8A8 | [ ] | 가중치/활성값 모두 8bit. 안전한 선택 |
| Scale / Zero-point | [ ] | 양자화 매핑 파라미터. 정밀도의 핵심 |
| Per-channel | [ ] | 채널별 독립 양자화. 정밀하지만 오버헤드 있음 |
| Calibration | [ ] | 대표 데이터로 양자화 범위 결정 |
| SQNR | [ ] | 양자화 품질 지표. bit당 ~6dB 개선 |
| Group Quantization | [ ] | 가중치를 그룹으로 나눠 각각 양자화 |
| Fake Quantization | [ ] | QAT에서 사용. 양자화→역양자화로 오차 시뮬레이션 |
