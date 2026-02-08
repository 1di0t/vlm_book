---
---

# 5.4 패턴 비교 분석

5.1~5.3절에서 다룬 Cross-Attention, Projection, Early Fusion 세 가지 VLM 아키텍처 패턴을 종합 비교한다. 학습 효율성, 추론 속도, 표현력, 구현 복잡도 관점에서 분석하고, 특히 OCR 태스크와 의료 문서 OCR에 최적인 패턴을 논의한다.

---

## 핵심 용어

| 용어 | 정의 | 관련 섹션 |
|------|------|-----------|
| **Information Bottleneck** | 입력 정보를 제한된 차원으로 압축할 때 발생하는 정보 손실 | Cross-Attention, Projection |
| **Modality Gap** | Vision과 Language feature space 사이의 의미적 거리 | Feature Alignment |
| **Compute-Performance Tradeoff** | 연산 비용 대비 성능 향상의 균형점 | 전체 비교 |
| **Task-Specific Optimality** | 특정 태스크(OCR, VQA 등)에 최적화된 아키텍처 선택 | OCR 분석 |
| **Scalability** | 모델 크기, 데이터, 해상도 증가에 대한 확장성 | 전체 비교 |

---

## 5.4.1 종합 비교표

### 아키텍처 특성 비교

| 항목 | Cross-Attention | Projection | Early Fusion |
|------|----------------|------------|--------------|
| **대표 모델** | Flamingo, BLIP-2 | LLaVA, LLaVA-1.5 | Fuyu |
| **Vision Encoder** | 사용 (Frozen) | 사용 (Frozen) | 없음 (직접 패치 처리) |
| **LLM 수정** | 레이어 사이에 모듈 삽입 | 수정 없음 (입력만 변경) | 전체가 통합 모델 |
| **Visual Token 수** | 압축됨 (32~64) | 유지 또는 감소 (144~576) | 원본 그대로 (수백~수천) |
| **정보 흐름** | 단방향 (vision→text) | 단방향 (vision→text) | 양방향 (vision↔text) |
| **Connector 파라미터** | ~100M (Q-Former) | ~4~21M (Linear/MLP) | 0 (통합) |

### 성능 지표 비교

| 지표 | Cross-Attention | Projection | Early Fusion |
|------|----------------|------------|--------------|
| **학습 효율성** | ★★★★★ | ★★★★☆ | ★★☆☆☆ |
| **추론 속도** | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ |
| **표현력** | ★★★★☆ | ★★★☆☆ | ★★★★★ |
| **구현 복잡도** | ★★☆☆☆ (복잡) | ★★★★★ (단순) | ★★★☆☆ (중간) |
| **사전학습 활용** | ★★★★★ | ★★★★★ | ★☆☆☆☆ |
| **고해상도 처리** | ★★★☆☆ | ★★★★☆ | ★★★★★ |
| **Few-shot 성능** | ★★★★★ | ★★★☆☆ | ★★★★☆ |

---

## 수학적 원리: 정보 병목 분석

### Information Bottleneck 이론

각 VLM 패턴을 Information Bottleneck 관점에서 분석한다. $X$가 이미지 입력, $Y$가 텍스트 출력, $Z$가 중간 표현일 때:

$$
\mathcal{L}_{\text{IB}} = I(X; Z) - \beta \cdot I(Z; Y)
$$

여기서:
- $I(X; Z)$: 입력 이미지와 중간 표현 사이의 상호 정보량 (압축 정도)
- $I(Z; Y)$: 중간 표현과 출력 텍스트 사이의 상호 정보량 (유용한 정보 보존)
- $\beta$: 압축과 정보 보존의 균형을 조절하는 하이퍼파라미터

### 패턴별 정보 병목

#### Cross-Attention (Q-Former)

Q-Former는 $N_v$개의 visual token을 $M$개로 압축한다. 이 과정에서 명시적인 정보 병목이 발생한다.

$$
I(X; Z_{\text{Q-Former}}) \leq M \cdot D \cdot \log_2(L)
$$

여기서 $L$은 feature의 양자화 레벨이다.

$$
\text{Compression Ratio} = \frac{N_v \cdot D_{\text{vision}}}{M \cdot D_{\text{Q-Former}}} = \frac{257 \times 1408}{32 \times 768} \approx 14.7
$$

Q-Former의 핵심: **태스크에 관련된 정보만 선택적으로 보존**한다. 학습 가능한 query가 "무엇을 보존할지"를 결정한다.

$$
Z_{\text{compressed}} = \text{softmax}\left(\frac{Q_{\text{learn}} K_{\text{img}}^T}{\sqrt{d_k}}\right) V_{\text{img}}
$$

Attention 가중치가 정보 선택의 게이트 역할을 한다.

#### Projection (LLaVA)

Projection 방식에서는 토큰 수를 유지하므로 공간적 정보 손실이 적다.

$$
I(X; Z_{\text{proj}}) = I(X; WZ_{\text{vision}} + b) \leq I(X; Z_{\text{vision}})
$$

데이터 처리 부등식(Data Processing Inequality)에 의해, 선형 변환을 거치면 정보량은 감소하거나 같다.

단, 토큰 수 $N_v$를 유지하므로:

$$
\text{Compression Ratio} = \frac{D_{\text{vision}}}{D_{\text{LLM}}} = \frac{1024}{4096} = 0.25 \quad (\text{차원 확장})
$$

차원이 오히려 늘어나므로, 공간적 정보 병목은 거의 없다. 대신 **LLM의 시퀀스 길이를 많이 차지**하는 게 비용이다.

#### Early Fusion

Early Fusion에서는 명시적인 정보 병목이 없다. 원본 이미지 패치가 그대로 모델에 입력된다.

$$
I(X; Z_{\text{early}}) \approx I(X; X) = H(X)
$$

이론적으로 입력 정보를 최대한 보존하지만:
1. 모델이 vision 사전학습 없이 이미지를 이해해야 함
2. 충분한 학습 데이터 없이는 정보를 효과적으로 활용 못함

### 정보 효율성 비교

$$
\text{Information Efficiency} = \frac{I(Z; Y)}{|Z|}
$$

| 패턴 | $|Z|$ (토큰 수) | $I(Z;Y)$ (상대적) | 효율성 |
|------|----------------|-------------------|--------|
| Cross-Attention | 32 (Q-Former) | 0.85 | **매우 높음** |
| Projection | 576 (ViT-L/14@336) | 0.95 | 중간 |
| Early Fusion | 2304 (672px) | 0.98 | 낮음 |

Cross-Attention이 **토큰당 정보 밀도**가 가장 높다. 적은 수의 토큰으로 필요한 정보를 효율적으로 압축한다.

---

## 5.4.2 연산 비용 상세 분석

### FLOPs 비교

```python
import torch
import math
from typing import Optional


def compute_flops(
    pattern: str,
    image_size: int = 336,
    patch_size: int = 14,
    dim_vision: int = 1024,
    dim_llm: int = 4096,
    num_llm_layers: int = 32,
    text_len: int = 256,
    num_query_tokens: int = 32,
) -> dict:
    """
    각 VLM 패턴의 FLOPs 추정.

    Returns:
        dict with breakdown of computation costs
    """
    num_patches = (image_size // patch_size) ** 2  # 576

    if pattern == "cross_attention":
        # Vision Encoder FLOPs (frozen, 추론만)
        vision_flops = num_patches * dim_vision * dim_vision * 4 * 24  # ViT-L 24 layers
        # Q-Former FLOPs
        qformer_flops = num_query_tokens * num_patches * dim_vision * 12  # 12 layers
        # LLM FLOPs (text만, visual은 cross-attn으로)
        llm_self_attn = text_len * text_len * dim_llm * 2 * num_llm_layers
        llm_cross_attn = text_len * num_query_tokens * dim_llm * 2 * num_llm_layers
        llm_ffn = text_len * dim_llm * dim_llm * 8 * num_llm_layers

        return {
            "vision_encoder": vision_flops,
            "connector": qformer_flops,
            "llm_attention": llm_self_attn + llm_cross_attn,
            "llm_ffn": llm_ffn,
            "total": vision_flops + qformer_flops + llm_self_attn + llm_cross_attn + llm_ffn,
        }

    elif pattern == "projection":
        # Vision Encoder FLOPs (frozen)
        vision_flops = num_patches * dim_vision * dim_vision * 4 * 24
        # MLP Connector FLOPs
        connector_flops = num_patches * dim_vision * dim_llm * 2  # 2-layer MLP
        # LLM FLOPs (visual + text 전체 시퀀스)
        total_len = num_patches + text_len
        llm_self_attn = total_len * total_len * dim_llm * 2 * num_llm_layers
        llm_ffn = total_len * dim_llm * dim_llm * 8 * num_llm_layers

        return {
            "vision_encoder": vision_flops,
            "connector": connector_flops,
            "llm_attention": llm_self_attn,
            "llm_ffn": llm_ffn,
            "total": vision_flops + connector_flops + llm_self_attn + llm_ffn,
        }

    elif pattern == "early_fusion":
        # Vision Encoder 없음
        # 패치 임베딩 (단순 선형 변환)
        patch_dim = patch_size * patch_size * 3
        embed_flops = num_patches * patch_dim * dim_llm
        # Transformer FLOPs (전체 시퀀스)
        total_len = num_patches + text_len
        self_attn = total_len * total_len * dim_llm * 2 * num_llm_layers
        ffn = total_len * dim_llm * dim_llm * 8 * num_llm_layers

        return {
            "vision_encoder": 0,
            "connector": embed_flops,
            "llm_attention": self_attn,
            "llm_ffn": ffn,
            "total": embed_flops + self_attn + ffn,
        }

    raise ValueError(f"Unknown pattern: {pattern}")


def compare_all_patterns():
    """세 패턴의 연산 비용 비교"""
    patterns = ["cross_attention", "projection", "early_fusion"]

    print("=" * 80)
    print("VLM 패턴별 FLOPs 비교 (이미지: 336px, 텍스트: 256 토큰)")
    print("=" * 80)

    for pattern in patterns:
        flops = compute_flops(pattern)
        total_tflops = flops["total"] / 1e12

        print(f"\n{pattern.upper()}")
        print(f"  Vision Encoder: {flops['vision_encoder']/1e9:>10.1f} GFLOPs")
        print(f"  Connector:      {flops['connector']/1e9:>10.1f} GFLOPs")
        print(f"  LLM Attention:  {flops['llm_attention']/1e9:>10.1f} GFLOPs")
        print(f"  LLM FFN:        {flops['llm_ffn']/1e9:>10.1f} GFLOPs")
        print(f"  Total:          {total_tflops:>10.2f} TFLOPs")
```

### 추론 시간 비교

```python
def estimate_inference_time():
    """GPU 기준 추론 시간 추정"""
    # A100 80GB 기준 (312 TFLOPS FP16)
    gpu_tflops = 312

    image_sizes = [224, 336, 672, 1344]

    print(f"{'해상도':<8} {'패턴':<18} {'Visual Tokens':>14} {'LLM 입력 길이':>14} {'예상 시간(ms)':>14}")
    print("-" * 75)

    for img_size in image_sizes:
        patch_size = 14
        num_patches = (img_size // patch_size) ** 2
        text_len = 256

        for pattern, name, token_count in [
            ("cross_attention", "Cross-Attention", 32),
            ("projection", "Projection", num_patches),
            ("early_fusion", "Early Fusion", num_patches),
        ]:
            if pattern == "cross_attention":
                llm_input = text_len  # visual은 cross-attn으로
                effective_len = text_len + token_count
            else:
                llm_input = token_count + text_len
                effective_len = llm_input

            flops = compute_flops(
                pattern,
                image_size=img_size,
                text_len=text_len,
            )
            time_ms = flops["total"] / (gpu_tflops * 1e12) * 1000

            print(f"{img_size}px   {name:<18} {token_count:>14} {effective_len:>14} {time_ms:>14.1f}")

        print()


if __name__ == "__main__":
    compare_all_patterns()
    print()
    estimate_inference_time()
```

---

## 5.4.3 학습 전략 비교

### 학습 파라미터와 데이터 요구량

| 항목 | Cross-Attention | Projection | Early Fusion |
|------|----------------|------------|--------------|
| **학습 파라미터** | Cross-Attn layers + Perceiver (~2-5% of total) | Connector (~0.06-0.3%) | 전체 모델 (100%) |
| **필요 데이터** | 대규모 (수십M 이미지-텍스트 쌍) | 중규모 (수백K~수M) | 초대규모 (수B 토큰) |
| **학습 시간** | 수백 GPU-days | 수십 GPU-hours | 수천 GPU-days |
| **Stage 수** | 1~2 | 2 (align + instruct) | 1 (end-to-end) |

### 학습 안정성

```python
def analyze_training_stability():
    """각 패턴의 학습 안정성 분석"""

    stability_factors = {
        "Cross-Attention (Flamingo)": {
            "Frozen Backbone": "Vision + LLM 모두 frozen → 매우 안정",
            "Gated Residual": "tanh(α) gate로 점진적 주입 → 안정적 수렴",
            "학습 파라미터": "전체의 ~3% → gradient 안정",
            "리스크": "Gate가 0에 수렴할 수 있음 (dead gate)",
        },
        "Projection (LLaVA)": {
            "Stage 1": "Connector만 학습 → 안정적 정렬",
            "Stage 2": "LLM도 학습 → learning rate 주의 필요",
            "학습 파라미터": "Stage 2에서 LLM 전체 → gradient 클 수 있음",
            "리스크": "Stage 2에서 catastrophic forgetting 가능",
        },
        "Early Fusion (Fuyu)": {
            "End-to-End": "전체 모델 학습 → 대규모 데이터 필수",
            "사전학습 없음": "Vision 표현을 처음부터 학습 → 초기 불안정",
            "학습 파라미터": "전체 (100%) → 학습률 스케줄링 중요",
            "리스크": "모달리티 불균형 (텍스트 우세)",
        },
    }

    for pattern, factors in stability_factors.items():
        print(f"\n{'='*60}")
        print(f"  {pattern}")
        print(f"{'='*60}")
        for factor, desc in factors.items():
            print(f"  [{factor}] {desc}")
```

---

## 5.4.4 표현력 분석

### Attention 패턴 비교

```python
def compare_attention_patterns():
    """각 패턴의 attention 구조 차이 시각화"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    m, n = 8, 6  # visual tokens, text tokens
    total = m + n

    # 1. Cross-Attention
    ax = axes[0]
    ax.set_title("Cross-Attention\n(Flamingo, BLIP-2)", fontsize=12)
    # Text self-attention (causal)
    text_mask = np.tril(np.ones((n, n)))
    # Cross-attention (text→visual)
    cross_mask = np.ones((n, m))

    full = np.zeros((total, total))
    full[m:, :m] = cross_mask * 0.6     # cross-attn (blue)
    full[m:, m:] = text_mask * 0.3      # text self-attn (green)

    im = ax.imshow(full, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
    ax.axhline(m - 0.5, color='black', linewidth=2)
    ax.axvline(m - 0.5, color='black', linewidth=2)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    ax.set_xticks([m//2, m + n//2])
    ax.set_xticklabels(["Visual", "Text"])
    ax.set_yticks([m//2, m + n//2])
    ax.set_yticklabels(["Visual", "Text"])

    # 2. Projection (LLM의 Self-Attention)
    ax = axes[1]
    ax.set_title("Projection\n(LLaVA)", fontsize=12)

    full = np.zeros((total, total))
    # Visual tokens: full attention to each other
    full[:m, :m] = np.ones((m, m)) * 0.4
    # Text tokens: attend to all visual + causal text
    full[m:, :m] = np.ones((n, m)) * 0.6
    full[m:, m:] = np.tril(np.ones((n, n))) * 0.3

    ax.imshow(full, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
    ax.axhline(m - 0.5, color='black', linewidth=2)
    ax.axvline(m - 0.5, color='black', linewidth=2)
    ax.set_xlabel("Key")
    ax.set_xticks([m//2, m + n//2])
    ax.set_xticklabels(["Visual", "Text"])
    ax.set_yticks([m//2, m + n//2])
    ax.set_yticklabels(["Visual", "Text"])

    # 3. Early Fusion
    ax = axes[2]
    ax.set_title("Early Fusion\n(Fuyu)", fontsize=12)

    full = np.zeros((total, total))
    # Visual: full bidirectional
    full[:m, :m] = np.ones((m, m)) * 0.5
    # Visual→Text: 0 (visual doesn't see text)
    # Text→Visual: full
    full[m:, :m] = np.ones((n, m)) * 0.7
    # Text: causal
    full[m:, m:] = np.tril(np.ones((n, n))) * 0.4

    ax.imshow(full, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
    ax.axhline(m - 0.5, color='black', linewidth=2)
    ax.axvline(m - 0.5, color='black', linewidth=2)
    ax.set_xlabel("Key")
    ax.set_xticks([m//2, m + n//2])
    ax.set_xticklabels(["Visual", "Text"])
    ax.set_yticks([m//2, m + n//2])
    ax.set_yticklabels(["Visual", "Text"])

    plt.tight_layout()
    plt.savefig("attention_pattern_comparison.png", dpi=150)
    plt.show()
```

### 표현력의 수학적 비교

**Cross-Attention**: 텍스트가 시각 정보를 선택적으로 참조

$$
h_t^{(l)} = \text{SelfAttn}(h_t^{(l-1)}) + \text{CrossAttn}(h_t^{(l-1)}, z_v)
$$

텍스트 레이어 $l$에서 visual feature $z_v$에 대한 접근이 cross-attention을 통해서만 이뤄진다. Visual feature 자체는 텍스트 정보에 의해 업데이트되지 않는다.

**Projection**: Visual token이 LLM의 self-attention에 참여

$$
[h_v; h_t]^{(l)} = \text{SelfAttn}([h_v; h_t]^{(l-1)})
$$

모든 토큰이 같은 self-attention에 참여하지만, visual token은 prefix로 동작하여 causal mask에 의해 text의 영향을 받지 않는다.

**Early Fusion**: 양방향 상호작용

$$
[h_v; h_t]^{(l)} = \text{BiDirectionalAttn}([h_v; h_t]^{(l-1)})
$$

Visual token도 text의 영향을 받을 수 있어 가장 풍부한 표현력을 가진다. 단, causal 모델에서는 이 장점이 제한된다.

---

## 5.4.5 OCR 태스크에 최적인 패턴

### OCR의 특수 요구사항

| 요구사항 | 중요도 | 이유 |
|----------|--------|------|
| 고해상도 처리 | ★★★★★ | 작은 글자 인식에 필수 |
| 공간 정보 보존 | ★★★★★ | 텍스트 위치와 레이아웃 이해 |
| 세밀한 패치-문자 대응 | ★★★★☆ | 각 패치가 어떤 문자인지 정밀 매핑 |
| 추론 속도 | ★★★★☆ | 실시간/배치 처리 요구 |
| 학습 데이터 효율성 | ★★★☆☆ | 도메인별 데이터가 제한적 |

### 패턴별 OCR 적합성

```python
def analyze_ocr_suitability():
    """OCR 태스크에 대한 각 패턴의 적합성 분석"""

    criteria = {
        "고해상도 처리": {
            "Cross-Attention": ("C", "Q-Former가 토큰을 압축하므로 세밀한 정보 손실 우려"),
            "Projection": ("A", "Visual token 수 유지, 고해상도 ViT 활용 가능"),
            "Early Fusion": ("B+", "원본 해상도 처리 가능하나 메모리 비용 높음"),
        },
        "공간 정보 보존": {
            "Cross-Attention": ("C", "32개 query로 압축 → 공간 정보 손실"),
            "Projection": ("A", "패치 토큰 그대로 유지 → 공간 구조 보존"),
            "Early Fusion": ("A", "패치 + newline으로 2D 구조 인코딩"),
        },
        "패치-문자 대응": {
            "Cross-Attention": ("B", "Cross-attention 가중치로 대응 가능"),
            "Projection": ("A", "Self-attention 내 visual-text 대응"),
            "Early Fusion": ("A", "양방향 attention으로 정밀 대응"),
        },
        "추론 속도": {
            "Cross-Attention": ("A", "32 토큰만 LLM에 입력 → 빠름"),
            "Projection": ("B", "576 토큰 → 보통"),
            "Early Fusion": ("C", "수천 토큰 → 느림"),
        },
        "학습 효율성": {
            "Cross-Attention": ("B", "Q-Former 사전학습 필요"),
            "Projection": ("A", "간단한 2-stage, 적은 데이터로 가능"),
            "Early Fusion": ("C", "대규모 데이터 필요, Vision 사전학습 없음"),
        },
    }

    print("=" * 80)
    print("OCR 태스크 적합성 분석")
    print("=" * 80)

    for criterion, scores in criteria.items():
        print(f"\n[{criterion}]")
        for pattern, (grade, reason) in scores.items():
            print(f"  {pattern:<20} [{grade}] {reason}")

    print("\n" + "=" * 80)
    print("종합 평가: OCR에는 Projection 방식(LLaVA 계열)이 가장 적합")
    print("=" * 80)
```

### 최적 패턴: Projection (LLaVA 계열)

OCR에 Projection 방식이 가장 적합한 이유:

1. **공간 정보 보존**: ViT의 각 패치 토큰이 그대로 LLM에 전달되어, 이미지의 어느 위치에 어떤 문자가 있는지 정보가 유지된다.

2. **고해상도 ViT 활용**: CLIP ViT-L/14@336 등 강력한 사전학습 Vision Encoder의 고해상도 feature를 활용할 수 있다.

3. **학습 효율성**: Connector만 학습하면 되므로 적은 OCR 데이터로도 도메인 적응이 가능하다.

4. **균형 잡힌 추론 속도**: 576개 visual token은 Cross-Attention의 32개보다 많지만, Early Fusion의 수천 개보다 훨씬 적다.

5. **구현 단순성**: LLM 구조를 수정할 필요 없이, 입력만 변경하면 되므로 기존 LLM을 쉽게 VLM으로 확장 가능하다.

```python
class OCROptimalArchitecture(torch.nn.Module):
    """
    OCR에 최적화된 VLM 아키텍처.
    Projection 방식 기반으로 고해상도 처리를 강화.

    핵심 설계 결정:
    1. Vision Encoder: 고해상도 ViT (336px 이상)
    2. Connector: MLP (비선형 매핑)
    3. LLM: 7B 이상 (충분한 언어 능력)
    4. 해상도: Dynamic resolution 지원
    """
    def __init__(
        self,
        vision_encoder: torch.nn.Module,
        llm: torch.nn.Module,
        dim_vision: int = 1024,
        dim_llm: int = 4096,
        max_image_size: int = 672,
        patch_size: int = 14,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.llm = llm
        self.patch_size = patch_size
        self.max_image_size = max_image_size

        # MLP Connector (LLaVA-1.5 스타일)
        self.connector = torch.nn.Sequential(
            torch.nn.Linear(dim_vision, dim_llm),
            torch.nn.GELU(),
            torch.nn.Linear(dim_llm, dim_llm),
        )

        # 고해상도를 위한 위치 보간
        max_patches = (max_image_size // patch_size) ** 2
        self.pos_interpolation = torch.nn.Parameter(
            torch.randn(1, max_patches, dim_vision) * 0.02
        )

        # Vision Encoder freeze
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """OCR 추론"""
        # 1. Vision Encoder (고해상도)
        with torch.no_grad():
            visual_features = self.vision_encoder(images)
            if visual_features.size(1) > 1:
                visual_features = visual_features[:, 1:, :]

        # 2. 위치 정보 보강
        num_patches = visual_features.size(1)
        if num_patches <= self.pos_interpolation.size(1):
            pos = self.pos_interpolation[:, :num_patches, :]
        else:
            pos = torch.nn.functional.interpolate(
                self.pos_interpolation.transpose(1, 2),
                size=num_patches,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)
        visual_features = visual_features + pos

        # 3. MLP Projection
        visual_tokens = self.connector(visual_features)

        # 4. LLM 입력 구성
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        if attention_mask is not None:
            visual_mask = torch.ones(
                visual_tokens.size(0), visual_tokens.size(1),
                device=attention_mask.device, dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        # 5. LLM forward
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs
```

---

## 5.4.6 의료 문서 OCR 관점

### 의료 문서의 특수 요구사항

| 요구사항 | 설명 | 기술적 함의 |
|----------|------|-------------|
| **정확도** | 약품명, 용량 한 글자 오류도 치명적 | 고해상도 + 세밀한 패치 인식 |
| **레이아웃 이해** | 처방전, 검사 결과지의 구조적 정보 | 공간 정보 보존 필수 |
| **필기체 인식** | 의사 수기 처방 | 강한 시각적 특징 추출 |
| **규정 준수** | HIPAA, 개인정보보호법 | 온프레미스 배포 가능해야 |
| **다양한 양식** | 병원마다 다른 문서 형식 | 범용적 레이아웃 이해 |
| **다국어** | 영문 약품명 + 한글 진단명 혼재 | 다국어 토크나이저 |

### 최적 아키텍처 선택

```python
def medical_ocr_architecture_decision():
    """의료 문서 OCR을 위한 아키텍처 의사결정 트리"""

    decision_tree = """
    의료 문서 OCR 아키텍처 선택 가이드
    ================================================

    Q1: 실시간 처리가 필요한가?
    ├─ YES → Q2
    └─ NO  → Q3

    Q2: 처리할 문서 종류가 단순한가? (인쇄체 위주)
    ├─ YES → Projection (LLaVA-1.5) + 경량 ViT
    │        - 이유: 빠른 추론, 충분한 정확도
    │        - 예: ViT-B/14 + MLP + LLaMA-7B
    └─ NO  → Projection (LLaVA-1.5) + 고해상도 ViT
             - 이유: 필기체 인식에 더 많은 visual token 필요
             - 예: ViT-L/14@336 + MLP + LLaMA-13B

    Q3: 복잡한 레이아웃 이해가 필요한가? (표, 다단 등)
    ├─ YES → Projection + 동적 해상도
    │        - 이유: 다양한 크기의 문서 처리
    │        - 예: InternVL 스타일 dynamic resolution
    └─ NO  → Projection (기본)
             - 이유: 단순 텍스트 추출에 과도한 복잡도 불필요

    ================================================

    Cross-Attention을 피하는 이유:
    - Q-Former의 토큰 압축(257→32)이 작은 글자 정보를 손실시킬 수 있음
    - 의료 문서에서는 모든 텍스트가 중요 (약품명, 용량 등)
    - 압축 과정에서 숫자/특수문자 정보 손실 위험

    Early Fusion을 피하는 이유:
    - 대규모 의료 OCR 학습 데이터 확보 어려움
    - Vision 사전학습 없이 의료 이미지 이해 학습은 비효율적
    - 고해상도 의료 문서에서 메모리 비용이 과도함
    - 온프레미스 배포 시 GPU 자원 제한

    Projection이 최적인 이유:
    1. CLIP/DINOv2 등 강력한 사전학습 vision encoder 활용
    2. 적은 의료 OCR 데이터로도 fine-tuning 가능
    3. 공간 정보 보존으로 레이아웃 이해 가능
    4. 합리적인 추론 비용으로 온프레미스 배포 가능
    5. LLM의 의료 지식 활용 (약품명 맥락 이해 등)
    """
    print(decision_tree)
```

### 의료 OCR 파이프라인 예시

```python
class MedicalDocumentOCR(torch.nn.Module):
    """
    의료 문서 OCR 특화 VLM.

    Projection 방식 기반으로 다음을 추가:
    1. 고해상도 동적 처리 (처방전 크기 다양)
    2. 후처리: 약품명 교차 검증
    3. 신뢰도 점수 출력
    """
    def __init__(
        self,
        vision_encoder: torch.nn.Module,
        llm: torch.nn.Module,
        dim_vision: int = 1024,
        dim_llm: int = 4096,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.llm = llm

        # MLP Connector
        self.connector = torch.nn.Sequential(
            torch.nn.LayerNorm(dim_vision),
            torch.nn.Linear(dim_vision, dim_llm),
            torch.nn.GELU(),
            torch.nn.Linear(dim_llm, dim_llm),
        )

        # 신뢰도 예측 헤드
        self.confidence_head = torch.nn.Sequential(
            torch.nn.Linear(dim_llm, dim_llm // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_llm // 4, 1),
            torch.nn.Sigmoid(),
        )

        # Freeze vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> dict:
        # Vision encoding (frozen)
        with torch.no_grad():
            visual_features = self.vision_encoder(images)
            if visual_features.size(1) > 1:
                visual_features = visual_features[:, 1:, :]

        # Projection
        visual_tokens = self.connector(visual_features)

        # LLM 입력 구성
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        if attention_mask is not None:
            visual_mask = torch.ones(
                visual_tokens.size(0), visual_tokens.size(1),
                device=attention_mask.device, dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        # LLM forward
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        # 신뢰도 계산
        hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        confidence = self.confidence_head(hidden_states.mean(dim=1))

        return {
            "logits": outputs.logits if hasattr(outputs, 'logits') else outputs[0],
            "confidence": confidence,
        }

    def postprocess_medical_text(
        self,
        decoded_text: str,
        drug_database: list[str],
    ) -> dict:
        """
        의료 텍스트 후처리.
        약품명 검증, 용량 파싱 등.

        Args:
            decoded_text: OCR 결과 텍스트
            drug_database: 약품명 데이터베이스

        Returns:
            dict with validated results
        """
        import re

        result = {
            "raw_text": decoded_text,
            "drugs": [],
            "dosages": [],
            "warnings": [],
        }

        # 용량 패턴 추출 (예: "500mg", "0.5ml")
        dosage_pattern = r'(\d+\.?\d*)\s*(mg|ml|g|mcg|iu|unit)'
        dosages = re.findall(dosage_pattern, decoded_text, re.IGNORECASE)
        result["dosages"] = [
            {"amount": float(d[0]), "unit": d[1].lower()}
            for d in dosages
        ]

        # 약품명 매칭 (편집 거리 기반)
        words = decoded_text.split()
        for word in words:
            word_clean = word.strip(".,;:()")
            if len(word_clean) < 3:
                continue
            best_match = self._find_closest_drug(word_clean, drug_database)
            if best_match:
                result["drugs"].append({
                    "ocr_text": word_clean,
                    "matched_drug": best_match["name"],
                    "confidence": best_match["similarity"],
                })
                if best_match["similarity"] < 0.9:
                    result["warnings"].append(
                        f"'{word_clean}' → '{best_match['name']}' "
                        f"(유사도: {best_match['similarity']:.2f}, 검증 필요)"
                    )

        return result

    def _find_closest_drug(
        self,
        word: str,
        database: list[str],
        threshold: float = 0.7,
    ) -> dict | None:
        """편집 거리 기반 가장 유사한 약품명 찾기"""
        best_sim = 0
        best_name = None

        word_lower = word.lower()
        for drug in database:
            drug_lower = drug.lower()
            # 단순 유사도 (Jaccard)
            set1 = set(word_lower)
            set2 = set(drug_lower)
            sim = len(set1 & set2) / max(len(set1 | set2), 1)

            if sim > best_sim:
                best_sim = sim
                best_name = drug

        if best_sim >= threshold:
            return {"name": best_name, "similarity": best_sim}
        return None
```

---

## 5.4.7 최종 권장 사항

### 태스크별 패턴 선택 가이드

| 태스크 | 권장 패턴 | 이유 |
|--------|----------|------|
| **일반 VQA** | Projection | 범용적, 구현 단순, 충분한 성능 |
| **문서 OCR** | Projection (고해상도) | 공간 정보 보존, 세밀한 인식 |
| **의료 문서 OCR** | Projection + 후처리 | 학습 효율 + 정확도 검증 |
| **Few-shot 이미지 이해** | Cross-Attention | Flamingo 스타일 interleaved 입력 |
| **동영상 이해** | Cross-Attention | 프레임 수 압축 필수 |
| **Multimodal Chat** | Projection | LLaVA/GPT-4V 계열 성능 입증 |
| **원본 해상도 필수** | Early Fusion | 이미지 크기 변경 없이 처리 |

### 실무 체크리스트

```python
def print_deployment_checklist():
    """VLM 배포 전 체크리스트"""
    checklist = """
    ┌─────────────────────────────────────────────────────────┐
    │           VLM 아키텍처 선택 체크리스트                    │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │  [ ] 1. 태스크 정의                                     │
    │      - 입력 이미지 해상도 범위 확인                       │
    │      - 필요한 출력 형태 (텍스트, 좌표, 분류 등)            │
    │      - 정확도 요구 수준                                  │
    │                                                         │
    │  [ ] 2. 리소스 제약                                     │
    │      - GPU 메모리 (Vision tokens 수에 따라 결정)          │
    │      - 추론 시간 제약 (실시간 vs 배치)                    │
    │      - 학습 데이터 양                                    │
    │                                                         │
    │  [ ] 3. 사전학습 모델 선택                               │
    │      - Vision Encoder: CLIP, SigLIP, DINOv2             │
    │      - LLM: LLaMA, Mistral, Qwen                       │
    │      - 라이선스 확인                                     │
    │                                                         │
    │  [ ] 4. Connector 설계                                  │
    │      - Linear vs MLP (기본은 MLP 권장)                   │
    │      - Token 수 조절 필요 여부                           │
    │      - 위치 정보 보존 전략                               │
    │                                                         │
    │  [ ] 5. 학습 전략                                       │
    │      - Stage 1: Feature alignment                       │
    │      - Stage 2: Instruction tuning                      │
    │      - 도메인 특화 데이터 준비                            │
    │                                                         │
    │  [ ] 6. 배포 최적화                                     │
    │      - 양자화 (INT4/INT8)                               │
    │      - KV-Cache 최적화                                  │
    │      - 배치 처리 전략                                    │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
    """
    print(checklist)


if __name__ == "__main__":
    medical_ocr_architecture_decision()
    print_deployment_checklist()
```

---

## 용어 체크리스트

학습 후 아래 질문에 답할 수 있는지 확인하라.

| 체크 | 용어 | 자가 점검 질문 |
|------|------|----------------|
| ☐ | Information Bottleneck | Q-Former의 토큰 압축(257→32)이 OCR에서 왜 문제가 되는가? |
| ☐ | Modality Gap | Vision feature space와 Language feature space의 거리를 줄이는 방법은? |
| ☐ | Compute-Performance Tradeoff | 세 패턴 중 "성능 대비 비용"이 가장 좋은 패턴과 그 이유는? |
| ☐ | 고해상도 처리 | 고해상도 문서 이미지에서 각 패턴의 메모리 사용량 차이를 계산할 수 있는가? |
| ☐ | 공간 정보 보존 | 약품명/용량 인식에 공간 정보가 왜 중요한가? (레이아웃 이해) |
| ☐ | 사전학습 활용 | CLIP ViT를 활용할 때 Projection이 Cross-Attention보다 간단한 이유는? |
| ☐ | 2-Stage Training | Projection 방식의 Stage 1과 Stage 2에서 학습 대상이 다른 이유는? |
| ☐ | 의료 OCR 정확도 | 의료 문서에서 한 글자 오류가 치명적인 이유를 실제 사례로 설명할 수 있는가? |
| ☐ | 배포 제약 | 온프레미스 배포 시 Early Fusion보다 Projection이 유리한 이유는? |
| ☐ | 패턴 선택 기준 | 새로운 태스크가 주어졌을 때 아키텍처 선택 기준을 설명할 수 있는가? |

---

## 챕터 5 정리

이 챕터에서 다룬 세 가지 VLM 아키텍처 패턴의 핵심을 요약한다.

| 패턴 | 핵심 메커니즘 | 강점 | 약점 | 대표 모델 |
|------|-------------|------|------|-----------|
| **Cross-Attention** | 별도 모듈로 시각 정보 주입 | 학습 효율, 토큰 압축 | 정보 손실 가능 | Flamingo, BLIP-2 |
| **Projection** | 선형/MLP 차원 변환 | 구현 단순, 공간 보존 | 토큰 수 많음 | LLaVA, LLaVA-1.5 |
| **Early Fusion** | 입력 단계 통합 | 양방향 상호작용 | 메모리 비용, 데이터 요구 | Fuyu |

**OCR, 특히 의료 문서 OCR에는 Projection 방식이 최적이다.** 공간 정보를 보존하면서, 사전학습된 Vision Encoder의 강력한 feature를 활용하고, 적은 도메인 데이터로도 fine-tuning이 가능하기 때문이다.

---

## 다음 챕터

[Chapter 6: OCR 특화 VLM](../chapter06-ocr-vlm/)에서 OCR에 특화된 VLM 모델들(Donut, Pix2Struct, Nougat 등)을 상세히 다룬다.
