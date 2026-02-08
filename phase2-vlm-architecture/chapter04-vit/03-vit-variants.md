---
---

# 4.3 ViT Variants

ViT 원논문 이후 수많은 변형이 등장했다. 모델 크기를 스케일링한 ViT-B/L/H, 데이터 효율성을 높인 DeiT, 계층적 구조를 도입한 Swin Transformer까지. 각 변형의 설계 철학과 수학적 차이를 이해하는 게 핵심이다.

---

> **핵심 용어 박스**
>
> | 용어 | 정의 |
> |------|------|
> | **ViT-B/L/H** | ViT의 Base/Large/Huge 모델. 레이어 수, 헤드 수, 임베딩 차원이 다름 |
> | **DeiT** | Data-efficient Image Transformer. ImageNet 단독 학습 가능한 ViT |
> | **Swin Transformer** | Shifted Window 기반의 계층적 Vision Transformer |
> | **Window Attention** | 전체 이미지가 아닌 로컬 윈도우 내에서만 Attention을 계산 |
> | **Shifted Window** | 윈도우를 이동시켜 인접 윈도우 간 정보 교환을 가능하게 함 |
> | **Hierarchical Feature Map** | CNN처럼 해상도를 점진적으로 줄이며 채널을 늘리는 구조 |

---

## 4.3.1 ViT 모델 스케일링

### 모델 크기별 설정

ViT 원논문(Dosovitskiy et al., 2020)에서 정의한 세 가지 크기:

| 모델 | 레이어 L | 히든 차원 D | MLP 차원 | 헤드 수 | 파라미터 수 |
|------|:--------:|:-----------:|:--------:|:-------:|:-----------:|
| ViT-B/16 | 12 | 768 | 3072 | 12 | 86M |
| ViT-L/16 | 24 | 1024 | 4096 | 16 | 307M |
| ViT-H/14 | 32 | 1280 | 5120 | 16 | 632M |

`B/16`에서 16은 패치 크기 $P = 16$을 의미한다.

---

## 수학적 원리

### 파라미터 수 계산

Transformer 인코더 한 레이어의 파라미터 수를 계산해보자.

**Multi-Head Self-Attention (MSA):**

$$
\text{MSA params} = 4D^2 + 4D
$$

- $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$: 각 $D^2$ (3개)
- $W_O \in \mathbb{R}^{D \times D}$: $D^2$ (1개)
- 바이어스: $4D$

**MLP (FFN):**

$$
\text{MLP params} = 2 \times D \times 4D + D + 4D = 8D^2 + 5D
$$

- 첫 번째 Linear: $D \times 4D + 4D$
- 두 번째 Linear: $4D \times D + D$

**LayerNorm (2개):**

$$
\text{LN params} = 2 \times 2D = 4D
$$

**레이어 합계:**

$$
\text{params per layer} = 4D^2 + 4D + 8D^2 + 5D + 4D = 12D^2 + 13D
$$

**전체 모델 (Patch Embedding + L layers):**

$$
\text{Total} \approx P^2 \cdot C \cdot D + D + L \times (12D^2 + 13D) + (N+1) \cdot D
$$

검증:

```python
def count_vit_params(
    img_size: int = 224,
    patch_size: int = 16,
    in_channels: int = 3,
    embed_dim: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
    num_classes: int = 1000,
) -> dict:
    """ViT 모델의 파라미터 수를 이론적으로 계산한다."""

    N = (img_size // patch_size) ** 2  # 패치 수
    mlp_dim = int(embed_dim * mlp_ratio)

    # Patch Embedding: Conv2d(C, D, P, P) + bias
    patch_embed = in_channels * patch_size * patch_size * embed_dim + embed_dim

    # CLS Token
    cls_token = embed_dim

    # Position Embedding
    pos_embed = (N + 1) * embed_dim

    # 각 Transformer 레이어
    msa = 4 * embed_dim * embed_dim + 4 * embed_dim  # Q,K,V,O + biases
    mlp = embed_dim * mlp_dim + mlp_dim + mlp_dim * embed_dim + embed_dim  # 2 Linears
    ln = 2 * (2 * embed_dim)  # 2 LayerNorms, 각각 weight + bias
    per_layer = msa + mlp + ln

    # 분류 헤드: LayerNorm + Linear
    head_ln = 2 * embed_dim
    head_linear = embed_dim * num_classes + num_classes

    total = patch_embed + cls_token + pos_embed + num_layers * per_layer + head_ln + head_linear

    return {
        "Patch Embedding": patch_embed,
        "CLS Token": cls_token,
        "Position Embedding": pos_embed,
        "레이어당 파라미터": per_layer,
        "Transformer 전체": num_layers * per_layer,
        "분류 헤드": head_ln + head_linear,
        "총 파라미터": total,
        "총 파라미터 (M)": total / 1e6,
    }


# ViT 변형별 파라미터 비교
configs = {
    "ViT-B/16": {"embed_dim": 768,  "num_layers": 12, "num_heads": 12, "patch_size": 16},
    "ViT-L/16": {"embed_dim": 1024, "num_layers": 24, "num_heads": 16, "patch_size": 16},
    "ViT-H/14": {"embed_dim": 1280, "num_layers": 32, "num_heads": 16, "patch_size": 14},
}

for name, cfg in configs.items():
    result = count_vit_params(**cfg)
    print(f"{name}: {result['총 파라미터 (M)']:.1f}M")
```

### Swin Transformer의 Window Attention

ViT의 Self-Attention은 모든 패치 간 계산: $O(N^2 D)$. 이미지가 커지면 $N$이 급격히 증가해서 비현실적이다.

Swin Transformer는 이미지를 $M \times M$ 크기의 윈도우로 나누고, 각 윈도우 내에서만 Attention을 계산한다.

**표준 Self-Attention:**

$$
\text{Complexity} = O(N^2 \cdot D)
$$

여기서 $N = \frac{H}{P} \times \frac{W}{P}$

**Window Self-Attention (W-MSA):**

$$
\text{Complexity} = O\left(\frac{N}{M^2} \cdot M^4 \cdot D\right) = O(M^2 \cdot N \cdot D)
$$

분석:
- 전체 패치 수 $N$개를 $M^2$개씩 묶은 윈도우 $N/M^2$개로 분할
- 각 윈도우 내 Attention: $O((M^2)^2 \cdot D) = O(M^4 D)$
- 전체: $\frac{N}{M^2} \times M^4 D = M^2 N D$

$M$은 고정값(보통 7)이므로, 전체 복잡도는 $N$에 선형이다. $O(N^2)$에서 $O(N)$으로 극적인 개선.

### Shifted Window

Window Attention의 문제: 윈도우 경계를 넘는 정보 교환 불가.

해결: 연속된 레이어에서 윈도우를 $(M/2, M/2)$만큼 이동(shift):

```
Layer l (Regular Window):      Layer l+1 (Shifted Window):
┌────┬────┬────┬────┐          ┌──┬─────┬─────┬──┐
│ W1 │ W2 │ W3 │ W4 │          │  │     │     │  │
├────┼────┼────┼────┤          ├──┼─────┼─────┼──┤
│ W5 │ W6 │ W7 │ W8 │    →     │  │ SW1 │ SW2 │  │
├────┼────┼────┼────┤          ├──┼─────┼─────┼──┤
│ W9 │W10 │W11 │W12 │          │  │ SW3 │ SW4 │  │
├────┼────┼────┼────┤          ├──┼─────┼─────┼──┤
│W13 │W14 │W15 │W16 │          │  │     │     │  │
└────┴────┴────┴────┘          └──┴─────┴─────┴──┘
```

Shifted Window에서 경계 영역은 cyclic shift + masking으로 효율적으로 처리한다.

### Hierarchical Feature Map

Swin은 CNN처럼 4개 Stage로 구성되며, 각 Stage에서:
- 해상도가 2배 감소 (Patch Merging)
- 채널이 2배 증가

$$
\text{Stage 1}: \frac{H}{4} \times \frac{W}{4},\; C
$$
$$
\text{Stage 2}: \frac{H}{8} \times \frac{W}{8},\; 2C
$$
$$
\text{Stage 3}: \frac{H}{16} \times \frac{W}{16},\; 4C
$$
$$
\text{Stage 4}: \frac{H}{32} \times \frac{W}{32},\; 8C
$$

이 계층 구조 덕분에 FPN (Feature Pyramid Network)과 호환되어, Object Detection, Segmentation 등 dense prediction 태스크에 직접 사용할 수 있다.

---

## 4.3.2 DeiT: 데이터 효율적 ViT

### 핵심 아이디어

ViT 원논문은 JFT-300M (3억 장)으로 사전 학습해야 좋은 성능을 냈다. DeiT(Touvron et al., 2021)는 ImageNet-1K (130만 장)만으로 ViT를 효과적으로 학습하는 방법을 제시했다.

핵심 기법:
1. **Knowledge Distillation**: CNN Teacher(RegNet)의 지식을 ViT Student에 전이
2. **Distillation Token**: CLS Token 외에 별도의 Distillation Token 추가
3. **Strong Data Augmentation**: RandAugment, Mixup, CutMix, Random Erasing

### Distillation Token

```
입력: [CLS, dist, patch_1, patch_2, ..., patch_N]

Transformer Encoder

출력: [z_cls, z_dist, z_1, z_2, ..., z_N]
       ↓       ↓
   CE Loss  Distill Loss
  (Label)   (Teacher)
```

학습 시 두 가지 손실을 동시에 최적화:

$$
\mathcal{L} = (1 - \lambda)\; \mathcal{L}_{CE}(y_{cls},\; y_{true}) + \lambda\; \mathcal{L}_{KD}(y_{dist},\; y_{teacher})
$$

### DeiT 모델 크기

| 모델 | D | L | Heads | 파라미터 | ImageNet Top-1 |
|------|:-:|:-:|:-----:|:--------:|:--------------:|
| DeiT-Ti | 192 | 12 | 3 | 5M | 72.2% |
| DeiT-S | 384 | 12 | 6 | 22M | 79.8% |
| DeiT-B | 768 | 12 | 12 | 86M | 81.8% |

---

## 4.3.3 Swin Transformer 구현

### Window Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class WindowAttention(nn.Module):
    """Window-based Multi-Head Self-Attention.

    Swin Transformer의 핵심. 고정 크기 윈도우 내에서만 Attention 계산.
    """

    def __init__(
        self,
        dim: int,
        window_size: int = 7,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative Position Bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # 상대 위치 인덱스 미리 계산
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, Wh, Ww)
        coords_flat = coords.reshape(2, -1)  # (2, Wh*Ww)

        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative_coords[:, :, 0] += window_size - 1  # 음수 → 양수로 이동
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1  # 2D → 1D 인덱스
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (num_windows*B, N, C) 윈도우별 토큰
            mask: (num_windows, N, N) Attention Mask (Shifted Window용)
        Returns:
            (num_windows*B, N, C)
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B_, heads, N, N)

        # Relative Position Bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_position_bias

        # Shifted Window Masking
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(B_ // num_windows, num_windows, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
```

### Window Partition & Reverse

```python
def window_partition(
    x: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """이미지 특징맵을 윈도우로 분할한다.

    Args:
        x: (B, H, W, C)
        window_size: 윈도우 크기 M
    Returns:
        windows: (num_windows*B, M, M, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """분할된 윈도우를 원래 이미지 형태로 복원한다.

    Args:
        windows: (num_windows*B, M, M, C)
        window_size: 윈도우 크기 M
        H, W: 원본 높이/너비
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x
```

### Swin Transformer Block

```python
class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.

    Regular Window와 Shifted Window를 번갈아 사용한다.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C)
            H, W: 특징맵의 높이/너비
        Returns:
            (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W, f"입력 길이 {L} != H*W={H*W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic Shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # 실제 구현에서는 attention mask도 필요하지만 여기선 생략
            attn_mask = None
        else:
            shifted_x = x
            attn_mask = None

        # Window Partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window Attention
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Window Reverse
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse Cyclic Shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # Residual + MLP
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x
```

### Patch Merging (다운샘플링)

```python
class PatchMerging(nn.Module):
    """Patch Merging Layer.

    인접 2×2 패치를 합쳐서 해상도를 절반으로 줄이고
    채널을 2배로 늘린다. CNN의 stride-2 conv와 유사.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C)
        Returns:
            (B, H/2 * W/2, 2C)
        """
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # 2×2 인접 패치 수집
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)  # (B, H/2*W/2, 4C)

        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2*W/2, 2C)

        return x
```

---

## 4.3.4 모델 비교표

### 아키텍처 비교

| 속성 | ViT-B/16 | ViT-L/16 | Swin-B | DeiT-B |
|------|:--------:|:--------:|:------:|:------:|
| 파라미터 (M) | 86 | 307 | 88 | 86 |
| FLOPs (G) | 17.6 | 61.6 | 15.4 | 17.6 |
| ImageNet Top-1 | 77.9* | 76.5* | 83.5 | 81.8 |
| 사전학습 데이터 | JFT-300M | JFT-300M | ImageNet-1K | ImageNet-1K |
| Feature Pyramid | 불가 | 불가 | 가능 | 불가 |
| Detection 호환 | 어려움 | 어려움 | 직접 가능 | 어려움 |
| Attention 복잡도 | $O(N^2)$ | $O(N^2)$ | $O(N)$ | $O(N^2)$ |
| PE 방식 | 1D Learnable | 1D Learnable | Relative | 1D Learnable |

*ViT-B/L의 ImageNet-1K 단독 학습 성능. JFT 사전학습 후 fine-tune하면 각각 84.2%, 85.2%

### 설계 철학 비교

```python
# 각 모델의 핵심 설계를 코드로 요약

# ViT: 단순하고 범용적
class ViT:
    """
    - 이미지 → 패치 시퀀스 → Transformer 인코더
    - CLS 토큰으로 분류
    - Global Attention (O(N²))
    - 대규모 데이터 필수
    """
    pass

# DeiT: ViT + 학습 효율
class DeiT:
    """
    - ViT 아키텍처 동일
    - + Distillation Token (Teacher 모델에서 지식 전이)
    - + 강력한 Data Augmentation
    - ImageNet-1K 단독 학습 가능
    """
    pass

# Swin: 계층적 + 효율적
class Swin:
    """
    - 계층적 Feature Map (FPN 호환)
    - Window Attention (O(N))
    - Shifted Window (윈도우 간 정보 교환)
    - Relative Position Bias
    - Detection/Segmentation에 강함
    """
    pass
```

### FLOPs vs 정확도 트렌드

```python
import matplotlib.pyplot as plt
import numpy as np


def plot_model_comparison():
    """ViT 변형 모델들의 FLOPs vs Top-1 정확도를 비교한다."""

    models = {
        'DeiT-Ti': {'flops': 1.3,  'acc': 72.2, 'params': 5,   'marker': 'o', 'color': 'blue'},
        'DeiT-S':  {'flops': 4.6,  'acc': 79.8, 'params': 22,  'marker': 'o', 'color': 'blue'},
        'DeiT-B':  {'flops': 17.6, 'acc': 81.8, 'params': 86,  'marker': 'o', 'color': 'blue'},
        'ViT-B/16': {'flops': 17.6, 'acc': 84.2, 'params': 86, 'marker': 's', 'color': 'red'},
        'ViT-L/16': {'flops': 61.6, 'acc': 85.2, 'params': 307, 'marker': 's', 'color': 'red'},
        'Swin-T':   {'flops': 4.5,  'acc': 81.3, 'params': 29,  'marker': '^', 'color': 'green'},
        'Swin-S':   {'flops': 8.7,  'acc': 83.0, 'params': 50,  'marker': '^', 'color': 'green'},
        'Swin-B':   {'flops': 15.4, 'acc': 83.5, 'params': 88,  'marker': '^', 'color': 'green'},
        'Swin-L':   {'flops': 34.5, 'acc': 86.3, 'params': 197, 'marker': '^', 'color': 'green'},
    }

    fig, ax = plt.subplots(figsize=(12, 8))

    for name, info in models.items():
        ax.scatter(
            info['flops'], info['acc'],
            s=info['params'] * 3,  # 파라미터 수에 비례하는 크기
            marker=info['marker'],
            c=info['color'],
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
        )
        ax.annotate(
            name, (info['flops'], info['acc']),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
        )

    ax.set_xlabel("FLOPs (G)", fontsize=14)
    ax.set_ylabel("ImageNet Top-1 Accuracy (%)", fontsize=14)
    ax.set_title("ViT Variants: FLOPs vs Accuracy", fontsize=16)
    ax.legend(
        ['DeiT', 'ViT (JFT pretrain)', 'Swin'],
        fontsize=12, loc='lower right',
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("vit_variants_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 4.3.5 최신 ViT 변형 요약

| 모델 | 연도 | 핵심 기여 |
|------|------|----------|
| ViT | 2020 | 패치 기반 이미지 토큰화 + 표준 Transformer |
| DeiT | 2021 | Distillation Token + 데이터 효율적 학습 |
| Swin | 2021 | Window Attention + Hierarchical Feature Map |
| BEiT | 2021 | BERT 스타일의 이미지 사전학습 (Masked Image Modeling) |
| MAE | 2022 | 높은 마스킹 비율(75%)의 효율적 사전학습 |
| EVA | 2023 | 10억 파라미터 스케일 ViT + CLIP 지식 |
| DINOv2 | 2023 | Self-supervised ViT, 범용 시각 특징 |
| SigLIP | 2023 | Sigmoid Loss 기반 이미지-텍스트 대조학습 |

---

## 용어 체크리스트

학습 후 아래 항목을 설명할 수 있는지 점검하라.

- [ ] **ViT-B/L/H**: 각 모델의 레이어 수, 히든 차원, 파라미터 수를 대략적으로 아는가?
- [ ] **파라미터 공식**: $12D^2 + 13D$가 어떻게 유도되는지 MSA와 MLP를 분리해서 설명할 수 있는가?
- [ ] **DeiT**: Distillation Token의 역할과 학습 방식을 설명할 수 있는가?
- [ ] **Window Attention**: $O(N^2) \rightarrow O(N)$으로 복잡도가 줄어드는 이유를 수식으로 유도할 수 있는가?
- [ ] **Shifted Window**: 왜 윈도우를 이동시키는지, cyclic shift의 원리를 아는가?
- [ ] **Patch Merging**: 2x2 인접 패치를 합쳐 해상도를 줄이는 과정을 코드로 구현할 수 있는가?
- [ ] **Hierarchical Feature Map**: 왜 Swin이 Detection/Segmentation에 유리한지 설명할 수 있는가?
- [ ] **모델 선택 기준**: 태스크와 데이터 규모에 따라 어떤 ViT 변형을 선택해야 하는지 판단 기준이 있는가?
