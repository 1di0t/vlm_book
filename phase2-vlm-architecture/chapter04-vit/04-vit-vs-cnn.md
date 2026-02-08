---
---

# 4.4 ViT vs CNN

ViT와 CNN은 이미지를 처리하는 근본적으로 다른 철학을 가진다. CNN은 "이미지는 로컬 패턴의 조합"이라는 강한 귀납 편향을 내장하고, ViT는 "데이터가 알아서 패턴을 찾게 하자"는 유연한 접근을 택한다. 이 차이가 성능, 데이터 효율성, 표현력에 어떤 영향을 미치는지 분석한다.

---

> **핵심 용어 박스**
>
> | 용어 | 정의 |
> |------|------|
> | **Inductive Bias** | 모델 아키텍처에 내장된 사전 가정. 학습 전부터 존재하는 편향 |
> | **Locality** | 인접한 픽셀이 서로 관련 있다는 가정. CNN 커널의 핵심 |
> | **Translation Equivariance** | 입력을 이동시키면 출력도 동일하게 이동하는 성질. CNN의 weight sharing에서 유래 |
> | **Global Receptive Field** | 한 번의 연산으로 입력 전체를 볼 수 있는 범위. Self-Attention의 특성 |
> | **Inductive Bias-Data Tradeoff** | 강한 편향은 적은 데이터에서 유리하지만, 많은 데이터에서는 표현력을 제한 |

---

## 4.4.1 Inductive Bias 분석

### CNN의 Inductive Bias

CNN에는 두 가지 핵심적인 귀납 편향이 있다:

**1. Locality (지역성)**

커널 크기 $k$에 의해 수용 영역이 제한된다. $3 \times 3$ 커널이면 각 출력 픽셀은 인접 9개 픽셀만 참조한다.

$$
y_{ij} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} W_{mn} \cdot x_{i+m,\; j+n} + b
$$

이미지에서 인접 픽셀은 대부분 관련이 있으므로, 이 가정은 합리적이다. 하지만 장거리 의존성을 잡으려면 레이어를 깊게 쌓아야 한다.

```
CNN의 수용 영역 확장:
Layer 1: 3×3     (지역)
Layer 2: 5×5     (좀 더 넓게)
Layer 3: 7×7     (더 넓게)
  ...
Layer N: 전체 이미지 (매우 깊은 네트워크 필요)
```

**2. Translation Equivariance (이동 등변성)**

같은 커널 가중치를 모든 위치에서 공유하므로, 입력이 이동하면 출력도 동일하게 이동한다:

$$
f(\text{shift}(x)) = \text{shift}(f(x))
$$

고양이가 이미지 왼쪽에 있든 오른쪽에 있든, 같은 필터로 감지한다. 이 성질은 적은 파라미터로 효율적인 학습을 가능하게 한다.

### ViT의 Inductive Bias (부재)

ViT에는 거의 inductive bias가 없다:

| 속성 | CNN | ViT |
|------|-----|-----|
| Locality | 커널 크기로 강제 | 없음 (모든 패치 참조 가능) |
| Translation Equivariance | Weight sharing으로 자동 | 없음 (Position Embedding에 의존) |
| Scale Invariance | Pooling으로 부분적 | 없음 |
| 2D 구조 인식 | 2D Convolution으로 내장 | 패치 분할 시에만 (이후는 1D 시퀀스) |

ViT에서 이미지 구조에 대한 유일한 가정은:
1. 이미지를 패치로 나눈다는 것 (아주 약한 locality)
2. Position Embedding으로 위치 정보를 넣는다는 것

이 최소한의 bias 덕분에 ViT는 더 유연한 표현을 학습할 수 있지만, 그만큼 많은 데이터가 필요하다.

---

## 수학적 원리

### CNN의 수학적 편향

**Convolution 연산 (2D):**

$$
(f * g)(i, j) = \sum_{m} \sum_{n} f(m, n) \cdot g(i-m, j-n)
$$

이 연산은 본질적으로:
- 로컬: $f(m,n)$의 범위가 커널 크기로 제한
- 등변: weight sharing → $g$가 위치에 무관

**유효 수용 영역 (Effective Receptive Field):**

$L$개 레이어, 커널 크기 $k$일 때 이론적 수용 영역:

$$
\text{RF} = 1 + L \times (k - 1)
$$

하지만 실제 유효 수용 영역은 이론값보다 훨씬 작다. 가우시안 분포를 따르며, 중심에 편중된다.

### ViT의 Self-Attention 분석

**Attention의 수용 영역:**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

한 번의 Attention으로 모든 패치를 참조한다. 수용 영역 = 전체 이미지.

$$
\text{RF}_{\text{ViT}} = \text{전체 이미지} \quad (\text{첫 레이어부터})
$$

**Attention 패턴의 레이어별 변화:**

실험적으로 관찰된 패턴:

```
초기 레이어 (Layer 1-4):
- Attention이 로컬에 집중 (인접 패치)
- CNN의 초기 레이어와 유사한 역할
- 엣지, 텍스처 등 저수준 특징 추출

중간 레이어 (Layer 5-8):
- 로컬 + 글로벌 혼합
- 일부 헤드는 로컬, 일부는 글로벌
- 객체 부분 간 관계 학습

후기 레이어 (Layer 9-12):
- 글로벌 Attention 패턴
- 의미적으로 관련된 영역 간 연결
- 장거리 의존성 포착
```

이 패턴은 ViT가 **데이터에서** locality를 학습한다는 것을 보여준다. CNN은 이걸 아키텍처로 강제하고, ViT는 학습으로 발견한다.

### Attention Distance 측정

각 레이어에서 Attention의 평균 거리를 계산한다:

$$
d_{\text{attn}}^l = \frac{1}{H} \sum_{h=1}^{H} \sum_{i} \sum_{j} A_{ij}^{l,h} \cdot \|pos_i - pos_j\|_2
$$

여기서:
- $A_{ij}^{l,h}$: 레이어 $l$, 헤드 $h$에서 토큰 $i$가 $j$에 주는 attention weight
- $\|pos_i - pos_j\|_2$: 패치 $i$와 $j$ 사이의 2D 유클리드 거리

초기 레이어에서 $d_{\text{attn}}$이 작고, 후기 레이어에서 크다면, ViT가 점진적으로 수용 영역을 확장하는 것이다.

### Inductive Bias-Data Tradeoff

핵심 관계:

$$
\text{Test Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
$$

- **강한 Inductive Bias (CNN)**: Bias 큼, Variance 작음 → 적은 데이터에서 안정적
- **약한 Inductive Bias (ViT)**: Bias 작음, Variance 큼 → 많은 데이터에서 유리

이를 데이터 규모별로 보면:

$$
\text{성능}(n) = \begin{cases}
\text{CNN} > \text{ViT} & \text{if } n < n_{\text{threshold}} \\
\text{ViT} \geq \text{CNN} & \text{if } n \geq n_{\text{threshold}}
\end{cases}
$$

$n_{\text{threshold}}$는 대략 수백만 장 수준이다. JFT-300M 같은 초대규모 데이터셋에서 ViT가 CNN을 압도하는 이유다.

---

## 4.4.2 Attention 패턴 분석 코드

### Attention Map 추출

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class ViTWithAttnMaps(nn.Module):
    """Attention Map을 추출할 수 있는 간단한 ViT.

    학습용이 아닌, Attention 패턴 분석을 위한 구현.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # CLS Token & Position Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlockWithAttn(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        B = x.shape[0]

        # Patch Embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # CLS Token Prepend
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Position Embedding
        x = x + self.pos_embed

        # Transformer
        attention_maps = []
        for block in self.blocks:
            x, attn = block(x)
            if return_attention:
                attention_maps.append(attn.detach())

        x = self.norm(x)

        if return_attention:
            return x, attention_maps  # attention_maps: list of (B, heads, N+1, N+1)
        return x


class TransformerBlockWithAttn(nn.Module):
    """Attention Map을 반환하는 Transformer Block."""

    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttentionWithMap(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x):
        attn_out, attn_map = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_map


class MultiHeadAttentionWithMap(nn.Module):
    """Attention Weight를 반환하는 MHA."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out, attn  # attn: (B, heads, N, N)
```

### Attention Distance 계산

```python
def compute_attention_distance(
    attention_maps: list,
    num_patches_h: int = 14,
    num_patches_w: int = 14,
) -> np.ndarray:
    """각 레이어, 각 헤드의 평균 Attention Distance를 계산한다.

    Args:
        attention_maps: list of (B, heads, N+1, N+1) 텐서
        num_patches_h, num_patches_w: 패치 그리드 크기
    Returns:
        distances: (num_layers, num_heads) 배열
    """
    num_layers = len(attention_maps)
    num_heads = attention_maps[0].shape[1]
    N = num_patches_h * num_patches_w

    # 패치 좌표 계산 (CLS 제외)
    coords = np.array([
        (i // num_patches_w, i % num_patches_w)
        for i in range(N)
    ], dtype=np.float32)

    # 패치 간 거리 행렬
    dist_matrix = np.sqrt(
        ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    )  # (N, N)

    distances = np.zeros((num_layers, num_heads))

    for layer_idx, attn in enumerate(attention_maps):
        # CLS 토큰 제외: 패치 → 패치 Attention만 사용
        attn_patch = attn[:, :, 1:, 1:].cpu().numpy()  # (B, heads, N, N)
        attn_mean = attn_patch.mean(axis=0)  # (heads, N, N)

        for head in range(num_heads):
            # 가중 평균 거리
            weighted_dist = (attn_mean[head] * dist_matrix).sum() / attn_mean[head].sum()
            distances[layer_idx, head] = weighted_dist

    return distances


def plot_attention_distance(distances: np.ndarray):
    """레이어별 Attention Distance를 시각화한다.

    초기 레이어는 로컬, 후기 레이어는 글로벌 패턴을 보인다.
    """
    num_layers, num_heads = distances.shape

    fig, ax = plt.subplots(figsize=(12, 6))

    # 각 헤드를 개별 점으로 표시
    for head in range(num_heads):
        ax.plot(
            range(num_layers), distances[:, head],
            'o-', alpha=0.3, markersize=4, color='steelblue',
        )

    # 헤드 평균
    mean_dist = distances.mean(axis=1)
    ax.plot(
        range(num_layers), mean_dist,
        'r-s', linewidth=2, markersize=8, label='헤드 평균',
    )

    ax.set_xlabel("레이어", fontsize=14)
    ax.set_ylabel("평균 Attention Distance (패치 단위)", fontsize=14)
    ax.set_title("레이어별 Attention Distance", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(num_layers))

    plt.tight_layout()
    plt.savefig("attention_distance.png", dpi=150, bbox_inches='tight')
    plt.show()
```

### Attention Map 시각화

```python
def visualize_attention_maps(
    attention_maps: list,
    image: torch.Tensor,
    patch_size: int = 16,
    selected_layers: list = None,
    selected_heads: list = None,
):
    """특정 레이어/헤드의 Attention Map을 이미지 위에 오버레이한다.

    Args:
        attention_maps: list of (B, heads, N+1, N+1)
        image: (3, H, W) 원본 이미지 텐서
        patch_size: 패치 크기
        selected_layers: 시각화할 레이어 인덱스 리스트
        selected_heads: 시각화할 헤드 인덱스 리스트
    """
    if selected_layers is None:
        selected_layers = [0, 3, 7, 11]  # 초기, 중초기, 중후기, 최종
    if selected_heads is None:
        selected_heads = [0]

    img_np = image.permute(1, 2, 0).numpy()
    H, W = img_np.shape[:2]
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    fig, axes = plt.subplots(
        len(selected_heads), len(selected_layers),
        figsize=(4 * len(selected_layers), 4 * len(selected_heads)),
    )

    if len(selected_heads) == 1:
        axes = axes[np.newaxis, :]
    if len(selected_layers) == 1:
        axes = axes[:, np.newaxis]

    for row, head in enumerate(selected_heads):
        for col, layer in enumerate(selected_layers):
            ax = axes[row, col]

            # CLS → 패치 Attention (CLS가 각 패치에 주는 가중치)
            attn = attention_maps[layer][0, head, 0, 1:]  # (N,)
            attn = attn.reshape(num_patches_h, num_patches_w).cpu().numpy()

            # 원본 이미지 해상도로 업스케일
            attn_upscale = np.kron(attn, np.ones((patch_size, patch_size)))

            ax.imshow(img_np, alpha=0.5)
            ax.imshow(attn_upscale, cmap='hot', alpha=0.5, interpolation='bilinear')
            ax.set_title(f"Layer {layer}, Head {head}", fontsize=11)
            ax.axis('off')

    plt.suptitle("CLS Token의 Attention Map (레이어별 변화)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig("attention_maps_overlay.png", dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 4.4.3 데이터 규모별 성능 비교

### 실험 시뮬레이션

ViT 원논문(Figure 4)에서 보고된 트렌드를 재현하는 코드.

```python
def plot_data_scaling_comparison():
    """데이터 규모에 따른 CNN vs ViT 성능 곡선.

    ViT 원논문의 핵심 발견을 시각화한다:
    - 적은 데이터: CNN > ViT
    - 대규모 데이터: ViT > CNN
    """

    # 데이터 포인트 (근사값, 원논문 기반)
    data_sizes = [1, 10, 30, 100, 300]  # 백만 장 단위
    data_labels = ['1M', '10M', '30M', '100M', '300M']

    # ImageNet Top-1 Accuracy (근사)
    resnet_152 = [77.0, 79.5, 80.5, 81.0, 81.3]   # 포화 경향
    vit_b16    = [73.0, 78.0, 80.0, 82.0, 84.2]    # 지속 상승
    vit_l16    = [71.0, 77.0, 80.5, 83.5, 85.2]    # 더 급격한 상승
    vit_h14    = [68.0, 75.0, 80.0, 84.0, 86.5]    # 최대 상한

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(data_sizes, resnet_152, 'b-o', linewidth=2, markersize=8, label='ResNet-152')
    ax.plot(data_sizes, vit_b16, 'r-s', linewidth=2, markersize=8, label='ViT-B/16')
    ax.plot(data_sizes, vit_l16, 'g-^', linewidth=2, markersize=8, label='ViT-L/16')
    ax.plot(data_sizes, vit_h14, 'm-D', linewidth=2, markersize=8, label='ViT-H/14')

    # 교차점 표시
    ax.axvline(x=30, color='gray', linestyle='--', alpha=0.5)
    ax.annotate(
        '≈ 교차점\n(~30M 이미지)',
        xy=(30, 80.3), xytext=(60, 76),
        fontsize=11,
        arrowprops=dict(arrowstyle='->', color='gray'),
    )

    # 영역 표시
    ax.axvspan(0.5, 30, alpha=0.05, color='blue', label='CNN 유리 영역')
    ax.axvspan(30, 350, alpha=0.05, color='red', label='ViT 유리 영역')

    ax.set_xscale('log')
    ax.set_xlabel("사전학습 데이터 규모 (백만 장)", fontsize=14)
    ax.set_ylabel("ImageNet Top-1 Accuracy (%)", fontsize=14)
    ax.set_title("데이터 규모에 따른 CNN vs ViT 성능 변화", fontsize=16)
    ax.set_xticks(data_sizes)
    ax.set_xticklabels(data_labels)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(65, 88)

    plt.tight_layout()
    plt.savefig("data_scaling_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
```

### 비교 요약 테이블

```python
def print_comparison_table():
    """CNN vs ViT 종합 비교 테이블을 출력한다."""

    comparison = [
        ("Inductive Bias", "강함 (locality + equivariance)", "약함 (패치 분할 정도)"),
        ("수용 영역", "점진적 확장 (레이어 깊이에 비례)", "첫 레이어부터 글로벌"),
        ("데이터 효율성", "높음 (적은 데이터에서도 학습)", "낮음 (대규모 데이터 필요)"),
        ("표현력 상한", "제한적 (locality 가정에 의해)", "높음 (데이터 의존적)"),
        ("파라미터 효율성", "높음 (weight sharing)", "상대적으로 낮음"),
        ("연산 복잡도", "O(K²·C²·HW)", "O(N²·D) + O(N·D²)"),
        ("Feature Map", "계층적 (자연스럽게 다중 스케일)", "단일 스케일 (Swin 제외)"),
        ("Detection/Seg", "직접 사용 가능 (FPN)", "CLS만 (Swin 등 변형 필요)"),
        ("전이 학습", "ImageNet 사전학습이면 충분", "JFT-300M 급 또는 DeiT 기법 필요"),
        ("해상도 유연성", "임의 해상도 입력 가능", "PE 보간 필요"),
    ]

    print(f"{'속성':<20} | {'CNN':<35} | {'ViT':<35}")
    print("-" * 95)
    for attr, cnn, vit in comparison:
        print(f"{attr:<20} | {cnn:<35} | {vit:<35}")
```

---

## 4.4.4 하이브리드 접근: CNN + ViT

순수 CNN과 순수 ViT의 장점을 결합하는 하이브리드 모델도 활발히 연구되고 있다.

### 하이브리드 아키텍처 패턴

```
패턴 1: CNN → ViT (ViT 원논문의 Hybrid)
┌─────────┐    ┌──────────────────┐
│ ResNet   │ →  │ Transformer      │
│ (특징    │    │ Encoder          │
│  추출)   │    │ (글로벌 관계)     │
└─────────┘    └──────────────────┘

패턴 2: ViT + Conv (LeViT, CoAtNet)
┌──────────┐   ┌──────────┐   ┌──────────┐
│ Conv     │ → │ Attention│ → │ Conv     │
│ Stage    │   │ Stage    │   │ + Attn   │
└──────────┘   └──────────┘   └──────────┘

패턴 3: Conv 내장 Attention (CvT, CMT)
┌──────────────────────────────────┐
│ Convolutional Token Embedding    │
│ + Convolutional Projection       │
│ + Transformer Block              │
└──────────────────────────────────┘
```

### 하이브리드 구현 예시

```python
class HybridViT(nn.Module):
    """CNN 백본 + Transformer 인코더 하이브리드 모델.

    CNN으로 저수준 특징을 추출한 뒤,
    Transformer로 글로벌 관계를 모델링한다.
    """

    def __init__(
        self,
        cnn_out_channels: int = 256,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 8,
        num_classes: int = 1000,
    ):
        super().__init__()

        # CNN 백본 (간단한 3-stage)
        self.cnn = nn.Sequential(
            # Stage 1: 224 → 112
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Stage 2: 56 → 28
            self._make_conv_block(64, 128, stride=2),

            # Stage 3: 28 → 14
            self._make_conv_block(128, cnn_out_channels, stride=2),
        )

        # CNN 출력 → Transformer 입력 투영
        self.proj = nn.Conv2d(cnn_out_channels, embed_dim, kernel_size=1)

        # CLS Token & Position Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 14×14 = 196 패치 (224 입력, CNN 출력 14×14)
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))

        # Transformer 레이어
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        # 분류 헤드
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _make_conv_block(in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # CNN 특징 추출
        x = self.cnn(x)              # (B, cnn_out, 14, 14)
        x = self.proj(x)             # (B, D, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (B, 196, D)

        # CLS Token Prepend
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)   # (B, 197, D)

        # Position Embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # 분류
        x = self.norm(x[:, 0])  # CLS 토큰만
        x = self.head(x)

        return x


# 테스트
if __name__ == "__main__":
    model = HybridViT(
        cnn_out_channels=256,
        embed_dim=768,
        num_heads=12,
        num_layers=8,
        num_classes=1000,
    )
    img = torch.randn(2, 3, 224, 224)
    out = model(img)
    print(f"입력: {img.shape}")     # (2, 3, 224, 224)
    print(f"출력: {out.shape}")     # (2, 1000)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터: {total_params / 1e6:.1f}M")
```

---

## 4.4.5 OCR에서의 ViT vs CNN

OCR 태스크에서의 특수성을 고려한 비교.

| 속성 | CNN 기반 OCR | ViT 기반 OCR |
|------|:-----------:|:------------:|
| 글자 크기 변화 대응 | 멀티스케일 FPN | Attention으로 자동 |
| 긴 텍스트 줄 처리 | 슬라이딩 윈도우 필요 | 패치 시퀀스로 자연스러움 |
| 레이아웃 이해 | 별도 모듈 필요 | 글로벌 Attention으로 내장 |
| 소규모 학습 데이터 | 유리 (inductive bias) | 불리 (사전학습 필수) |
| 멀티모달 확장 | 별도 인코더 필요 | VLM으로 자연스럽게 확장 |

현대 OCR 모델(Donut, Pix2Struct, Nougat, GOT-OCR)은 대부분 ViT 기반이다. 대규모 사전학습으로 inductive bias 부재 문제를 해결하고, Transformer의 유연성을 활용한다.

```python
# OCR 관점에서의 모델 선택 가이드
ocr_model_guide = {
    "소규모 데이터 (< 10K)": {
        "권장": "CNN (CRNN, SVTR-Tiny)",
        "이유": "적은 데이터에서 안정적",
    },
    "중규모 데이터 (10K ~ 100K)": {
        "권장": "Hybrid (CNN+Transformer) 또는 DeiT 기반",
        "이유": "CNN 특징 + Transformer 유연성",
    },
    "대규모 데이터 (> 100K)": {
        "권장": "ViT 기반 VLM (사전학습 활용)",
        "이유": "표현력 상한이 높고 멀티모달 확장 용이",
    },
    "사전학습 모델 Fine-tuning": {
        "권장": "ViT 기반 (Donut, Pix2Struct 등)",
        "이유": "이미 대규모 데이터로 학습됨, 소규모 데이터로도 fine-tune 가능",
    },
}

for scenario, info in ocr_model_guide.items():
    print(f"\n[{scenario}]")
    print(f"  권장: {info['권장']}")
    print(f"  이유: {info['이유']}")
```

---

## 4.4.6 실전 벤치마크 요약

### ImageNet-1K 분류 성능 (224×224, Top-1)

| 모델 | 파라미터 | FLOPs | Top-1 (%) | 사전학습 |
|------|:--------:|:-----:|:---------:|:--------:|
| ResNet-50 | 25M | 4.1G | 76.2 | ImageNet-1K |
| ResNet-152 | 60M | 11.6G | 78.3 | ImageNet-1K |
| EfficientNet-B7 | 66M | 37G | 84.3 | ImageNet-1K |
| ViT-B/16 | 86M | 17.6G | 77.9 | ImageNet-1K |
| ViT-B/16 | 86M | 17.6G | 84.2 | JFT-300M |
| DeiT-B | 86M | 17.6G | 81.8 | ImageNet-1K |
| Swin-B | 88M | 15.4G | 83.5 | ImageNet-1K |
| ConvNeXt-B | 89M | 15.4G | 83.8 | ImageNet-1K |

핵심 관찰:
1. 같은 데이터(ImageNet-1K)에서 ViT-B < DeiT-B < Swin-B ≈ ConvNeXt-B
2. 대규모 사전학습(JFT) 시 ViT-B가 모든 CNN을 추월
3. ConvNeXt는 "ViT의 학습 기법을 적용한 순수 CNN"으로, CNN도 충분히 강력할 수 있음을 보여줌

---

## 용어 체크리스트

학습 후 아래 항목을 설명할 수 있는지 점검하라.

- [ ] **Inductive Bias**: CNN의 locality와 translation equivariance가 무엇이고, 왜 적은 데이터에서 유리한지 설명할 수 있는가?
- [ ] **ViT의 약한 편향**: ViT에 내장된 유일한 inductive bias(패치 분할, PE)가 무엇인지 아는가?
- [ ] **Bias-Variance Tradeoff**: 강한 편향이 적은 데이터에서 왜 유리하고, 많은 데이터에서 왜 불리한지 수식으로 설명할 수 있는가?
- [ ] **Attention Distance**: 초기 레이어는 로컬, 후기 레이어는 글로벌 패턴을 보이는 이유를 이해하는가?
- [ ] **데이터 교차점**: CNN과 ViT의 성능이 역전되는 데이터 규모(~수천만 장)를 알고 있는가?
- [ ] **하이브리드 모델**: CNN 백본 + Transformer 인코더 하이브리드의 장점과 대표 모델을 아는가?
- [ ] **Global Receptive Field**: Self-Attention이 첫 레이어부터 전체 이미지를 볼 수 있는 이유를 수식으로 설명할 수 있는가?
- [ ] **OCR 관점**: OCR에서 ViT 기반 모델이 주류가 된 이유와, 소규모 데이터에서의 전략을 설명할 수 있는가?
- [ ] **ConvNeXt**: "모던 CNN"이 ViT에 근접한 성능을 내는 이유와 시사점을 이해하는가?
