---
---

# 4.1 Patch Embedding

이미지를 Transformer에 넣으려면 먼저 "토큰"으로 바꿔야 한다. NLP에서 문장을 토큰으로 분할하듯, ViT는 이미지를 고정 크기 패치로 분할한 뒤 벡터로 변환한다. 이 과정이 Patch Embedding이다.

---

> **핵심 용어 박스**
>
> | 용어 | 정의 |
> |------|------|
> | **Patch** | 이미지를 격자 형태로 나눈 정사각형 조각. 보통 $P \times P$ 크기 |
> | **Embedding** | 고차원 입력을 고정 차원 벡터 공간으로 매핑하는 변환 |
> | **Linear Projection** | 패치를 1차원으로 펼친 뒤 행렬 곱으로 임베딩 차원에 사영 |
> | **Flatten** | 2D 패치 $(P, P, C)$를 1D 벡터 $(P^2 \cdot C)$로 재배열 |

---

## 4.1.1 왜 패치인가?

### Transformer의 입력 제약

Transformer는 시퀀스를 처리한다. 텍스트는 자연스럽게 토큰 시퀀스지만, 이미지는 2D 그리드다. 선택지는 두 가지:

1. **픽셀 단위**: $224 \times 224 = 50,176$개 토큰 → Self-Attention의 $O(N^2)$ 복잡도로 현실적이지 않다
2. **패치 단위**: $16 \times 16$ 패치면 $14 \times 14 = 196$개 토큰 → 연산량이 합리적

```
이미지 (224 × 224 × 3)
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │    16×16 패치
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤    → 14×14 = 196개
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
        ↓ Flatten + Linear Projection
[z_1, z_2, z_3, ..., z_196]  각 z_i ∈ R^D
```

### 패치 크기 P의 트레이드오프

| 패치 크기 P | 패치 수 N (224×224) | 시퀀스 길이 | Attention 메모리 | 해상도 |
|:-----------:|:-------------------:|:-----------:|:----------------:|:------:|
| 32 | 49 | 짧음 | $O(49^2)$ | 낮음 |
| 16 | 196 | 보통 | $O(196^2)$ | 보통 |
| 14 | 256 | 김 | $O(256^2)$ | 높음 |
| 8 | 784 | 매우 김 | $O(784^2)$ | 매우 높음 |

P가 작을수록 세밀한 정보를 보존하지만, 시퀀스 길이 $N$이 급격히 증가한다. Self-Attention은 $O(N^2)$ 복잡도이므로 메모리와 연산량이 폭증한다.

---

## 수학적 원리

### 이미지 분할

이미지 $x \in \mathbb{R}^{H \times W \times C}$를 크기 $P \times P$의 패치로 분할한다:

$$
x \in \mathbb{R}^{H \times W \times C} \rightarrow x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}
$$

여기서 패치의 총 개수는:

$$
N = \frac{H \times W}{P^2}
$$

예를 들어 $H = W = 224$, $P = 16$이면:

$$
N = \frac{224 \times 224}{16^2} = \frac{50176}{256} = 196
$$

각 패치의 차원은 $P^2 \cdot C = 16^2 \times 3 = 768$이다.

### 선형 투영 (Linear Projection)

각 패치를 Transformer의 임베딩 차원 $D$로 매핑한다. 투영 행렬 $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$를 사용:

$$
z_0 = [x_{\text{class}};\; x_p^1 E;\; x_p^2 E;\; \ldots;\; x_p^N E] + E_{pos}
$$

여기서:
- $x_{\text{class}} \in \mathbb{R}^D$: CLS 토큰 (4.2에서 상세 설명)
- $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$: 학습 가능한 투영 행렬
- $E_{pos} \in \mathbb{R}^{(N+1) \times D}$: 위치 임베딩
- $z_0 \in \mathbb{R}^{(N+1) \times D}$: Transformer 인코더의 입력

이 선형 투영은 사실상 `nn.Linear(P²·C, D)` 연산과 동일하다.

### Conv2d를 이용한 등가 구현

패치 분할 + 선형 투영을 하나의 Conv2d로 대체할 수 있다:

$$
\text{Conv2d}(C_{\text{in}}=C,\; C_{\text{out}}=D,\; \text{kernel\_size}=P,\; \text{stride}=P)
$$

이 Convolution은 stride = kernel_size이므로 겹치지 않는(non-overlapping) 패치를 추출하면서 동시에 선형 투영을 수행한다. 수학적으로:

$$
z_i = W_{\text{conv}} * x_p^i + b = \text{vec}(x_p^i)^T E + b
$$

Conv2d 방식이 unfold + Linear보다 메모리 효율이 좋고, cuDNN 최적화 혜택을 받는다.

### CNN과의 비교

| 속성 | CNN (Conv Layer) | ViT (Patch Embedding) |
|------|------------------|----------------------|
| 수용 영역 | 로컬 (커널 크기 $k$) | 글로벌 (전체 시퀀스) |
| 특징 추출 | 계층적 (여러 레이어 쌓기) | 패치 단위 → Attention |
| 파라미터 공유 | 공간적 weight sharing | 위치별 독립적 attention |
| Inductive Bias | 강함 (locality, translation equivariance) | 약함 (데이터에서 학습) |

---

## 4.1.2 구현

### 방법 1: Conv2d 기반 (표준)

실전에서 가장 많이 쓰는 방식이다. Conv2d의 kernel_size와 stride를 패치 크기 P로 설정하면 패치 분할과 선형 투영을 한 번에 수행한다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbeddingConv(nn.Module):
    """Conv2d 기반 Patch Embedding.

    패치 분할 + 선형 투영을 단일 Conv2d로 처리한다.
    메모리 효율이 좋고 cuDNN 최적화 혜택을 받는다.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # stride = kernel_size = patch_size → non-overlapping 패치
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 이미지 텐서
        Returns:
            (B, N, D) 패치 임베딩 시퀀스
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, (
            f"입력 크기 ({H}×{W})가 기대값 ({self.img_size}×{self.img_size})과 다르다"
        )

        # (B, C, H, W) → (B, D, H/P, W/P)
        x = self.proj(x)

        # (B, D, H/P, W/P) → (B, D, N) → (B, N, D)
        x = x.flatten(2).transpose(1, 2)

        return x


# 테스트
if __name__ == "__main__":
    model = PatchEmbeddingConv(img_size=224, patch_size=16, embed_dim=768)
    img = torch.randn(2, 3, 224, 224)
    out = model(img)
    print(f"입력: {img.shape}")       # (2, 3, 224, 224)
    print(f"출력: {out.shape}")       # (2, 196, 768)
    print(f"패치 수: {model.num_patches}")  # 196
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
```

### 방법 2: Unfold 기반 (교육용)

패치 분할(unfold)과 선형 투영(Linear)을 명시적으로 분리한 구현. 내부 동작을 이해하기 좋다.

```python
class PatchEmbeddingUnfold(nn.Module):
    """Unfold + Linear 기반 Patch Embedding.

    패치 분할과 선형 투영 단계를 명시적으로 분리하여
    내부 동작의 이해를 돕는다.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        # 선형 투영: (P²·C) → D
        self.linear = nn.Linear(patch_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, N, D)
        """
        B, C, H, W = x.shape
        P = self.patch_size

        # Step 1: 패치 분할 — einops로 명확하게
        # (B, C, H, W) → (B, N, P²·C)
        x = rearrange(
            x,
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=P, p2=P
        )
        # 이 시점에서 x.shape = (B, N, P²·C) = (B, 196, 768)

        # Step 2: 선형 투영
        # (B, N, P²·C) → (B, N, D)
        x = self.linear(x)

        return x


# Conv2d와 Unfold+Linear의 등가성 검증
if __name__ == "__main__":
    torch.manual_seed(42)

    conv_embed = PatchEmbeddingConv(img_size=224, patch_size=16, embed_dim=768)
    unfold_embed = PatchEmbeddingUnfold(img_size=224, patch_size=16, embed_dim=768)

    # Conv2d 가중치를 Unfold+Linear로 복사
    with torch.no_grad():
        # conv weight: (D, C, P, P) → (D, P²·C) → (P²·C, D)로 변환
        w = conv_embed.proj.weight.data  # (768, 3, 16, 16)
        w = w.reshape(768, -1).T         # (768, 768) → (768, 768)
        unfold_embed.linear.weight.data = w.T
        unfold_embed.linear.bias.data = conv_embed.proj.bias.data

    img = torch.randn(1, 3, 224, 224)
    out_conv = conv_embed(img)
    out_unfold = unfold_embed(img)

    diff = (out_conv - out_unfold).abs().max().item()
    print(f"Conv2d vs Unfold+Linear 최대 차이: {diff:.2e}")
    # → 부동소수점 오차 수준 (≈ 1e-6)
```

### 방법 3: einops 기반 (간결)

```python
from einops.layers.torch import Rearrange


class PatchEmbeddingEinops(nn.Module):
    """einops를 활용한 간결한 Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.embedding = nn.Sequential(
            Rearrange(
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1=patch_size, p2=patch_size
            ),
            nn.LayerNorm(patch_dim),  # 일부 구현에서 사전 정규화 추가
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)
```

---

## 4.1.3 시각화

패치 분할 과정을 시각적으로 확인하는 코드다.

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def visualize_patches(
    image_path: str,
    patch_size: int = 16,
    img_size: int = 224,
):
    """이미지의 패치 분할 과정을 시각화한다."""

    # 이미지 로드 및 전처리
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img)  # (3, 224, 224)

    num_patches_per_side = img_size // patch_size
    N = num_patches_per_side ** 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1) 원본 이미지 + 그리드
    axes[0].imshow(img_tensor.permute(1, 2, 0).numpy())
    axes[0].set_title(f"원본 + {patch_size}×{patch_size} 그리드", fontsize=14)
    for i in range(num_patches_per_side + 1):
        axes[0].axhline(y=i * patch_size, color='red', linewidth=0.5)
        axes[0].axvline(x=i * patch_size, color='red', linewidth=0.5)
    axes[0].axis('off')

    # 2) 개별 패치 나열
    patches = img_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # patches: (3, num_h, num_w, P, P)
    patches = patches.permute(1, 2, 0, 3, 4)  # (num_h, num_w, 3, P, P)

    # 처음 16개 패치 표시
    grid_rows, grid_cols = 4, 4
    patch_grid = np.zeros((grid_rows * patch_size, grid_cols * patch_size, 3))
    for r in range(grid_rows):
        for c in range(grid_cols):
            patch = patches[r, c].permute(1, 2, 0).numpy()
            y_start = r * patch_size
            x_start = c * patch_size
            patch_grid[y_start:y_start+patch_size, x_start:x_start+patch_size] = patch

    axes[1].imshow(patch_grid)
    axes[1].set_title(f"처음 16개 패치 (총 {N}개)", fontsize=14)
    for i in range(grid_rows + 1):
        axes[1].axhline(y=i * patch_size, color='white', linewidth=1)
        axes[1].axvline(x=i * patch_size, color='white', linewidth=1)
    axes[1].axis('off')

    # 3) 패치 → 벡터 변환 시각화
    # 각 패치를 flatten한 벡터의 L2 노름으로 히트맵 생성
    flat_patches = patches.reshape(num_patches_per_side, num_patches_per_side, -1)
    norms = flat_patches.norm(dim=-1).numpy()

    im = axes[2].imshow(norms, cmap='viridis', interpolation='nearest')
    axes[2].set_title("패치 벡터 L2 노름 히트맵", fontsize=14)
    axes[2].set_xlabel("패치 열 인덱스")
    axes[2].set_ylabel("패치 행 인덱스")
    plt.colorbar(im, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.savefig("patch_embedding_viz.png", dpi=150, bbox_inches='tight')
    plt.show()

    print(f"이미지 크기: {img_size}×{img_size}")
    print(f"패치 크기: {patch_size}×{patch_size}")
    print(f"패치 수: {N} ({num_patches_per_side}×{num_patches_per_side})")
    print(f"패치 벡터 차원: {patch_size * patch_size * 3}")
```

---

## 4.1.4 Embedding 차원의 영향

임베딩 차원 $D$가 커질수록 모델의 표현력이 증가하지만, 파라미터도 비례해서 증가한다.

```python
def analyze_patch_embedding_params(
    img_size: int = 224,
    patch_sizes: list = [8, 14, 16, 32],
    embed_dims: list = [384, 768, 1024],
    in_channels: int = 3,
):
    """패치 크기와 임베딩 차원에 따른 파라미터 수를 분석한다."""

    print(f"{'P':>4} | {'D':>6} | {'N':>6} | {'Proj 파라미터':>14} | {'총 입력 차원':>12}")
    print("-" * 60)

    for P in patch_sizes:
        for D in embed_dims:
            N = (img_size // P) ** 2
            patch_dim = in_channels * P * P
            proj_params = patch_dim * D + D  # weight + bias

            print(f"{P:>4} | {D:>6} | {N:>6} | {proj_params:>14,} | {patch_dim:>12}")


# 실행 결과:
#    P |      D |      N |    Proj 파라미터 |     총 입력 차원
# ------------------------------------------------------------
#    8 |    384 |    784 |         73,728 |          192
#    8 |    768 |    784 |        147,456 |          192
#   16 |    384 |    196 |        295,296 |          768
#   16 |    768 |    196 |        590,592 |          768
#   32 |    768 |     49 |      2,360,064 |        3,072
#   32 |   1024 |     49 |      3,146,752 |        3,072
```

---

## 4.1.5 Overlapping Patch Embedding

일부 변형(PVT, CvT 등)에서는 stride < kernel_size로 패치를 겹쳐서 추출한다. 경계 정보 손실을 줄이는 효과가 있다.

```python
class OverlappingPatchEmbed(nn.Module):
    """Overlapping Patch Embedding.

    stride < kernel_size로 패치 경계의 정보 손실을 완화한다.
    PVT, CvT 등의 모델에서 사용된다.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 7,
        stride: int = 4,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # 출력 크기 계산
        self.num_patches_h = (img_size + 2 * (patch_size // 2) - patch_size) // stride + 1
        self.num_patches_w = self.num_patches_h
        self.num_patches = self.num_patches_h * self.num_patches_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                   # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)   # (B, N, D)
        x = self.norm(x)
        return x
```

---

## 4.1.6 실전 팁

### 패치 크기 선택 가이드

```
데이터셋 크기       권장 패치 크기
─────────────────────────────
< 10K 이미지        P = 32 (시퀀스 짧게, 오버피팅 방지)
10K ~ 100K          P = 16 (표준)
100K ~ 1M           P = 16 or 14
> 1M (ImageNet+)    P = 14 or 8 (세밀한 특징 포착)
```

### 메모리 사용량 추정

Self-Attention의 메모리는 시퀀스 길이에 제곱 비례한다:

$$
\text{Memory} \propto N^2 \cdot D = \left(\frac{HW}{P^2}\right)^2 \cdot D
$$

P를 반으로 줄이면 $N$이 4배, 메모리가 16배 증가한다. 고해상도 이미지를 다룰 때 반드시 고려해야 한다.

```python
def estimate_attention_memory(
    img_size: int = 224,
    patch_size: int = 16,
    embed_dim: int = 768,
    num_heads: int = 12,
    num_layers: int = 12,
    batch_size: int = 32,
    dtype_bytes: int = 4,  # float32 = 4 bytes
) -> dict:
    """Self-Attention의 메모리 사용량을 추정한다."""

    N = (img_size // patch_size) ** 2 + 1  # +1 for CLS token

    # Attention score matrix: (B, heads, N, N) per layer
    attn_mem_per_layer = batch_size * num_heads * N * N * dtype_bytes
    total_attn_mem = attn_mem_per_layer * num_layers

    # 임베딩 메모리: (B, N, D) per layer
    embed_mem_per_layer = batch_size * N * embed_dim * dtype_bytes

    return {
        "시퀀스 길이 N": N,
        "레이어당 Attention 메모리 (MB)": attn_mem_per_layer / (1024 ** 2),
        "전체 Attention 메모리 (MB)": total_attn_mem / (1024 ** 2),
        "레이어당 Embedding 메모리 (MB)": embed_mem_per_layer / (1024 ** 2),
    }


# 다양한 설정 비교
for P in [32, 16, 8]:
    result = estimate_attention_memory(patch_size=P)
    print(f"\nP={P}: N={result['시퀀스 길이 N']}")
    print(f"  Attention 메모리: {result['전체 Attention 메모리 (MB)']:.1f} MB")
```

---

## 용어 체크리스트

학습 후 아래 항목을 설명할 수 있는지 점검하라.

- [ ] **Patch**: 이미지를 $P \times P$ 크기로 분할한 조각이 뭔지 설명할 수 있는가?
- [ ] **Flatten**: 2D 패치를 1D 벡터로 변환하는 과정을 이해하는가?
- [ ] **Linear Projection**: 패치 벡터 $(P^2 \cdot C)$를 임베딩 차원 $D$로 변환하는 행렬 곱을 수식으로 쓸 수 있는가?
- [ ] **Conv2d 등가성**: stride = kernel_size인 Conv2d가 패치 분할 + 선형 투영과 동일한 이유를 설명할 수 있는가?
- [ ] **패치 수 N**: $N = HW / P^2$ 공식을 유도하고, P가 반으로 줄면 N이 4배 증가하는 이유를 설명할 수 있는가?
- [ ] **메모리 복잡도**: Attention의 $O(N^2)$ 메모리가 패치 크기 P와 어떻게 연결되는지 설명할 수 있는가?
- [ ] **Overlapping Patch**: stride < kernel_size일 때의 장단점을 설명할 수 있는가?
- [ ] **세 가지 구현 방식**: Conv2d, Unfold+Linear, einops 방식의 차이와 각각의 장단점을 아는가?
