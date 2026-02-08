# 4.2 CLS Token & Position Embedding

패치를 임베딩한 뒤, 두 가지 핵심 요소를 더해야 한다. 첫째, 전체 이미지를 대표하는 CLS 토큰. 둘째, 패치 간 공간 관계를 알려주는 위치 임베딩이다. 이 두 요소가 없으면 ViT는 "순서 없는 패치 집합"만 보게 된다.

---

> **핵심 용어 박스**
>
> | 용어 | 정의 |
> |------|------|
> | **CLS Token** | 시퀀스 맨 앞에 추가되는 학습 가능한 벡터. 분류 시 이미지 전체 표현으로 사용 |
> | **Learnable Embedding** | 고정값이 아닌, 역전파로 학습되는 임베딩 벡터 |
> | **Absolute Position Embedding** | 각 위치에 고유한 벡터를 더하는 방식 (1D 순서 기반) |
> | **Relative Position Embedding** | 토큰 간 상대 거리를 기반으로 하는 위치 인코딩 |
> | **2D Position Embedding** | 행(row)과 열(col)을 분리하여 2D 공간 정보를 인코딩 |

---

## 4.2.1 CLS Token

### 왜 CLS Token이 필요한가?

BERT에서 빌려온 아이디어다. Transformer 인코더를 거치면 모든 토큰이 서로의 정보를 aggregation한다. 이때 CLS 토큰은 특정 패치에 귀속되지 않으므로, 전체 시퀀스의 요약 역할을 하게 된다.

```
입력:  [CLS, patch_1, patch_2, ..., patch_N]
         ↓    ↓        ↓              ↓
Transformer Encoder (L layers)
         ↓    ↓        ↓              ↓
출력:  [z_cls, z_1,   z_2,    ...,   z_N]
         ↓
Classification Head → 클래스 예측
```

### 대안: Global Average Pooling

CLS 토큰 대신 모든 패치 토큰의 평균을 사용할 수도 있다.

$$
z_{\text{pool}} = \frac{1}{N} \sum_{i=1}^{N} z_i^L
$$

| 방식 | 장점 | 단점 |
|------|------|------|
| CLS Token | 추가 파라미터 1개, BERT와 일관성 | 학습 초기 불안정할 수 있음 |
| GAP (Global Average Pooling) | 안정적, 추가 파라미터 없음 | Dense prediction에 부적합할 수 있음 |

실험적으로 둘의 성능 차이는 크지 않다. DeiT는 CLS + Distillation Token을 동시에 사용한다.

---

## 수학적 원리

### CLS Token 정의

CLS 토큰은 학습 가능한 벡터다:

$$
x_{\text{class}} \in \mathbb{R}^D
$$

패치 임베딩 시퀀스의 맨 앞에 prepend한다:

$$
z_0 = [x_{\text{class}};\; x_p^1 E;\; x_p^2 E;\; \ldots;\; x_p^N E] + E_{pos}
$$

최종 출력의 CLS 위치를 분류에 사용:

$$
y = \text{MLP}(z_L^0)
$$

여기서 $z_L^0$은 $L$번째(마지막) Transformer 레이어의 CLS 토큰 출력이다.

### 1D Position Embedding (Absolute)

ViT 원논문의 기본 방식. 학습 가능한 파라미터 행렬:

$$
E_{pos} \in \mathbb{R}^{(N+1) \times D}
$$

$N+1$인 이유는 CLS 토큰을 포함하기 때문이다. 각 위치 $i$에 대해:

$$
z_0^i = x_p^i E + E_{pos}[i] \quad (i = 0, 1, \ldots, N)
$$

여기서 $i = 0$은 CLS 토큰 위치다.

이 방식의 한계: 패치의 2D 공간 배치를 1D 순서로 압축하므로 행/열 정보가 암묵적으로만 학습된다.

### 2D Position Embedding

행(row)과 열(col)을 분리해서 인코딩한다:

$$
E_{pos}^{2D}(i, j) = [E_{row}(i);\; E_{col}(j)]
$$

여기서:
- $E_{row} \in \mathbb{R}^{H_p \times D/2}$: 행 방향 위치 임베딩
- $E_{col} \in \mathbb{R}^{W_p \times D/2}$: 열 방향 위치 임베딩
- $H_p = H/P$, $W_p = W/P$: 행/열 패치 수

concatenation이므로 전체 차원은 $D/2 + D/2 = D$를 유지한다.

### Sinusoidal Position Embedding (고정)

Transformer 원논문의 사인/코사인 인코딩을 2D로 확장:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

2D 확장 시 행과 열에 각각 적용:

$$
PE_{2D}(r, c) = [PE_{sin/cos}(r);\; PE_{sin/cos}(c)]
$$

고정 인코딩은 학습이 불필요하지만, 학습 가능한 PE가 일반적으로 약간 더 좋은 성능을 보인다.

### Relative Position Embedding

절대 위치가 아닌 토큰 간 상대 거리를 인코딩:

$$
A_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} + b_{i-j}
$$

여기서 $b_{i-j}$는 상대 위치 $(i-j)$에 대한 학습 가능한 바이어스다. 2D로 확장하면:

$$
b_{(r_i - r_j, c_i - c_j)}
$$

행 차이와 열 차이를 모두 고려한다.

### Position Embedding 보간 (Interpolation)

학습 시 $224 \times 224$ (패치 수 $14 \times 14 = 196$)로 훈련했는데, 추론 시 $384 \times 384$ (패치 수 $24 \times 24 = 576$)를 쓰고 싶다면?

위치 임베딩을 bicubic interpolation으로 보간한다:

$$
E_{pos}^{new} = \text{Interpolate}_{bicubic}\left(
    \text{reshape}(E_{pos}, [H_p, W_p, D]),\;
    [H_p^{new}, W_p^{new}]
\right)
$$

구체적으로:
1. $E_{pos} \in \mathbb{R}^{(N+1) \times D}$에서 CLS 토큰 분리
2. 나머지를 $(H_p, W_p, D)$로 reshape
3. bicubic interpolation으로 $(H_p^{new}, W_p^{new}, D)$로 리사이즈
4. 다시 flatten하고 CLS 토큰 재결합

---

## 4.2.2 구현

### CLS Token + 1D Learnable PE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLSPositionEmbedding(nn.Module):
    """CLS Token과 1D Learnable Position Embedding.

    ViT 원논문의 표준 구현.
    """

    def __init__(
        self,
        num_patches: int = 196,
        embed_dim: int = 768,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_patches = num_patches

        # CLS Token: 학습 가능한 벡터
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position Embedding: (N+1) 위치에 대한 학습 가능한 벡터
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        # 초기화
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) 패치 임베딩
        Returns:
            (B, N+1, D) CLS 토큰 + 위치 정보가 추가된 시퀀스
        """
        B = x.shape[0]

        # CLS Token을 배치 크기만큼 확장
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)

        # 패치 시퀀스 앞에 CLS 토큰 추가
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # Position Embedding 더하기
        x = x + self.pos_embed  # (B, N+1, D)

        x = self.dropout(x)
        return x


# 테스트
if __name__ == "__main__":
    embed = CLSPositionEmbedding(num_patches=196, embed_dim=768)
    patch_tokens = torch.randn(2, 196, 768)
    out = embed(patch_tokens)
    print(f"입력: {patch_tokens.shape}")  # (2, 196, 768)
    print(f"출력: {out.shape}")           # (2, 197, 768)
    print(f"CLS 토큰 파라미터: {embed.cls_token.shape}")  # (1, 1, 768)
    print(f"PE 파라미터: {embed.pos_embed.shape}")         # (1, 197, 768)
```

### 2D Position Embedding

```python
class PositionEmbedding2D(nn.Module):
    """2D Position Embedding.

    행(row)과 열(col) 방향을 분리하여 인코딩한다.
    2D 공간 구조를 더 명시적으로 반영한다.
    """

    def __init__(
        self,
        num_patches_h: int = 14,
        num_patches_w: int = 14,
        embed_dim: int = 768,
    ):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim은 짝수여야 한다 (행/열 분할)"

        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        half_dim = embed_dim // 2

        # 행/열 각각에 대한 학습 가능한 임베딩
        self.row_embed = nn.Parameter(torch.zeros(1, num_patches_h, half_dim))
        self.col_embed = nn.Parameter(torch.zeros(1, num_patches_w, half_dim))

        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)

    def forward(self) -> torch.Tensor:
        """
        Returns:
            (1, H_p * W_p, D) 2D 위치 임베딩
        """
        H, W = self.num_patches_h, self.num_patches_w

        # 행 임베딩: (1, H, D/2) → (1, H, W, D/2)
        row_emb = self.row_embed.unsqueeze(2).expand(-1, -1, W, -1)

        # 열 임베딩: (1, W, D/2) → (1, H, W, D/2)
        col_emb = self.col_embed.unsqueeze(1).expand(-1, H, -1, -1)

        # 합치기: (1, H, W, D)
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)

        # (1, H*W, D)로 flatten
        pos_emb = pos_emb.reshape(1, H * W, -1)

        return pos_emb


# 테스트
if __name__ == "__main__":
    pe_2d = PositionEmbedding2D(num_patches_h=14, num_patches_w=14, embed_dim=768)
    pos = pe_2d()
    print(f"2D PE 출력: {pos.shape}")  # (1, 196, 768)
```

### Sinusoidal 2D Position Embedding (고정)

```python
import math
import numpy as np


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_h: int,
    grid_w: int,
    cls_token: bool = True,
) -> np.ndarray:
    """2D Sinusoidal Position Embedding 생성.

    MAE (Masked Autoencoder) 논문에서 사용된 방식.

    Args:
        embed_dim: 임베딩 차원 (4의 배수여야 함)
        grid_h: 패치 그리드 높이
        grid_w: 패치 그리드 너비
        cls_token: CLS 토큰 위치 포함 여부
    Returns:
        pos_embed: (grid_h*grid_w (+1), embed_dim) numpy 배열
    """
    assert embed_dim % 4 == 0, "embed_dim은 4의 배수여야 한다"

    half_dim = embed_dim // 2
    quarter_dim = embed_dim // 4

    # 각 축의 주파수 계산
    omega = np.arange(quarter_dim, dtype=np.float64) / quarter_dim
    omega = 1.0 / (10000.0 ** omega)  # (quarter_dim,)

    # 그리드 좌표 생성
    grid_r = np.arange(grid_h, dtype=np.float64)  # (grid_h,)
    grid_c = np.arange(grid_w, dtype=np.float64)  # (grid_w,)

    # 외적으로 모든 (position, frequency) 조합 생성
    pos_r = grid_r[:, None] * omega[None, :]  # (grid_h, quarter_dim)
    pos_c = grid_c[:, None] * omega[None, :]  # (grid_w, quarter_dim)

    # sin/cos 적용
    emb_r = np.concatenate([np.sin(pos_r), np.cos(pos_r)], axis=1)  # (grid_h, half_dim)
    emb_c = np.concatenate([np.sin(pos_c), np.cos(pos_c)], axis=1)  # (grid_w, half_dim)

    # 2D 그리드로 확장
    emb_r = np.repeat(emb_r[:, None, :], grid_w, axis=1)  # (grid_h, grid_w, half_dim)
    emb_c = np.repeat(emb_c[None, :, :], grid_h, axis=0)  # (grid_h, grid_w, half_dim)

    # 합치기 → (grid_h, grid_w, embed_dim)
    pos_embed = np.concatenate([emb_r, emb_c], axis=2)

    # (grid_h * grid_w, embed_dim)
    pos_embed = pos_embed.reshape(grid_h * grid_w, embed_dim)

    if cls_token:
        # CLS 토큰 위치: 0 벡터
        cls_pos = np.zeros((1, embed_dim))
        pos_embed = np.concatenate([cls_pos, pos_embed], axis=0)

    return pos_embed


# 테스트
if __name__ == "__main__":
    pe = get_2d_sincos_pos_embed(embed_dim=768, grid_h=14, grid_w=14, cls_token=True)
    print(f"Sinusoidal 2D PE: {pe.shape}")  # (197, 768)
```

### Position Embedding 보간 (Interpolation)

```python
def interpolate_pos_embed(
    pos_embed: torch.Tensor,
    new_num_patches_h: int,
    new_num_patches_w: int,
    old_num_patches_h: int = 14,
    old_num_patches_w: int = 14,
    has_cls_token: bool = True,
) -> torch.Tensor:
    """학습된 PE를 새로운 해상도에 맞게 보간한다.

    학습: 224×224 (14×14 패치) → 추론: 384×384 (24×24 패치)처럼
    해상도가 달라질 때 사용.

    Args:
        pos_embed: (1, N_old+1, D) 또는 (1, N_old, D)
        new_num_patches_h: 새 해상도의 패치 행 수
        new_num_patches_w: 새 해상도의 패치 열 수
        old_num_patches_h: 기존 해상도의 패치 행 수
        old_num_patches_w: 기존 해상도의 패치 열 수
        has_cls_token: CLS 토큰 포함 여부
    Returns:
        (1, N_new+1, D) 또는 (1, N_new, D) 보간된 PE
    """
    D = pos_embed.shape[-1]

    if has_cls_token:
        cls_pos = pos_embed[:, :1, :]       # (1, 1, D)
        patch_pos = pos_embed[:, 1:, :]     # (1, N_old, D)
    else:
        patch_pos = pos_embed

    # (1, N_old, D) → (1, D, H_old, W_old)
    patch_pos = patch_pos.reshape(1, old_num_patches_h, old_num_patches_w, D)
    patch_pos = patch_pos.permute(0, 3, 1, 2)  # (1, D, H_old, W_old)

    # Bicubic Interpolation
    patch_pos = F.interpolate(
        patch_pos,
        size=(new_num_patches_h, new_num_patches_w),
        mode='bicubic',
        align_corners=False,
    )
    # (1, D, H_new, W_new)

    # 다시 시퀀스로
    patch_pos = patch_pos.permute(0, 2, 3, 1)  # (1, H_new, W_new, D)
    patch_pos = patch_pos.reshape(1, -1, D)     # (1, N_new, D)

    if has_cls_token:
        return torch.cat([cls_pos, patch_pos], dim=1)  # (1, N_new+1, D)

    return patch_pos


# 테스트
if __name__ == "__main__":
    # 14×14 → 24×24 보간
    old_pe = torch.randn(1, 197, 768)  # 196 patches + 1 CLS
    new_pe = interpolate_pos_embed(
        old_pe,
        new_num_patches_h=24, new_num_patches_w=24,
        old_num_patches_h=14, old_num_patches_w=14,
    )
    print(f"기존 PE: {old_pe.shape}")  # (1, 197, 768)
    print(f"보간 PE: {new_pe.shape}")  # (1, 577, 768)  → 576 + 1 CLS
```

---

## 4.2.3 시각화

### Position Embedding 유사도 히트맵

학습된 PE의 코사인 유사도를 시각화하면, PE가 2D 공간 구조를 얼마나 학습했는지 확인할 수 있다.

```python
import matplotlib.pyplot as plt
import torch
import numpy as np


def visualize_position_embedding(
    pos_embed: torch.Tensor,
    num_patches_h: int = 14,
    num_patches_w: int = 14,
    has_cls_token: bool = True,
    selected_indices: list = None,
):
    """위치 임베딩의 코사인 유사도를 히트맵으로 시각화한다.

    특정 패치 위치를 선택하면, 해당 위치와 다른 모든 위치 간의
    유사도를 2D 히트맵으로 보여준다.
    """
    if has_cls_token:
        patch_pos = pos_embed[0, 1:, :].detach()  # (N, D)
    else:
        patch_pos = pos_embed[0].detach()

    N = patch_pos.shape[0]

    # 코사인 유사도 계산
    patch_pos_norm = F.normalize(patch_pos, dim=-1)
    cos_sim = patch_pos_norm @ patch_pos_norm.T  # (N, N)

    if selected_indices is None:
        # 4개 대표 위치 선택: 좌상단, 우상단, 중앙, 좌하단
        selected_indices = [
            0,                                      # 좌상단
            num_patches_w - 1,                      # 우상단
            N // 2 + num_patches_w // 2,           # 중앙
            (num_patches_h - 1) * num_patches_w,   # 좌하단
        ]

    fig, axes = plt.subplots(1, len(selected_indices), figsize=(5 * len(selected_indices), 5))

    for idx, sel in enumerate(selected_indices):
        sim = cos_sim[sel].reshape(num_patches_h, num_patches_w).numpy()

        im = axes[idx].imshow(sim, cmap='RdBu_r', vmin=-1, vmax=1)
        # 선택 위치 표시
        row, col = sel // num_patches_w, sel % num_patches_w
        axes[idx].plot(col, row, 'k*', markersize=15)
        axes[idx].set_title(f"위치 ({row},{col})과의 유사도", fontsize=12)
        axes[idx].axis('off')

    plt.colorbar(im, ax=axes, shrink=0.6, label='코사인 유사도')
    plt.suptitle("Position Embedding 코사인 유사도", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig("pos_embed_similarity.png", dpi=150, bbox_inches='tight')
    plt.show()


def visualize_pe_pca(
    pos_embed: torch.Tensor,
    num_patches_h: int = 14,
    num_patches_w: int = 14,
    has_cls_token: bool = True,
    n_components: int = 3,
):
    """PCA로 Position Embedding을 2D/3D로 투영하여 시각화한다.

    가까운 패치 위치의 PE가 임베딩 공간에서도 가까운지 확인.
    """
    from sklearn.decomposition import PCA

    if has_cls_token:
        patch_pos = pos_embed[0, 1:, :].detach().numpy()
    else:
        patch_pos = pos_embed[0].detach().numpy()

    N = patch_pos.shape[0]

    pca = PCA(n_components=n_components)
    pos_pca = pca.fit_transform(patch_pos)  # (N, n_components)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for comp in range(min(3, n_components)):
        values = pos_pca[:, comp].reshape(num_patches_h, num_patches_w)
        im = axes[comp].imshow(values, cmap='coolwarm', interpolation='nearest')
        axes[comp].set_title(
            f"PC{comp+1} (분산 설명: {pca.explained_variance_ratio_[comp]:.1%})",
            fontsize=12,
        )
        axes[comp].set_xlabel("열")
        axes[comp].set_ylabel("행")
        plt.colorbar(im, ax=axes[comp], shrink=0.8)

    plt.suptitle("Position Embedding PCA 분석", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig("pos_embed_pca.png", dpi=150, bbox_inches='tight')
    plt.show()

    print(f"상위 {n_components}개 PC의 누적 분산 설명: "
          f"{pca.explained_variance_ratio_[:n_components].sum():.1%}")
```

### 1D vs 2D PE 비교 시각화

```python
def compare_1d_vs_2d_pe(
    num_patches_h: int = 14,
    num_patches_w: int = 14,
    embed_dim: int = 768,
):
    """1D와 2D Position Embedding의 학습 결과 차이를 비교한다."""

    N = num_patches_h * num_patches_w

    # 1D PE 시뮬레이션 (랜덤 초기화 후 정규화)
    torch.manual_seed(42)
    pe_1d = torch.randn(1, N, embed_dim) * 0.02
    pe_1d_norm = F.normalize(pe_1d[0], dim=-1)
    sim_1d = (pe_1d_norm @ pe_1d_norm.T).numpy()

    # 2D PE 시뮬레이션
    pe_2d_model = PositionEmbedding2D(num_patches_h, num_patches_w, embed_dim)
    pe_2d = pe_2d_model()
    pe_2d_norm = F.normalize(pe_2d[0], dim=-1)
    sim_2d = (pe_2d_norm @ pe_2d_norm.T).detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(sim_1d, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title("1D PE 유사도 행렬", fontsize=14)
    axes[0].set_xlabel("패치 인덱스")
    axes[0].set_ylabel("패치 인덱스")

    im = axes[1].imshow(sim_2d, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title("2D PE 유사도 행렬", fontsize=14)
    axes[1].set_xlabel("패치 인덱스")
    axes[1].set_ylabel("패치 인덱스")

    plt.colorbar(im, ax=axes, shrink=0.6, label='코사인 유사도')
    plt.tight_layout()
    plt.savefig("1d_vs_2d_pe.png", dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 4.2.4 PE 방식별 비교

| 방식 | 학습 여부 | 해상도 변환 | 2D 구조 인식 | 사용 모델 |
|------|:---------:|:-----------:|:------------:|----------|
| 1D Learnable | 학습 | 보간 필요 | 암묵적 | ViT 원논문 |
| 2D Learnable | 학습 | 보간 필요 | 명시적 | ViT (실험) |
| Sinusoidal 고정 | 고정 | 자동 확장 | 명시적 | MAE |
| Relative PE | 학습 | 자동 일반화 | 쌍별 거리 | Swin, BEiT |
| RoPE (Rotary) | 고정 구조 | 자동 확장 | 가능 | 최신 VLM |

ViT 원논문에서 1D learnable과 2D learnable의 성능 차이는 미미했다. 하지만 해상도가 크게 바뀌는 transfer learning 시나리오에서는 2D PE나 보간이 중요해진다.

---

## 4.2.5 실전 팁

### 보간 시 주의사항

```python
def safe_interpolate_pos_embed(
    model: nn.Module,
    checkpoint_pos_embed: torch.Tensor,
    new_img_size: int,
    patch_size: int,
):
    """체크포인트의 PE를 새 해상도에 안전하게 보간한다.

    Transfer learning에서 해상도 변경 시 사용.
    """
    old_num = checkpoint_pos_embed.shape[1] - 1  # CLS 제외
    new_num_h = new_img_size // patch_size
    new_num_w = new_img_size // patch_size
    new_num = new_num_h * new_num_w

    if old_num == new_num:
        print("해상도 변경 없음. PE 그대로 사용.")
        return checkpoint_pos_embed

    old_side = int(old_num ** 0.5)
    print(f"PE 보간: {old_side}×{old_side} ({old_num}) → "
          f"{new_num_h}×{new_num_w} ({new_num})")

    new_pe = interpolate_pos_embed(
        checkpoint_pos_embed,
        new_num_patches_h=new_num_h,
        new_num_patches_w=new_num_w,
        old_num_patches_h=old_side,
        old_num_patches_w=old_side,
    )

    return new_pe
```

### CLS Token 초기화 전략

```python
# 방법 1: 0 초기화 (ViT 원논문)
cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

# 방법 2: Truncated Normal 초기화 (DeiT)
cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
nn.init.trunc_normal_(cls_token, std=0.02)

# 방법 3: Xavier Uniform 초기화
cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
nn.init.xavier_uniform_(cls_token)
```

일반적으로 truncated normal ($\sigma = 0.02$) 초기화가 가장 안정적이다.

---

## 용어 체크리스트

학습 후 아래 항목을 설명할 수 있는지 점검하라.

- [ ] **CLS Token**: 왜 별도의 학습 가능한 토큰을 시퀀스 앞에 추가하는지 설명할 수 있는가?
- [ ] **CLS vs GAP**: CLS Token과 Global Average Pooling의 차이점과 트레이드오프를 아는가?
- [ ] **1D Absolute PE**: $E_{pos} \in \mathbb{R}^{(N+1) \times D}$가 의미하는 바를 수식으로 설명할 수 있는가?
- [ ] **2D PE**: 행/열 분리 인코딩 $[E_{row}(i); E_{col}(j)]$의 구현 방식을 아는가?
- [ ] **Sinusoidal PE**: 사인/코사인 기반 고정 위치 인코딩의 수식과 장단점을 설명할 수 있는가?
- [ ] **Relative PE**: 절대 위치 대신 상대 거리 기반 인코딩이 왜 필요한지 아는가?
- [ ] **PE 보간**: 학습 해상도와 다른 해상도에서 bicubic interpolation으로 PE를 조정하는 과정을 구현할 수 있는가?
- [ ] **Learnable vs Fixed**: 학습 가능한 PE와 고정 PE의 장단점을 비교할 수 있는가?
