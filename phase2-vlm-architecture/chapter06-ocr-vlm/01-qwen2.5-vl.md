---
---

# 6.1 Qwen2.5-VL 심층 분석

> **Qwen2.5-VL**은 Alibaba Qwen 팀이 2024년에 공개한 Vision-Language Model이다.
> Dynamic Resolution, M-RoPE, Token Merging 등 OCR 성능을 극대화하는 핵심 기법들을 통합했다.

---

## 핵심 용어

| 용어 | 정의 | 왜 중요한가 |
|------|------|-------------|
| **Dynamic Resolution** | 입력 이미지의 원본 비율을 유지하면서 가변 개수의 패치를 생성하는 기법 | 문서 이미지는 크기와 비율이 제각각이라 고정 해상도로는 정보 손실이 크다 |
| **NaViT** (Native Resolution ViT) | Google이 제안한 패딩 없이 다양한 해상도를 처리하는 ViT 변형 | Qwen2.5-VL의 Dynamic Resolution 설계에 영감을 준 핵심 선행 연구 |
| **Token Merging** | 인접한 비전 토큰들을 병합하여 시퀀스 길이를 줄이는 기법 | LLM에 입력되는 비전 토큰 수를 줄여 추론 속도와 메모리 효율을 확보 |
| **M-RoPE** (Multimodal RoPE) | 시간·높이·너비 3축으로 확장한 회전 위치 인코딩 | 이미지의 2D 공간 구조와 비디오의 시간 축을 동시에 인코딩 |

---

## 아키텍처 개요

Qwen2.5-VL의 전체 파이프라인은 다음과 같다:

```
Input Image (임의 해상도)
    │
    ▼
[Dynamic Resolution Preprocessing]
    │  - 원본 비율 유지
    │  - 가변 패치 수 생성
    ▼
[ViT Encoder (ViT-G/14 변형)]
    │  - 패치 임베딩
    │  - M-RoPE 위치 인코딩
    ▼
[Token Merging (2×2 → 1)]
    │  - 비전 토큰 수 1/4 축소
    ▼
[MLP Projector]
    │  - 비전 → 언어 차원 매핑
    ▼
[Qwen2.5 LLM Backbone]
    │  - 비전 토큰 + 텍스트 토큰 통합
    │  - M-RoPE로 멀티모달 위치 인코딩
    ▼
Output Text
```

### 모델 규모별 구성

| 구성요소 | Qwen2.5-VL-3B | Qwen2.5-VL-7B | Qwen2.5-VL-72B |
|----------|---------------|---------------|-----------------|
| ViT Encoder | ViT-G/14 (675M) | ViT-G/14 (675M) | ViT-G/14 (675M) |
| LLM Backbone | Qwen2.5-3B | Qwen2.5-7B | Qwen2.5-72B |
| Token Merging | 2×2 병합 | 2×2 병합 | 2×2 병합 |
| 최대 이미지 토큰 | 16,384 | 16,384 | 16,384 |
| 총 파라미터 | ~3.7B | ~7.7B | ~72.7B |

---

## 수학적 원리

### 1. Dynamic Resolution

기존 ViT는 모든 이미지를 $224 \times 224$ 같은 고정 크기로 리사이즈했다. 문서 이미지에서 이건 치명적이다. A4 세로 문서를 정사각형으로 찌그러뜨리면 글자가 뭉개진다.

**Dynamic Resolution의 핵심 아이디어**: 원본 비율을 유지하면서, 총 패치 수만 제한한다.

이미지 크기가 $(H, W)$이고 패치 크기가 $p$일 때:

$$
n_h = \left\lceil \frac{H}{p} \right\rceil, \quad n_w = \left\lceil \frac{W}{p} \right\rceil
$$

총 패치 수:

$$
N_{\text{patches}} = n_h \times n_w
$$

**제약 조건**: 총 패치 수가 최대 한도 $N_{\max}$를 넘지 않도록 스케일링한다.

$$
\text{if } N_{\text{patches}} > N_{\max}: \quad s = \sqrt{\frac{N_{\max}}{N_{\text{patches}}}}, \quad H' = \lfloor s \cdot H \rfloor, \quad W' = \lfloor s \cdot W \rfloor
$$

이때 비율 $\frac{H'}{W'} \approx \frac{H}{W}$가 유지된다. 핵심은 **비율 보존**이다.

```python
import math
from typing import Tuple


def dynamic_resolution(
    height: int,
    width: int,
    patch_size: int = 14,
    max_patches: int = 16384,
    min_patches: int = 256,
    patch_multiple: int = 2,  # Token Merging 단위와 맞춤
) -> Tuple[int, int]:
    """
    Qwen2.5-VL 스타일 Dynamic Resolution 계산.

    원본 비율을 유지하면서 패치 수가 [min_patches, max_patches] 범위에
    들도록 리사이즈 크기를 결정한다.

    Args:
        height: 원본 이미지 높이
        width: 원본 이미지 너비
        patch_size: ViT 패치 크기 (픽셀)
        max_patches: 최대 패치 수
        min_patches: 최소 패치 수
        patch_multiple: 패치 수가 이 값의 배수가 되도록 정렬

    Returns:
        (new_height, new_width): 리사이즈할 크기
    """
    aspect_ratio = width / height

    # 현재 패치 수 계산
    n_h = math.ceil(height / patch_size)
    n_w = math.ceil(width / patch_size)
    n_patches = n_h * n_w

    # 패치 수가 범위를 벗어나면 스케일링
    if n_patches > max_patches:
        scale = math.sqrt(max_patches / n_patches)
        n_h = max(1, round(n_h * scale))
        n_w = max(1, round(n_w * scale))
    elif n_patches < min_patches:
        scale = math.sqrt(min_patches / n_patches)
        n_h = max(1, round(n_h * scale))
        n_w = max(1, round(n_w * scale))

    # patch_multiple의 배수로 정렬 (Token Merging과 호환)
    n_h = max(patch_multiple, (n_h // patch_multiple) * patch_multiple)
    n_w = max(patch_multiple, (n_w // patch_multiple) * patch_multiple)

    new_height = n_h * patch_size
    new_width = n_w * patch_size

    return new_height, new_width


# 사용 예시
examples = [
    (1920, 1080, "세로 문서"),   # 세로가 긴 문서
    (800, 1200, "가로 문서"),     # 가로가 긴 문서
    (3000, 3000, "정사각형"),     # 큰 정사각형
    (200, 200, "작은 이미지"),    # 작은 이미지
]

for h, w, desc in examples:
    new_h, new_w = dynamic_resolution(h, w)
    n_patches = (new_h // 14) * (new_w // 14)
    print(f"[{desc}] {h}×{w} → {new_h}×{new_w} (패치 수: {n_patches})")
```

### 2. M-RoPE (Multimodal Rotary Position Embedding)

표준 RoPE는 1D 시퀀스 위치만 인코딩한다. 하지만 이미지는 2D 공간이고, 비디오는 3D(시간+2D)다. M-RoPE는 이 문제를 해결한다.

#### 표준 RoPE 복습

1D 위치 $m$에 대한 RoPE:

$$
\text{RoPE}(x, m) = x \cdot e^{im\theta}
$$

여기서 $\theta_j = 10000^{-2j/d}$이고, 복소수 표현을 사용한다. 실수 구현에서는:

$$
R_{\theta, m} = \begin{pmatrix} \cos m\theta_1 & -\sin m\theta_1 \\ \sin m\theta_1 & \cos m\theta_1 \\ & & \cos m\theta_2 & -\sin m\theta_2 \\ & & \sin m\theta_2 & \cos m\theta_2 \\ & & & & \ddots \end{pmatrix}
$$

#### M-RoPE 확장

M-RoPE는 차원을 3등분하여 각각 시간($t$), 높이($h$), 너비($w$) 축의 위치를 인코딩한다.

$d$차원 벡터를 3등분: $d = d_t + d_h + d_w$, 보통 $d_t = d_h = d_w = d/3$.

위치 인덱스 $(m_t, m_h, m_w)$에 대해:

$$
\text{M-RoPE}(x, m_t, m_h, m_w) = \begin{bmatrix} \text{RoPE}(x_{[0:d/3]}, m_t) \\ \text{RoPE}(x_{[d/3:2d/3]}, m_h) \\ \text{RoPE}(x_{[2d/3:d]}, m_w) \end{bmatrix}
$$

Attention score 계산:

$$
\text{score}(q_i, k_j) = \text{M-RoPE}(q, m_t^i, m_h^i, m_w^i) \cdot \text{M-RoPE}(k, n_t^j, n_h^j, n_w^j)^T
$$

풀어쓰면:

$$
\text{score} = f(q, m_t, m_h, m_w) \cdot f(k, n_t, n_h, n_w)^T
$$

**각 모달리티별 위치 인덱스 할당**:

| 모달리티 | $m_t$ | $m_h$ | $m_w$ |
|----------|-------|-------|-------|
| 텍스트 | 토큰 순서 $i$ | 토큰 순서 $i$ | 토큰 순서 $i$ |
| 이미지 | 0 (단일 프레임) | 패치 행 인덱스 | 패치 열 인덱스 |
| 비디오 | 프레임 인덱스 | 패치 행 인덱스 | 패치 열 인덱스 |

텍스트 토큰은 3축 모두 동일한 위치값을 쓰므로 표준 RoPE와 동일하게 동작한다.

```python
import torch
import torch.nn as nn


class MultimodalRoPE(nn.Module):
    """
    M-RoPE: 시간(t), 높이(h), 너비(w) 3축 회전 위치 인코딩.

    Qwen2.5-VL에서 텍스트/이미지/비디오 토큰의 위치를 통합 인코딩한다.
    """

    def __init__(self, dim: int, max_position: int = 32768, base: float = 10000.0):
        super().__init__()
        assert dim % 6 == 0, f"dim({dim})은 6의 배수여야 한다 (3축 × 2 for sin/cos)"

        self.dim = dim
        self.dim_per_axis = dim // 3  # 각 축에 할당되는 차원

        # 각 축별 주파수 계산
        half_dim = self.dim_per_axis // 2
        freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        self.register_buffer("freq", freq)

    def _compute_rope(
        self, positions: torch.Tensor, dim_offset: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        단일 축에 대한 RoPE cos/sin 계산.

        Args:
            positions: (batch, seq_len) 위치 인덱스
            dim_offset: 차원 내 오프셋 (사용하지 않지만 확장성 위해)

        Returns:
            cos, sin: 각각 (batch, seq_len, dim_per_axis // 2)
        """
        # positions: (B, S) -> (B, S, 1)
        # freq: (D/6,) -> (1, 1, D/6)
        angles = positions.unsqueeze(-1).float() * self.freq.unsqueeze(0).unsqueeze(0)
        return angles.cos(), angles.sin()

    def _apply_rotary(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        회전 변환 적용. x의 마지막 차원을 짝수/홀수로 나눠 회전한다.

        Args:
            x: (batch, heads, seq_len, dim_per_axis)
            cos, sin: (batch, seq_len, dim_per_axis // 2)
        """
        # (B, S, D//2) -> (B, 1, S, D//2) for head broadcast
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # x를 짝수/홀수 인덱스로 분리
        x_even = x[..., 0::2]  # (B, H, S, D//4)
        x_odd = x[..., 1::2]

        # 회전 적용
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # 인터리브하여 원래 형태로
        out = torch.stack([rotated_even, rotated_odd], dim=-1)
        return out.flatten(-2)  # (..., D//2, 2) -> (..., D)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids_t: torch.Tensor,
        position_ids_h: torch.Tensor,
        position_ids_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        M-RoPE 적용.

        Args:
            q, k: (batch, heads, seq_len, dim)
            position_ids_t: (batch, seq_len) 시간 축 위치
            position_ids_h: (batch, seq_len) 높이 축 위치
            position_ids_w: (batch, seq_len) 너비 축 위치

        Returns:
            q_rotated, k_rotated: M-RoPE가 적용된 Q, K
        """
        d = self.dim_per_axis

        # 각 축별 cos/sin 계산
        cos_t, sin_t = self._compute_rope(position_ids_t, 0)
        cos_h, sin_h = self._compute_rope(position_ids_h, 1)
        cos_w, sin_w = self._compute_rope(position_ids_w, 2)

        # Q, K를 3등분
        q_t, q_h, q_w = q[..., :d], q[..., d:2*d], q[..., 2*d:]
        k_t, k_h, k_w = k[..., :d], k[..., d:2*d], k[..., 2*d:]

        # 각 축에 RoPE 적용
        q_t = self._apply_rotary(q_t, cos_t, sin_t)
        q_h = self._apply_rotary(q_h, cos_h, sin_h)
        q_w = self._apply_rotary(q_w, cos_w, sin_w)

        k_t = self._apply_rotary(k_t, cos_t, sin_t)
        k_h = self._apply_rotary(k_h, cos_h, sin_h)
        k_w = self._apply_rotary(k_w, cos_w, sin_w)

        # 결합
        q_out = torch.cat([q_t, q_h, q_w], dim=-1)
        k_out = torch.cat([k_t, k_h, k_w], dim=-1)

        return q_out, k_out


# ---- 사용 예시 ----
def demo_mrope():
    batch, heads, seq_len, dim = 1, 8, 100, 96  # dim은 6의 배수

    mrope = MultimodalRoPE(dim=dim)

    q = torch.randn(batch, heads, seq_len, dim)
    k = torch.randn(batch, heads, seq_len, dim)

    # 이미지 토큰 (10×10 패치 = 100 토큰)
    # 시간: 모두 0 (단일 이미지)
    # 높이: 0~9 반복
    # 너비: 0,0,...,0, 1,1,...,1, ... 9,9,...,9
    pos_t = torch.zeros(batch, seq_len, dtype=torch.long)
    pos_h = torch.arange(10).repeat(10).unsqueeze(0)    # [0,1,...,9,0,1,...,9,...]
    pos_w = torch.arange(10).repeat_interleave(10).unsqueeze(0)  # [0,0,...,1,1,...,9,9,...]

    q_rot, k_rot = mrope(q, k, pos_t, pos_h, pos_w)
    print(f"Q shape: {q_rot.shape}")  # (1, 8, 100, 96)
    print(f"K shape: {k_rot.shape}")  # (1, 8, 100, 96)

    # 인접 패치 간 attention score가 먼 패치보다 높은지 확인
    scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / (dim ** 0.5)
    print(f"Score shape: {scores.shape}")  # (1, 8, 100, 100)


demo_mrope()
```

### 3. Token Merging

ViT가 뽑아낸 비전 토큰 수가 수천~수만 개에 달하면 LLM의 컨텍스트 윈도우를 심하게 잡아먹는다. Token Merging은 인접한 $2 \times 2$ 패치 토큰을 하나로 합쳐서 토큰 수를 $\frac{1}{4}$로 줄인다.

패치 그리드가 $(n_h, n_w)$일 때, Token Merging 후:

$$
n_h' = \frac{n_h}{2}, \quad n_w' = \frac{n_w}{2}, \quad N' = \frac{N}{4}
$$

병합 방식은 단순 평균 또는 학습 가능한 가중합이다:

$$
z_{\text{merged}} = \text{MLP}\left(\frac{1}{4} \sum_{i \in \{2 \times 2\}} z_i\right) \quad \text{또는} \quad z_{\text{merged}} = \text{MLP}(\text{concat}(z_{00}, z_{01}, z_{10}, z_{11}))
$$

Qwen2.5-VL에서는 concat + MLP 방식을 사용한다. 4개 토큰의 차원 $d$를 concat하면 $4d$가 되고, MLP가 다시 $d$로 압축한다.

```python
import torch
import torch.nn as nn


class TokenMerging2x2(nn.Module):
    """
    2×2 패치 토큰을 하나로 병합하는 모듈.

    Qwen2.5-VL의 비전 토큰 압축에 사용된다.
    4개 인접 토큰을 concat한 뒤 MLP로 차원을 복원한다.
    """

    def __init__(self, dim: int, out_dim: int = None):
        super().__init__()
        out_dim = out_dim or dim

        # 4개 토큰 concat (4*dim) → out_dim
        self.merge_proj = nn.Sequential(
            nn.LayerNorm(4 * dim),
            nn.Linear(4 * dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """
        Args:
            x: (batch, n_tokens, dim) — 비전 토큰
            grid_h: 패치 그리드 높이
            grid_w: 패치 그리드 너비

        Returns:
            merged: (batch, n_tokens // 4, out_dim)
        """
        B, N, D = x.shape
        assert N == grid_h * grid_w, f"토큰 수({N}) != grid({grid_h}×{grid_w})"
        assert grid_h % 2 == 0 and grid_w % 2 == 0, "그리드 크기는 짝수여야 한다"

        # (B, N, D) → (B, grid_h, grid_w, D)
        x = x.view(B, grid_h, grid_w, D)

        # 2×2 블록으로 재배열
        # (B, grid_h//2, 2, grid_w//2, 2, D)
        x = x.view(B, grid_h // 2, 2, grid_w // 2, 2, D)

        # (B, grid_h//2, grid_w//2, 2, 2, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        # 2×2 블록의 4개 토큰을 concat
        # (B, grid_h//2 * grid_w//2, 4*D)
        x = x.view(B, (grid_h // 2) * (grid_w // 2), 4 * D)

        # MLP로 차원 복원
        merged = self.merge_proj(x)

        return merged


# ---- 사용 예시 ----
def demo_token_merging():
    batch = 2
    grid_h, grid_w = 20, 30  # 20×30 = 600 패치
    dim = 1024

    tokens = torch.randn(batch, grid_h * grid_w, dim)

    merger = TokenMerging2x2(dim=dim, out_dim=dim)
    merged = merger(tokens, grid_h, grid_w)

    print(f"병합 전: {tokens.shape}")   # (2, 600, 1024)
    print(f"병합 후: {merged.shape}")   # (2, 150, 1024) → 1/4로 축소
    print(f"토큰 수 비율: {merged.shape[1] / tokens.shape[1]:.2f}")  # 0.25


demo_token_merging()
```

---

## ViT Encoder 상세

Qwen2.5-VL의 ViT는 표준 ViT-G/14를 기반으로 하되 몇 가지 변형이 있다:

1. **패치 크기**: 14×14 픽셀
2. **위치 인코딩**: 학습 가능한 절대 위치 인코딩 대신 **2D RoPE** 사용
3. **출력**: Token Merging 전 비전 토큰 시퀀스

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PatchEmbed(nn.Module):
    """패치 임베딩: 이미지를 패치로 분할하고 선형 임베딩한다."""

    def __init__(self, patch_size: int = 14, in_channels: int = 3, embed_dim: int = 1664):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, C, H, W) — Dynamic Resolution으로 전처리된 이미지

        Returns:
            patches: (B, n_patches, embed_dim)
            grid_h, grid_w: 패치 그리드 크기
        """
        B, C, H, W = x.shape
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size

        # (B, C, H, W) → (B, embed_dim, grid_h, grid_w)
        x = self.proj(x)
        # (B, embed_dim, grid_h, grid_w) → (B, n_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        return x, grid_h, grid_w


class Qwen25VLViT(nn.Module):
    """Qwen2.5-VL의 Vision Encoder 간소화 구현."""

    def __init__(
        self,
        patch_size: int = 14,
        embed_dim: int = 1664,
        depth: int = 48,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        out_dim: int = 3584,  # LLM hidden dim (7B 기준)
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim)

        # Transformer 블록 (간소화)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                activation=F.silu,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Token Merging
        self.token_merger = TokenMerging2x2(dim=embed_dim, out_dim=out_dim)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            pixel_values: (B, C, H, W)

        Returns:
            vision_tokens: (B, n_merged, out_dim) — LLM에 입력할 비전 토큰
            grid_h, grid_w: 병합 후 그리드 크기
        """
        # 패치 임베딩
        x, grid_h, grid_w = self.patch_embed(pixel_values)

        # Transformer 블록 통과
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Token Merging (2×2 → 1)
        x = self.token_merger(x, grid_h, grid_w)

        merged_h = grid_h // 2
        merged_w = grid_w // 2

        return x, merged_h, merged_w
```

---

## 위치 인코딩 통합: 텍스트 + 이미지

LLM backbone에서 텍스트와 비전 토큰이 하나의 시퀀스로 합쳐질 때, M-RoPE의 위치 인덱스가 어떻게 할당되는지가 핵심이다.

```
시퀀스: [텍스트_1, ..., 텍스트_k, <img_start>, 비전_1, ..., 비전_m, <img_end>, 텍스트_{k+1}, ...]

위치 할당 예시 (k=3, 비전 토큰 4×4=16):

토큰          | pos_t | pos_h | pos_w
------------- |-------|-------|------
텍스트_1      |  0    |  0    |  0
텍스트_2      |  1    |  1    |  1
텍스트_3      |  2    |  2    |  2
<img_start>   |  3    |  3    |  3
비전_(0,0)    |  3    |  3    |  3
비전_(0,1)    |  3    |  3    |  4
비전_(0,2)    |  3    |  3    |  5
비전_(0,3)    |  3    |  3    |  6
비전_(1,0)    |  3    |  4    |  3
비전_(1,1)    |  3    |  4    |  4
...           |  ...  |  ...  |  ...
비전_(3,3)    |  3    |  6    |  6
<img_end>     |  7    |  7    |  7
텍스트_4      |  8    |  8    |  8
```

**핵심 포인트**:
- 텍스트 토큰: 3축 모두 동일한 순차 인덱스 → 1D RoPE와 동치
- 이미지 토큰: 시간 축은 고정, 높이/너비 축은 2D 그리드 좌표
- 이미지 종료 후 텍스트 재개 시: 인덱스가 이미지의 최대 인덱스 + 1부터 계속

```python
import torch
from typing import List, Tuple


def build_mrope_position_ids(
    token_types: List[str],
    image_grids: List[Tuple[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    멀티모달 시퀀스에 대한 M-RoPE 위치 인덱스를 생성한다.

    Args:
        token_types: 각 토큰의 타입 리스트. "text" 또는 "image_start"
        image_grids: 각 이미지의 (grid_h, grid_w) 크기 리스트

    Returns:
        pos_t, pos_h, pos_w: 각각 (1, total_seq_len) 텐서
    """
    pos_t, pos_h, pos_w = [], [], []

    current_pos = 0
    img_idx = 0

    i = 0
    while i < len(token_types):
        if token_types[i] == "text":
            pos_t.append(current_pos)
            pos_h.append(current_pos)
            pos_w.append(current_pos)
            current_pos += 1
            i += 1

        elif token_types[i] == "image_start":
            grid_h, grid_w = image_grids[img_idx]
            img_idx += 1

            # <img_start> 토큰
            start_pos = current_pos
            pos_t.append(start_pos)
            pos_h.append(start_pos)
            pos_w.append(start_pos)
            i += 1

            # 이미지 토큰들
            max_h, max_w = start_pos, start_pos
            for row in range(grid_h):
                for col in range(grid_w):
                    pos_t.append(start_pos)           # 시간 고정
                    pos_h.append(start_pos + row)      # 행 좌표
                    pos_w.append(start_pos + col)      # 열 좌표
                    max_h = max(max_h, start_pos + row)
                    max_w = max(max_w, start_pos + col)
                    i += 1

            # <img_end> 토큰 이후 텍스트 재개 위치
            current_pos = max(max_h, max_w) + 1

            # <img_end> 토큰
            pos_t.append(current_pos)
            pos_h.append(current_pos)
            pos_w.append(current_pos)
            current_pos += 1
            i += 1

    return (
        torch.tensor([pos_t], dtype=torch.long),
        torch.tensor([pos_h], dtype=torch.long),
        torch.tensor([pos_w], dtype=torch.long),
    )


# ---- 사용 예시 ----
token_types = (
    ["text"] * 5 +                          # "이 문서의 내용은"
    ["image_start"] +                        # <img_start>
    ["image"] * (4 * 6) +                    # 4×6 비전 토큰
    ["image_end"] +                          # <img_end>
    ["text"] * 3                              # "을 요약하면"
)

pos_t, pos_h, pos_w = build_mrope_position_ids(
    token_types=token_types,
    image_grids=[(4, 6)],
)

print(f"시퀀스 길이: {pos_t.shape[1]}")
print(f"pos_t 범위: [{pos_t.min()}, {pos_t.max()}]")
print(f"pos_h 범위: [{pos_h.min()}, {pos_h.max()}]")
print(f"pos_w 범위: [{pos_w.min()}, {pos_w.max()}]")
```

---

## OCR 성능 분석

Qwen2.5-VL은 문서 OCR 벤치마크에서 최고 수준의 성능을 보인다.

### 벤치마크 성능

| 벤치마크 | Qwen2.5-VL-7B | GPT-4o | Gemini 1.5 Pro | InternVL2.5-8B |
|----------|---------------|--------|----------------|----------------|
| DocVQA | 94.5% | 92.8% | 93.1% | 91.6% |
| ChartQA | 87.3% | 85.7% | 86.8% | 83.9% |
| TextVQA | 84.9% | 77.4% | 78.7% | 80.6% |
| OCRBench | 877 | 736 | 754 | 822 |

### OCR에 강한 이유

1. **Dynamic Resolution**: 문서의 작은 글씨도 충분한 패치 수로 커버
2. **M-RoPE**: 문서 내 텍스트의 2D 위치 관계를 정확히 인코딩
3. **대규모 학습 데이터**: 문서 이해 관련 데이터를 대량 포함
4. **Token Merging 후에도 유지되는 공간 정보**: 2×2 병합이 지역 정보를 보존

---

## 실전 활용: 문서 OCR 추론

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch


def run_qwen25vl_ocr(
    image_path: str,
    prompt: str = "이 문서의 모든 텍스트를 추출해라. 표가 있으면 마크다운 표로 변환해라.",
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    max_new_tokens: int = 4096,
) -> str:
    """
    Qwen2.5-VL로 문서 OCR을 수행한다.

    Args:
        image_path: 문서 이미지 경로
        prompt: OCR 프롬프트
        model_name: 모델 이름
        max_new_tokens: 최대 생성 토큰 수

    Returns:
        추출된 텍스트
    """
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # 입력 부분 제거하고 생성된 텍스트만 디코딩
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return result


# 사용법 (GPU 환경에서 실행)
# text = run_qwen25vl_ocr("medical_document.png")
# print(text)
```

---

## 한계점과 개선 방향

| 한계 | 설명 | 개선 방향 |
|------|------|----------|
| 메모리 소비 | 고해상도 이미지 → 많은 토큰 → 큰 KV cache | FlashAttention, KV cache 양자화 |
| 추론 속도 | 토큰 수에 비례하여 느려짐 | 적응적 Token Merging 비율 |
| 소형 텍스트 | 매우 작은 글씨는 여전히 어려움 | 다중 스케일 패치 추출 |
| 손글씨 | 인쇄체 대비 정확도 하락 | 손글씨 특화 fine-tuning |

---

## 용어 체크리스트

아래 용어들을 설명할 수 있으면 이 챕터를 이해한 거다.

- [ ] **Dynamic Resolution**: 왜 고정 해상도가 문서 OCR에 나쁜지, 가변 해상도가 어떻게 동작하는지 설명할 수 있는가?
- [ ] **NaViT**: Native Resolution ViT의 핵심 아이디어를 설명할 수 있는가?
- [ ] **M-RoPE**: 3축(t, h, w) RoPE가 텍스트/이미지/비디오 토큰에 각각 어떻게 적용되는지 설명할 수 있는가?
- [ ] **Token Merging**: 2×2 병합의 수학적 과정과 왜 필요한지 설명할 수 있는가?
- [ ] **패치 수 제한**: $N_{\max}$를 넘지 않도록 스케일링하는 수식을 유도할 수 있는가?
- [ ] **위치 인덱스 할당**: 멀티모달 시퀀스에서 텍스트·이미지 토큰의 M-RoPE 위치가 어떻게 결정되는지 설명할 수 있는가?
- [ ] **OCR 성능 요인**: Qwen2.5-VL이 문서 OCR에서 강한 4가지 이유를 열거할 수 있는가?
