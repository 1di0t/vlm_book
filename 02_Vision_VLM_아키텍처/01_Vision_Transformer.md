# Chapter 2.1: Vision Transformer (ViT)

## 개요

Vision Transformer(ViT)는 NLP의 Transformer를 이미지 분류에 적용한 모델이다. CNN의 지배적 위치를 흔들며 Vision 분야의 패러다임을 전환했다.

> **핵심 논문**: Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
> - arXiv: https://arxiv.org/abs/2010.11929
> - ICLR 2021

---

## 1. ViT 아키텍처

### 1.1 Patch Embedding

이미지를 고정 크기 패치로 분할하여 시퀀스로 변환:

```
이미지 (H × W × C) → 패치 (N × P² × C) → 임베딩 (N × D)

N = (H × W) / P²  (패치 개수)
P = 패치 크기 (보통 16 또는 14)
```

**예시:**
- 이미지: 224 × 224 × 3
- 패치 크기: 16 × 16
- 패치 개수: (224/16)² = 196
- 각 패치: 16 × 16 × 3 = 768 차원

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Conv2d로 패치 분할 + Linear projection 동시 수행
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        x: (batch, channels, height, width)
        returns: (batch, num_patches, embed_dim)
        """
        # (B, C, H, W) → (B, D, H/P, W/P)
        x = self.projection(x)

        # (B, D, H/P, W/P) → (B, D, N) → (B, N, D)
        x = x.flatten(2).transpose(1, 2)

        return x
```

### 1.2 [CLS] Token과 Position Embedding

```python
class ViTEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # [CLS] token: 분류를 위한 학습 가능한 토큰
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embedding: 패치 + CLS 토큰
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        # 초기화
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        batch_size = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)

        # CLS token 추가
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # Position embedding
        x = x + self.pos_embed

        x = self.dropout(x)
        return x
```

### 1.3 해상도 변경 시 Position Embedding Interpolation

학습 시와 다른 해상도의 이미지를 처리할 때:

```python
def interpolate_pos_embed(pos_embed, new_size, old_size):
    """
    Position embedding을 새 해상도에 맞게 보간

    pos_embed: (1, old_num_patches + 1, D)
    new_size: 새 이미지 크기 (H, W)
    old_size: 학습 시 이미지 크기
    """
    # CLS token 분리
    cls_token = pos_embed[:, :1, :]
    patch_pos_embed = pos_embed[:, 1:, :]

    # 2D로 reshape
    old_h = old_w = int(patch_pos_embed.size(1) ** 0.5)
    patch_pos_embed = patch_pos_embed.reshape(1, old_h, old_w, -1).permute(0, 3, 1, 2)

    # Bicubic interpolation
    new_h, new_w = new_size[0] // patch_size, new_size[1] // patch_size
    patch_pos_embed = F.interpolate(
        patch_pos_embed,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=False
    )

    # 다시 1D로
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)

    # CLS token 다시 결합
    return torch.cat([cls_token, patch_pos_embed], dim=1)
```

---

## 2. ViT Variants

### 2.1 모델 크기

| Model | Layers | Hidden Dim | Heads | Params |
|-------|--------|------------|-------|--------|
| ViT-Ti (Tiny) | 12 | 192 | 3 | 5.7M |
| ViT-S (Small) | 12 | 384 | 6 | 22M |
| ViT-B (Base) | 12 | 768 | 12 | 86M |
| ViT-L (Large) | 24 | 1024 | 16 | 304M |
| ViT-H (Huge) | 32 | 1280 | 16 | 632M |

### 2.2 DeiT (Data-efficient Image Transformer)

> **논문**: Touvron et al. (2021). "Training data-efficient image transformers & distillation through attention"
> - arXiv: https://arxiv.org/abs/2012.12877
> - ICML 2021

**핵심 기여:**
- ImageNet만으로 ViT 학습 성공 (대규모 JFT 데이터 불필요)
- Knowledge Distillation을 위한 Distillation Token 도입
- 강력한 데이터 증강 (RandAugment, Mixup, CutMix)

```python
class DeiT(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 기본 ViT 구조
        self.patch_embed = PatchEmbedding(...)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Distillation token (teacher 지식 학습용)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 두 개의 classification head
        self.head = nn.Linear(embed_dim, num_classes)      # CLS token용
        self.head_dist = nn.Linear(embed_dim, num_classes)  # Dist token용

    def forward(self, x):
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        dist_tokens = self.dist_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)

        x = x + self.pos_embed
        x = self.transformer(x)

        # 두 토큰으로 각각 예측
        cls_out = self.head(x[:, 0])
        dist_out = self.head_dist(x[:, 1])

        if self.training:
            return cls_out, dist_out
        else:
            return (cls_out + dist_out) / 2  # 앙상블
```

### 2.3 Swin Transformer

> **논문**: Liu et al. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
> - arXiv: https://arxiv.org/abs/2103.14030
> - ICCV 2021 Best Paper

**핵심 특징:**

1. **Hierarchical Feature Maps**
   - 단계별로 해상도 감소 (CNN처럼)
   - FPN, U-Net 등과 호환

2. **Shifted Window Attention**
   - 윈도우 내에서만 attention (O(n) 복잡도)
   - Shifted windows로 윈도우 간 연결

```python
class WindowAttention(nn.Module):
    """Window-based Multi-head Self-Attention"""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )

    def forward(self, x, mask=None):
        B_, N, C = x.shape  # N = window_size * window_size

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.num_heads)

        # Relative position bias
        attn = attn + self.relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        return self.proj(x)


def window_partition(x, window_size):
    """이미지를 윈도우로 분할"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """윈도우를 다시 이미지로 결합"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
```

---

## 3. ViT vs CNN

### 3.1 Inductive Bias 비교

| 특성 | CNN | ViT |
|------|-----|-----|
| **Locality** | 강함 (커널 크기 제한) | 약함 (전역 attention) |
| **Translation Equivariance** | 내재적 | Position embedding 필요 |
| **Scale Invariance** | Pooling으로 부분 획득 | 없음 |
| **데이터 효율성** | 높음 | 낮음 (대규모 데이터 필요) |

### 3.2 데이터 요구량

ViT 논문의 핵심 발견:

- **ImageNet (1.3M 이미지)**: CNN > ViT
- **ImageNet-21K (14M 이미지)**: ViT ≈ CNN
- **JFT-300M (300M 이미지)**: ViT > CNN

```
"When trained on mid-sized datasets such as ImageNet without strong regularization,
these models yield modest accuracies of a few percentage points below ResNets
of comparable size. This seemingly discouraging outcome may be expected:
Transformers lack some of the inductive biases inherent to CNNs..."
```

### 3.3 Attention Map 시각화

```python
def visualize_attention(model, image):
    """ViT attention map 시각화"""
    # Forward pass with attention weights
    model.eval()
    with torch.no_grad():
        output, attn_weights = model(image, return_attention=True)

    # 마지막 레이어의 [CLS] token attention
    attn = attn_weights[-1]  # (B, heads, N+1, N+1)
    attn = attn[:, :, 0, 1:]  # CLS token의 패치 attention (B, heads, N)

    # 평균 또는 특정 head
    attn = attn.mean(dim=1)  # (B, N)

    # 2D로 reshape
    num_patches = int(attn.size(-1) ** 0.5)
    attn = attn.reshape(1, num_patches, num_patches)

    # Upsample to image size
    attn = F.interpolate(attn.unsqueeze(1), size=image.shape[-2:], mode='bilinear')

    return attn.squeeze()
```

---

## 4. 전체 ViT 구현

```python
class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Patch embedding + Position embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_layer_weights)

    def _init_layer_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]  # CLS token

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1, attn_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-LN
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
```

---

## 핵심 참고 자료

### 논문
- **ViT** (Dosovitskiy et al., 2020)
  - https://arxiv.org/abs/2010.11929

- **DeiT** (Touvron et al., 2021)
  - https://arxiv.org/abs/2012.12877
  - GitHub: https://github.com/facebookresearch/deit

- **Swin Transformer** (Liu et al., 2021)
  - https://arxiv.org/abs/2103.14030
  - GitHub: https://github.com/microsoft/Swin-Transformer

### 구현
- **timm (PyTorch Image Models)**
  - https://github.com/huggingface/pytorch-image-models
  - 다양한 ViT variant 구현 제공

- **HuggingFace Transformers**
  - https://huggingface.co/docs/transformers/model_doc/vit

---

## 핵심 요약

| 구성요소 | 역할 | 핵심 파라미터 |
|---------|------|--------------|
| Patch Embedding | 이미지→시퀀스 | patch_size (16, 14) |
| [CLS] Token | 분류용 임베딩 | 학습 가능 |
| Position Embedding | 위치 정보 | 2D interpolation 가능 |
| Transformer Blocks | 특징 추출 | depth, heads, mlp_ratio |
| Classification Head | 최종 분류 | [CLS] token 사용 |
