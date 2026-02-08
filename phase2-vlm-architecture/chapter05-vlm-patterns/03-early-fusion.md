---
---

# 5.3 Early Fusion 방식

Early Fusion은 이미지와 텍스트를 **초기 단계에서 하나의 시퀀스로 결합**하여 동일한 Transformer에 입력하는 방식이다. 별도의 Vision Encoder 없이, 이미지 패치를 직접 토큰화하여 텍스트 토큰과 함께 처리한다. Cross-Attention이나 Projection 방식과 달리 **모달리티 간 양방향 상호작용**이 모든 레이어에서 일어난다.

---

## 핵심 용어

| 용어 | 정의 | 관련 모델 |
|------|------|-----------|
| **Early Fusion** | 이미지와 텍스트를 입력 단계에서 하나의 시퀀스로 결합하여 단일 모델로 처리하는 방식 | Fuyu, Unified-IO |
| **Token Concatenation** | Visual token과 text token을 하나의 시퀀스로 이어 붙이는 연산 | 전체 Early Fusion 계열 |
| **Unified Embedding** | 이미지와 텍스트를 동일한 임베딩 공간에 매핑하는 통합 임베딩 레이어 | Fuyu |
| **Interleaved Sequence** | 이미지와 텍스트 토큰이 교차 배치된 시퀀스 (여러 이미지 포함 가능) | Flamingo, Fuyu |
| **Patch Embedding** | 이미지 패치를 직접 선형 변환하여 토큰으로 만드는 레이어 (ViT 사전학습 없이) | Fuyu |

---

## 5.3.1 Early Fusion의 기본 원리

### 다른 방식과의 차이

```
Cross-Attention (5.1절):
  이미지 → Vision Encoder → cross-attn으로 LLM에 주입
  (이미지/텍스트가 별도 경로로 처리된 후 cross-attention으로 연결)

Projection (5.2절):
  이미지 → Vision Encoder → Projection → LLM 입력 앞에 붙임
  (Vision Encoder로 먼저 처리 후 LLM에 입력)

Early Fusion (5.3절):
  이미지 패치 + 텍스트 → 통합 시퀀스 → 단일 Transformer
  (처음부터 하나의 모델이 양쪽을 함께 처리)
```

### 아키텍처 개요

```
이미지 (H×W×3)
    ↓ 패치 분할 (P×P)
    ↓ 선형 임베딩
visual tokens: v_1, v_2, ..., v_m      (m = HW/P²)
                    ↓
    [v_1, ..., v_m, <sep>, t_1, ..., t_n]  ← 통합 시퀀스
                    ↓
    Transformer (Self-Attention × L layers)
                    ↓
              텍스트 출력 생성
```

핵심 포인트: **별도의 Vision Encoder(ViT)를 사전학습하지 않고**, 이미지 패치를 직접 선형 변환으로 토큰화한다.

---

## 수학적 원리

### 토큰 결합 (Token Concatenation)

이미지에서 $m$개의 패치 토큰, 텍스트에서 $n$개의 토큰을 추출하여 하나의 시퀀스로 결합한다.

$$
Z = [v_1, v_2, \ldots, v_m, \langle\text{sep}\rangle, t_1, t_2, \ldots, t_n]
$$

여기서:
- $v_i \in \mathbb{R}^D$: $i$번째 visual token (이미지 패치의 임베딩)
- $t_j \in \mathbb{R}^D$: $j$번째 text token
- $\langle\text{sep}\rangle \in \mathbb{R}^D$: 모달리티 구분 토큰
- 전체 시퀀스 길이: $L = m + 1 + n$

### Self-Attention으로 양방향 상호작용

결합된 시퀀스에 Self-Attention을 적용하면, 모든 토큰 쌍 사이에 attention이 계산된다.

$$
\text{Attention}(Z) = \text{softmax}\left(\frac{(ZW_Q)(ZW_K)^T}{\sqrt{d_k}}\right)(ZW_V)
$$

### Attention Matrix 구조

전체 attention 행렬 $A \in \mathbb{R}^{(m+n) \times (m+n)}$은 4개 블록으로 분해할 수 있다:

$$
A = \begin{bmatrix}
A_{\text{v→v}} & A_{\text{v→t}} \\
A_{\text{t→v}} & A_{\text{t→t}}
\end{bmatrix}
$$

| 블록 | 크기 | 의미 |
|------|------|------|
| $A_{\text{v→v}} \in \mathbb{R}^{m \times m}$ | visual-visual | 이미지 패치 간 공간적 관계 학습 |
| $A_{\text{v→t}} \in \mathbb{R}^{m \times n}$ | visual-text | 이미지가 텍스트 정보를 참조 |
| $A_{\text{t→v}} \in \mathbb{R}^{n \times m}$ | text-visual | 텍스트가 이미지 정보를 참조 (OCR에서 핵심) |
| $A_{\text{t→t}} \in \mathbb{R}^{n \times n}$ | text-text | 텍스트 내부 언어 모델링 |

Cross-Attention 방식에서는 $A_{\text{t→v}}$만 있지만, Early Fusion에서는 4개 블록 모두 동시에 학습된다. 이것이 **양방향 상호작용**의 핵심이다.

### 메모리 비용 분석

Self-Attention의 메모리 복잡도는 시퀀스 길이의 제곱에 비례한다.

$$
\text{Memory} = O\left((m + n)^2\right)
$$

구체적 예시:

| 설정 | $m$ (visual) | $n$ (text) | 총 길이 | Attention 행렬 크기 |
|------|-------------|-----------|---------|-------------------|
| 저해상도 (224px, P=16) | 196 | 256 | 452 | 204K |
| 중해상도 (336px, P=14) | 576 | 256 | 832 | 692K |
| 고해상도 (672px, P=14) | 2304 | 512 | 2816 | 7.9M |
| 초고해상도 (1344px, P=14) | 9216 | 512 | 9728 | 94.6M |

고해상도 문서 이미지에서는 visual token이 수천 개가 되므로, 메모리 비용이 급격히 증가한다. 이 때문에 **Flash Attention** 같은 효율적 attention 구현이 필수적이다.

### Causal Masking in Early Fusion

Autoregressive 생성 모델에서는 causal mask를 적용한다. Early Fusion에서의 causal mask 구조:

$$
M_{ij} = \begin{cases}
1 & \text{if } i \geq j \\
0 & \text{if } i < j
\end{cases}
$$

단, 이미지 패치는 모두 "과거"에 해당하므로 텍스트 토큰이 모든 이미지 패치에 attend할 수 있다:

$$
M = \begin{bmatrix}
\text{Full Attention (visual)} & 0 \\
\text{Full Attend to Visual} & \text{Causal (text)}
\end{bmatrix}
$$

---

## 5.3.2 Fuyu 아키텍처

### 설계 철학

Fuyu (Adept AI)의 핵심 설계 원칙:
1. **No Vision Encoder**: 사전학습된 ViT를 사용하지 않는다
2. **Native Resolution**: 이미지를 고정 크기로 resize하지 않고 원본 해상도 그대로 처리
3. **Simple Architecture**: 이미지 패치 → 선형 임베딩 → decoder-only Transformer

```
이미지 (임의 해상도)
    ↓ 패치 분할 (30×30)
    ↓ 선형 임베딩 (patch_dim → model_dim)
    ↓ 행 구분 토큰 <newline> 삽입
[row1_patches, <newline>, row2_patches, <newline>, ..., <image_end>, text_tokens]
    ↓
Decoder-only Transformer
    ↓
텍스트 출력
```

### 핵심 코드

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class FuyuPatchEmbedding(nn.Module):
    """
    Fuyu 스타일 패치 임베딩.
    이미지를 패치로 분할하고 선형 변환으로 토큰화한다.
    사전학습된 Vision Encoder를 사용하지 않는다.
    """
    def __init__(
        self,
        patch_size: int = 30,
        in_channels: int = 3,
        embed_dim: int = 4096,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 패치를 선형 변환으로 임베딩
        # Conv2d로 구현 (stride=patch_size로 비중첩 패치 추출)
        self.proj = nn.Linear(
            patch_size * patch_size * in_channels,
            embed_dim,
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Args:
            images: (batch, 3, H, W) - 임의 해상도

        Returns:
            patch_embeds: (batch, num_patches, embed_dim)
            h_patches: 세로 패치 수
            w_patches: 가로 패치 수
        """
        batch_size, channels, H, W = images.shape
        p = self.patch_size

        # 패치 크기에 맞게 패딩
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, (0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w

        h_patches = H // p
        w_patches = W // p

        # 패치 추출: (batch, 3, H, W) → (batch, h*w, patch_size²*3)
        patches = images.unfold(2, p, p).unfold(3, p, p)
        # (batch, 3, h_patches, w_patches, p, p)
        patches = patches.contiguous().view(
            batch_size, channels, h_patches, w_patches, p * p
        )
        patches = patches.permute(0, 2, 3, 1, 4).contiguous()
        # (batch, h_patches, w_patches, channels, p*p)
        patches = patches.view(batch_size, h_patches * w_patches, channels * p * p)
        # (batch, num_patches, patch_dim)

        # 선형 임베딩
        patch_embeds = self.proj(patches)

        return patch_embeds, h_patches, w_patches
```

---

## 5.3.3 Early Fusion VLM 전체 구현

```python
class EarlyFusionVLM(nn.Module):
    """
    Early Fusion 방식의 Vision-Language Model.

    이미지 패치를 직접 토큰화하여 텍스트 토큰과 함께
    단일 Transformer에 입력한다.

    특징:
    - 별도 Vision Encoder 없음
    - Native resolution 지원
    - 양방향 vision-language 상호작용
    - 단순한 아키텍처
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 4096,
        num_heads: int = 32,
        num_layers: int = 32,
        patch_size: int = 30,
        max_seq_len: int = 16384,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size

        # 이미지 패치 임베딩
        self.patch_embed = FuyuPatchEmbedding(
            patch_size=patch_size,
            embed_dim=dim,
        )

        # 텍스트 토큰 임베딩
        self.text_embed = nn.Embedding(vocab_size, dim)

        # 특수 토큰 임베딩
        self.newline_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.image_end_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Positional encoding (학습 가능)
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_mult, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.text_embed.weight

    def create_image_sequence(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        이미지를 패치 시퀀스로 변환.
        각 행 끝에 <newline> 토큰을 삽입한다.

        Args:
            images: (batch, 3, H, W)

        Returns:
            image_tokens: (batch, total_visual_len, dim)
        """
        batch_size = images.size(0)
        patch_embeds, h_patches, w_patches = self.patch_embed(images)
        # patch_embeds: (batch, h*w, dim)

        # 2D 그리드로 재배열
        patches_2d = patch_embeds.view(batch_size, h_patches, w_patches, self.dim)

        # 각 행 끝에 newline 토큰 삽입
        rows = []
        for row_idx in range(h_patches):
            row_patches = patches_2d[:, row_idx, :, :]  # (batch, w_patches, dim)
            newline = self.newline_embed.expand(batch_size, -1, -1)
            rows.append(row_patches)
            rows.append(newline)

        # image_end 토큰 추가
        image_end = self.image_end_embed.expand(batch_size, -1, -1)
        rows.append(image_end)

        image_tokens = torch.cat(rows, dim=1)
        return image_tokens

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            images: (batch, 3, H, W) or None
            input_ids: (batch, seq_len) - 텍스트 토큰 ID
            labels: (batch, seq_len) - 학습 정답

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        device = input_ids.device if input_ids is not None else images.device
        embeds_list = []

        # 이미지 토큰
        if images is not None:
            image_tokens = self.create_image_sequence(images)
            embeds_list.append(image_tokens)

        # 텍스트 토큰
        if input_ids is not None:
            text_tokens = self.text_embed(input_ids)
            embeds_list.append(text_tokens)

        # 시퀀스 결합
        hidden = torch.cat(embeds_list, dim=1)
        seq_len = hidden.size(1)

        # Positional encoding
        positions = torch.arange(seq_len, device=device)
        hidden = hidden + self.pos_embed(positions)

        # Causal mask (visual tokens에는 full attention, text에는 causal)
        causal_mask = self._create_early_fusion_mask(
            num_visual=embeds_list[0].size(1) if images is not None else 0,
            total_len=seq_len,
            device=device,
        )

        # Transformer layers
        for layer in self.layers:
            hidden = layer(hidden, mask=causal_mask)

        hidden = self.norm(hidden)
        logits = self.head(hidden)

        result = {"logits": logits}

        # Loss 계산 (텍스트 부분만)
        if labels is not None:
            num_visual = embeds_list[0].size(1) if images is not None else 0
            # Visual token 부분의 logits는 loss에서 제외
            text_logits = logits[:, num_visual:, :]
            shift_logits = text_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    def _create_early_fusion_mask(
        self,
        num_visual: int,
        total_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Early Fusion용 attention mask.

        구조:
        - Visual tokens: 서로 bidirectional attention (full)
        - Text tokens: visual tokens에 대해 full attend
        - Text tokens: 다른 text tokens에 대해 causal
        """
        mask = torch.zeros(total_len, total_len, device=device)

        # Visual-Visual: full attention
        mask[:num_visual, :num_visual] = 1.0

        # Text-Visual: full attend (text가 모든 visual에 attend)
        mask[num_visual:, :num_visual] = 1.0

        # Text-Text: causal mask
        num_text = total_len - num_visual
        causal = torch.tril(torch.ones(num_text, num_text, device=device))
        mask[num_visual:, num_visual:] = causal

        # Visual-Text: visual은 text를 보지 못함 (prefix로 동작)
        # mask[:num_visual, num_visual:] = 0  (이미 0)

        # 0 → -inf, 1 → 0 변환 (additive mask)
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)

        return mask


class TransformerBlock(nn.Module):
    """Transformer Decoder Block (Pre-Norm)"""
    def __init__(
        self,
        dim: int = 4096,
        num_heads: int = 32,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-Attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=mask)
        x = residual + x

        # FFN
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x
```

---

## 5.3.4 Interleaved Image-Text 처리

### 여러 이미지 포함 시퀀스

실제 문서에서는 여러 이미지와 텍스트가 교차된다. Early Fusion은 이를 자연스럽게 처리한다.

```python
class InterleavedEarlyFusion(nn.Module):
    """
    여러 이미지와 텍스트가 교차된 시퀀스를 처리하는 Early Fusion 모듈.

    예시 입력:
    [img1_patches, <sep>, "이 그래프는", img2_patches, <sep>, "를 보여준다"]
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 4096,
        patch_size: int = 30,
    ):
        super().__init__()
        self.dim = dim
        self.patch_embed = FuyuPatchEmbedding(patch_size=patch_size, embed_dim=dim)
        self.text_embed = nn.Embedding(vocab_size, dim)
        self.sep_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # 모달리티 타입 임베딩
        self.modality_embed = nn.Embedding(2, dim)  # 0: visual, 1: text

    def build_interleaved_sequence(
        self,
        images: list[torch.Tensor],
        text_segments: list[torch.Tensor],
        interleave_order: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        교차 시퀀스 구성.

        Args:
            images: 이미지 리스트 [(3, H1, W1), (3, H2, W2), ...]
            text_segments: 텍스트 토큰 리스트 [(seq1,), (seq2,), ...]
            interleave_order: 순서 지정 ["image", "text", "image", "text"]

        Returns:
            sequence: (1, total_len, dim)
            modality_ids: (1, total_len) - 모달리티 식별자
        """
        embeds = []
        modality_ids = []
        img_idx = 0
        text_idx = 0

        for item_type in interleave_order:
            if item_type == "image" and img_idx < len(images):
                img = images[img_idx].unsqueeze(0)  # (1, 3, H, W)
                img_tokens, _, _ = self.patch_embed(img)
                embeds.append(img_tokens)
                modality_ids.extend([0] * img_tokens.size(1))

                # 구분 토큰
                embeds.append(self.sep_token)
                modality_ids.append(0)
                img_idx += 1

            elif item_type == "text" and text_idx < len(text_segments):
                text_ids = text_segments[text_idx].unsqueeze(0)  # (1, seq_len)
                text_tokens = self.text_embed(text_ids)
                embeds.append(text_tokens)
                modality_ids.extend([1] * text_tokens.size(1))
                text_idx += 1

        sequence = torch.cat(embeds, dim=1)
        modality_ids_tensor = torch.tensor(
            modality_ids, device=sequence.device
        ).unsqueeze(0)

        # 모달리티 임베딩 추가
        mod_embeds = self.modality_embed(modality_ids_tensor)
        sequence = sequence + mod_embeds

        return sequence, modality_ids_tensor
```

---

## 5.3.5 메모리 최적화 기법

### Flash Attention 통합

Early Fusion의 가장 큰 문제는 긴 시퀀스로 인한 메모리 사용량이다. Flash Attention으로 $O(n^2)$ → $O(n)$ 메모리로 줄일 수 있다.

```python
class FlashAttentionBlock(nn.Module):
    """
    Flash Attention을 활용한 메모리 효율적 Transformer Block.
    PyTorch 2.0+의 scaled_dot_product_attention 사용.
    """
    def __init__(
        self,
        dim: int = 4096,
        num_heads: int = 32,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        residual = x

        # Pre-Norm
        x = self.norm1(x)

        # QKV 계산
        qkv = self.qkv(x).view(
            batch_size, seq_len, 3, self.num_heads, self.dim_head
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (batch, heads, seq, dim_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # PyTorch 2.0+ Flash Attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=0.0,
        )

        # Concat heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, dim)
        attn_output = self.out_proj(attn_output)

        x = residual + attn_output

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x
```

### 메모리 사용량 비교

```python
def compare_memory_usage():
    """Standard Attention vs Flash Attention 메모리 비교"""
    import gc

    results = {}

    for seq_len in [1024, 2048, 4096, 8192]:
        for attn_type in ["standard", "flash"]:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

            dim = 4096
            num_heads = 32
            batch_size = 1

            x = torch.randn(batch_size, seq_len, dim)

            if attn_type == "standard":
                block = TransformerBlock(dim, num_heads)
            else:
                block = FlashAttentionBlock(dim, num_heads)

            if torch.cuda.is_available():
                x = x.cuda()
                block = block.cuda()

                torch.cuda.reset_peak_memory_stats()
                _ = block(x)
                peak_mem = torch.cuda.max_memory_allocated() / 1e6
                results[f"{attn_type}_{seq_len}"] = peak_mem
                print(f"{attn_type:>8} | seq_len={seq_len:>5} | Peak Memory: {peak_mem:.1f}MB")
            else:
                print(f"{attn_type:>8} | seq_len={seq_len:>5} | (CPU - memory tracking not available)")

    return results
```

---

## 5.3.6 Early Fusion의 장단점

### 장점

| 장점 | 설명 |
|------|------|
| **아키텍처 단순성** | Vision Encoder 불필요, 단일 모델로 처리 |
| **양방향 상호작용** | 모든 레이어에서 vision↔text 양방향 attention |
| **Native Resolution** | 이미지 크기를 고정하지 않고 원본 그대로 처리 가능 |
| **학습 단순성** | 별도 사전학습 단계 없이 end-to-end 학습 |
| **Interleaved 지원** | 여러 이미지와 텍스트 교차 배치 자연스럽게 처리 |

### 단점

| 단점 | 설명 |
|------|------|
| **메모리 비용** | $O((m+n)^2)$로, visual token이 많으면 폭발적 증가 |
| **사전학습 활용 불가** | CLIP, DINOv2 같은 강력한 vision encoder 재사용 못함 |
| **학습 데이터 요구량** | Vision Encoder 없으므로 대량의 이미지-텍스트 데이터 필요 |
| **추론 속도** | 긴 시퀀스로 인해 autoregressive 생성이 느림 |

### OCR에서의 고려사항

```python
def estimate_early_fusion_cost_for_ocr():
    """OCR 문서 이미지에서의 Early Fusion 비용 추정"""
    patch_size = 30

    # 일반적인 문서 이미지 해상도들
    resolutions = [
        ("A4 300dpi", 2480, 3508),
        ("A4 150dpi", 1240, 1754),
        ("처방전 사진", 1080, 1440),
        ("명함 스캔", 600, 1000),
    ]

    print(f"{'문서 타입':<16} {'해상도':<12} {'패치 수':>8} {'시퀀스 길이':>10} {'Attn 행렬 (MB)':>14}")
    print("-" * 65)

    for name, h, w in resolutions:
        h_patches = math.ceil(h / patch_size)
        w_patches = math.ceil(w / patch_size)
        num_patches = h_patches * w_patches
        # newline 토큰 + image_end 토큰
        num_visual = num_patches + h_patches + 1
        # 텍스트 (OCR 결과) 약 512 토큰
        num_text = 512
        total = num_visual + num_text

        # Attention 행렬 크기 (float16)
        attn_size_mb = total * total * 2 / 1e6  # float16 = 2 bytes

        print(f"{name:<16} {h}x{w:<8} {num_patches:>8} {total:>10} {attn_size_mb:>14.1f}")


if __name__ == "__main__":
    estimate_early_fusion_cost_for_ocr()
```

출력 예시:
```
문서 타입          해상도        패치 수    시퀀스 길이   Attn 행렬 (MB)
-----------------------------------------------------------------
A4 300dpi        2480x3508      9672      10268           211.0
A4 150dpi        1240x1754      2436       2990            17.9
처방전 사진       1080x1440      1728       2278            10.4
명함 스캔         600x1000        680       1213             2.9
```

고해상도 문서는 Early Fusion에서 메모리 부담이 크다. 이 때문에 실무에서는 Projection 방식이나 토큰 압축 기법을 함께 사용하는 경우가 많다.

---

## 5.3.7 개선된 Early Fusion: 하이브리드 접근

### Vision Encoder + Early Fusion

순수 Early Fusion의 학습 효율성 문제를 해결하기 위해, Vision Encoder로 먼저 처리한 후 Early Fusion을 적용하는 하이브리드 방식도 있다.

```python
class HybridEarlyFusion(nn.Module):
    """
    하이브리드 Early Fusion.

    1단계: Vision Encoder로 이미지 feature 추출 (토큰 수 감소)
    2단계: 추출된 visual token과 text token을 결합하여 Early Fusion

    순수 Early Fusion 대비:
    - Vision Encoder의 사전학습 지식 활용 가능
    - Visual token 수 감소로 메모리 절약
    - 양방향 상호작용은 유지
    """
    def __init__(
        self,
        vision_encoder: nn.Module,
        dim_vision: int = 1024,
        dim: int = 4096,
        vocab_size: int = 32000,
        num_heads: int = 32,
        num_layers: int = 32,
        freeze_vision: bool = True,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.vision_proj = nn.Linear(dim_vision, dim)
        self.text_embed = nn.Embedding(vocab_size, dim)

        # 모달리티 임베딩 (visual=0, text=1)
        self.modality_embed = nn.Embedding(2, dim)

        self.layers = nn.ModuleList([
            FlashAttentionBlock(dim, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> dict:
        # Vision Encoder (frozen)
        with torch.no_grad():
            visual_features = self.vision_encoder(images)

        # Projection
        visual_tokens = self.vision_proj(visual_features)
        text_tokens = self.text_embed(input_ids)

        # 모달리티 임베딩
        batch_size = images.size(0)
        num_visual = visual_tokens.size(1)
        num_text = text_tokens.size(1)

        vis_mod = self.modality_embed(
            torch.zeros(batch_size, num_visual, dtype=torch.long, device=images.device)
        )
        txt_mod = self.modality_embed(
            torch.ones(batch_size, num_text, dtype=torch.long, device=images.device)
        )

        visual_tokens = visual_tokens + vis_mod
        text_tokens = text_tokens + txt_mod

        # Early Fusion: 결합
        hidden = torch.cat([visual_tokens, text_tokens], dim=1)

        # Transformer layers (모든 토큰이 서로 attend)
        for layer in self.layers:
            hidden = layer(hidden, is_causal=False)

        hidden = self.norm(hidden)
        logits = self.head(hidden)

        return {"logits": logits}
```

---

## 5.3.8 실습: Early Fusion 동작 확인

```python
def test_early_fusion():
    """Early Fusion VLM 기본 동작 테스트"""
    torch.manual_seed(42)

    # 설정 (작은 모델)
    model = EarlyFusionVLM(
        vocab_size=1000,
        dim=256,
        num_heads=8,
        num_layers=4,
        patch_size=16,
        max_seq_len=4096,
        ff_mult=4,
    )

    # 더미 데이터
    batch_size = 2
    images = torch.randn(batch_size, 3, 64, 64)  # 작은 이미지
    input_ids = torch.randint(0, 1000, (batch_size, 20))
    labels = torch.randint(0, 1000, (batch_size, 20))

    # Forward
    result = model(images=images, input_ids=input_ids, labels=labels)

    print(f"Input image:  {images.shape}")
    print(f"Input text:   {input_ids.shape}")
    print(f"Logits shape: {result['logits'].shape}")
    print(f"Loss:         {result['loss'].item():.4f}")

    # 패치 수 계산
    num_patches = (64 // 16) * (64 // 16)  # 4x4 = 16
    num_newlines = 64 // 16  # 4
    num_special = 1  # image_end
    total_visual = num_patches + num_newlines + num_special
    print(f"\nVisual tokens: {total_visual} (patches={num_patches}, newlines={num_newlines})")
    print(f"Text tokens:   {input_ids.size(1)}")
    print(f"Total seq len: {result['logits'].size(1)}")

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {total_params:,}")

    # Attention matrix 크기 추정
    total_seq = total_visual + input_ids.size(1)
    attn_elements = total_seq * total_seq
    print(f"Attention matrix: {total_seq}x{total_seq} = {attn_elements:,} elements")


if __name__ == "__main__":
    test_early_fusion()
```

---

## 5.3.9 Attention 패턴 분석

```python
def analyze_attention_blocks(
    attn_weights: torch.Tensor,
    num_visual: int,
    num_text: int,
):
    """
    Early Fusion attention 행렬의 4개 블록 분석.

    Args:
        attn_weights: (total, total) - attention 가중치
        num_visual: visual token 수
        num_text: text token 수
    """
    import matplotlib.pyplot as plt

    # 4개 블록 추출
    v2v = attn_weights[:num_visual, :num_visual]           # visual→visual
    v2t = attn_weights[:num_visual, num_visual:]           # visual→text
    t2v = attn_weights[num_visual:, :num_visual]           # text→visual
    t2t = attn_weights[num_visual:, num_visual:]           # text→text

    blocks = {
        "Visual→Visual": v2v,
        "Visual→Text": v2t,
        "Text→Visual": t2v,
        "Text→Text": t2t,
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for idx, (name, block) in enumerate(blocks.items()):
        ax = axes[idx]
        data = block.detach().cpu().numpy()
        im = ax.imshow(data, cmap='Blues', aspect='auto')
        ax.set_title(name)
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig("early_fusion_attention_blocks.png", dpi=150)
    plt.show()

    # 각 블록의 평균 attention 값
    print("\nAttention 블록별 평균값:")
    for name, block in blocks.items():
        mean_val = block.mean().item()
        max_val = block.max().item()
        print(f"  {name}: mean={mean_val:.4f}, max={max_val:.4f}")
```

---

## 용어 체크리스트

학습 후 아래 질문에 답할 수 있는지 확인하라.

| 체크 | 용어 | 자가 점검 질문 |
|------|------|----------------|
| ☐ | Early Fusion | Cross-Attention / Projection 방식과 비교하여 Early Fusion의 정보 흐름 차이를 그릴 수 있는가? |
| ☐ | Token Concatenation | visual token과 text token을 결합할 때 separator 토큰이 왜 필요한가? |
| ☐ | Unified Embedding | 이미지 패치와 텍스트를 같은 차원으로 임베딩하는 것의 의미는? |
| ☐ | Attention Matrix 4블록 | $A_{\text{v→v}}, A_{\text{v→t}}, A_{\text{t→v}}, A_{\text{t→t}}$ 각각의 역할을 설명할 수 있는가? |
| ☐ | 메모리 복잡도 | $O((m+n)^2)$에서 $m$이 클 때 어떤 문제가 생기는가? Flash Attention은 어떻게 해결하는가? |
| ☐ | Causal Masking | Early Fusion에서 visual token에 full attention, text token에 causal mask를 적용하는 이유는? |
| ☐ | Native Resolution | 고정 크기 resize 대비 원본 해상도 처리의 장단점은? OCR에서 어떤 이점이 있는가? |
| ☐ | Interleaved Sequence | 여러 이미지와 텍스트가 교차된 시퀀스를 Early Fusion이 자연스럽게 처리할 수 있는 이유는? |

---

## 다음 단계

[5.4 패턴 비교 분석](04-pattern-comparison.md)에서 Cross-Attention, Projection, Early Fusion 세 방식을 종합 비교하고, OCR 태스크에 최적인 패턴을 분석한다.
