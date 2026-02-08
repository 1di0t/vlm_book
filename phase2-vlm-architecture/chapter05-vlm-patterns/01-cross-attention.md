# 5.1 Cross-Attention 방식

Vision-Language Model에서 Cross-Attention은 텍스트 토큰이 이미지 피처를 "질의"하여 시각 정보를 흡수하는 핵심 메커니즘이다. LLM의 각 레이어에 cross-attention 모듈을 삽입하거나, 별도의 어댑터 모듈(Q-Former)로 시각 정보를 압축해서 LLM에 전달한다.

---

## 핵심 용어

| 용어 | 정의 | 관련 모델 |
|------|------|-----------|
| **Cross-Attention** | Query가 한 모달리티, Key·Value가 다른 모달리티에서 오는 attention | Transformer Decoder |
| **Perceiver** | 고정 길이 latent array로 임의 길이 입력을 압축하는 아키텍처 | Perceiver IO |
| **Q-Former** | 학습 가능한 query 벡터로 visual feature를 압축하는 BLIP-2의 핵심 모듈 | BLIP-2 |
| **Gated Cross-Attention** | tanh 게이트로 cross-attention 출력을 조절하여 LLM을 안정적으로 학습하는 기법 | Flamingo |
| **Frozen Backbone** | 사전학습된 vision encoder/LLM 가중치를 고정하고 어댑터만 학습하는 전략 | Flamingo, BLIP-2 |

---

## 5.1.1 Cross-Attention의 기본 원리

### Self-Attention과의 차이

Self-Attention에서는 Q, K, V가 모두 같은 시퀀스에서 나온다. Cross-Attention에서는 Q가 한쪽(텍스트), K·V가 다른 쪽(이미지)에서 온다.

```
Self-Attention:   Q, K, V ← 같은 시퀀스 X
Cross-Attention:  Q ← 텍스트 H_text,  K, V ← 이미지 Z_vision
```

### 동작 흐름

```
이미지 → Vision Encoder → Z_vision (N_v × D_v)
                                ↓ K, V
텍스트 → LLM Layer → H_text ─→ Cross-Attention → 시각 정보가 주입된 텍스트 표현
                       (Q)          ↓
                              다음 LLM Layer
```

---

## 수학적 원리

### 기본 Cross-Attention 수식

텍스트 hidden state $H \in \mathbb{R}^{N_t \times D}$를 Query로, 이미지 피처 $Z \in \mathbb{R}^{N_v \times D_v}$를 Key·Value로 사용한다.

$$
Q = H W_Q, \quad K = Z W_K, \quad V = Z W_V
$$

$$
\text{CrossAttn}(Q_{\text{text}}, K_{\text{img}}, V_{\text{img}}) = \text{softmax}\left(\frac{Q_{\text{text}} K_{\text{img}}^T}{\sqrt{d_k}}\right) V_{\text{img}}
$$

여기서:
- $W_Q \in \mathbb{R}^{D \times d_k}$: 텍스트 Query 프로젝션
- $W_K \in \mathbb{R}^{D_v \times d_k}$: 비전 Key 프로젝션
- $W_V \in \mathbb{R}^{D_v \times d_v}$: 비전 Value 프로젝션
- $d_k$: Key 차원 (scaling factor)

Attention 행렬 $A \in \mathbb{R}^{N_t \times N_v}$로, 각 텍스트 토큰이 어떤 이미지 패치에 주목하는지 나타낸다.

### Q-Former (BLIP-2)

BLIP-2의 핵심 아이디어는 학습 가능한 query 벡터 $Q_{\text{learn}} \in \mathbb{R}^{M \times D}$를 도입하여, 수백 개의 visual token을 $M$개(보통 32)의 압축된 표현으로 변환하는 것이다.

$$
Q_{\text{learn}} \in \mathbb{R}^{M \times D}, \quad M \ll N_v
$$

$$
Z_{\text{compressed}} = \text{CrossAttn}(Q_{\text{learn}}, K_{\text{img}}, V_{\text{img}}) \in \mathbb{R}^{M \times D}
$$

Q-Former는 BERT 기반 아키텍처로, self-attention과 cross-attention을 교차하여 수행한다:

$$
\hat{Q} = \text{SelfAttn}(Q_{\text{learn}}) + Q_{\text{learn}}
$$

$$
Z_{\text{out}} = \text{CrossAttn}(\hat{Q}, Z_{\text{vision}}, Z_{\text{vision}}) + \hat{Q}
$$

이렇게 하면 Vision Encoder의 출력 토큰 수 $N_v$(예: ViT-G의 257개)를 $M=32$개로 압축해서, LLM이 처리할 시퀀스 길이를 대폭 줄인다. 정보 압축 비율은 $N_v / M$이다.

### Gated Cross-Attention (Flamingo)

Flamingo는 기존 LLM에 cross-attention을 삽입할 때 학습 초기의 불안정을 막기 위해 tanh 게이트를 사용한다.

$$
h = h + \tanh(\alpha) \cdot \text{CrossAttn}(h, v, v)
$$

여기서:
- $h \in \mathbb{R}^{N_t \times D}$: LLM의 hidden state
- $v \in \mathbb{R}^{N_v \times D_v}$: vision features
- $\alpha \in \mathbb{R}$: 학습 가능한 스칼라 파라미터, **초기값 0**

$\alpha = 0$일 때 $\tanh(0) = 0$이므로 cross-attention 출력이 완전히 차단된다. 학습이 진행되면서 $\alpha$가 점진적으로 커져 시각 정보를 서서히 주입한다. 이 전략 덕분에 **사전학습된 LLM의 언어 능력을 보존**하면서 시각 정보를 안전하게 통합할 수 있다.

### 파라미터 효율성 분석

Cross-Attention 방식의 핵심 장점은 파라미터 효율성이다.

| 컴포넌트 | 파라미터 상태 | 비고 |
|----------|-------------|------|
| Vision Encoder | **Frozen** | ViT-G 등 대형 모델 재사용 |
| LLM | **Frozen** | Chinchilla 등 기존 모델 유지 |
| Cross-Attention Layer | **Trainable** | 새로 삽입된 레이어만 학습 |
| Gate Parameter $\alpha$ | **Trainable** | 레이어당 스칼라 1개 |

학습 파라미터 비율:

$$
\text{Trainable Ratio} = \frac{|\theta_{\text{xattn}}|}{|\theta_{\text{vision}}| + |\theta_{\text{LLM}}| + |\theta_{\text{xattn}}|}
$$

Flamingo의 경우 전체 파라미터의 약 2~5%만 학습한다.

---

## 5.1.2 Flamingo 아키텍처

### 구조 개요

```
이미지 → Vision Encoder (Frozen) → Perceiver Resampler → visual tokens (64개)
                                                              ↓
텍스트 → LLM Layer 1 → Gated XAttn → LLM Layer 2 → Gated XAttn → ... → 출력
         (Frozen)     (Trainable)   (Frozen)      (Trainable)
```

Flamingo의 핵심 설계:
1. **Perceiver Resampler**: 가변 길이 visual feature를 고정 길이(64)로 압축
2. **Gated Cross-Attention Dense**: LLM의 매 레이어마다 삽입
3. **Interleaved Image-Text**: 여러 이미지와 텍스트를 교차 배치하여 few-shot 학습 지원

### Perceiver Resampler

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PerceiverResampler(nn.Module):
    """
    Flamingo의 Perceiver Resampler.
    가변 길이 visual feature를 고정 길이 latent로 압축한다.
    """
    def __init__(
        self,
        dim: int = 1024,
        num_latents: int = 64,
        num_heads: int = 16,
        num_layers: int = 6,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                # Cross-Attention: latent(Q) attends to visual features(K,V)
                nn.MultiheadAttention(dim, num_heads, batch_first=True),
                nn.LayerNorm(dim),
                # Feed-Forward
                nn.Sequential(
                    nn.Linear(dim, dim * ff_mult),
                    nn.GELU(),
                    nn.Linear(dim * ff_mult, dim),
                ),
                nn.LayerNorm(dim),
            ]))

        self.norm_out = nn.LayerNorm(dim)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (batch, N_v, dim) - Vision Encoder 출력

        Returns:
            latents: (batch, num_latents, dim) - 압축된 visual tokens
        """
        batch_size = visual_features.size(0)
        # latent를 batch 차원으로 확장
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        for xattn, norm1, ff, norm2 in self.layers:
            # Cross-Attention: latent이 visual features에 attend
            latents_normed = norm1(latents)
            xattn_out, _ = xattn(
                query=latents_normed,
                key=visual_features,
                value=visual_features,
            )
            latents = latents + xattn_out

            # Feed-Forward
            latents = latents + ff(norm2(latents))

        return self.norm_out(latents)
```

---

## 5.1.3 Gated Cross-Attention 구현

```python
class GatedCrossAttention(nn.Module):
    """
    Flamingo 스타일 Gated Cross-Attention.

    기존 LLM 레이어 사이에 삽입되어 visual 정보를 주입한다.
    tanh(alpha) 게이트로 학습 초기에는 영향을 0으로 만들어
    사전학습된 LLM의 성능을 보존한다.
    """
    def __init__(
        self,
        dim: int = 4096,
        dim_visual: int = 1024,
        num_heads: int = 32,
        dim_head: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        inner_dim = num_heads * dim_head

        # Query는 텍스트 hidden state에서
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        # Key, Value는 visual features에서
        self.to_k = nn.Linear(dim_visual, inner_dim, bias=False)
        self.to_v = nn.Linear(dim_visual, inner_dim, bias=False)
        # Output projection
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # Layer norms
        self.norm_text = nn.LayerNorm(dim)
        self.norm_visual = nn.LayerNorm(dim_visual)

        # Gating parameter: 초기값 0 → tanh(0) = 0
        self.gate = nn.Parameter(torch.zeros(1))

        self.scale = dim_head ** -0.5

    def forward(
        self,
        text_hidden: torch.Tensor,
        visual_features: torch.Tensor,
        visual_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            text_hidden: (batch, seq_len, dim) - LLM hidden state
            visual_features: (batch, num_visual, dim_visual) - 시각 특징
            visual_mask: (batch, num_visual) - 유효한 visual token 마스크

        Returns:
            output: (batch, seq_len, dim) - 시각 정보가 주입된 hidden state
        """
        batch_size, seq_len, _ = text_hidden.shape

        # Pre-LayerNorm
        text_normed = self.norm_text(text_hidden)
        visual_normed = self.norm_visual(visual_features)

        # Q, K, V 계산
        q = self.to_q(text_normed)  # (batch, seq_len, inner_dim)
        k = self.to_k(visual_normed)  # (batch, num_visual, inner_dim)
        v = self.to_v(visual_normed)

        # Multi-head 분리: (batch, heads, seq, dim_head)
        q = q.view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Visual mask 적용
        if visual_mask is not None:
            # (batch, 1, 1, num_visual)로 확장
            mask = visual_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~mask.bool(), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Head 합치기
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        xattn_out = self.to_out(attn_output)

        # Gated residual: h = h + tanh(α) · CrossAttn(h, v)
        output = text_hidden + torch.tanh(self.gate) * xattn_out

        return output
```

### 게이트 동작 분석

```python
def analyze_gate_behavior():
    """Gate parameter의 학습 과정 시뮬레이션"""
    import matplotlib.pyplot as plt

    alpha_values = torch.linspace(-3, 3, 100)
    gate_values = torch.tanh(alpha_values)

    # 학습 초기: alpha ≈ 0 → gate ≈ 0 (시각 정보 차단)
    # 학습 중기: alpha → 1 → gate ≈ 0.76 (시각 정보 부분 주입)
    # 학습 후기: alpha → 2 → gate ≈ 0.96 (시각 정보 거의 전부 주입)

    print(f"alpha=0.0 → gate={torch.tanh(torch.tensor(0.0)):.4f}")
    print(f"alpha=0.5 → gate={torch.tanh(torch.tensor(0.5)):.4f}")
    print(f"alpha=1.0 → gate={torch.tanh(torch.tensor(1.0)):.4f}")
    print(f"alpha=2.0 → gate={torch.tanh(torch.tensor(2.0)):.4f}")
```

---

## 5.1.4 Q-Former (BLIP-2) 구현

```python
class QFormerLayer(nn.Module):
    """
    Q-Former의 단일 레이어.
    Self-Attention → Cross-Attention → FFN 구조.
    """
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        dim_visual: int = 1408,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Self-Attention over queries
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_self = nn.LayerNorm(dim)

        # Cross-Attention: queries attend to visual features
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
            kdim=dim_visual, vdim=dim_visual,
        )
        self.norm_cross = nn.LayerNorm(dim)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(
        self,
        queries: torch.Tensor,
        visual_features: torch.Tensor,
        query_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: (batch, M, dim) - 학습 가능한 query tokens
            visual_features: (batch, N_v, dim_visual) - vision encoder 출력
            query_mask: (M, M) - self-attention mask (optional)

        Returns:
            queries: (batch, M, dim) - 업데이트된 query tokens
        """
        # Self-Attention
        residual = queries
        queries = self.norm_self(queries)
        queries, _ = self.self_attn(
            queries, queries, queries, attn_mask=query_mask
        )
        queries = residual + queries

        # Cross-Attention
        residual = queries
        queries = self.norm_cross(queries)
        queries, _ = self.cross_attn(
            query=queries,
            key=visual_features,
            value=visual_features,
        )
        queries = residual + queries

        # FFN
        residual = queries
        queries = residual + self.ffn(self.norm_ffn(queries))

        return queries


class QFormer(nn.Module):
    """
    BLIP-2 Q-Former.

    학습 가능한 query 벡터 M개로 Vision Encoder의 출력(N_v개 토큰)을
    M개의 압축된 표현으로 변환한다. M << N_v이므로 LLM의 입력 길이를
    대폭 줄이면서 핵심 시각 정보를 보존한다.
    """
    def __init__(
        self,
        num_queries: int = 32,
        dim: int = 768,
        dim_visual: int = 1408,
        num_heads: int = 12,
        num_layers: int = 12,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 학습 가능한 query embeddings: Q_learn ∈ R^{M × D}
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, dim) * 0.02
        )

        # Q-Former layers
        self.layers = nn.ModuleList([
            QFormerLayer(dim, num_heads, dim_visual, ff_mult, dropout)
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(dim)

        # LLM projection: Q-Former dim → LLM dim
        self.llm_proj = nn.Linear(dim, 4096)  # 예: LLaMA-7B dim

    def forward(
        self,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: (batch, N_v, dim_visual)
                예: ViT-G 출력 (batch, 257, 1408)

        Returns:
            query_output: (batch, M, D_llm)
                압축된 visual representation, LLM에 입력 가능
        """
        batch_size = visual_features.size(0)

        # Query tokens를 batch 차원으로 확장
        queries = self.query_tokens.expand(batch_size, -1, -1)

        # Q-Former layers 통과
        for layer in self.layers:
            queries = layer(queries, visual_features)

        queries = self.norm_out(queries)

        # LLM 차원으로 프로젝션
        query_output = self.llm_proj(queries)

        return query_output
```

### 사용 예시

```python
def demo_qformer():
    """Q-Former 동작 데모"""
    # Vision Encoder 출력 시뮬레이션 (ViT-G)
    batch_size = 2
    num_patches = 257  # 16x16 patches + CLS
    dim_visual = 1408  # ViT-G 차원

    visual_features = torch.randn(batch_size, num_patches, dim_visual)

    # Q-Former 생성
    qformer = QFormer(
        num_queries=32,      # 32개 query로 압축
        dim=768,             # Q-Former 내부 차원
        dim_visual=1408,     # ViT-G 출력 차원
        num_layers=12,
        num_heads=12,
    )

    # Forward
    query_output = qformer(visual_features)
    print(f"Input:  {visual_features.shape}")   # (2, 257, 1408)
    print(f"Output: {query_output.shape}")       # (2, 32, 4096)
    print(f"압축 비율: {num_patches}→32 ({num_patches/32:.1f}x)")

    # 파라미터 수 확인
    total_params = sum(p.numel() for p in qformer.parameters())
    trainable_params = sum(p.numel() for p in qformer.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")


if __name__ == "__main__":
    demo_qformer()
```

---

## 5.1.5 BLIP-2 전체 파이프라인

### 2단계 학습 전략

**Stage 1: Vision-Language Representation Learning**

```
이미지 → ViT-G (Frozen) → visual features → Q-Former → query output
                                                          ↓
                                              3가지 학습 목표:
                                              1. Image-Text Contrastive (ITC)
                                              2. Image-grounded Text Generation (ITG)
                                              3. Image-Text Matching (ITM)
```

**Stage 2: Vision-to-Language Generative Learning**

```
이미지 → ViT-G (Frozen) → visual features → Q-Former (Stage 1 초기화)
                                                  ↓ projection
                                              LLM (Frozen)
                                                  ↓
                                              텍스트 생성
```

### Stage 2 코드

```python
class BLIP2(nn.Module):
    """
    BLIP-2 Stage 2: Q-Former + Frozen LLM.
    간략화한 구현으로 핵심 구조를 보여준다.
    """
    def __init__(
        self,
        vision_encoder: nn.Module,
        qformer: QFormer,
        llm: nn.Module,
        freeze_vision: bool = True,
        freeze_llm: bool = True,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.qformer = qformer
        self.llm = llm

        # Vision Encoder freeze
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # LLM freeze
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            images: (batch, 3, H, W)
            input_ids: (batch, seq_len) - 텍스트 토큰 ID
            attention_mask: (batch, seq_len)
        """
        # 1. Vision Encoder (frozen)
        with torch.no_grad():
            visual_features = self.vision_encoder(images)
            # (batch, N_v, D_vision)

        # 2. Q-Former로 visual token 압축 (trainable)
        query_output = self.qformer(visual_features)
        # (batch, M, D_llm)

        # 3. LLM에 visual tokens + text tokens 전달
        # query_output을 text embedding 앞에 prepend
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        # (batch, seq_len, D_llm)

        inputs_embeds = torch.cat([query_output, text_embeds], dim=1)
        # (batch, M + seq_len, D_llm)

        # Attention mask 확장
        if attention_mask is not None:
            visual_mask = torch.ones(
                query_output.size(0), query_output.size(1),
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        # 4. LLM forward (frozen, query_output의 gradient만 흐름)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        return outputs
```

---

## 5.1.6 Flamingo vs BLIP-2 비교

| 측면 | Flamingo | BLIP-2 |
|------|----------|--------|
| **Cross-Attention 위치** | LLM 매 레이어에 삽입 | 별도 Q-Former 모듈 |
| **Visual Token 수** | 64 (Perceiver) | 32 (Q-Former) |
| **게이팅** | tanh(α) 게이트 | 없음 |
| **LLM 수정** | 레이어 사이에 모듈 추가 | 입력 앞에 prepend만 |
| **학습 데이터** | 대규모 이미지-텍스트 쌍 | 2단계 학습 |
| **Few-shot** | interleaved로 자연스럽게 지원 | prompt 방식 |
| **Vision Encoder** | NFNet (Frozen) | ViT-G (Frozen) |
| **LLM** | Chinchilla (Frozen) | OPT/FlanT5 (Frozen) |

### 정보 흐름 차이

**Flamingo**: 시각 정보가 LLM의 **모든 레이어**에서 직접 주입

$$
h_l = \text{LLMLayer}_l(h_{l-1}) + \tanh(\alpha_l) \cdot \text{CrossAttn}_l(h_{l-1}, v)
$$

**BLIP-2**: 시각 정보가 LLM의 **입력 단계**에서만 주입

$$
\text{input} = [z_1^{\text{visual}}, \ldots, z_M^{\text{visual}}, t_1, \ldots, t_n]
$$

Flamingo 방식은 깊은 레이어에서도 시각 정보에 직접 접근할 수 있어 표현력이 더 높지만, BLIP-2 방식은 LLM을 전혀 수정하지 않아 구현이 간단하다.

---

## 5.1.7 OCR에서의 Cross-Attention 활용

### 문서 이미지 이해

OCR 태스크에서 Cross-Attention은 텍스트 디코더가 이미지의 특정 영역에 집중하게 한다.

```python
class OCRCrossAttentionHead(nn.Module):
    """
    OCR에 특화된 Cross-Attention 헤드.
    문서 이미지에서 텍스트 영역에 높은 attention을 부여한다.
    """
    def __init__(
        self,
        dim_text: int = 768,
        dim_visual: int = 1024,
        num_heads: int = 12,
        max_text_len: int = 512,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim_text,
            num_heads=num_heads,
            kdim=dim_visual,
            vdim=dim_visual,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim_text)
        self.ffn = nn.Sequential(
            nn.Linear(dim_text, dim_text * 4),
            nn.GELU(),
            nn.Linear(dim_text * 4, dim_text),
        )
        self.norm_ffn = nn.LayerNorm(dim_text)

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: 업데이트된 텍스트 features
            attn_weights: attention 가중치 (시각화용)
        """
        residual = text_features
        text_normed = self.norm(text_features)
        attn_output, attn_weights = self.cross_attn(
            query=text_normed,
            key=image_features,
            value=image_features,
        )
        output = residual + attn_output
        output = output + self.ffn(self.norm_ffn(output))

        return output, attn_weights
```

### Attention 시각화로 OCR 해석

```python
import matplotlib.pyplot as plt
import numpy as np


def visualize_ocr_attention(
    attn_weights: torch.Tensor,
    image: np.ndarray,
    decoded_tokens: list[str],
    patch_size: int = 16,
):
    """
    OCR Cross-Attention 가중치를 원본 이미지 위에 시각화.

    Args:
        attn_weights: (num_tokens, num_patches) - attention 가중치
        image: (H, W, 3) - 원본 이미지
        decoded_tokens: 디코딩된 문자 리스트
        patch_size: ViT 패치 크기
    """
    H, W = image.shape[:2]
    h_patches = H // patch_size
    w_patches = W // patch_size

    num_display = min(len(decoded_tokens), 8)
    fig, axes = plt.subplots(2, num_display, figsize=(num_display * 3, 6))

    for i in range(num_display):
        # 원본 이미지
        axes[0, i].imshow(image)
        axes[0, i].set_title(f"'{decoded_tokens[i]}'")
        axes[0, i].axis('off')

        # Attention heatmap
        attn_map = attn_weights[i].detach().cpu().numpy()
        attn_map = attn_map.reshape(h_patches, w_patches)
        attn_map = np.kron(attn_map, np.ones((patch_size, patch_size)))

        axes[1, i].imshow(image)
        axes[1, i].imshow(attn_map, alpha=0.5, cmap='hot')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig("ocr_attention_vis.png", dpi=150)
    plt.show()
```

---

## 5.1.8 실습: Gated Cross-Attention 동작 확인

```python
def test_gated_cross_attention():
    """Gated Cross-Attention 모듈 테스트"""
    torch.manual_seed(42)

    # 설정
    batch_size = 2
    seq_len = 10       # 텍스트 시퀀스 길이
    num_visual = 64    # visual token 수
    dim_text = 4096    # LLM hidden dim
    dim_visual = 1024  # vision encoder dim

    # 더미 데이터
    text_hidden = torch.randn(batch_size, seq_len, dim_text)
    visual_features = torch.randn(batch_size, num_visual, dim_visual)

    # 모듈 생성
    gated_xattn = GatedCrossAttention(
        dim=dim_text,
        dim_visual=dim_visual,
        num_heads=32,
        dim_head=128,
    )

    # 초기 상태: gate = 0
    print(f"초기 gate value: {gated_xattn.gate.item():.4f}")
    print(f"초기 tanh(gate): {torch.tanh(gated_xattn.gate).item():.4f}")

    # Forward
    output = gated_xattn(text_hidden, visual_features)
    print(f"Input shape:  {text_hidden.shape}")
    print(f"Output shape: {output.shape}")

    # 초기에는 출력이 입력과 거의 같아야 함 (gate ≈ 0)
    diff = (output - text_hidden).abs().mean().item()
    print(f"Input-Output 차이 (초기): {diff:.6f}")

    # Gate를 수동으로 변경해서 테스트
    with torch.no_grad():
        gated_xattn.gate.fill_(2.0)
    output_gated = gated_xattn(text_hidden, visual_features)
    diff_gated = (output_gated - text_hidden).abs().mean().item()
    print(f"Input-Output 차이 (gate=2.0): {diff_gated:.6f}")

    # 파라미터 수
    total = sum(p.numel() for p in gated_xattn.parameters())
    print(f"GatedCrossAttention 파라미터 수: {total:,}")


if __name__ == "__main__":
    test_gated_cross_attention()
```

---

## 용어 체크리스트

학습 후 아래 질문에 답할 수 있는지 확인하라.

| 체크 | 용어 | 자가 점검 질문 |
|------|------|----------------|
| ☐ | Cross-Attention | Self-Attention과 Cross-Attention의 Q, K, V 출처 차이를 설명할 수 있는가? |
| ☐ | Perceiver Resampler | 왜 가변 길이 visual feature를 고정 길이로 압축하는가? LLM 입력에 어떤 이점이 있는가? |
| ☐ | Q-Former | Q-Former의 학습 가능한 query가 하는 역할은? 왜 32개면 충분한가? |
| ☐ | Gated Cross-Attention | tanh gate의 초기값을 0으로 설정하는 이유를 수식으로 설명할 수 있는가? |
| ☐ | Frozen Backbone | Vision Encoder와 LLM을 freeze하고 어댑터만 학습하면 왜 파라미터 효율적인가? |
| ☐ | Information Bottleneck | $N_v$개 visual token을 $M$개로 압축할 때 정보 손실과 효율의 트레이드오프를 설명할 수 있는가? |
| ☐ | BLIP-2 2단계 학습 | Stage 1(표현 학습)과 Stage 2(생성 학습)의 목적 차이를 설명할 수 있는가? |
| ☐ | Flamingo vs BLIP-2 | 두 모델의 시각 정보 주입 위치 차이가 표현력에 미치는 영향을 비교할 수 있는가? |

---

## 다음 단계

[5.2 Projection/Connector 방식](02-projection.md)에서 더 단순하지만 효과적인 시각-언어 연결 방식을 다룬다.
