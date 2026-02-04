# Chapter 1.2: Transformer 아키텍처

## 개요

2017년 Google의 "Attention Is All You Need" 논문에서 소개된 Transformer는 현대 NLP와 Vision의 핵심 아키텍처이다. RNN/LSTM의 순차적 처리 한계를 극복하고 병렬 처리를 가능하게 했다.

> **핵심 논문**: Vaswani et al. (2017). "Attention Is All You Need"
> - arXiv: https://arxiv.org/abs/1706.03762
> - NeurIPS 2017, 173,000+ citations (2025년 기준)

---

## 1. Self-Attention 메커니즘

### 1.1 핵심 개념

Self-Attention은 시퀀스 내 모든 위치 간의 관계를 동시에 계산한다.

**직관적 이해:**
- 문장 "The cat sat on the mat because it was tired"에서
- "it"이 무엇을 지칭하는지 파악하기 위해 문장 전체를 참조
- "cat"과 "it" 사이의 attention weight가 높아짐

### 1.2 Query, Key, Value (Q, K, V)

입력 임베딩 X로부터 세 가지 벡터를 생성:

```
Q = X × W_Q  (Query: "내가 찾고 싶은 정보")
K = X × W_K  (Key: "내가 가진 정보의 인덱스")
V = X × W_V  (Value: "실제 정보 내용")
```

**비유:**
- 도서관에서 책 찾기
- Query: 찾고 싶은 주제
- Key: 책 제목/분류
- Value: 책 내용

### 1.3 Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**단계별 분석:**

1. **QK^T**: Query와 Key의 유사도 계산
   - 출력: (seq_len, seq_len) 행렬
   - 각 위치가 다른 모든 위치와의 관련성 점수

2. **/ √d_k**: Scaling
   - d_k가 커지면 dot product 값이 커짐
   - Softmax의 gradient가 작아지는 것 방지
   - 논문: "We suspect that for large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients"

3. **softmax**: 확률 분포 변환
   - 각 행의 합이 1이 되도록 정규화
   - 중요한 위치에 더 높은 가중치

4. **× V**: 가중 평균
   - Attention weights로 Value를 가중합

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: (batch, num_heads, seq_len, d_k)
        K: (batch, num_heads, seq_len, d_k)
        V: (batch, num_heads, seq_len, d_v)
        mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
    Returns:
        output: (batch, num_heads, seq_len, d_v)
        attention_weights: (batch, num_heads, seq_len, seq_len)
    """
    d_k = Q.size(-1)

    # Step 1: QK^T
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, heads, seq, seq)

    # Step 2: Scale
    scores = scores / math.sqrt(d_k)

    # Step 3: Mask (optional)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 4: Softmax
    attention_weights = F.softmax(scores, dim=-1)

    # Step 5: Weighted sum with V
    output = torch.matmul(attention_weights, V)

    return output, attention_weights
```

### 1.4 Multi-Head Attention

단일 attention 대신 여러 개의 attention head를 병렬로 사용:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

where head_i = Attention(Q × W_Q^i, K × W_K^i, V × W_V^i)
```

**장점:**
- 다양한 관점에서 정보 추출
- 각 head가 다른 패턴 학습 (syntax, semantics, coreference 등)
- 원래 dimension d_model을 h개로 나눠 연산량 유지

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections
        Q = self.W_Q(Q)  # (batch, seq, d_model)
        K = self.W_K(K)
        V = self.W_V(V)

        # Reshape to (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_O(attn_output)

        return output, attn_weights
```

---

## 2. Attention Mask

### 2.1 Padding Mask

가변 길이 시퀀스 처리 시 패딩 토큰 무시:

```python
def create_padding_mask(seq, pad_token_id=0):
    """
    seq: (batch, seq_len)
    returns: (batch, 1, 1, seq_len)
    """
    mask = (seq != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask.float()
```

### 2.2 Causal Mask (Look-ahead Mask)

Autoregressive 모델에서 미래 토큰 참조 방지:

```python
def create_causal_mask(seq_len):
    """
    returns: (1, 1, seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.unsqueeze(0).unsqueeze(0)

# 결과 예시 (seq_len=4):
# [[0, -inf, -inf, -inf],
#  [0,   0,  -inf, -inf],
#  [0,   0,    0,  -inf],
#  [0,   0,    0,    0]]
```

---

## 3. Positional Encoding

Transformer는 순서 정보가 없으므로 위치 정보를 주입해야 함.

### 3.1 Sinusoidal Positional Encoding (원래 논문)

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**특징:**
- 학습 없이 고정된 패턴 사용
- 임의의 긴 시퀀스로 확장 가능
- 상대적 위치 관계 표현 가능 (PE(pos+k)를 PE(pos)의 선형 함수로 표현)

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]
```

### 3.2 Learned Positional Embedding

위치별 임베딩을 학습:

```python
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.embedding(positions)
```

### 3.3 RoPE (Rotary Position Embedding)

> **논문**: Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding"
> - arXiv: https://arxiv.org/abs/2104.09864
> - LLaMA, Qwen 등 현대 LLM의 표준

**핵심 아이디어:**
- 위치 정보를 회전 행렬로 인코딩
- 상대 위치가 dot product에 자연스럽게 반영됨

```
q_m = R(θ_m) × q  (m번째 위치의 query)
k_n = R(θ_n) × k  (n번째 위치의 key)

q_m^T × k_n = q^T × R(θ_m)^T × R(θ_n) × k
            = q^T × R(θ_n - θ_m) × k
```

위치 차이(n-m)만 남음 → 상대적 위치 인코딩

```python
def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """RoPE 주파수 사전 계산"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """RoPE 적용"""
    # x를 complex로 변환
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 회전 적용
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim/2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
```

---

## 4. Transformer Block 구성요소

### 4.1 Layer Normalization

> **논문**: Ba et al. (2016). "Layer Normalization"
> - arXiv: https://arxiv.org/abs/1607.06450

**Batch Norm vs Layer Norm:**
- Batch Norm: batch 차원에서 정규화 → 배치 크기 의존
- Layer Norm: feature 차원에서 정규화 → 배치 크기 무관

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

### 4.2 RMSNorm

LLaMA 등 현대 LLM에서 사용:

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)
```

### 4.3 Pre-LN vs Post-LN

**Post-LN (원래 Transformer):**
```
x = x + Attention(LN(x))  # ❌ 오타, 원래는:
x = LN(x + Attention(x))
```

**Pre-LN (현대 표준):**
```
x = x + Attention(LN(x))
```

Pre-LN의 장점:
- 더 안정적인 학습
- Learning rate warmup 덜 필요
- 더 깊은 모델 학습 가능

### 4.4 Feed-Forward Network (MLP)

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))
```

**SwiGLU (LLaMA 스타일):**
```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

### 4.5 Residual Connection

```
output = x + sublayer(x)
```

- Gradient flow 개선
- 깊은 네트워크 학습 가능
- Identity mapping 학습 용이

---

## 5. Encoder vs Decoder

### 5.1 Encoder-only (BERT 스타일)

- Bidirectional attention
- 입력 전체를 한 번에 처리
- 용도: 분류, NER, 문장 임베딩

```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # FFN with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x
```

### 5.2 Decoder-only (GPT 스타일)

- Causal (unidirectional) attention
- Autoregressive generation
- 용도: 텍스트 생성, LLM

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        # Causal self-attention
        attn_output, _ = self.self_attn(x, x, x, causal_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # FFN
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x
```

### 5.3 Encoder-Decoder (원래 Transformer, T5)

- Encoder: 입력 처리
- Decoder: Cross-attention으로 encoder 출력 참조
- 용도: 번역, 요약

```python
class EncoderDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, causal_mask, cross_mask):
        # Causal self-attention
        attn_output, _ = self.self_attn(x, x, x, causal_mask)
        x = self.norm1(x + attn_output)

        # Cross-attention
        cross_output, _ = self.cross_attn(x, encoder_output, encoder_output, cross_mask)
        x = self.norm2(x + cross_output)

        # FFN
        x = self.norm3(x + self.ffn(x))

        return x
```

---

## 6. 전체 Transformer 구현

```python
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.output_proj.weight = self.embedding.weight

    def forward(self, x):
        seq_len = x.size(1)

        # Embedding + Positional Encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Causal mask
        causal_mask = create_causal_mask(seq_len).to(x.device)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask)

        x = self.norm(x)

        # Output projection
        logits = self.output_proj(x)

        return logits
```

---

## 핵심 참고 자료

### 논문
- **Attention Is All You Need** (Vaswani et al., 2017)
  - https://arxiv.org/abs/1706.03762
  - 원조 Transformer 논문

- **RoFormer** (Su et al., 2021)
  - https://arxiv.org/abs/2104.09864
  - Rotary Position Embedding

- **Layer Normalization** (Ba et al., 2016)
  - https://arxiv.org/abs/1607.06450

### 튜토리얼
- **The Illustrated Transformer** by Jay Alammar
  - https://jalammar.github.io/illustrated-transformer/

- **Andrej Karpathy's "Let's build GPT"**
  - https://karpathy.ai/zero-to-hero.html
  - GitHub: https://github.com/karpathy/nanoGPT

### 강의
- **Stanford CS224N Lecture 8: Transformers**
  - https://web.stanford.edu/class/cs224n/

---

## 핵심 요약

| 구성요소 | 역할 | 현대 변형 |
|---------|------|----------|
| Self-Attention | 시퀀스 내 관계 학습 | FlashAttention |
| Multi-Head | 다양한 관점 학습 | Grouped Query Attention |
| Positional Encoding | 위치 정보 주입 | RoPE |
| Layer Norm | 학습 안정화 | RMSNorm |
| FFN | 비선형 변환 | SwiGLU |
| Residual Connection | Gradient flow | 동일 |
