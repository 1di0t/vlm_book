# 2.3 Transformer Block 구성요소

Transformer Block은 Self-Attention만으로 구성되지 않는다. Layer Normalization, Feed-Forward Network, Residual Connection이 함께 작동한다.

## 2.3.1 전체 구조

```
Input
  │
  ├──────────────────────────┐
  │                          │ (Residual)
  ▼                          │
Layer Norm (Pre-LN)          │
  │                          │
  ▼                          │
Multi-Head Attention         │
  │                          │
  ▼                          │
  + ◄────────────────────────┘
  │
  ├──────────────────────────┐
  │                          │ (Residual)
  ▼                          │
Layer Norm (Pre-LN)          │
  │                          │
  ▼                          │
Feed-Forward Network         │
  │                          │
  ▼                          │
  + ◄────────────────────────┘
  │
  ▼
Output
```

## 2.3.2 Layer Normalization

### Batch Norm vs Layer Norm

| | Batch Normalization | Layer Normalization |
|---|---------------------|---------------------|
| 정규화 축 | Batch 차원 | Feature 차원 |
| 시퀀스 길이 | 가변 길이 문제 | 문제 없음 |
| 배치 크기 1 | 불안정 | 안정적 |
| 주 사용처 | CNN | Transformer |

### Layer Norm 수식

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

- $\mu$: 각 샘플의 평균
- $\sigma^2$: 각 샘플의 분산
- $\gamma, \beta$: 학습 가능한 스케일, 시프트 파라미터

### 구현

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

### Pre-LN vs Post-LN

**Post-LN (원래 Transformer)**:
```
x → Attention → Add → LayerNorm → FFN → Add → LayerNorm
```

**Pre-LN (현재 표준)**:
```
x → LayerNorm → Attention → Add → LayerNorm → FFN → Add
```

```python
# Post-LN
class PostLNBlock(nn.Module):
    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ffn(x))
        return x

# Pre-LN
class PreLNBlock(nn.Module):
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

**Pre-LN의 장점**:
- 학습 안정성 향상
- Learning rate warmup 덜 필요
- 깊은 모델에서 gradient flow 개선

## 2.3.3 RMSNorm

### Layer Norm의 간소화 버전

LLaMA, Qwen 등 최신 모델이 사용:

$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}}
$$

- 평균을 빼지 않음 (re-centering 없음)
- 계산 효율성 향상

### 구현

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # RMS (Root Mean Square) 계산
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)
```

### 비교

| | Layer Norm | RMS Norm |
|---|------------|----------|
| 파라미터 | $2d$ ($\gamma, \beta$) | $d$ ($\gamma$만) |
| 연산 | mean, var 계산 | mean of squares만 |
| 성능 | 기준 | 유사하거나 약간 우수 |
| 사용 모델 | BERT, GPT-2 | LLaMA, Qwen, Mistral |

## 2.3.4 Feed-Forward Network (FFN)

### 기본 구조

Position-wise Feed-Forward Network:

$$
\text{FFN}(x) = \text{Act}(xW_1 + b_1)W_2 + b_2
$$

- $W_1$: $d_{model} \rightarrow d_{ff}$ (확장)
- $W_2$: $d_{ff} \rightarrow d_{model}$ (축소)
- 일반적으로 $d_{ff} = 4 \times d_{model}$

### 구현

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

### 활성화 함수 선택

| 함수 | 수식 | 사용 모델 |
|------|------|----------|
| ReLU | $\max(0, x)$ | 원래 Transformer |
| GELU | $x \cdot \Phi(x)$ | BERT, GPT |
| SiLU/Swish | $x \cdot \sigma(x)$ | LLaMA, Mistral |

```python
# GELU (Gaussian Error Linear Unit)
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)
    ))

# SiLU / Swish
def silu(x):
    return x * torch.sigmoid(x)
```

### SwiGLU (Gated Linear Unit)

LLaMA에서 사용하는 구조:

$$
\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xW_3)W_2
$$

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

### FFN의 역할

Attention이 "어디를 볼지" 결정한다면, FFN은 "무엇을 기억할지" 결정한다.

연구에 따르면 FFN은:
- 사실적 지식 저장 (key-value memory)
- 토큰별 독립적 변환 수행

## 2.3.5 Residual Connection

### Skip Connection

$$
\text{Output} = \text{Layer}(x) + x
$$

### 왜 필요한가?

1. **Gradient Vanishing 방지**: 깊은 네트워크에서 gradient가 직접 전파됨
2. **Identity Mapping**: 레이어가 "아무것도 하지 않기"를 학습할 수 있음
3. **학습 안정화**: 초기 학습 시 레이어 출력이 작아도 괜찮음

```python
class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Args:
            x: input
            sublayer: attention or ffn function
        """
        return x + self.dropout(sublayer(x))
```

## 2.3.6 Dropout

### 위치

Transformer에서 Dropout이 적용되는 위치:

1. **Attention weights**: Softmax 후
2. **Sublayer output**: Attention/FFN 출력
3. **Embedding**: 입력 임베딩

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention block (Pre-LN)
        attn_out, _ = self.attention(
            self.norm1(x), self.norm1(x), self.norm1(x), mask
        )
        x = x + self.dropout(attn_out)

        # FFN block (Pre-LN)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)

        return x
```

## 2.3.7 전체 Transformer Block 구현

```python
class TransformerBlock(nn.Module):
    """
    Complete Transformer Block with Pre-LN

    Components:
    - Multi-Head Self-Attention
    - RMSNorm
    - SwiGLU Feed-Forward
    - Residual Connections
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int = None,
        dropout: float = 0.1,
        use_swiglu: bool = True
    ):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        # Attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.attn_norm = RMSNorm(d_model)

        # FFN
        if use_swiglu:
            # SwiGLU는 d_ff를 2/3로 조정 (파라미터 수 맞추기)
            self.ffn = SwiGLU(d_model, int(d_ff * 2 / 3))
        else:
            self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ffn_norm = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention with residual
        normed = self.attn_norm(x)
        attn_output, attn_weights = self.attention(normed, normed, normed, mask)
        x = x + self.dropout(attn_output)

        # FFN with residual
        normed = self.ffn_norm(x)
        ffn_output = self.ffn(normed)
        x = x + self.dropout(ffn_output)

        return x, attn_weights
```

## 2.3.8 하이퍼파라미터

### 일반적인 설정

| 모델 | d_model | num_heads | d_ff | layers |
|------|---------|-----------|------|--------|
| BERT-base | 768 | 12 | 3072 | 12 |
| GPT-2 | 768 | 12 | 3072 | 12 |
| LLaMA-7B | 4096 | 32 | 11008 | 32 |
| Qwen-7B | 4096 | 32 | 11008 | 32 |

### 파라미터 수 계산

```python
def count_parameters(d_model, num_heads, d_ff, num_layers, vocab_size):
    """Transformer 파라미터 수 계산"""

    # Embedding
    embed_params = vocab_size * d_model

    # Per layer
    attention_params = 4 * d_model * d_model  # Q, K, V, O
    ffn_params = 2 * d_model * d_ff           # up, down
    norm_params = 2 * d_model                  # 2 norms per layer

    layer_params = attention_params + ffn_params + norm_params

    # Total
    total = embed_params + num_layers * layer_params

    return total

# 예시: BERT-base
params = count_parameters(768, 12, 3072, 12, 30000)
print(f"파라미터 수: {params:,}")  # 약 109M
```

## 2.3.9 실습 체크리스트

- [ ] Layer Norm 직접 구현
- [ ] RMSNorm 구현 및 비교
- [ ] SwiGLU FFN 구현
- [ ] 전체 Transformer Block 조립
- [ ] 파라미터 수 계산해보기

## 2.3.10 핵심 요약

| 구성요소 | 역할 |
|---------|------|
| Multi-Head Attention | 시퀀스 내 관계 모델링 |
| Layer/RMS Norm | 학습 안정화, gradient flow |
| FFN (SwiGLU) | 비선형 변환, 지식 저장 |
| Residual Connection | 깊은 네트워크 학습 가능 |
| Dropout | 과적합 방지 |

## 다음 단계

[2.4 Encoder vs Decoder](04-encoder-decoder.md)에서 Transformer의 두 가지 구조를 비교한다.
