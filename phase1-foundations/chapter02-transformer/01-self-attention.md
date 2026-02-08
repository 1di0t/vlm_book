---
---

# 2.1 Self-Attention 메커니즘

Self-Attention은 Transformer의 핵심이다. 시퀀스 내 모든 위치가 서로를 "주목"하여 문맥을 이해하는 메커니즘이다.

## 2.1.1 Attention의 직관적 이해

### 번역 문제로 이해하기

"The cat sat on the mat because **it** was tired."

여기서 "it"이 무엇을 가리키는지 알려면 문장 전체를 봐야 한다. Attention은 "it"이 "cat"에 주목하도록 학습한다.

### Self-Attention vs 전통적 접근

| 방식 | 문맥 범위 | 병렬화 | 장거리 의존성 |
|------|----------|--------|--------------|
| RNN | 순차적 누적 | 불가능 | 어려움 (vanishing gradient) |
| CNN | 고정 윈도우 | 가능 | 레이어 쌓아야 함 |
| Self-Attention | 전체 시퀀스 | 가능 | 직접 연결 |

## 2.1.2 Query, Key, Value

### 개념

Self-Attention은 데이터베이스 검색에 비유할 수 있다:

- **Query (Q)**: "내가 찾고 싶은 것" - 현재 토큰이 어떤 정보를 원하는지
- **Key (K)**: "나를 설명하는 태그" - 각 토큰이 어떤 정보를 제공하는지
- **Value (V)**: "실제 내용" - 각 토큰이 전달할 정보

### 수식

입력 시퀀스 $X \in \mathbb{R}^{n \times d}$ (n개 토큰, d차원)에서:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

- $W_Q, W_K \in \mathbb{R}^{d \times d_k}$
- $W_V \in \mathbb{R}^{d \times d_v}$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class QKVProjection(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        Q = self.W_q(x)  # (batch, seq_len, d_k)
        K = self.W_k(x)  # (batch, seq_len, d_k)
        V = self.W_v(x)  # (batch, seq_len, d_v)
        return Q, K, V
```

### 직관적 예시

문장: "I love cats"

```
토큰    Query 의미                Key 의미              Value 의미
----    -----------               --------              ----------
"I"     "주어를 찾아"             "나는 주어야"          주어 정보
"love"  "목적어가 뭐지?"           "나는 동사야"          동사 정보
"cats"  "나를 수식하는 건?"        "나는 명사야"          목적어 정보
```

## 2.1.3 Scaled Dot-Product Attention

### 핵심 수식

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 단계별 분석

**Step 1: 유사도 계산 ($QK^T$)**

```python
# Q: (batch, seq_len, d_k)
# K: (batch, seq_len, d_k)
scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq_len, seq_len)
```

결과: 각 토큰 쌍의 유사도 행렬

```
         I    love   cats
I      [0.9,  0.3,   0.1]
love   [0.2,  0.8,   0.5]
cats   [0.1,  0.4,   0.9]
```

**Step 2: 스케일링 ($/ \sqrt{d_k}$)**

왜 필요한가?

$d_k$가 크면 내적 값이 커지고, softmax가 극단적인 분포를 만든다.

$$
\text{Var}(q \cdot k) = d_k \cdot \text{Var}(q_i) \cdot \text{Var}(k_i) \approx d_k
$$

$\sqrt{d_k}$로 나누면 분산이 1에 가까워진다.

```python
d_k = K.size(-1)
scores = scores / (d_k ** 0.5)
```

**Step 3: Softmax (행 단위 정규화)**

각 Query에 대해 확률 분포로 변환:

```python
attention_weights = F.softmax(scores, dim=-1)  # 각 행의 합 = 1
```

**Step 4: Value 가중 평균**

```python
output = torch.matmul(attention_weights, V)  # (batch, seq_len, d_v)
```

### 전체 구현

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention

    Args:
        Q: (batch, seq_len, d_k) 또는 (batch, heads, seq_len, d_k)
        K: (batch, seq_len, d_k) 또는 (batch, heads, seq_len, d_k)
        V: (batch, seq_len, d_v) 또는 (batch, heads, seq_len, d_v)
        mask: attention mask (optional)

    Returns:
        output: (batch, seq_len, d_v)
        attention_weights: (batch, seq_len, seq_len)
    """
    d_k = K.size(-1)

    # Step 1 & 2: 유사도 계산 + 스케일링
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    # Masking (optional)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 3: Softmax
    attention_weights = F.softmax(scores, dim=-1)

    # Step 4: Value 가중 평균
    output = torch.matmul(attention_weights, V)

    return output, attention_weights
```

## 2.1.4 Attention Mask

### Padding Mask

패딩 토큰은 attention에서 무시해야 한다.

```python
def create_padding_mask(seq, pad_idx=0):
    """
    패딩 위치를 마스킹

    Args:
        seq: (batch, seq_len) 토큰 ID
        pad_idx: 패딩 토큰 ID

    Returns:
        mask: (batch, 1, 1, seq_len) - broadcasting용
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask.float()

# 예시
tokens = torch.tensor([[1, 2, 3, 0, 0],  # 0은 패딩
                       [4, 5, 0, 0, 0]])
mask = create_padding_mask(tokens)
# 결과: [[[[1, 1, 1, 0, 0]]],
#        [[[1, 1, 0, 0, 0]]]]
```

### Causal Mask (Look-ahead Mask)

Decoder에서 미래 토큰을 보지 못하게 막는다.

```python
def create_causal_mask(seq_len):
    """
    Causal mask: 현재 위치 이전만 볼 수 있음

    Returns:
        mask: (1, 1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)

# 예시: seq_len=4
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
```

### Mask 적용

```python
# scores: (batch, seq_len, seq_len)
# mask: 0인 위치를 -inf로
scores = scores.masked_fill(mask == 0, float('-inf'))
# softmax 후 -inf → 0
```

## 2.1.5 Multi-Head Attention

### 왜 Multi-Head인가?

단일 Attention은 하나의 "관점"만 학습한다. 여러 Head를 두면:

- Head 1: 문법적 관계 (주어-동사)
- Head 2: 의미적 관계 (대명사-선행사)
- Head 3: 위치 관계 (인접 단어)
- ...

### 수식

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

### 구현

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 모든 head를 한 번에 계산하기 위한 projection
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projection
        Q = self.W_q(Q)  # (batch, seq_len, d_model)
        K = self.W_k(K)
        V = self.W_v(V)

        # Split into heads
        Q = self.split_heads(Q, batch_size)  # (batch, heads, seq_len, d_k)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled dot-product attention
        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Concat heads
        output = output.transpose(1, 2).contiguous()  # (batch, seq_len, heads, d_k)
        output = output.view(batch_size, -1, self.d_model)  # (batch, seq_len, d_model)

        # Final linear
        output = self.W_o(output)

        return output, attention_weights
```

### 파라미터 수

- 단일 Head: $4 \times d_{model} \times d_k = 4d_{model}^2 / h$
- Multi-Head (h개): $4d_{model}^2$ (동일!)

Multi-Head는 추가 파라미터 없이 여러 관점을 학습한다.

## 2.1.6 Self-Attention의 특성

### 순서 불변성 (Permutation Equivariance)

Self-Attention 자체는 순서 정보가 없다:

$$
\text{Attention}(\pi(X)) = \pi(\text{Attention}(X))
$$

입력 순서를 바꾸면 출력 순서도 같이 바뀐다.
→ **Positional Encoding**이 필요한 이유

### 계산 복잡도

- 시간: $O(n^2 \cdot d)$ (n: 시퀀스 길이)
- 공간: $O(n^2)$ (attention 행렬)

긴 시퀀스에서 병목이 된다 → Flash Attention 등 최적화 필요

### 표현력

Self-Attention은 **any-to-any** 연결이 가능:
- 첫 번째 토큰 ↔ 마지막 토큰 직접 연결
- RNN은 순차적으로 전파해야 함
- CNN은 receptive field 내에서만

## 2.1.7 Attention 시각화

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens):
    """
    Attention weights 시각화

    Args:
        attention_weights: (seq_len, seq_len)
        tokens: 토큰 리스트
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights.detach().numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        annot=True,
        fmt='.2f'
    )
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.title('Attention Weights')
    plt.show()

# 예시 사용
# tokens = ["I", "love", "cats"]
# visualize_attention(attention_weights[0, 0], tokens)
```

## 2.1.8 Self-Attention vs Cross-Attention

| 구분 | Self-Attention | Cross-Attention |
|------|----------------|-----------------|
| Q 출처 | 같은 시퀀스 | Decoder |
| K, V 출처 | 같은 시퀀스 | Encoder 출력 |
| 사용처 | Encoder, Decoder | Encoder-Decoder 연결 |
| 목적 | 시퀀스 내 관계 | 시퀀스 간 관계 |

Cross-Attention (VLM에서 중요):

```python
# Vision-Language Cross-Attention
# Q: Language (text tokens)
# K, V: Vision (image features)
output = cross_attention(
    Q=text_features,      # 언어 모델의 query
    K=image_features,     # 이미지의 key
    V=image_features      # 이미지의 value
)
```

## 2.1.9 실습 체크리스트

- [ ] Scaled Dot-Product Attention 직접 구현
- [ ] Multi-Head Attention 구현
- [ ] Padding Mask, Causal Mask 적용
- [ ] Attention weights 시각화
- [ ] 간단한 문장에서 attention 패턴 분석

## 2.1.10 핵심 요약

| 개념 | 역할 |
|------|------|
| Query | 현재 토큰이 찾고 싶은 정보 |
| Key | 각 토큰이 제공하는 정보의 "태그" |
| Value | 실제 전달할 정보 |
| Scaling | Softmax 안정화 ($/ \sqrt{d_k}$) |
| Multi-Head | 여러 관점에서 attention 학습 |
| Mask | 패딩, 미래 토큰 차단 |

## 다음 단계

[2.2 Positional Encoding](02-positional-encoding.md)에서 Self-Attention에 순서 정보를 부여하는 방법을 다룬다.
