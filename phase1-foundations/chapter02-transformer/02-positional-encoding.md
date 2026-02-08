---
---

# 2.2 Positional Encoding

Self-Attention은 순서를 모른다. "I love you"와 "You love I"를 구분하려면 위치 정보가 필요하다.

## 2.2.1 왜 필요한가?

### Self-Attention의 순서 불변성

```python
# 두 입력이 같은 attention 결과를 낸다 (순서만 바뀜)
input1 = ["The", "cat", "sat"]
input2 = ["sat", "cat", "The"]

# Self-Attention만으로는 구분 불가!
```

### RNN과의 차이

- **RNN**: 순차 처리 → 암묵적 위치 정보
- **Transformer**: 병렬 처리 → 명시적 위치 정보 필요

## 2.2.2 Sinusoidal Positional Encoding

### 원래 Transformer의 방식

"Attention Is All You Need" 논문에서 제안:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

- $pos$: 토큰 위치 (0, 1, 2, ...)
- $i$: 차원 인덱스
- $d_{model}$: 임베딩 차원

### 구현

```python
import torch
import math

def sinusoidal_positional_encoding(max_len, d_model):
    """
    Sinusoidal Positional Encoding 생성

    Args:
        max_len: 최대 시퀀스 길이
        d_model: 임베딩 차원

    Returns:
        pe: (max_len, d_model)
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

    # div_term = 10000^(2i/d_model)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
    pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스

    return pe

# 예시
pe = sinusoidal_positional_encoding(100, 512)
print(f"Shape: {pe.shape}")  # (100, 512)
```

### 시각화

```python
import matplotlib.pyplot as plt

pe = sinusoidal_positional_encoding(100, 64)

plt.figure(figsize=(12, 6))
plt.pcolormesh(pe.numpy(), cmap='RdBu')
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.colorbar()
plt.title('Sinusoidal Positional Encoding')
plt.show()
```

### 특징

1. **주기적 패턴**: 각 차원이 다른 주파수의 사인/코사인
   - 낮은 차원: 빠른 주기 (인접 위치 구분)
   - 높은 차원: 느린 주기 (먼 위치 구분)

2. **상대 위치 표현 가능**:
   $$PE_{pos+k} = f(PE_{pos})$$
   선형 변환으로 상대 위치를 표현할 수 있다.

3. **외삽 가능**: 학습에 없던 긴 시퀀스에도 적용 가능

## 2.2.3 Learned Positional Embedding

### 학습 가능한 위치 임베딩

```python
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x + positional embedding
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pe(positions)
```

### 장단점

| | Sinusoidal | Learned |
|---|------------|---------|
| 외삽 | 가능 | 불가능 (학습 범위 내) |
| 파라미터 | 0 | max_len × d_model |
| 성능 | 비슷 | 약간 우수 (짧은 시퀀스) |
| 사용 모델 | 원래 Transformer | BERT, GPT |

## 2.2.4 Rotary Position Embedding (RoPE)

### 현재 LLM의 표준

LLaMA, Qwen, Mistral 등 최신 모델이 사용하는 방식.

### 핵심 아이디어

위치를 **더하지 않고 회전**시킨다.

Query와 Key에 위치 정보를 인코딩:

$$
f_q(x_m, m) = R_{\Theta, m} W_q x_m
$$

$$
f_k(x_n, n) = R_{\Theta, n} W_k x_n
$$

내적하면:

$$
f_q(x_m, m)^T f_k(x_n, n) = x_m^T W_q^T R_{\Theta, n-m} W_k x_n
$$

**결과**: 내적 결과가 **상대 위치** $n-m$에만 의존!

### 2D 회전 행렬

$$
R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

각 차원 쌍에 대해 다른 각도로 회전:

$$
\theta_i = m \cdot 10000^{-2i/d}
$$

### 구현

```python
def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """
    RoPE 주파수 미리 계산

    Args:
        dim: 임베딩 차원 (짝수)
        max_seq_len: 최대 시퀀스 길이
        theta: base frequency

    Returns:
        freqs_cis: (max_seq_len, dim//2) 복소수
    """
    # 주파수 계산
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # 위치별 각도
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)  # (max_seq_len, dim//2)

    # 복소수 형태로 (cos + i*sin)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    """
    RoPE 적용

    Args:
        xq: Query (batch, seq_len, heads, dim)
        xk: Key (batch, seq_len, heads, dim)
        freqs_cis: 미리 계산된 주파수

    Returns:
        rotated xq, xk
    """
    # 실수 → 복소수로 reshape
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 회전 적용 (복소수 곱)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim//2)
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)
```

### 실수 연산 버전

```python
def rotate_half(x):
    """x를 반으로 나눠 회전 준비"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    RoPE 적용 (실수 연산)

    Args:
        q, k: (batch, heads, seq_len, dim)
        cos, sin: (seq_len, dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
```

### RoPE의 장점

1. **상대 위치 인코딩**: 절대 위치가 아닌 상대 위치 학습
2. **외삽 능력**: 학습보다 긴 시퀀스에 일반화 (NTK scaling으로 개선)
3. **계산 효율**: 파라미터 추가 없음
4. **호환성**: 기존 Attention 구조 유지

## 2.2.5 2D RoPE for Vision

Vision Transformer에서는 2D 위치 정보가 필요하다.

### 개념

이미지 패치의 (x, y) 좌표를 각각 인코딩:

$$
R_{2D}(x, y) = R_x \otimes R_y
$$

### 구현 (Qwen-VL 스타일)

```python
def get_2d_rotary_pos_embed(dim, height, width):
    """
    2D RoPE for Vision

    Args:
        dim: 임베딩 차원
        height: 이미지 높이 (패치 수)
        width: 이미지 너비 (패치 수)

    Returns:
        pos_embed: (height*width, dim)
    """
    assert dim % 4 == 0, "dim must be divisible by 4"

    # x, y 좌표 생성
    y_pos = torch.arange(height).unsqueeze(1).repeat(1, width).flatten()
    x_pos = torch.arange(width).unsqueeze(0).repeat(height, 1).flatten()

    # 각 좌표에 대해 RoPE
    dim_quarter = dim // 4
    freqs = 1.0 / (10000 ** (torch.arange(0, dim_quarter, 2).float() / dim_quarter))

    # x 방향 인코딩
    x_angles = torch.outer(x_pos.float(), freqs)
    x_embed = torch.cat([x_angles.sin(), x_angles.cos()], dim=-1)

    # y 방향 인코딩
    y_angles = torch.outer(y_pos.float(), freqs)
    y_embed = torch.cat([y_angles.sin(), y_angles.cos()], dim=-1)

    # 결합
    pos_embed = torch.cat([x_embed, y_embed], dim=-1)

    return pos_embed
```

## 2.2.6 ALiBi (Attention with Linear Biases)

### 다른 접근법

위치를 임베딩으로 더하지 않고, Attention score에 bias 추가:

$$
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} - m \cdot |i-j|\right)
$$

- $m$: head별 다른 기울기
- $|i-j|$: 토큰 간 거리

### 특징

- 외삽 능력 우수
- 구현 단순
- BLOOM 등에서 사용

```python
def get_alibi_slopes(num_heads):
    """ALiBi slopes 계산"""
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        return get_slopes_power_of_2(num_heads)
    else:
        closest_power = 2 ** math.floor(math.log2(num_heads))
        return (
            get_slopes_power_of_2(closest_power) +
            get_alibi_slopes(2 * closest_power)[0::2][:num_heads - closest_power]
        )
```

## 2.2.7 위치 인코딩 비교

| 방식 | 상대 위치 | 외삽 | 파라미터 | 사용 모델 |
|------|----------|------|----------|----------|
| Sinusoidal | 간접적 | 가능 | 0 | 원래 Transformer |
| Learned | X | X | 많음 | BERT, GPT-2 |
| RoPE | O | 가능 | 0 | LLaMA, Qwen, Mistral |
| ALiBi | O | 우수 | 0 | BLOOM |

## 2.2.8 실습 체크리스트

- [ ] Sinusoidal PE 구현 및 시각화
- [ ] RoPE 구현 (복소수/실수 버전)
- [ ] 2D RoPE for Vision 이해
- [ ] 위치 인코딩 유무에 따른 성능 비교

## 2.2.9 핵심 요약

| 개념 | 핵심 |
|------|------|
| 필요성 | Self-Attention은 순서 불변 |
| Sinusoidal | 사인/코사인 주기 함수 |
| Learned | 학습 가능한 임베딩 테이블 |
| RoPE | 회전으로 상대 위치 인코딩 (현재 표준) |
| 2D RoPE | Vision에서 x, y 좌표 인코딩 |

## 다음 단계

[2.3 Transformer Block 구성요소](03-transformer-block.md)에서 Layer Normalization, FFN 등을 다룬다.
