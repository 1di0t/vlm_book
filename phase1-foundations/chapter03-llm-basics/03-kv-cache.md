# 3.3 KV Cache

Autoregressive generation에서 KV Cache는 필수적인 최적화다. 없으면 토큰 생성마다 전체 시퀀스를 다시 계산해야 한다.

## 3.3.1 왜 KV Cache가 필요한가?

### 중복 계산 문제

토큰 4개를 생성한다고 가정:

**Without KV Cache**:
```
Step 1: [A] → 계산
Step 2: [A, B] → A, B 모두 계산 (A 중복!)
Step 3: [A, B, C] → A, B, C 모두 계산 (A, B 중복!)
Step 4: [A, B, C, D] → 전부 계산 (A, B, C 중복!)
```

**With KV Cache**:
```
Step 1: [A] → A 계산, K_A, V_A 캐시
Step 2: [B] → B만 계산, 캐시된 K, V 사용
Step 3: [C] → C만 계산, 캐시된 K, V 사용
Step 4: [D] → D만 계산, 캐시된 K, V 사용
```

### 계산 복잡도 비교

| 생성 토큰 수 | Without Cache | With Cache |
|-------------|---------------|------------|
| n | O(n² × L) | O(n × L) |

L = 레이어 수. **n배 빨라진다.**

## 3.3.2 KV Cache 동작 원리

### Attention에서의 역할

```
Q: 현재 토큰의 Query (새로 계산)
K: 모든 토큰의 Key (이전 것 캐시 + 현재 것 추가)
V: 모든 토큰의 Value (이전 것 캐시 + 현재 것 추가)

Attention(Q_new, K_cached + K_new, V_cached + V_new)
```

### 구현

```python
class AttentionWithKVCache(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None, use_cache=True):
        """
        Args:
            x: (batch, seq_len, d_model) - Prefill 시 전체, Generate 시 1
            kv_cache: (past_key, past_value) 튜플
            use_cache: 캐시 사용 여부

        Returns:
            output: (batch, seq_len, d_model)
            new_kv_cache: (new_key, new_value)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Q, K, V 계산 (현재 토큰만)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Head 분리
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # KV Cache 사용
        if kv_cache is not None:
            past_key, past_value = kv_cache
            K = torch.cat([past_key, K], dim=2)  # seq_len 차원으로 concat
            V = torch.cat([past_value, V], dim=2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Causal mask (현재 위치까지만)
        # Q의 위치에서 K의 모든 과거 위치 참조 가능
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        # Head 합치기
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        # 새 캐시 반환
        new_kv_cache = (K, V) if use_cache else None

        return output, new_kv_cache
```

### 전체 모델에서의 사용

```python
class TransformerWithKVCache(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlockWithCache(d_model, num_heads)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, past_key_values=None, use_cache=True):
        """
        Args:
            input_ids: (batch, seq_len)
            past_key_values: List of (key, value) per layer
            use_cache: 캐시 사용 여부

        Returns:
            logits: (batch, seq_len, vocab_size)
            new_past_key_values: List of (key, value)
        """
        x = self.embed(input_ids)

        new_past_key_values = []

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            x, new_kv = layer(x, kv_cache=past_kv, use_cache=use_cache)
            new_past_key_values.append(new_kv)

        logits = self.lm_head(x)

        return logits, new_past_key_values if use_cache else None
```

## 3.3.3 메모리 사용량

### 계산 공식

```
KV Cache 메모리 = 2 × batch × num_layers × seq_len × num_heads × d_k × bytes_per_element
```

### 예시: LLaMA-7B

```python
# LLaMA-7B 파라미터
batch_size = 1
num_layers = 32
seq_len = 4096
num_heads = 32
d_k = 128  # d_model / num_heads = 4096 / 32
bytes_per_element = 2  # FP16

# KV Cache 메모리 (Key + Value)
kv_cache_memory = 2 * batch_size * num_layers * seq_len * num_heads * d_k * bytes_per_element
kv_cache_memory_gb = kv_cache_memory / (1024**3)

print(f"KV Cache: {kv_cache_memory_gb:.2f} GB")  # ~2 GB per batch

# 시퀀스 길이에 따른 증가
for seq_len in [2048, 4096, 8192, 32768, 128000]:
    mem = 2 * 1 * 32 * seq_len * 32 * 128 * 2 / (1024**3)
    print(f"seq_len={seq_len:6d}: {mem:.2f} GB")
```

## 3.3.4 Prefill vs Decode

### 두 단계

1. **Prefill**: 프롬프트 전체 처리, KV Cache 초기화
2. **Decode**: 토큰 하나씩 생성, KV Cache 업데이트

```
┌─────────────────────────────────────────────────────┐
│                     Prefill Phase                    │
│  "What is the capital of France?"                   │
│  → 전체 시퀀스 병렬 처리                              │
│  → KV Cache 초기화                                   │
│  → Compute-bound (GPU 연산 중심)                     │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│                     Decode Phase                     │
│  토큰 1개씩 생성: "The" → "capital" → "is" → ...    │
│  → KV Cache 읽기/쓰기                                │
│  → Memory-bound (메모리 대역폭 중심)                  │
└─────────────────────────────────────────────────────┘
```

### 특성 비교

| | Prefill | Decode |
|---|---------|--------|
| 처리 토큰 | 프롬프트 전체 | 1개씩 |
| 병렬성 | 높음 | 낮음 |
| 병목 | Compute | Memory |
| Batch 효율 | 높음 | 높음 (continuous batching) |

## 3.3.5 KV Cache 최적화

### 1. Quantized KV Cache

KV를 INT8/INT4로 저장:

```python
# FP16 → INT8 (메모리 50% 절약)
def quantize_kv_cache(key, value):
    # Per-tensor quantization
    key_scale = key.abs().max() / 127
    value_scale = value.abs().max() / 127

    key_int8 = (key / key_scale).round().to(torch.int8)
    value_int8 = (value / value_scale).round().to(torch.int8)

    return key_int8, value_int8, key_scale, value_scale

def dequantize_kv_cache(key_int8, value_int8, key_scale, value_scale):
    key = key_int8.float() * key_scale
    value = value_int8.float() * value_scale
    return key, value
```

### 2. Sliding Window Attention

고정 윈도우만 캐시 (Mistral):

```python
def sliding_window_attention(Q, K, V, window_size=4096):
    """
    최근 window_size 토큰만 attend
    KV Cache 크기 고정
    """
    seq_len = K.size(2)
    if seq_len > window_size:
        K = K[:, :, -window_size:, :]
        V = V[:, :, -window_size:, :]

    return scaled_dot_product_attention(Q, K, V)
```

### 3. Multi-Query Attention (MQA)

Key, Value를 모든 head가 공유:

```python
# 기존: num_heads개의 K, V
# MQA: 1개의 K, V (모든 head 공유)

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)  # num_heads개
        self.W_k = nn.Linear(d_model, self.d_k)  # 1개만!
        self.W_v = nn.Linear(d_model, self.d_k)  # 1개만!
        self.W_o = nn.Linear(d_model, d_model)
```

**메모리 절약**: KV Cache 1/num_heads

### 4. Grouped-Query Attention (GQA)

MQA와 MHA의 중간 (LLaMA 2):

```python
# num_heads = 32, num_kv_heads = 8
# 4개의 Q head가 1개의 K, V head 공유
# KV Cache: 1/4로 감소

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        # ...
```

## 3.3.6 PagedAttention (vLLM)

### 개념

KV Cache를 고정 크기 "페이지"로 관리:

```
┌──────────────────────────────────────┐
│           KV Cache Pool              │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐        │
│  │ P1 │ │ P2 │ │ P3 │ │Free│        │
│  └────┘ └────┘ └────┘ └────┘        │
│                                      │
│  Request A: [P1] → [P2] → ...       │
│  Request B: [P3] → [P1] → ...       │
│                                      │
│  비연속 메모리 허용 → 단편화 감소    │
└──────────────────────────────────────┘
```

### 장점

- 메모리 단편화 감소
- 동적 배치 크기
- 프리픽스 공유 가능

## 3.3.7 실습 체크리스트

- [ ] KV Cache 있는/없는 추론 시간 비교
- [ ] 시퀀스 길이별 KV Cache 메모리 계산
- [ ] GQA 구현
- [ ] Sliding Window Attention 이해

## 3.3.8 핵심 요약

| 개념 | 설명 |
|------|------|
| KV Cache | K, V 저장하여 중복 계산 방지 |
| Prefill | 프롬프트 처리, 캐시 초기화 |
| Decode | 토큰 생성, 캐시 업데이트 |
| MQA/GQA | KV head 공유로 캐시 감소 |
| PagedAttention | 페이지 단위 캐시 관리 |

## 다음 단계

[3.4 Inference 특성](04-inference.md)에서 추론의 성능 특성을 다룬다.
