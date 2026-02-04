# Chapter 1.3: LLM 기초

## 개요

Large Language Model(LLM)의 핵심 개념인 토큰화, 자기회귀 생성, KV Cache를 다룬다.

---

## 1. Tokenization

### 1.1 토큰화 방식 비교

| 방식 | 장점 | 단점 | 예시 |
|------|------|------|------|
| Character-level | OOV 없음, 작은 vocab | 긴 시퀀스, 의미 손실 | 'h','e','l','l','o' |
| Word-level | 의미 보존 | OOV 문제, 큰 vocab | 'hello', 'world' |
| **Subword** | 균형 잡힌 해결책 | 토큰화 일관성 필요 | 'hel', 'lo' |

### 1.2 BPE (Byte Pair Encoding)

> **논문**: Sennrich et al. (2016). "Neural Machine Translation of Rare Words with Subword Units"
> - arXiv: https://arxiv.org/abs/1508.07909

**알고리즘:**
1. 문자 수준으로 시작
2. 가장 빈번한 인접 쌍을 병합
3. 원하는 vocab size까지 반복

```python
def train_bpe(corpus, vocab_size):
    """
    BPE 토크나이저 학습
    """
    # 초기화: 문자 수준
    vocab = set(char for word in corpus for char in word)
    word_freqs = Counter(corpus)

    # 단어를 문자로 분리
    splits = {word: list(word) for word in word_freqs}

    while len(vocab) < vocab_size:
        # 인접 쌍 빈도 계산
        pair_freqs = Counter()
        for word, freq in word_freqs.items():
            symbols = splits[word]
            for i in range(len(symbols) - 1):
                pair_freqs[(symbols[i], symbols[i+1])] += freq

        if not pair_freqs:
            break

        # 가장 빈번한 쌍 선택
        best_pair = max(pair_freqs, key=pair_freqs.get)

        # 새 토큰 추가
        new_token = best_pair[0] + best_pair[1]
        vocab.add(new_token)

        # 모든 단어에서 병합 수행
        for word in splits:
            symbols = splits[word]
            new_symbols = []
            i = 0
            while i < len(symbols):
                if (i < len(symbols) - 1 and
                    symbols[i] == best_pair[0] and
                    symbols[i+1] == best_pair[1]):
                    new_symbols.append(new_token)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            splits[word] = new_symbols

    return vocab, splits
```

### 1.3 SentencePiece

> **논문**: Kudo & Richardson (2018). "SentencePiece: A simple and language independent subword tokenizer"
> - arXiv: https://arxiv.org/abs/1808.06226
> - GitHub: https://github.com/google/sentencepiece

**특징:**
- 언어 독립적 (전처리 불필요)
- 공백을 특수 문자(▁)로 처리
- BPE와 Unigram 모델 지원

```python
import sentencepiece as spm

# 학습
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    model_type='bpe',  # 또는 'unigram'
    character_coverage=0.9995,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

# 사용
sp = spm.SentencePieceProcessor()
sp.load('tokenizer.model')

text = "Hello, world!"
tokens = sp.encode(text, out_type=str)
# ['▁Hello', ',', '▁world', '!']

ids = sp.encode(text, out_type=int)
# [1234, 5, 678, 9]

decoded = sp.decode(ids)
# "Hello, world!"
```

### 1.4 Vocab Size 트레이드오프

| Vocab Size | 시퀀스 길이 | 메모리 | OOV 처리 |
|------------|------------|--------|----------|
| 작음 (8K) | 길어짐 | 적음 | 어려움 |
| 보통 (32K) | 적절 | 적절 | 적절 |
| 큼 (128K) | 짧아짐 | 많음 | 우수 |

**현대 LLM vocab size:**
- GPT-4: ~100K
- LLaMA: 32K
- Qwen: 150K+

---

## 2. Autoregressive Generation

### 2.1 Next-Token Prediction

LLM은 다음 토큰의 확률 분포를 예측:

```
P(x_t | x_1, x_2, ..., x_{t-1})
```

**학습:**
```python
def compute_loss(model, input_ids):
    """
    Causal Language Modeling Loss
    """
    # input_ids: [BOS, t1, t2, t3, t4]
    # labels:    [t1, t2, t3, t4, EOS]

    logits = model(input_ids[:, :-1])  # (batch, seq-1, vocab)
    labels = input_ids[:, 1:]           # (batch, seq-1)

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=pad_token_id
    )
    return loss
```

### 2.2 생성 전략

#### Greedy Decoding

```python
def greedy_decode(model, input_ids, max_length):
    for _ in range(max_length):
        logits = model(input_ids)[:, -1, :]  # 마지막 위치
        next_token = logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == eos_token_id:
            break

    return input_ids
```

**문제점:** 항상 같은 출력, 다양성 부족

#### Temperature Sampling

```python
def sample_with_temperature(logits, temperature=1.0):
    """
    temperature < 1: 더 확정적 (sharp distribution)
    temperature > 1: 더 다양함 (flat distribution)
    """
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
```

#### Top-k Sampling

```python
def top_k_sampling(logits, k=50):
    """
    상위 k개 토큰만 샘플링 대상
    """
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    probs = F.softmax(top_k_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    next_token = top_k_indices.gather(-1, sampled_idx)
    return next_token
```

#### Top-p (Nucleus) Sampling

```python
def top_p_sampling(logits, p=0.9):
    """
    누적 확률이 p에 도달할 때까지의 토큰만 샘플링
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)

    # 누적 확률 p 초과하는 토큰 제거
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    probs = F.softmax(sorted_logits, dim=-1)

    sampled_idx = torch.multinomial(probs, num_samples=1)
    next_token = sorted_indices.gather(-1, sampled_idx)
    return next_token
```

#### Beam Search

```python
def beam_search(model, input_ids, beam_size=5, max_length=50):
    """
    여러 경로를 동시에 탐색
    """
    batch_size = input_ids.size(0)

    # 초기 beam
    beam_scores = torch.zeros(batch_size, beam_size, device=input_ids.device)
    beam_seqs = input_ids.unsqueeze(1).repeat(1, beam_size, 1)

    for step in range(max_length):
        # 각 beam에 대해 logits 계산
        logits = model(beam_seqs.view(-1, beam_seqs.size(-1)))[:, -1, :]
        vocab_size = logits.size(-1)

        # Log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.view(batch_size, beam_size, vocab_size)

        # 누적 점수
        next_scores = beam_scores.unsqueeze(-1) + log_probs

        # 상위 beam_size 선택
        next_scores = next_scores.view(batch_size, -1)
        top_scores, top_indices = torch.topk(next_scores, beam_size, dim=-1)

        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size

        # Beam 업데이트
        beam_seqs = beam_seqs.gather(1, beam_indices.unsqueeze(-1).expand_as(beam_seqs))
        beam_seqs = torch.cat([beam_seqs, token_indices.unsqueeze(-1)], dim=-1)
        beam_scores = top_scores

    # 최고 점수 beam 반환
    best_beam = beam_scores.argmax(dim=-1)
    return beam_seqs.gather(1, best_beam.view(-1, 1, 1).expand(-1, 1, beam_seqs.size(-1))).squeeze(1)
```

### 2.3 생성 파라미터 권장값

| 용도 | Temperature | Top-p | Top-k |
|------|-------------|-------|-------|
| 사실 기반 응답 | 0.0-0.3 | 0.1-0.5 | 10-20 |
| 일반 대화 | 0.5-0.7 | 0.8-0.9 | 40-50 |
| 창작/브레인스토밍 | 0.8-1.2 | 0.95 | 100+ |
| 코드 생성 | 0.0-0.2 | 0.5-0.8 | 20-40 |

---

## 3. KV Cache

### 3.1 필요성

Autoregressive 생성에서 매 스텝마다 전체 시퀀스를 다시 계산하면 비효율적:

```
Step 1: Attention(Q[1], K[1], V[1])
Step 2: Attention(Q[1:2], K[1:2], V[1:2])  # K[1], V[1] 재계산
Step 3: Attention(Q[1:3], K[1:3], V[1:3])  # K[1:2], V[1:2] 재계산
...
```

**KV Cache 사용:**
```
Step 1: cache_k[1], cache_v[1] 저장
Step 2: K = [cache_k[1], K[2]], V = [cache_v[1], V[2]]
Step 3: K = [cache_k[1:2], K[3]], V = [cache_v[1:2], V[3]]
```

### 3.2 구현

```python
class KVCache:
    def __init__(self):
        self.cache_k = None
        self.cache_v = None

    def update(self, new_k, new_v):
        if self.cache_k is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.cat([self.cache_k, new_k], dim=2)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=2)
        return self.cache_k, self.cache_v

    def get_seq_len(self):
        return 0 if self.cache_k is None else self.cache_k.size(2)


class CachedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None, use_cache=False):
        batch_size, seq_len, _ = x.shape

        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if kv_cache is not None:
            K, V = kv_cache.update(K, V)

        # Attention 계산 (Q는 현재 토큰만, K/V는 전체 캐시)
        attn_output, _ = scaled_dot_product_attention(Q, K, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(attn_output)

        return output
```

### 3.3 메모리 사용량 계산

```
KV Cache Memory = 2 × num_layers × batch_size × seq_len × num_heads × head_dim × bytes_per_element
```

**예시 (LLaMA-7B, FP16):**
```
2 × 32 × 1 × 2048 × 32 × 128 × 2 bytes
= 1,073,741,824 bytes
= 1 GB (per sequence)
```

### 3.4 Inference Phases

**1. Prefill Phase:**
- 전체 프롬프트를 한 번에 처리
- KV Cache 초기화
- Compute-bound (GPU 연산 병목)

**2. Decode Phase:**
- 한 번에 하나의 토큰 생성
- KV Cache 사용
- Memory-bound (메모리 대역폭 병목)

```python
def generate_with_cache(model, prompt_ids, max_new_tokens):
    """
    KV Cache를 사용한 효율적 생성
    """
    # Phase 1: Prefill
    kv_caches = [KVCache() for _ in range(model.num_layers)]
    logits = model.forward_with_cache(prompt_ids, kv_caches, use_cache=True)
    next_token = sample(logits[:, -1, :])

    generated = [next_token.item()]

    # Phase 2: Decode
    for _ in range(max_new_tokens - 1):
        # 새 토큰만 입력 (이전 토큰은 cache에)
        logits = model.forward_with_cache(
            next_token.unsqueeze(1),
            kv_caches,
            use_cache=True
        )
        next_token = sample(logits[:, -1, :])
        generated.append(next_token.item())

        if next_token.item() == eos_token_id:
            break

    return generated
```

### 3.5 KV Cache 최적화 기법

> **Survey**: "A Survey on Large Language Model Acceleration based on KV Cache Management"
> - arXiv: https://arxiv.org/abs/2412.19442

| 기법 | 방법 | 메모리 절감 |
|------|------|------------|
| **PagedAttention** | 페이지 단위 동적 할당 | 20-40% |
| **Sliding Window** | 고정 크기 윈도우만 유지 | 시퀀스 길이에 무관 |
| **KV Quantization** | INT8/FP8 양자화 | 50% |
| **Token Pruning** | 중요도 낮은 토큰 제거 | 가변 |

---

## 4. Batch Size와 Throughput

### 4.1 Prefill vs Decode Batching

**Prefill:**
- 다양한 길이 프롬프트 처리
- Padding으로 인한 비효율

**Decode:**
- 모든 시퀀스가 1 토큰씩 생성
- 배치 처리 용이

### 4.2 Continuous Batching

새 요청이 들어오면 진행 중인 배치에 동적 추가:

```python
class ContinuousBatcher:
    def __init__(self, model, max_batch_size):
        self.model = model
        self.max_batch_size = max_batch_size
        self.active_sequences = []
        self.pending_requests = []

    def add_request(self, prompt_ids):
        self.pending_requests.append(prompt_ids)

    def step(self):
        # 완료된 시퀀스 제거
        self.active_sequences = [
            seq for seq in self.active_sequences
            if not seq.is_finished
        ]

        # 새 요청 추가
        while (len(self.active_sequences) < self.max_batch_size
               and self.pending_requests):
            new_seq = self.pending_requests.pop(0)
            self.active_sequences.append(new_seq)

        # 배치 처리
        if self.active_sequences:
            batch = self.prepare_batch()
            outputs = self.model.forward_batch(batch)
            self.update_sequences(outputs)
```

---

## 핵심 참고 자료

### 논문
- **GPT-3** (Brown et al., 2020). "Language Models are Few-Shot Learners"
  - https://arxiv.org/abs/2005.14165
  - NeurIPS 2020

- **BPE** (Sennrich et al., 2016). "Neural Machine Translation of Rare Words with Subword Units"
  - https://arxiv.org/abs/1508.07909

- **SentencePiece** (Kudo & Richardson, 2018)
  - https://arxiv.org/abs/1808.06226

- **KV Cache Survey** (2024)
  - https://arxiv.org/abs/2412.19442

### 튜토리얼
- **Andrej Karpathy's minBPE**
  - https://github.com/karpathy/minbpe
  - BPE tokenizer 직접 구현

- **HuggingFace Tokenizers Documentation**
  - https://huggingface.co/docs/tokenizers

---

## 핵심 요약

| 개념 | 역할 | 현대 기법 |
|------|------|----------|
| Tokenization | 텍스트→토큰 변환 | BPE, SentencePiece |
| Autoregressive | 순차적 생성 | KV Cache |
| Sampling | 다양한 출력 생성 | Top-p, Temperature |
| KV Cache | 추론 효율화 | PagedAttention |
| Batching | 처리량 향상 | Continuous Batching |
