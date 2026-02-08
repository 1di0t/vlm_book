---
---

# 3.2 Autoregressive Generation

LLM은 텍스트를 한 토큰씩 순차적으로 생성한다. 이것이 Autoregressive Generation이다.

## 3.2.1 Next Token Prediction

### 핵심 원리

$$
P(x_1, x_2, ..., x_n) = \prod_{t=1}^{n} P(x_t | x_1, x_2, ..., x_{t-1})
$$

- 각 토큰은 이전 토큰들에 조건부로 생성
- 모델은 "다음에 올 확률이 높은 토큰"을 예측

### 예시

```
입력: "The cat sat on the"
      ↓ 모델
출력 확률: {"mat": 0.3, "floor": 0.2, "bed": 0.15, ...}
      ↓ 샘플링
선택: "mat"
      ↓ 반복
새 입력: "The cat sat on the mat"
```

## 3.2.2 Generation 과정

### 기본 알고리즘

```python
def generate(model, tokenizer, prompt, max_new_tokens=50):
    """기본 Autoregressive Generation"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    for _ in range(max_new_tokens):
        # 1. Forward pass
        with torch.no_grad():
            logits = model(input_ids).logits

        # 2. 마지막 토큰의 확률 분포
        next_token_logits = logits[:, -1, :]

        # 3. 확률로 변환
        probs = F.softmax(next_token_logits, dim=-1)

        # 4. 샘플링 (가장 간단: argmax)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)

        # 5. 시퀀스에 추가
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # 6. EOS 체크
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0])
```

## 3.2.3 Sampling 전략

### Greedy Decoding

항상 가장 확률 높은 토큰 선택:

```python
def greedy_decode(logits):
    return torch.argmax(logits, dim=-1)
```

**장점**: 결정론적, 빠름
**단점**: 다양성 없음, 반복 문제

### Temperature Sampling

Temperature로 분포의 "날카로움" 조절:

$$
P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

```python
def temperature_sample(logits, temperature=1.0):
    """Temperature가 낮으면 greedy, 높으면 uniform에 가까움"""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# T=0.1: 거의 greedy
# T=1.0: 원래 분포
# T=2.0: 더 평평한 분포 (다양성↑)
```

### Top-k Sampling

상위 k개 토큰만 고려:

```python
def top_k_sample(logits, k=50):
    """상위 k개 토큰에서만 샘플링"""
    values, indices = torch.topk(logits, k)

    # 나머지는 -inf로 마스킹
    logits_filtered = torch.full_like(logits, float('-inf'))
    logits_filtered.scatter_(1, indices, values)

    probs = F.softmax(logits_filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### Top-p (Nucleus) Sampling

누적 확률이 p가 될 때까지의 토큰만 고려:

```python
def top_p_sample(logits, p=0.9):
    """누적 확률 p까지의 토큰에서 샘플링"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # p를 초과하는 토큰 제거
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = float('-inf')

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### 조합 사용

실제로는 여러 전략을 조합:

```python
def sample_with_all_strategies(logits, temperature=0.8, top_k=50, top_p=0.9):
    # 1. Temperature 적용
    logits = logits / temperature

    # 2. Top-k 필터링
    if top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(1, indices, values)
        logits = logits_filtered

    # 3. Top-p 필터링
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')

    # 4. 샘플링
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

## 3.2.4 Beam Search

여러 후보를 동시에 유지하며 최적 시퀀스 탐색:

```python
def beam_search(model, tokenizer, prompt, beam_width=5, max_length=50):
    """Beam Search 구현"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # 초기 beam: (시퀀스, 누적 로그 확률)
    beams = [(input_ids, 0.0)]

    for _ in range(max_length):
        all_candidates = []

        for seq, score in beams:
            if seq[0, -1].item() == tokenizer.eos_token_id:
                all_candidates.append((seq, score))
                continue

            with torch.no_grad():
                logits = model(seq).logits[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)

            # 상위 beam_width개 토큰
            top_log_probs, top_indices = torch.topk(log_probs, beam_width)

            for i in range(beam_width):
                new_seq = torch.cat([seq, top_indices[:, i:i+1]], dim=-1)
                new_score = score + top_log_probs[0, i].item()
                all_candidates.append((new_seq, new_score))

        # 상위 beam_width개 유지
        beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # 모든 beam이 EOS면 종료
        if all(b[0][0, -1].item() == tokenizer.eos_token_id for b in beams):
            break

    return tokenizer.decode(beams[0][0][0])
```

**장점**: 더 좋은 전체 시퀀스
**단점**: 다양성 부족, 계산 비용

## 3.2.5 Repetition Penalty

반복을 줄이기 위한 페널티:

```python
def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    """이미 생성된 토큰에 페널티 적용"""
    for token_id in set(generated_ids.tolist()):
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty
    return logits
```

### Frequency Penalty vs Presence Penalty

- **Frequency Penalty**: 등장 횟수에 비례해 페널티
- **Presence Penalty**: 등장 여부에 따라 고정 페널티

```python
def apply_frequency_penalty(logits, token_counts, penalty=0.5):
    """등장 빈도에 비례한 페널티"""
    for token_id, count in token_counts.items():
        logits[0, token_id] -= penalty * count
    return logits

def apply_presence_penalty(logits, seen_tokens, penalty=0.5):
    """등장 여부에 따른 페널티"""
    for token_id in seen_tokens:
        logits[0, token_id] -= penalty
    return logits
```

## 3.2.6 Stopping Criteria

### 기본 조건

```python
def should_stop(generated_ids, tokenizer, max_length):
    # EOS 토큰
    if generated_ids[-1] == tokenizer.eos_token_id:
        return True

    # 최대 길이
    if len(generated_ids) >= max_length:
        return True

    return False
```

### Custom Stop Strings

```python
def check_stop_strings(text, stop_strings):
    """특정 문자열에서 생성 중단"""
    for stop in stop_strings:
        if stop in text:
            return True
    return False

# 예시
stop_strings = ["###", "\n\n", "<|endoftext|>"]
```

## 3.2.7 Streaming Generation

토큰 단위로 실시간 출력:

```python
def generate_streaming(model, tokenizer, prompt, max_new_tokens=100):
    """Streaming Generation (generator)"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]

        # 샘플링
        next_token = top_p_sample(next_token_logits)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # 토큰 yield
        token_text = tokenizer.decode(next_token[0])
        yield token_text

        if next_token.item() == tokenizer.eos_token_id:
            break

# 사용
for token in generate_streaming(model, tokenizer, "Once upon a time"):
    print(token, end="", flush=True)
```

## 3.2.8 HuggingFace generate()

실제 사용 시 HuggingFace의 `generate()` 활용:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

input_ids = tokenizer.encode("Hello, I am", return_tensors="pt")

# Greedy
output = model.generate(input_ids, max_new_tokens=50)

# Sampling
output = model.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)

# Beam Search
output = model.generate(
    input_ids,
    max_new_tokens=50,
    num_beams=5,
    early_stopping=True
)

print(tokenizer.decode(output[0]))
```

## 3.2.9 Generation 파라미터 가이드

| 파라미터 | 범위 | 효과 |
|---------|------|------|
| temperature | 0.1-2.0 | 낮으면 결정론적, 높으면 창의적 |
| top_k | 1-100 | 낮으면 안전, 높으면 다양 |
| top_p | 0.1-1.0 | 낮으면 집중, 높으면 다양 |
| repetition_penalty | 1.0-2.0 | 높으면 반복 감소 |
| num_beams | 1-10 | 높으면 품질↑, 다양성↓ |

### 태스크별 권장 설정

| 태스크 | temperature | top_p | 설명 |
|--------|-------------|-------|------|
| 코드 생성 | 0.2-0.4 | 0.9 | 정확성 중요 |
| 창작 글쓰기 | 0.7-1.0 | 0.9 | 창의성 중요 |
| 챗봇 | 0.6-0.8 | 0.9 | 균형 |
| 요약 | 0.3-0.5 | 0.9 | 사실성 중요 |

## 3.2.10 실습 체크리스트

- [ ] Greedy, Temperature, Top-k, Top-p 구현
- [ ] 각 전략의 출력 비교
- [ ] Beam Search 구현
- [ ] HuggingFace generate() 사용
- [ ] Streaming generation 구현

## 3.2.11 핵심 요약

| 개념 | 설명 |
|------|------|
| Autoregressive | 토큰 하나씩 순차 생성 |
| Temperature | 분포의 날카로움 조절 |
| Top-k | 상위 k개 토큰만 고려 |
| Top-p | 누적 확률 p까지 고려 |
| Beam Search | 여러 후보 유지하며 탐색 |

## 다음 단계

[3.3 KV Cache](03-kv-cache.md)에서 효율적인 추론을 위한 캐싱을 다룬다.
