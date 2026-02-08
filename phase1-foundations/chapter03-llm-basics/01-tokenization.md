---
---

# 3.1 Tokenization

LLM은 텍스트를 직접 처리하지 못한다. 먼저 토큰(숫자 ID)으로 변환해야 한다. 이 과정이 Tokenization이다.

## 3.1.1 토큰화의 필요성

### 왜 토큰화가 필요한가?

```
텍스트: "Hello, world!"
    ↓ Tokenization
토큰 ID: [15496, 11, 995, 0]
    ↓ Embedding
벡터: [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
```

신경망은 숫자만 처리할 수 있다. 텍스트 → 숫자 변환이 필수.

### 토큰화 수준

| 수준 | 예시 | 어휘 크기 | 장점 | 단점 |
|------|------|----------|------|------|
| Character | H, e, l, l, o | ~100 | OOV 없음 | 시퀀스 길어짐 |
| Word | Hello, world | 수만~수십만 | 의미 단위 | OOV 문제 |
| Subword | Hel, lo, wor, ld | 수만 | 균형 | 복잡한 알고리즘 |

**현재 표준**: Subword 토큰화 (BPE, WordPiece, SentencePiece)

## 3.1.2 BPE (Byte Pair Encoding)

### 알고리즘 개요

1. 모든 단어를 문자 단위로 분리
2. 가장 빈번한 문자 쌍을 병합하여 새 토큰 생성
3. 원하는 어휘 크기까지 반복

### 예시

```
corpus: "low lower lowest"

Step 0: 초기 어휘 = {l, o, w, e, r, s, t, _}
        토큰화: l o w _, l o w e r _, l o w e s t _

Step 1: 가장 빈번한 쌍 = (l, o) → 새 토큰 'lo'
        토큰화: lo w _, lo w e r _, lo w e s t _

Step 2: 가장 빈번한 쌍 = (lo, w) → 새 토큰 'low'
        토큰화: low _, low e r _, low e s t _

Step 3: 가장 빈번한 쌍 = (low, _) → 새 토큰 'low_'
        ...
```

### 구현

```python
from collections import defaultdict
import re

def get_stats(vocab):
    """어휘에서 인접 토큰 쌍의 빈도 계산"""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """가장 빈번한 쌍을 병합"""
    new_vocab = {}
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in vocab:
        new_word = pattern.sub(''.join(pair), word)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def train_bpe(corpus, num_merges):
    """BPE 학습"""
    # 초기 어휘: 문자 단위
    vocab = defaultdict(int)
    for word in corpus:
        vocab[' '.join(list(word)) + ' </w>'] += 1

    merges = []

    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break

        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
        merges.append(best_pair)

        print(f"Merge {i+1}: {best_pair}")

    return vocab, merges

# 예시
corpus = ["low", "lower", "lowest", "newer", "wider"]
vocab, merges = train_bpe(corpus, num_merges=10)
```

### BPE 토큰화

```python
def tokenize_bpe(text, merges):
    """학습된 BPE로 토큰화"""
    tokens = list(text) + ['</w>']

    for pair in merges:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                tokens = tokens[:i] + [''.join(pair)] + tokens[i+2:]
            else:
                i += 1

    return tokens

# 예시
text = "lowest"
tokens = tokenize_bpe(text, merges)
print(f"'{text}' → {tokens}")
```

## 3.1.3 WordPiece (BERT)

### BPE와의 차이

- BPE: 가장 빈번한 쌍 선택
- WordPiece: 우도(likelihood)를 최대화하는 쌍 선택

$$
\text{score}(x, y) = \frac{\text{freq}(xy)}{\text{freq}(x) \times \text{freq}(y)}
$$

### 토큰화 특징

- 단어 시작이 아닌 서브워드는 `##` 접두사
- 예: "unbelievable" → ["un", "##believ", "##able"]

```python
# HuggingFace tokenizers 사용
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize("unbelievable")
print(tokens)  # ['un', '##bel', '##ie', '##va', '##ble']
```

## 3.1.4 SentencePiece

### 특징

- 언어 독립적 (전처리 없이 raw text 처리)
- Unigram LM 또는 BPE 알고리즘 사용
- 공백도 특수 문자로 처리 (▁)

### 사용 예시

```python
import sentencepiece as spm

# 학습
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='my_tokenizer',
    vocab_size=32000,
    model_type='bpe'  # 또는 'unigram'
)

# 로드 및 사용
sp = spm.SentencePieceProcessor()
sp.load('my_tokenizer.model')

text = "Hello, world!"
tokens = sp.encode_as_pieces(text)
print(tokens)  # ['▁Hello', ',', '▁world', '!']

ids = sp.encode_as_ids(text)
print(ids)  # [1234, 5, 678, 9]
```

## 3.1.5 현대 LLM의 토크나이저

### GPT 계열 (tiktoken)

```python
import tiktoken

# GPT-4 토크나이저
enc = tiktoken.encoding_for_model("gpt-4")

text = "Hello, world!"
tokens = enc.encode(text)
print(f"토큰 ID: {tokens}")
print(f"토큰 수: {len(tokens)}")

# 디코드
decoded = enc.decode(tokens)
print(f"복원: {decoded}")
```

### LLaMA / Qwen (HuggingFace)

```python
from transformers import AutoTokenizer

# LLaMA 토크나이저
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

text = "Hello, world!"
encoded = tokenizer(text, return_tensors="pt")
print(f"input_ids: {encoded['input_ids']}")
print(f"attention_mask: {encoded['attention_mask']}")

# 디코드
decoded = tokenizer.decode(encoded['input_ids'][0])
print(f"복원: {decoded}")
```

## 3.1.6 어휘 크기 (Vocab Size)

### 트레이드오프

| 어휘 크기 | 시퀀스 길이 | 임베딩 파라미터 | 희귀 토큰 |
|----------|------------|----------------|----------|
| 작음 (8K) | 길어짐 | 적음 | 잘 분해됨 |
| 큼 (128K) | 짧아짐 | 많음 | OOV 가능 |

### 모델별 어휘 크기

| 모델 | 어휘 크기 |
|------|----------|
| GPT-2 | 50,257 |
| BERT | 30,522 |
| LLaMA | 32,000 |
| LLaMA 2 | 32,000 |
| Qwen | 151,936 |
| GPT-4 | ~100,000 |

### 한국어 고려사항

```python
# 영어 vs 한국어 토큰 효율
text_en = "Hello, how are you?"
text_ko = "안녕하세요, 어떻게 지내세요?"

# 영어 최적화 토크나이저
tokens_en = tokenizer.encode(text_en)
tokens_ko = tokenizer.encode(text_ko)

print(f"영어 토큰 수: {len(tokens_en)}")  # ~5
print(f"한국어 토큰 수: {len(tokens_ko)}")  # ~15-20 (비효율)
```

다국어 지원 모델(Qwen, Gemma)은 한국어 토큰도 효율적.

## 3.1.7 특수 토큰

### 일반적인 특수 토큰

| 토큰 | 용도 |
|------|------|
| `<bos>` / `<s>` | 시퀀스 시작 |
| `<eos>` / `</s>` | 시퀀스 끝 |
| `<pad>` | 패딩 |
| `<unk>` | 미등록 토큰 |
| `<mask>` | 마스킹 (BERT) |

### 챗 모델 특수 토큰

```python
# LLaMA 2 Chat 형식
prompt = """<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

What is the capital of France? [/INST]"""

# Qwen Chat 형식
prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
"""
```

## 3.1.8 토큰화 주의사항

### 1. 공백 처리

```python
text1 = "hello"
text2 = " hello"  # 앞에 공백

tokens1 = tokenizer.encode(text1)
tokens2 = tokenizer.encode(text2)

# 다른 토큰이 될 수 있음!
print(f"'hello' → {tokens1}")
print(f"' hello' → {tokens2}")
```

### 2. 특수 문자

```python
# 이모지, 특수 기호 처리
text = "Hello 👋 World! 🌍"
tokens = tokenizer.encode(text)
print(f"이모지 포함: {len(tokens)} 토큰")
```

### 3. 숫자

```python
# 숫자는 자릿수별로 토큰화될 수 있음
numbers = ["123", "1234567890", "3.14159"]
for num in numbers:
    tokens = tokenizer.encode(num)
    print(f"{num} → {len(tokens)} 토큰")
```

## 3.1.9 실습 체크리스트

- [ ] BPE 알고리즘 직접 구현
- [ ] HuggingFace 토크나이저 사용법 익히기
- [ ] 다양한 텍스트의 토큰 수 비교
- [ ] 특수 토큰 처리 이해
- [ ] 한국어/영어 토큰 효율 비교

## 3.1.10 핵심 요약

| 개념 | 설명 |
|------|------|
| BPE | 빈번한 쌍 병합 (GPT 계열) |
| WordPiece | 우도 기반 병합 (BERT) |
| SentencePiece | 언어 독립적, raw text |
| Vocab Size | 토큰 수 vs 파라미터 트레이드오프 |
| 특수 토큰 | BOS, EOS, PAD, UNK |

## 다음 단계

[3.2 Autoregressive Generation](02-autoregressive.md)에서 토큰 생성 과정을 다룬다.
