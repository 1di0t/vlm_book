---
---

# 2.4 Encoder vs Decoder

Transformer 아키텍처는 세 가지 변형이 있다. 각각의 특성과 사용처를 이해하자.

## 2.4.1 세 가지 아키텍처

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Encoder-only   │  │  Decoder-only   │  │ Encoder-Decoder │
│                 │  │                 │  │                 │
│    [BERT]       │  │    [GPT]        │  │    [T5]         │
│                 │  │                 │  │                 │
│  양방향 문맥    │  │  단방향 (좌→우) │  │  입출력 분리    │
│                 │  │                 │  │                 │
│  분류, NER      │  │  텍스트 생성    │  │  번역, 요약     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## 2.4.2 Encoder-only (BERT 스타일)

### 특징

- **양방향 Attention**: 모든 토큰이 서로를 볼 수 있음
- **Masked Language Model (MLM)**: 일부 토큰을 마스킹하고 예측
- **[CLS] 토큰**: 문장 전체 표현

### Attention 패턴

```
       I    love   cats
I    [1.0   1.0    1.0]    ← 모든 위치 볼 수 있음
love [1.0   1.0    1.0]
cats [1.0   1.0    1.0]
```

### 구조

```python
class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Embedding
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        x = self.embedding(x) + self.pos_embedding(positions)

        # Encoder layers (양방향)
        for layer in self.layers:
            x, _ = layer(x, mask)

        return self.norm(x)
```

### 사용 사례

| 태스크 | 설명 |
|--------|------|
| 텍스트 분류 | [CLS] 토큰으로 분류 |
| NER | 각 토큰을 태깅 |
| 문장 유사도 | 두 문장의 [CLS] 비교 |
| QA (Extractive) | 답 위치 예측 |

### 대표 모델

- **BERT**: 원조 Encoder-only
- **RoBERTa**: BERT 학습 개선
- **ALBERT**: 파라미터 효율화
- **DeBERTa**: Disentangled Attention

## 2.4.3 Decoder-only (GPT 스타일)

### 특징

- **단방향 Attention**: 현재 위치 이전만 볼 수 있음 (Causal)
- **Autoregressive Generation**: 토큰을 하나씩 생성
- **Next Token Prediction**: 다음 토큰 예측

### Attention 패턴 (Causal Mask)

```
       I    love   cats
I    [1.0   0.0    0.0]    ← I는 자기만 봄
love [1.0   1.0    0.0]    ← love는 I, love 봄
cats [1.0   1.0    1.0]    ← cats는 모두 봄
```

### 구조

```python
class GPTDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(2048, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        seq_len = x.size(1)

        # Causal mask 생성
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        # Embedding
        positions = torch.arange(seq_len, device=x.device)
        x = self.embedding(x) + self.pos_embedding(positions)

        # Decoder layers (단방향)
        for layer in self.layers:
            x, _ = layer(x, causal_mask)

        x = self.norm(x)

        # LM head (다음 토큰 예측)
        logits = self.lm_head(x)

        return logits

    def generate(self, prompt, max_new_tokens=50, temperature=1.0):
        """Autoregressive generation"""
        self.eval()
        x = prompt

        for _ in range(max_new_tokens):
            logits = self.forward(x)
            next_token_logits = logits[:, -1, :] / temperature

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            x = torch.cat([x, next_token], dim=1)

        return x
```

### 사용 사례

| 태스크 | 설명 |
|--------|------|
| 텍스트 생성 | 주어진 프롬프트 이어쓰기 |
| 코드 생성 | 코드 자동 완성 |
| 대화 | 챗봇 응답 |
| 번역 (Zero-shot) | 프롬프트로 번역 지시 |

### 대표 모델

- **GPT 시리즈**: OpenAI (GPT-2, GPT-3, GPT-4)
- **LLaMA**: Meta
- **Mistral/Mixtral**: Mistral AI
- **Qwen**: Alibaba
- **Claude**: Anthropic

### 왜 Decoder-only가 대세인가?

1. **단순성**: 하나의 구조로 모든 태스크 처리
2. **스케일링 효율**: 파라미터 증가 시 성능 향상 예측 가능
3. **Few-shot Learning**: 프롬프트만으로 새 태스크 수행
4. **통합 학습**: 다양한 데이터로 단일 목적함수 학습

## 2.4.4 Encoder-Decoder (T5 스타일)

### 특징

- **Encoder**: 입력을 양방향으로 처리
- **Decoder**: 출력을 단방향으로 생성
- **Cross-Attention**: Decoder가 Encoder 출력을 참조

### 구조

```python
class TransformerEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers):
        super().__init__()

        # 공유 임베딩
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        # Decoder (Cross-Attention 포함)
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # Encode source
        enc_output = self.encode(src)

        # Decode target
        dec_output = self.decode(tgt, enc_output)

        return self.lm_head(dec_output)

    def encode(self, src):
        x = self.embedding(src)
        for layer in self.encoder_layers:
            x, _ = layer(x)
        return x

    def decode(self, tgt, enc_output):
        x = self.embedding(tgt)
        causal_mask = torch.tril(torch.ones(tgt.size(1), tgt.size(1)))

        for layer in self.decoder_layers:
            x = layer(x, enc_output, causal_mask)
        return x
```

### Decoder Block with Cross-Attention

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        # Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-Attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = FeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, causal_mask):
        # Self-Attention (causal)
        attn_out, _ = self.self_attention(x, x, x, causal_mask)
        x = x + attn_out
        x = self.norm1(x)

        # Cross-Attention (encoder 참조)
        cross_out, _ = self.cross_attention(
            Q=x,              # Decoder의 query
            K=enc_output,     # Encoder의 key
            V=enc_output      # Encoder의 value
        )
        x = x + cross_out
        x = self.norm2(x)

        # FFN
        x = x + self.ffn(x)
        x = self.norm3(x)

        return x
```

### 사용 사례

| 태스크 | 설명 |
|--------|------|
| 번역 | 입력 언어 → 출력 언어 |
| 요약 | 긴 문서 → 짧은 요약 |
| QA (Abstractive) | 질문 → 생성된 답 |
| STT/TTS | 음성 ↔ 텍스트 |

### 대표 모델

- **T5**: Text-to-Text Transfer Transformer
- **BART**: Denoising Autoencoder
- **mT5**: 다국어 T5
- **Whisper**: OpenAI 음성 인식

## 2.4.5 VLM에서의 아키텍처

Vision-Language Model은 주로 **Decoder-only + Vision Encoder**:

```
┌──────────────────────────────────────────┐
│                                          │
│  ┌──────────┐    ┌──────────────────┐   │
│  │  Vision  │    │                  │   │
│  │  Encoder │───▶│   LLM Decoder    │   │
│  │  (ViT)   │    │   (GPT-style)    │   │
│  └──────────┘    │                  │   │
│       ▲          │                  │   │
│       │          └──────────────────┘   │
│    [Image]              ▲               │
│                         │               │
│                      [Text]             │
└──────────────────────────────────────────┘
```

### 연결 방식

1. **Cross-Attention**: Flamingo
2. **Projection/Connector**: LLaVA
3. **Early Fusion**: Fuyu

## 2.4.6 아키텍처 비교 요약

| 특성 | Encoder-only | Decoder-only | Encoder-Decoder |
|------|--------------|--------------|-----------------|
| Attention | 양방향 | 단방향 (Causal) | Enc: 양방향, Dec: Causal + Cross |
| 주 학습 목표 | MLM | Next Token | Seq2Seq |
| 대표 모델 | BERT | GPT, LLaMA | T5, BART |
| 생성 능력 | 약함 | 강함 | 강함 |
| 분류 능력 | 강함 | Fine-tuning 필요 | 중간 |
| 현재 트렌드 | 특수 목적 | 범용 LLM | 특수 목적 |

## 2.4.7 Prefix LM

Decoder-only와 Encoder-Decoder의 중간:

```
Prefix (양방향)    Target (단방향)
[What is AI?]  →  [AI is...]
    ↑  ↑  ↑        ↑ ↑
  서로 참조      왼쪽만 참조
```

```python
def create_prefix_lm_mask(prefix_len, total_len):
    """Prefix LM mask"""
    mask = torch.zeros(total_len, total_len)

    # Prefix 부분: 양방향
    mask[:prefix_len, :prefix_len] = 1

    # Target 부분: prefix + causal
    for i in range(prefix_len, total_len):
        mask[i, :prefix_len] = 1  # prefix 볼 수 있음
        mask[i, prefix_len:i+1] = 1  # 자기까지 볼 수 있음

    return mask
```

U-PaLM, PaLM 2 등에서 사용.

## 2.4.8 실습 체크리스트

- [ ] BERT 스타일 Encoder 구현
- [ ] GPT 스타일 Decoder 구현
- [ ] Causal Mask 적용 확인
- [ ] Cross-Attention이 있는 Decoder 구현
- [ ] 세 아키텍처의 attention 패턴 시각화

## 2.4.9 핵심 요약

| 아키텍처 | 핵심 | 대표 사용처 |
|---------|------|------------|
| Encoder-only | 양방향 이해 | 분류, 추출 |
| Decoder-only | 단방향 생성 | LLM, 챗봇 |
| Encoder-Decoder | 입출력 분리 | 번역, 요약 |

**현재 트렌드**: Decoder-only가 범용 AI의 주류. VLM도 대부분 Decoder-only LLM 기반.

## 다음 단계

[Chapter 3: LLM 기초](../chapter03-llm-basics/01-tokenization.md)에서 Tokenization과 실제 LLM 동작을 다룬다.
