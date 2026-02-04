# B1. Attention 수식 유도

> Transformer의 핵심인 Self-Attention 메커니즘의 수학적 유도와 해석

---

## 1. Attention의 직관적 이해

### 1.1 왜 Attention이 필요한가?

**기존 방식의 한계 (RNN)**:
```
시퀀스: [단어1, 단어2, ..., 단어T]

RNN: 순차 처리 → 병목
- 먼 거리 의존성 학습 어려움
- 병렬화 불가
```

**Attention의 해결책**:
```
모든 위치를 직접 연결
- 거리와 무관하게 정보 접근
- 완전 병렬화 가능
```

### 1.2 Attention의 핵심 질문

```
"입력 시퀀스에서 어떤 부분에 집중할 것인가?"

Query (Q): 내가 찾는 것 - "무엇을 원하는가?"
Key (K): 정보의 인덱스 - "각 위치가 어떤 정보를 담고 있는가?"
Value (V): 실제 정보 - "해당 위치의 내용은 무엇인가?"
```

---

## 2. Scaled Dot-Product Attention 유도

### 2.1 수식

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

### 2.2 단계별 유도

#### Step 1: 유사도 측정

**목표**: Query와 각 Key 간의 관련성 측정

**내적 기반 유사도**:
```
score(q, k) = q · k = Σᵢ qᵢkᵢ

기하학적 의미:
q · k = ‖q‖ × ‖k‖ × cos(θ)

→ 방향이 비슷할수록 높은 점수
```

**행렬 형태**:
```
Q ∈ ℝᵀˣᵈᵏ (T개의 query, 각 d_k 차원)
K ∈ ℝᵀˣᵈᵏ (T개의 key, 각 d_k 차원)

Scores = QK^T ∈ ℝᵀˣᵀ

scores[i,j] = Query_i와 Key_j의 유사도
```

#### Step 2: Scaling (√d_k로 나누기)

**문제 상황**:
```
q, k의 각 성분이 평균 0, 분산 1인 독립 확률변수라 가정

q · k = Σᵢ qᵢkᵢ

기댓값: E[q · k] = Σᵢ E[qᵢ]E[kᵢ] = 0
분산: Var(q · k) = Σᵢ Var(qᵢkᵢ) = d_k × 1 = d_k
```

**d_k가 커지면**:
```
d_k = 64일 때, 내적의 표준편차 = √64 = 8
d_k = 512일 때, 내적의 표준편차 = √512 ≈ 22.6

→ 값이 매우 커질 수 있음
```

**Softmax 문제**:
```
softmax(x)_i = exp(x_i) / Σⱼ exp(x_j)

x 값이 크면:
- 가장 큰 값에 거의 모든 확률 집중
- gradient가 거의 0 (saturation)
```

**해결책: Scaling**:
```
scaled_scores = QK^T / √d_k

분산: Var(scaled_scores) = d_k / d_k = 1

→ softmax 입력의 분산을 1로 정규화
```

#### Step 3: Softmax 적용

**확률 분포 변환**:
```
attention_weights = softmax(scaled_scores)

성질:
- 모든 값 ≥ 0
- 각 행의 합 = 1
- 높은 score → 높은 가중치
```

**수식**:
```
attention_weights[i,j] = exp(score[i,j]) / Σₖ exp(score[i,k])

의미: Query_i가 Key_j에 부여하는 attention 가중치
```

#### Step 4: Value 가중합

**최종 출력**:
```
Output = attention_weights × V

output[i] = Σⱼ attention_weights[i,j] × V[j]

의미: Query_i의 출력은 모든 Value의 가중합
(attention_weights가 가중치 역할)
```

### 2.3 전체 수식 정리

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Shape 추적:
Q: [B, T_q, d_k]
K: [B, T_k, d_k]
V: [B, T_k, d_v]

QK^T: [B, T_q, d_k] × [B, d_k, T_k] = [B, T_q, T_k]
softmax: [B, T_q, T_k]
Output: [B, T_q, T_k] × [B, T_k, d_v] = [B, T_q, d_v]
```

---

## 3. Multi-Head Attention 유도

### 3.1 동기

**단일 Attention의 한계**:
```
하나의 attention만으로는 다양한 관계를 동시에 포착하기 어려움
- 문법적 관계 (주어-동사)
- 의미적 관계 (동의어, 반의어)
- 위치적 관계 (인접 단어)
```

**해결책: Multiple Heads**:
```
여러 개의 독립적인 attention을 병렬 수행
각 head가 다른 관계에 집중
```

### 3.2 수식 유도

**입력 투영**:
```
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

W_i^Q ∈ ℝᵈᵐᵒᵈᵉˡ ˣ ᵈᵏ
W_i^K ∈ ℝᵈᵐᵒᵈᵉˡ ˣ ᵈᵏ
W_i^V ∈ ℝᵈᵐᵒᵈᵉˡ ˣ ᵈᵛ

각 head는 다른 투영을 학습
```

**Head 결합**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W^O

Concat: [B, T, h × d_v]
W^O ∈ ℝ⁽ʰˣᵈᵛ⁾ ˣ ᵈᵐᵒᵈᵉˡ
Output: [B, T, d_model]
```

### 3.3 차원 설계

**일반적 설정**:
```
d_model = 768 (예: BERT-base)
h = 12 (head 수)
d_k = d_v = d_model / h = 64
```

**파라미터 수**:
```
단일 head: d_model × d_model × 4 (Q, K, V, O)
Multi-head: d_model × d_k × h × 3 + d_model × d_model
         = d_model × d_model × 4 (동일!)

→ 파라미터 수 동일하면서 표현력 증가
```

### 3.4 구현

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 1. 선형 투영
        Q = self.W_Q(Q)  # [B, T, d_model]
        K = self.W_K(K)
        V = self.W_V(V)

        # 2. Head 분할
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # [B, H, T, d_k]

        # 3. Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # [B, H, T, T]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)  # [B, H, T, d_k]

        # 4. Head 결합
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # [B, T, d_model]

        # 5. 출력 투영
        output = self.W_O(output)

        return output, attn_weights
```

---

## 4. Self-Attention vs Cross-Attention

### 4.1 Self-Attention

```
Q, K, V 모두 같은 입력에서 유도

X → W_Q → Q
X → W_K → K
X → W_V → V

용도: 시퀀스 내부의 관계 학습
예: "The cat sat on the mat" → "cat"과 "sat"의 관계
```

### 4.2 Cross-Attention

```
Q는 한 시퀀스, K와 V는 다른 시퀀스에서 유도

Decoder X → W_Q → Q
Encoder Y → W_K → K
Encoder Y → W_V → V

용도: 두 시퀀스 간의 관계 학습
예: 번역에서 source와 target 연결
```

### 4.3 VLM에서의 Cross-Attention

```
텍스트 토큰이 이미지 패치를 참조:

Text → W_Q → Q        # "이 사진에서"
Image → W_K → K       # 이미지 패치들
Image → W_V → V

→ 텍스트가 관련 이미지 영역에 집중
```

---

## 5. Causal (Masked) Attention 유도

### 5.1 필요성

**자기회귀 생성**:
```
"I love" → 다음 단어 예측

조건: 이전 토큰만 볼 수 있음
- "love"를 예측할 때 "I"만 참조
- 미래 토큰 참조 금지 (정보 누출 방지)
```

### 5.2 Causal Mask

```
T = 4일 때 mask:

     [1]  토큰1은 토큰1만 봄
     [2]  토큰2는 토큰1,2 봄
     [3]  토큰3은 토큰1,2,3 봄
     [4]  토큰4는 모두 봄

    1  2  3  4
1 [ 1  0  0  0 ]
2 [ 1  1  0  0 ]
3 [ 1  1  1  0 ]
4 [ 1  1  1  1 ]

= 하삼각 행렬 (lower triangular)
```

### 5.3 마스킹 적용

```
scores = QK^T / √d_k              # [T, T]
scores = scores.masked_fill(mask == 0, -inf)

softmax 후:
- -inf → exp(-inf) = 0
- 미래 토큰에 대한 attention = 0
```

### 5.4 구현

```python
def causal_attention(Q, K, V):
    """
    Q, K, V: [B, H, T, d_k]
    """
    T = Q.size(2)

    # Causal mask 생성
    mask = torch.tril(torch.ones(T, T, device=Q.device))

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output
```

---

## 6. Attention의 Backward

### 6.1 Forward 정리

```
S = QK^T / √d_k          # scores
A = softmax(S)           # attention weights
O = AV                   # output
```

### 6.2 Backward 유도

**∂L/∂V**:
```
O = AV
∂L/∂V = A^T × ∂L/∂O

Shape: [B, H, T, T]^T × [B, H, T, d_v] = [B, H, T, d_v]
```

**∂L/∂A**:
```
O = AV
∂L/∂A = ∂L/∂O × V^T

Shape: [B, H, T, d_v] × [B, H, d_v, T] = [B, H, T, T]
```

**∂L/∂S** (Softmax backward):
```
A = softmax(S)

∂L/∂S = A ⊙ (∂L/∂A - Σⱼ ∂L/∂Aⱼ × Aⱼ)

또는 행렬 형태:
∂L/∂S = A ⊙ ∂L/∂A - A ⊙ (A × ∂L/∂A^T × 1)
```

**∂L/∂Q, ∂L/∂K**:
```
S = QK^T / √d_k

∂L/∂Q = (∂L/∂S × K) / √d_k
∂L/∂K = (∂L/∂S^T × Q) / √d_k
```

### 6.3 메모리 효율적 구현 (FlashAttention)

```
기존: S, A 전체 저장 → O(T²) 메모리

FlashAttention:
- S, A를 블록 단위로 계산
- 전체 저장 없이 online softmax
- Backward에서 재계산

메모리: O(T²) → O(T)
```

---

## 7. Attention 변형들

### 7.1 Additive Attention (Bahdanau)

```
score(q, k) = v^T × tanh(W_q × q + W_k × k)

특징:
- 비선형 유사도
- Dot-product보다 표현력 높음
- 계산 비용 높음 (실무에서 잘 안 씀)
```

### 7.2 Relative Position Attention

```
score(q_i, k_j) = q_i · k_j + q_i · r_{i-j} + u · k_j + v · r_{i-j}

r_{i-j}: 상대 위치 임베딩
u, v: 학습 파라미터

장점: 길이 외삽 (학습 길이보다 긴 시퀀스 처리)
```

### 7.3 Rotary Position Embedding (RoPE)

```
q_m = R_m × q    # m 위치의 회전 적용
k_n = R_n × k    # n 위치의 회전 적용

q_m · k_n = q · R_m^T × R_n × k = q · R_{n-m} × k

장점:
- 상대 위치 정보가 내적에 자연스럽게 포함
- 길이 외삽에 유리
- LLaMA, Qwen 등에서 사용
```

### 7.4 Grouped Query Attention (GQA)

```
기존 MHA: H개 head, 각각 Q, K, V
GQA: H개 Q head, G개 K/V head (G < H)

예: H=32, G=8
- 각 Q head 그룹이 K/V를 공유
- 메모리 절감 (KV cache 감소)
```

---

## 8. 계산 복잡도 분석

### 8.1 시간 복잡도

```
Q, K, V: [B, T, d]
Attention 연산: O(BT²d)

분석:
- QK^T: [B, T, d] × [B, d, T] = O(BT²d)
- Softmax: O(BT²)
- (Softmax)V: [B, T, T] × [B, T, d] = O(BT²d)

T가 지배적 → O(T²)
```

### 8.2 공간 복잡도

```
Attention matrix: [B, H, T, T]
메모리: O(BHT²)

예: B=32, H=32, T=4096
→ 32 × 32 × 4096² × 4 bytes = 68 GB (!)
```

### 8.3 효율적 Attention

| 방법 | 시간 | 공간 | 특징 |
|:-----|:-----|:-----|:-----|
| 기본 | O(T²) | O(T²) | 정확 |
| FlashAttention | O(T²) | O(T) | IO 최적화 |
| Sparse | O(T√T) | O(T) | 근사 |
| Linear | O(T) | O(T) | 근사 |

---

## 9. 수학적 해석

### 9.1 Kernel 관점

```
Attention(Q, K, V) = softmax(QK^T) × V

Kernel trick 적용:
K(q, k) = exp(q · k / √d)

softmax_i = K(q, k_i) / Σⱼ K(q, k_j)

→ RBF kernel의 근사로 해석 가능
```

### 9.2 정보 검색 관점

```
Query: 검색 쿼리
Key: 문서 인덱스
Value: 문서 내용

Attention = weighted retrieval
→ 연속적인 정보 검색
```

### 9.3 그래프 관점

```
Attention matrix A는 가중 인접 행렬

A_ij = 노드 i → 노드 j의 연결 강도

Softmax → 각 노드에서 나가는 엣지의 합 = 1
→ 확률적 그래프 순회
```

---

## 10. 요약

### 10.1 핵심 수식

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

구성요소:
- QK^T: 유사도 계산
- √d_k: 스케일링 (분산 정규화)
- softmax: 확률 분포 변환
- ×V: 가중합
```

### 10.2 Multi-Head

```
MultiHead = Concat(head_1, ..., head_h) × W_O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

장점: 다양한 관계 동시 학습
```

### 10.3 Causal Mask

```
하삼각 마스크로 미래 토큰 차단
→ 자기회귀 생성 지원
```

### 10.4 복잡도

```
시간: O(T²d)
공간: O(T²)

→ 긴 시퀀스에서 병목
→ FlashAttention, Sparse Attention 등으로 해결
```

---

> 💡 **본문 연결**
> - [1.2 Transformer 아키텍처](../../01_딥러닝_Transformer_기초/02_Transformer_아키텍처.md)
> - [부록 A1: 선형대수](../A_수학_기초/A1_선형대수.md)
> - [부록 B2: Backpropagation 유도](B2_Backpropagation_유도.md)
