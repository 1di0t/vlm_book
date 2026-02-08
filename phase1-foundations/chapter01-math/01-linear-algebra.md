# 1.1 선형대수 (Linear Algebra)

딥러닝의 모든 연산은 행렬과 벡터로 표현된다. Transformer의 Attention 메커니즘을 이해하려면 선형대수가 필수다.

## 1.1.1 벡터 (Vector)

### 벡터의 정의

벡터는 크기와 방향을 가진 수학적 객체다. 딥러닝에서 벡터는 데이터 포인트, 임베딩, 특징(feature)을 표현한다.

```
v = [v₁, v₂, ..., vₙ]ᵀ
```

**예시: 단어 임베딩**
- "king" → [0.25, 0.87, -0.12, ..., 0.45] (768차원)
- "queen" → [0.28, 0.91, -0.08, ..., 0.42] (768차원)

### 벡터 연산

**덧셈과 스칼라 곱**

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 벡터 덧셈
c = a + b  # [5, 7, 9]

# 스칼라 곱
d = 2 * a  # [2, 4, 6]
```

**내적 (Dot Product)**

두 벡터의 유사도를 측정하는 핵심 연산. Attention의 Query-Key 유사도 계산에 사용된다.

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = |\mathbf{a}||\mathbf{b}|\cos\theta
$$

```python
# 내적 계산
dot_product = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32

# 또는
dot_product = a @ b  # 32
```

**내적의 기하학적 의미**
- 두 벡터가 같은 방향: 내적 > 0
- 두 벡터가 직교: 내적 = 0
- 두 벡터가 반대 방향: 내적 < 0

### 코사인 유사도 (Cosine Similarity)

벡터의 크기를 정규화한 유사도 측정. 임베딩 비교에 표준처럼 사용된다.

$$
\text{cosine\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}||\mathbf{b}|}
$$

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 예시
embedding_king = np.array([0.25, 0.87, -0.12])
embedding_queen = np.array([0.28, 0.91, -0.08])

sim = cosine_similarity(embedding_king, embedding_queen)  # 약 0.99
```

## 1.1.2 행렬 (Matrix)

### 행렬의 정의

행렬은 2차원 배열로, 벡터들의 집합이다. 딥러닝에서 가중치(weight), 배치 데이터, Attention 점수를 표현한다.

```
A = | a₁₁  a₁₂  ...  a₁ₙ |
    | a₂₁  a₂₂  ...  a₂ₙ |
    | ...  ...  ...  ... |
    | aₘ₁  aₘ₂  ...  aₘₙ |
```

**Shape 표기**: (m, n) = m개 행, n개 열

### 행렬 곱셈 (Matrix Multiplication)

딥러닝의 핵심 연산. 신경망의 선형 변환은 모두 행렬 곱셈이다.

$$
C = AB \quad \text{where} \quad c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}
$$

**Shape 규칙**
- A: (m, n)
- B: (n, p)
- C = AB: (m, p)

```python
A = np.array([[1, 2],
              [3, 4]])  # (2, 2)
B = np.array([[5, 6],
              [7, 8]])  # (2, 2)

C = A @ B  # 또는 np.matmul(A, B)
# [[19, 22],
#  [43, 50]]
```

**Attention에서의 행렬 곱**

```python
# Query, Key, Value 행렬
# Q: (batch, seq_len, d_k)
# K: (batch, seq_len, d_k)
# V: (batch, seq_len, d_v)

# Attention 점수 계산
scores = Q @ K.transpose(-2, -1)  # (batch, seq_len, seq_len)

# Attention 가중치 적용
output = scores @ V  # (batch, seq_len, d_v)
```

### 전치 행렬 (Transpose)

행과 열을 바꾼 행렬. Attention에서 Key 전치에 사용된다.

$$
(A^T)_{ij} = A_{ji}
$$

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # (2, 3)

A_T = A.T  # (3, 2)
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

### 역행렬 (Inverse Matrix)

$$
AA^{-1} = A^{-1}A = I
$$

역행렬은 선형 방정식 풀이나 최적화에 사용되지만, 딥러닝에서 직접 계산하는 경우는 드물다.

```python
A = np.array([[1, 2],
              [3, 4]])

A_inv = np.linalg.inv(A)
# [[-2.0,  1.0],
#  [ 1.5, -0.5]]

# 검증: A @ A_inv ≈ I
np.allclose(A @ A_inv, np.eye(2))  # True
```

### Broadcasting

NumPy/PyTorch에서 shape이 다른 배열 간 연산을 자동으로 확장하는 메커니즘.

```python
# (3, 4) + (4,) → (3, 4)
A = np.ones((3, 4))
b = np.array([1, 2, 3, 4])
C = A + b  # b가 (1, 4)로 확장되어 각 행에 더해짐

# (3, 1) + (1, 4) → (3, 4)
x = np.array([[1], [2], [3]])  # (3, 1)
y = np.array([1, 2, 3, 4])     # (4,) → (1, 4)
z = x + y  # (3, 4)
```

## 1.1.3 고유값과 고유벡터 (Eigenvalue & Eigenvector)

### 정의

행렬 A에 대해 다음을 만족하는 벡터 v와 스칼라 λ:

$$
A\mathbf{v} = \lambda\mathbf{v}
$$

- λ: 고유값 (eigenvalue)
- v: 고유벡터 (eigenvector)

**의미**: 행렬 A에 의한 변환에서 방향이 바뀌지 않고 크기만 λ배 변하는 특별한 벡터.

```python
A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"고유값: {eigenvalues}")      # [5., 2.]
print(f"고유벡터:\n{eigenvectors}")  # 열 단위로 고유벡터
```

### 딥러닝에서의 활용

1. **PCA (Principal Component Analysis)**: 차원 축소
2. **공분산 행렬 분석**: 데이터 분포 이해
3. **Attention 이해**: Attention 행렬의 고유벡터는 정보 흐름 패턴을 보여줌
4. **학습 안정성**: 가중치 행렬의 고유값이 1보다 크면 gradient explosion 위험

### Spectral Norm

행렬의 최대 고유값(singular value). 딥러닝에서 Lipschitz 조건 만족을 위해 사용된다.

$$
\|A\|_2 = \sigma_{\max}(A) = \sqrt{\lambda_{\max}(A^TA)}
$$

```python
# Spectral Norm 계산
def spectral_norm(A):
    return np.linalg.norm(A, ord=2)

A = np.random.randn(3, 3)
print(f"Spectral Norm: {spectral_norm(A)}")
```

## 1.1.4 행렬 분해 (Matrix Decomposition)

### SVD (Singular Value Decomposition)

모든 행렬을 세 행렬의 곱으로 분해:

$$
A = U\Sigma V^T
$$

- U: 왼쪽 특이벡터 (m × m, 직교 행렬)
- Σ: 특이값 대각 행렬 (m × n)
- V: 오른쪽 특이벡터 (n × n, 직교 행렬)

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

U, S, Vt = np.linalg.svd(A)

print(f"U shape: {U.shape}")    # (2, 2)
print(f"S: {S}")                # 특이값들
print(f"Vt shape: {Vt.shape}")  # (3, 3)

# 복원
S_matrix = np.zeros((2, 3))
np.fill_diagonal(S_matrix, S)
A_reconstructed = U @ S_matrix @ Vt
```

### LoRA와 Low-Rank Approximation

LoRA(Low-Rank Adaptation)는 SVD의 원리를 활용한다.

전체 가중치 W (d × d)를 업데이트하는 대신, 작은 행렬 두 개로 분해:

$$
\Delta W = BA
$$

- B: (d, r) 행렬
- A: (r, d) 행렬
- r << d (rank가 훨씬 작음)

```python
# Full fine-tuning: d*d 파라미터
d = 4096
full_params = d * d  # 16,777,216

# LoRA: r*(d+d) 파라미터
r = 8
lora_params = r * (d + d)  # 65,536

print(f"파라미터 감소율: {lora_params / full_params * 100:.2f}%")  # 0.39%
```

## 1.1.5 Attention과 선형대수

Self-Attention은 선형대수의 집약체다.

### Attention 수식

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**선형대수 관점 분석**:

1. **QKᵀ**: Query와 Key의 내적 → 유사도 행렬 (seq_len × seq_len)
2. **√dₖ로 나눔**: 내적 값 정규화 (값이 너무 커지면 softmax가 뾰족해짐)
3. **Softmax**: 행 단위 정규화 → 확률 분포로 변환
4. **×V**: 가중 평균 계산 → 최종 출력

```python
import numpy as np

def attention(Q, K, V):
    d_k = K.shape[-1]

    # Step 1: Q와 K의 내적 (유사도)
    scores = Q @ K.T  # (seq_len, seq_len)

    # Step 2: 스케일링
    scores = scores / np.sqrt(d_k)

    # Step 3: Softmax (행 단위)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Step 4: Value와 곱
    output = attention_weights @ V  # (seq_len, d_v)

    return output, attention_weights

# 예시
seq_len, d_k, d_v = 4, 8, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

output, weights = attention(Q, K, V)
print(f"Output shape: {output.shape}")      # (4, 8)
print(f"Attention weights:\n{weights}")     # (4, 4), 각 행 합 = 1
```

## 1.1.6 실습 체크리스트

- [ ] NumPy로 벡터 내적 계산하기
- [ ] 행렬 곱셈의 shape 규칙 이해하기
- [ ] Attention 수식을 코드로 구현하기
- [ ] 코사인 유사도로 단어 임베딩 비교하기
- [ ] SVD로 행렬 분해 후 복원하기

## 1.1.7 핵심 요약

| 개념 | 딥러닝에서의 역할 |
|------|------------------|
| 벡터 내적 | Query-Key 유사도 계산 |
| 행렬 곱셈 | 선형 변환, Attention |
| 전치 행렬 | Key 전치 (Kᵀ) |
| 코사인 유사도 | 임베딩 비교 |
| SVD / Low-Rank | LoRA 파라미터 효율화 |

## 다음 단계

[1.2 미적분](02-calculus.md)에서 Backpropagation의 핵심인 Chain Rule을 다룬다.
