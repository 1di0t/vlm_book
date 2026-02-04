# B2. Backpropagation 유도

> Neural Network 학습의 핵심 알고리즘인 역전파의 수학적 유도

---

## 1. 개요

### 1.1 Backpropagation이란?

**목적**: 손실 함수 L에 대한 모든 파라미터의 gradient 계산

```
∂L/∂W₁, ∂L/∂W₂, ..., ∂L/∂Wₙ 을 효율적으로 계산
```

**핵심 아이디어**: Chain Rule을 역방향으로 적용

### 1.2 왜 필요한가?

**Naive 접근 (수치 미분)**:
```
파라미터 수: N개
각 파라미터마다 2번의 forward pass 필요
총 복잡도: O(2N × Forward)
```

**Backpropagation**:
```
Forward 1번 + Backward 1번
총 복잡도: O(2 × Forward)
```

GPT-3 (175B 파라미터) 기준:
- 수치 미분: 3500억 번의 forward pass
- Backprop: 2번

---

## 2. 단일 뉴런에서의 유도

### 2.1 Setup

```
입력: x ∈ ℝ
가중치: w ∈ ℝ
편향: b ∈ ℝ
정답: y ∈ ℝ

Forward:
z = wx + b          (pre-activation)
a = σ(z)            (activation, σ = sigmoid)
L = (1/2)(a - y)²   (MSE loss)
```

### 2.2 Forward Pass 계산

```
x = 2, w = 0.5, b = 0.1, y = 1

z = 0.5 × 2 + 0.1 = 1.1
a = σ(1.1) = 1/(1 + e^(-1.1)) ≈ 0.75
L = (1/2)(0.75 - 1)² = 0.03125
```

### 2.3 Backward Pass 유도

**목표**: ∂L/∂w 계산

**Chain Rule 적용**:
```
∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w
```

**각 항 계산**:

1. **∂L/∂a** (손실 → 출력)
```
L = (1/2)(a - y)²
∂L/∂a = a - y = 0.75 - 1 = -0.25
```

2. **∂a/∂z** (출력 → pre-activation)
```
a = σ(z)
∂a/∂z = σ(z)(1 - σ(z)) = 0.75 × 0.25 = 0.1875
```

3. **∂z/∂w** (pre-activation → 가중치)
```
z = wx + b
∂z/∂w = x = 2
```

**최종 결과**:
```
∂L/∂w = (-0.25) × 0.1875 × 2 = -0.09375
```

**편향에 대한 gradient**:
```
∂L/∂b = ∂L/∂a × ∂a/∂z × ∂z/∂b
      = (-0.25) × 0.1875 × 1 = -0.046875
```

---

## 3. 2층 Neural Network 유도

### 3.1 Architecture

```
입력: x ∈ ℝⁿ (n차원 벡터)
은닉층: h = m 뉴런
출력층: o = k 클래스

층1: W₁ ∈ ℝⁿˣᵐ, b₁ ∈ ℝᵐ
층2: W₂ ∈ ℝᵐˣᵏ, b₂ ∈ ℝᵏ
```

### 3.2 Forward Pass

```
# 층 1
z₁ = W₁ᵀx + b₁        # [m]
a₁ = ReLU(z₁)          # [m]

# 층 2
z₂ = W₂ᵀa₁ + b₂       # [k]
ŷ = softmax(z₂)        # [k]

# 손실
L = CrossEntropy(y, ŷ) = -Σᵢ yᵢ log(ŷᵢ)
```

### 3.3 Backward Pass 유도

#### Step 1: 출력층 gradient (∂L/∂z₂)

Softmax + Cross-Entropy의 결합 미분:

```
∂L/∂z₂ = ŷ - y    # [k] 벡터

유도:
L = -Σᵢ yᵢ log(ŷᵢ)
ŷᵢ = softmax(z₂)ᵢ = exp(z₂ᵢ) / Σⱼ exp(z₂ⱼ)

∂L/∂z₂ⱼ = -Σᵢ yᵢ × (1/ŷᵢ) × ∂ŷᵢ/∂z₂ⱼ

Softmax Jacobian:
∂ŷᵢ/∂z₂ⱼ = ŷᵢ(δᵢⱼ - ŷⱼ)

대입 후 정리:
∂L/∂z₂ⱼ = ŷⱼ - yⱼ
```

#### Step 2: W₂의 gradient (∂L/∂W₂)

```
z₂ = W₂ᵀa₁ + b₂

∂L/∂W₂ = a₁ × (∂L/∂z₂)ᵀ    # [m, k] 행렬

배치 버전 (B개 샘플):
∂L/∂W₂ = (1/B) × A₁ᵀ × (Ŷ - Y)    # [m, k]
```

#### Step 3: b₂의 gradient (∂L/∂b₂)

```
∂L/∂b₂ = ∂L/∂z₂ = ŷ - y    # [k]

배치 버전:
∂L/∂b₂ = (1/B) × Σᵦ (ŷᵦ - yᵦ)    # [k]
```

#### Step 4: 은닉층으로 전파 (∂L/∂a₁)

```
∂L/∂a₁ = W₂ × ∂L/∂z₂    # [m]

z₂ = W₂ᵀa₁ 이므로
∂z₂/∂a₁ = W₂ᵀ
따라서 ∂L/∂a₁ = (W₂ᵀ)ᵀ × ∂L/∂z₂ = W₂ × ∂L/∂z₂
```

#### Step 5: ReLU 통과 (∂L/∂z₁)

```
∂L/∂z₁ = ∂L/∂a₁ ⊙ ReLU'(z₁)    # [m]

ReLU'(z) = {
    1 if z > 0
    0 if z ≤ 0
}

즉, ∂L/∂z₁ = ∂L/∂a₁ ⊙ (z₁ > 0)
```

#### Step 6: W₁의 gradient (∂L/∂W₁)

```
∂L/∂W₁ = x × (∂L/∂z₁)ᵀ    # [n, m]

배치 버전:
∂L/∂W₁ = (1/B) × Xᵀ × (∂L/∂Z₁)    # [n, m]
```

### 3.4 수식 요약

| 변수 | Gradient | Shape |
|:-----|:---------|:------|
| z₂ | ŷ - y | [B, k] |
| W₂ | (1/B) × A₁ᵀ × (Ŷ - Y) | [m, k] |
| b₂ | (1/B) × Σᵦ (ŷᵦ - yᵦ) | [k] |
| a₁ | (Ŷ - Y) × W₂ᵀ | [B, m] |
| z₁ | ∂L/∂a₁ ⊙ (Z₁ > 0) | [B, m] |
| W₁ | (1/B) × Xᵀ × ∂L/∂Z₁ | [n, m] |
| b₁ | (1/B) × Σᵦ ∂L/∂z₁ᵦ | [m] |

---

## 4. 일반화: L층 Network

### 4.1 Forward Pass (일반 형태)

```
for l = 1, 2, ..., L:
    z⁽ˡ⁾ = W⁽ˡ⁾ᵀa⁽ˡ⁻¹⁾ + b⁽ˡ⁾
    a⁽ˡ⁾ = σ⁽ˡ⁾(z⁽ˡ⁾)

여기서 a⁽⁰⁾ = x (입력)
```

### 4.2 Backward Pass (일반 형태)

```
# 출력층
δ⁽ᴸ⁾ = ∂L/∂z⁽ᴸ⁾ = (∂L/∂a⁽ᴸ⁾) ⊙ σ'⁽ᴸ⁾(z⁽ᴸ⁾)

# 역전파 (l = L-1, L-2, ..., 1)
δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾δ⁽ˡ⁺¹⁾) ⊙ σ'⁽ˡ⁾(z⁽ˡ⁾)

# Gradient 계산
∂L/∂W⁽ˡ⁾ = a⁽ˡ⁻¹⁾(δ⁽ˡ⁾)ᵀ
∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾
```

### 4.3 의사코드

```python
def backprop(network, x, y):
    """
    Backpropagation 알고리즘

    Args:
        network: 층들의 리스트 [(W1,b1), (W2,b2), ...]
        x: 입력 [batch, input_dim]
        y: 정답 [batch, output_dim]

    Returns:
        gradients: 각 층의 gradient
    """
    L = len(network)
    gradients = []

    # Forward pass (활성화 값 저장)
    activations = [x]  # a⁽⁰⁾ = x
    pre_activations = []

    a = x
    for l, (W, b) in enumerate(network):
        z = a @ W + b
        pre_activations.append(z)

        if l == L - 1:  # 마지막 층
            a = softmax(z)
        else:
            a = relu(z)
        activations.append(a)

    # Backward pass
    # 출력층: softmax + cross-entropy
    delta = activations[-1] - y  # ŷ - y

    for l in reversed(range(L)):
        W, b = network[l]
        a_prev = activations[l]

        # Gradient 계산
        dW = a_prev.T @ delta / len(x)
        db = delta.mean(axis=0)
        gradients.insert(0, (dW, db))

        # 이전 층으로 전파
        if l > 0:
            delta = (delta @ W.T) * relu_derivative(pre_activations[l-1])

    return gradients
```

---

## 5. Computational Graph 관점

### 5.1 개념

Neural Network를 연산의 그래프로 표현:

```
     x ──→ [×W₁] ──→ z₁ ──→ [ReLU] ──→ a₁
                                        │
                                        ↓
                              [×W₂] ──→ z₂
                                        │
            y ──────────────────────→ [Loss] ──→ L
```

### 5.2 Forward Mode vs Backward Mode

**Forward Mode (전방 미분)**:
- 입력 → 출력 방향으로 미분 전파
- 하나의 입력 변수에 대한 모든 출력의 미분
- 입력이 적고 출력이 많을 때 유리

**Backward Mode (역방향 미분)**:
- 출력 → 입력 방향으로 미분 전파
- 하나의 출력에 대한 모든 입력의 미분
- **Neural Network에 적합** (출력=손실 1개, 입력=파라미터 수억 개)

### 5.3 Local Gradient

각 연산 노드는 local gradient만 계산:

| 연산 | Forward | Local Gradient |
|:-----|:--------|:---------------|
| 덧셈 | c = a + b | ∂c/∂a = 1, ∂c/∂b = 1 |
| 곱셈 | c = a × b | ∂c/∂a = b, ∂c/∂b = a |
| ReLU | c = max(0, a) | ∂c/∂a = 1 if a > 0 else 0 |
| Sigmoid | c = σ(a) | ∂c/∂a = c(1-c) |
| MatMul | C = AB | ∂L/∂A = (∂L/∂C)Bᵀ |

---

## 6. 행렬 미분의 상세

### 6.1 선형 변환 z = Wx + b

**Forward**:
```
z: [batch, out]
W: [in, out]
x: [batch, in]
b: [out]

z = x @ W + b
```

**Backward**:
```
∂L/∂x = ∂L/∂z @ Wᵀ      # [batch, in]
∂L/∂W = xᵀ @ ∂L/∂z      # [in, out]
∂L/∂b = sum(∂L/∂z, axis=0)  # [out]
```

### 6.2 유도 (∂L/∂W)

```
z_ij = Σₖ x_ik × W_kj + b_j

∂z_ij/∂W_mn = x_im × δ_jn

∂L/∂W_mn = Σᵢⱼ (∂L/∂z_ij) × (∂z_ij/∂W_mn)
         = Σᵢ (∂L/∂z_in) × x_im
         = Σᵢ x_im × (∂L/∂z_in)
         = (xᵀ @ ∂L/∂z)_mn
```

### 6.3 PyTorch에서의 자동 미분

```python
import torch

# 자동 미분 활성화
x = torch.randn(32, 10, requires_grad=False)
W = torch.randn(10, 5, requires_grad=True)
b = torch.randn(5, requires_grad=True)

# Forward
z = x @ W + b
a = torch.relu(z)
loss = a.sum()

# Backward (자동 계산)
loss.backward()

# Gradient 확인
print(W.grad.shape)  # [10, 5]
print(b.grad.shape)  # [5]
```

---

## 7. Gradient Flow 문제

### 7.1 Vanishing Gradient

**원인**: 활성화 함수 미분값이 1보다 작음

```
Sigmoid: σ'(x) ≤ 0.25
Tanh: tanh'(x) ≤ 1

L층을 통과하면: gradient ≈ 0.25^L
L=10이면: gradient ≈ 10^(-6)
```

**해결책**:
- ReLU 사용 (미분 = 1)
- Residual Connection
- Batch/Layer Normalization
- 적절한 초기화

### 7.2 Exploding Gradient

**원인**: gradient가 기하급수적으로 증가

```
W의 최대 고유값 > 1이면 gradient 폭발
```

**해결책**:
- Gradient Clipping
- 적절한 초기화
- Layer Normalization

### 7.3 Gradient Clipping 구현

```python
def clip_gradient(grads, max_norm):
    """
    Gradient의 norm을 제한

    Args:
        grads: gradient 리스트
        max_norm: 최대 norm
    """
    total_norm = 0
    for g in grads:
        total_norm += (g ** 2).sum()
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g *= clip_coef

    return grads
```

---

## 8. 완전한 Python 구현

### 8.1 층 클래스

```python
import numpy as np

class Linear:
    """선형 층"""
    def __init__(self, in_features, out_features):
        # Xavier 초기화
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        self.b = np.zeros(out_features)
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T


class ReLU:
    """ReLU 활성화"""
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class Softmax:
    """Softmax 활성화"""
    def forward(self, x):
        exp_x = np.exp(x - x.max(axis=1, keepdims=True))
        self.out = exp_x / exp_x.sum(axis=1, keepdims=True)
        return self.out


class CrossEntropyLoss:
    """Cross-Entropy 손실"""
    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        batch_size = pred.shape[0]
        # target이 one-hot일 때
        loss = -np.sum(target * np.log(pred + 1e-8)) / batch_size
        return loss

    def backward(self):
        batch_size = self.pred.shape[0]
        return (self.pred - self.target) / batch_size
```

### 8.2 Network 클래스

```python
class NeuralNetwork:
    """2층 Neural Network"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.layers = [
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim),
            Softmax()
        ]
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self):
        dout = self.loss_fn.backward()
        for layer in reversed(self.layers[:-1]):  # softmax 제외
            dout = layer.backward(dout)

    def update(self, lr):
        for layer in self.layers:
            if hasattr(layer, 'W'):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db

    def train_step(self, x, y, lr=0.01):
        # Forward
        pred = self.forward(x)
        loss = self.loss_fn.forward(pred, y)

        # Backward
        self.backward()

        # Update
        self.update(lr)

        return loss
```

### 8.3 학습 루프

```python
# 데이터 생성 (XOR 문제)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[1,0], [0,1], [0,1], [1,0]])  # one-hot

# 모델 생성
model = NeuralNetwork(input_dim=2, hidden_dim=4, output_dim=2)

# 학습
for epoch in range(1000):
    loss = model.train_step(X, Y, lr=0.5)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 예측
pred = model.forward(X)
print("Predictions:", pred.argmax(axis=1))
print("Labels:", Y.argmax(axis=1))
```

---

## 9. Transformer에서의 Backpropagation

### 9.1 Attention의 Backward

```
Forward:
scores = Q @ K.T / sqrt(d_k)
weights = softmax(scores)
output = weights @ V

Backward:
∂L/∂V = weights.T @ ∂L/∂output
∂L/∂weights = ∂L/∂output @ V.T
∂L/∂scores = softmax_backward(∂L/∂weights)
∂L/∂Q = (∂L/∂scores @ K) / sqrt(d_k)
∂L/∂K = (∂L/∂scores.T @ Q) / sqrt(d_k)
```

### 9.2 메모리 효율적 Attention Backward

```
# FlashAttention의 핵심: 중간 attention matrix 저장 안 함

Forward:
- Q, K, V를 블록 단위로 처리
- softmax를 online으로 계산
- 저장: Q, K, V, output, softmax 정규화 상수만

Backward:
- 저장된 값으로 attention weights 재계산
- 메모리: O(N) vs 기존 O(N²)
```

---

## 10. 요약

### 10.1 Backpropagation 핵심 공식

| 층 | Forward | Backward (∂L/∂input) |
|:---|:--------|:---------------------|
| Linear | y = Wx + b | dW = xᵀ·dy, db = Σdy, dx = dy·Wᵀ |
| ReLU | y = max(0, x) | dx = dy ⊙ (x > 0) |
| Softmax | y = softmax(x) | (Combined with CE) |
| CE Loss | L = -Σy·log(ŷ) | dz = ŷ - y |

### 10.2 구현 체크리스트

- [ ] Forward에서 중간값 저장 (backward에서 필요)
- [ ] Backward는 역순으로 진행
- [ ] Shape 일치 확인
- [ ] Gradient Checking으로 검증
- [ ] Gradient Clipping 적용

### 10.3 PyTorch에서의 활용

```python
# 수동 구현 대신 autograd 사용
model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
for x, y in dataloader:
    optimizer.zero_grad()           # gradient 초기화
    output = model(x)               # forward
    loss = criterion(output, y)     # 손실 계산
    loss.backward()                 # backward (자동!)
    optimizer.step()                # 파라미터 업데이트
```

---

> 💡 **본문 연결**
> - [1.1 수학적 기초](../../01_딥러닝_Transformer_기초/01_수학적_기초.md)
> - [부록 A2: 미적분](../A_수학_기초/A2_미적분.md)
> - [부록 B1: Attention 수식 유도](B1_Attention_수식_유도.md)
