---
---

# 1.2 미적분 (Calculus)

신경망 학습의 핵심은 Gradient Descent다. Gradient를 계산하려면 미분이 필수고, 여러 층을 거치는 Backpropagation은 Chain Rule로 작동한다.

## 1.2.1 미분의 기초

### 도함수 (Derivative)

함수의 순간 변화율. 기울기를 나타낸다.

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

**기본 미분 공식**:

| 함수 | 도함수 |
|------|--------|
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln(x)$ | $1/x$ |
| $\sin(x)$ | $\cos(x)$ |
| $\cos(x)$ | $-\sin(x)$ |

```python
import numpy as np

def numerical_derivative(f, x, h=1e-5):
    """수치 미분 (검증용)"""
    return (f(x + h) - f(x - h)) / (2 * h)

# 예시: f(x) = x^2의 미분
f = lambda x: x ** 2
x = 3.0

# 수치 미분
numerical = numerical_derivative(f, x)  # ≈ 6.0

# 해석적 미분: f'(x) = 2x
analytical = 2 * x  # 6.0

print(f"수치 미분: {numerical:.6f}")
print(f"해석 미분: {analytical:.6f}")
```

### 편미분 (Partial Derivative)

다변수 함수에서 한 변수에 대해서만 미분. 나머지 변수는 상수 취급.

$$
\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h}
$$

```python
# f(x, y) = x^2 + 3xy + y^2
# ∂f/∂x = 2x + 3y
# ∂f/∂y = 3x + 2y

def f(x, y):
    return x**2 + 3*x*y + y**2

def partial_x(x, y):
    return 2*x + 3*y

def partial_y(x, y):
    return 3*x + 2*y

x, y = 2.0, 3.0
print(f"∂f/∂x at (2, 3): {partial_x(x, y)}")  # 13
print(f"∂f/∂y at (2, 3): {partial_y(x, y)}")  # 12
```

## 1.2.2 Chain Rule (연쇄 법칙)

### 핵심 개념

합성 함수의 미분. **Backpropagation의 수학적 기반**이다.

$$
\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)
$$

**직관적 이해**: 바깥 함수의 미분 × 안쪽 함수의 미분

### 예제: 합성 함수

$h(x) = (3x + 2)^2$ 의 미분

- 바깥 함수: $f(u) = u^2$ → $f'(u) = 2u$
- 안쪽 함수: $g(x) = 3x + 2$ → $g'(x) = 3$

$$
h'(x) = 2(3x + 2) \cdot 3 = 6(3x + 2)
$$

```python
# 수치 검증
h = lambda x: (3*x + 2)**2
h_prime = lambda x: 6 * (3*x + 2)

x = 1.0
print(f"수치 미분: {numerical_derivative(h, x):.6f}")  # ≈ 30
print(f"해석 미분: {h_prime(x):.6f}")                  # 30
```

### 다변수 Chain Rule

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

**신경망에서의 의미**:
- $L$: Loss 함수
- $y$: 중간 레이어 출력
- $x$: 이전 레이어 출력 또는 가중치

## 1.2.3 Gradient (기울기 벡터)

### 정의

다변수 함수의 모든 편미분을 모은 벡터. 함수가 **가장 빠르게 증가하는 방향**을 가리킨다.

$$
\nabla f = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right]
$$

```python
def compute_gradient(f, x, h=1e-5):
    """수치적 gradient 계산"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# f(x, y) = x^2 + y^2
f = lambda v: v[0]**2 + v[1]**2
x = np.array([3.0, 4.0])

grad = compute_gradient(f, x)
print(f"Gradient: {grad}")  # [6.0, 8.0] (= [2x, 2y])
```

### Gradient Descent

Loss를 줄이려면 Gradient의 **반대 방향**으로 이동한다.

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$

- $\theta$: 파라미터
- $\eta$: Learning Rate
- $\nabla L$: Loss의 Gradient

```python
def gradient_descent(f, initial, lr=0.1, steps=100):
    """간단한 Gradient Descent"""
    x = initial.copy()
    history = [x.copy()]

    for _ in range(steps):
        grad = compute_gradient(f, x)
        x = x - lr * grad
        history.append(x.copy())

    return x, history

# f(x, y) = x^2 + y^2 의 최솟값 찾기
f = lambda v: v[0]**2 + v[1]**2
initial = np.array([5.0, 5.0])

result, history = gradient_descent(f, initial, lr=0.1, steps=50)
print(f"최종 위치: {result}")  # ≈ [0, 0]
print(f"최솟값: {f(result):.6f}")  # ≈ 0
```

## 1.2.4 Jacobian 행렬

### 정의

다변수 함수의 모든 편미분을 행렬로 정리한 것.

$$
J = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

입력: n차원 → 출력: m차원 → Jacobian: (m × n)

```python
def compute_jacobian(f, x, h=1e-5):
    """수치적 Jacobian 계산"""
    n = len(x)
    f_x = f(x)
    m = len(f_x) if hasattr(f_x, '__len__') else 1

    jacobian = np.zeros((m, n))

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        jacobian[:, i] = (f(x_plus) - f(x_minus)) / (2 * h)

    return jacobian

# f(x, y) = [x^2 + y, xy]
# Jacobian: [[2x, 1], [y, x]]
f = lambda v: np.array([v[0]**2 + v[1], v[0]*v[1]])
x = np.array([2.0, 3.0])

J = compute_jacobian(f, x)
print(f"Jacobian:\n{J}")
# [[4, 1],   = [[2*2, 1],
#  [3, 2]]      [3, 2]]
```

### 딥러닝에서의 역할

- **Forward Pass**: 입력 → 출력 변환
- **Backward Pass**: Jacobian을 사용해 Gradient 전파

$$
\frac{\partial L}{\partial \mathbf{x}} = J^T \frac{\partial L}{\partial \mathbf{y}}
$$

## 1.2.5 Backpropagation 상세

### 단일 뉴런 예제

$$
y = \sigma(wx + b)
$$

- $\sigma$: 활성화 함수 (예: sigmoid)
- $w$: 가중치
- $b$: 편향

**Forward Pass**:
```
z = w*x + b
y = sigmoid(z)
L = loss(y, target)
```

**Backward Pass** (Chain Rule 적용):

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def mse_loss(pred, target):
    return 0.5 * (pred - target) ** 2

def mse_loss_derivative(pred, target):
    return pred - target

# Forward Pass
x = 2.0
w = 0.5
b = 0.1
target = 1.0

z = w * x + b           # z = 1.1
y = sigmoid(z)          # y ≈ 0.75
L = mse_loss(y, target) # L ≈ 0.031

# Backward Pass
dL_dy = mse_loss_derivative(y, target)  # ≈ -0.25
dy_dz = sigmoid_derivative(z)           # ≈ 0.186
dz_dw = x                               # 2.0
dz_db = 1                               # 1.0

dL_dw = dL_dy * dy_dz * dz_dw  # ≈ -0.093
dL_db = dL_dy * dy_dz * dz_db  # ≈ -0.047

print(f"dL/dw: {dL_dw:.4f}")
print(f"dL/db: {dL_db:.4f}")
```

### 다층 신경망 Backprop

```
입력 x → [W1] → z1 → [ReLU] → a1 → [W2] → z2 → [Softmax] → y → [Loss] → L
```

**Gradient 전파 (역방향)**:
1. $\partial L / \partial y$ (Loss → Softmax 출력)
2. $\partial L / \partial z_2$ (Softmax 미분 적용)
3. $\partial L / \partial W_2$ (z2 = W2 @ a1)
4. $\partial L / \partial a_1$ (이전 레이어로 전파)
5. ... 반복

```python
class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, dL_dz2):
        # dL/dW2, dL/db2
        self.dW2 = self.a1.T @ dL_dz2
        self.db2 = np.sum(dL_dz2, axis=0)

        # dL/da1
        dL_da1 = dL_dz2 @ self.W2.T

        # dL/dz1 (ReLU backward)
        dL_dz1 = dL_da1 * (self.z1 > 0)

        # dL/dW1, dL/db1
        self.dW1 = self.x.T @ dL_dz1
        self.db1 = np.sum(dL_dz1, axis=0)

        return dL_dz1
```

## 1.2.6 주요 활성화 함수의 미분

| 함수 | 수식 | 미분 |
|------|------|------|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ |
| Tanh | $\tanh(x)$ | $1 - \tanh^2(x)$ |
| ReLU | $\max(0, x)$ | $\begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ |
| Leaky ReLU | $\max(\alpha x, x)$ | $\begin{cases} 1 & x > 0 \\ \alpha & x \leq 0 \end{cases}$ |
| GELU | $x \cdot \Phi(x)$ | 복잡 (근사 사용) |

```python
# 활성화 함수와 미분 구현
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def gelu(x):
    """GELU (Gaussian Error Linear Unit) - Transformer에서 많이 사용"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
```

## 1.2.7 Softmax와 Cross-Entropy

### Softmax

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

```python
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # 수치 안정성
    return exp_z / np.sum(exp_z)

z = np.array([2.0, 1.0, 0.1])
probs = softmax(z)
print(f"확률: {probs}")  # [0.659, 0.242, 0.099]
print(f"합계: {np.sum(probs)}")  # 1.0
```

### Cross-Entropy Loss

$$
L = -\sum_i y_i \log(\hat{y}_i)
$$

- $y_i$: 정답 (one-hot)
- $\hat{y}_i$: 예측 확률

### Softmax + Cross-Entropy의 미분

놀랍게도 간단한 형태가 된다:

$$
\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i
$$

```python
def softmax_cross_entropy_backward(probs, target):
    """Softmax + Cross-Entropy의 gradient"""
    return probs - target  # 예측 - 정답

# 예시
probs = np.array([0.7, 0.2, 0.1])
target = np.array([1, 0, 0])  # 정답: 첫 번째 클래스

grad = softmax_cross_entropy_backward(probs, target)
print(f"Gradient: {grad}")  # [-0.3, 0.2, 0.1]
```

## 1.2.8 실습 체크리스트

- [ ] 수치 미분으로 해석적 미분 검증하기
- [ ] Chain Rule로 합성 함수 미분하기
- [ ] Gradient Descent로 간단한 함수 최적화
- [ ] 2-layer 신경망 backpropagation 구현
- [ ] Softmax + Cross-Entropy gradient 확인

## 1.2.9 핵심 요약

| 개념 | 딥러닝에서의 역할 |
|------|------------------|
| Chain Rule | Backpropagation의 수학적 기반 |
| Gradient | 파라미터 업데이트 방향 결정 |
| Jacobian | 다변수 함수의 gradient 전파 |
| Softmax + CE | 분류 문제의 표준 출력 + Loss |

## 다음 단계

[1.3 확률과 통계](03-probability.md)에서 확률분포, Cross-Entropy, KL Divergence를 다룬다.
