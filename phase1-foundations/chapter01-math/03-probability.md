# 1.3 확률과 통계 (Probability & Statistics)

딥러닝은 본질적으로 확률적 모델이다. 모델의 출력은 확률분포고, Loss 함수는 확률론적 개념에 기반한다.

## 1.3.1 확률의 기초

### 확률 공리

1. **비음수성**: $P(A) \geq 0$
2. **정규화**: $P(\Omega) = 1$ (전체 사건의 확률 = 1)
3. **가산 가법성**: 배반 사건들의 확률은 더할 수 있음

### 조건부 확률 (Conditional Probability)

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
$$

**LLM의 핵심**: 다음 토큰 예측은 조건부 확률이다.

$$
P(w_t | w_1, w_2, \ldots, w_{t-1})
$$

```python
# "I love" 다음에 올 단어의 조건부 확률 예시
vocab = ["you", "cats", "pizza", "the"]
probs = [0.4, 0.2, 0.15, 0.25]  # P(word | "I love")

print("P(word | 'I love'):")
for word, prob in zip(vocab, probs):
    print(f"  {word}: {prob}")
```

### 베이즈 정리 (Bayes' Theorem)

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

- $P(A)$: 사전 확률 (Prior)
- $P(A|B)$: 사후 확률 (Posterior)
- $P(B|A)$: 우도 (Likelihood)
- $P(B)$: 증거 (Evidence)

```python
# 예시: 스팸 필터
# P(스팸|"무료") = P("무료"|스팸) * P(스팸) / P("무료")

P_spam = 0.2                    # 전체 메일 중 스팸 비율
P_free_given_spam = 0.8         # 스팸에서 "무료" 등장 확률
P_free_given_not_spam = 0.1     # 정상 메일에서 "무료" 확률

P_free = P_free_given_spam * P_spam + P_free_given_not_spam * (1 - P_spam)
P_spam_given_free = (P_free_given_spam * P_spam) / P_free

print(f"P(스팸|'무료'): {P_spam_given_free:.3f}")  # 0.667
```

## 1.3.2 확률분포 (Probability Distributions)

### 이산 확률분포

#### Categorical Distribution (범주형 분포)

LLM의 출력이 바로 Categorical Distribution이다.

$$
P(X = k) = p_k, \quad \sum_{k=1}^{K} p_k = 1
$$

```python
import numpy as np

def categorical_sample(probs, n_samples=1):
    """Categorical 분포에서 샘플링"""
    return np.random.choice(len(probs), size=n_samples, p=probs)

# 단어 확률 분포
vocab = ["the", "a", "cat", "dog", "sat"]
probs = [0.3, 0.2, 0.25, 0.15, 0.1]

samples = categorical_sample(probs, 10)
print(f"샘플된 단어: {[vocab[i] for i in samples]}")
```

#### Bernoulli Distribution (베르누이 분포)

이진 결과 (0 또는 1)의 분포.

$$
P(X = 1) = p, \quad P(X = 0) = 1 - p
$$

```python
# Dropout은 Bernoulli 분포를 사용
def dropout(x, p=0.5, training=True):
    if not training:
        return x
    mask = np.random.binomial(1, 1-p, size=x.shape)  # Bernoulli
    return x * mask / (1 - p)  # Scale by 1/(1-p)
```

### 연속 확률분포

#### Gaussian (Normal) Distribution

$$
\mathcal{N}(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

```python
def gaussian_pdf(x, mu=0, sigma=1):
    """가우시안 확률밀도함수"""
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * \
           np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# 가중치 초기화에 사용
W = np.random.normal(loc=0, scale=0.02, size=(768, 768))
print(f"가중치 평균: {np.mean(W):.4f}")  # ≈ 0
print(f"가중치 표준편차: {np.std(W):.4f}")  # ≈ 0.02
```

**딥러닝에서의 활용**:
- 가중치 초기화 (Xavier, He initialization)
- VAE의 latent space
- Noise 추가 (data augmentation)

## 1.3.3 기댓값과 분산

### 기댓값 (Expected Value)

$$
\mathbb{E}[X] = \sum_x x \cdot P(X = x) \quad \text{(이산)}
$$

$$
\mathbb{E}[X] = \int x \cdot p(x) dx \quad \text{(연속)}
$$

```python
# 이산 기댓값
values = np.array([1, 2, 3, 4, 5, 6])  # 주사위
probs = np.array([1/6] * 6)

expected = np.sum(values * probs)
print(f"주사위 기댓값: {expected}")  # 3.5
```

### 분산 (Variance)

$$
\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

```python
# 분산 계산
variance = np.sum((values - expected)**2 * probs)
std_dev = np.sqrt(variance)

print(f"주사위 분산: {variance:.3f}")      # 2.917
print(f"주사위 표준편차: {std_dev:.3f}")  # 1.708
```

### Layer Normalization과의 연결

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer Normalization"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

# 예시
x = np.random.randn(2, 4)  # (batch, features)
gamma = np.ones(4)
beta = np.zeros(4)

normalized = layer_norm(x, gamma, beta)
print(f"정규화 후 평균: {np.mean(normalized, axis=-1)}")  # ≈ [0, 0]
print(f"정규화 후 분산: {np.var(normalized, axis=-1)}")   # ≈ [1, 1]
```

## 1.3.4 엔트로피 (Entropy)

### 정보 이론의 핵심

불확실성(정보량)의 측정. 확률분포가 얼마나 "예측 불가능"한지를 나타낸다.

$$
H(X) = -\sum_x P(x) \log P(x)
$$

**성질**:
- 균등 분포일 때 엔트로피 최대
- 확실한 결과(하나의 확률 = 1)일 때 엔트로피 = 0

```python
def entropy(probs, eps=1e-10):
    """엔트로피 계산"""
    probs = np.clip(probs, eps, 1)  # log(0) 방지
    return -np.sum(probs * np.log(probs))

# 예시 1: 균등 분포 (최대 불확실성)
uniform = np.array([0.25, 0.25, 0.25, 0.25])
print(f"균등 분포 엔트로피: {entropy(uniform):.3f}")  # 1.386 (= log(4))

# 예시 2: 확실한 분포 (불확실성 없음)
certain = np.array([1.0, 0.0, 0.0, 0.0])
print(f"확실한 분포 엔트로피: {entropy(certain):.3f}")  # 0.0

# 예시 3: 중간 분포
skewed = np.array([0.7, 0.1, 0.1, 0.1])
print(f"편향된 분포 엔트로피: {entropy(skewed):.3f}")  # 0.940
```

### 텍스트 생성에서의 Temperature

Temperature는 출력 분포의 엔트로피를 조절한다.

$$
P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

- $T < 1$: 분포가 뾰족해짐 (낮은 엔트로피, 확실한 선택)
- $T > 1$: 분포가 평평해짐 (높은 엔트로피, 다양한 선택)

```python
def softmax_with_temperature(logits, temperature=1.0):
    logits = logits / temperature
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

logits = np.array([2.0, 1.0, 0.5, 0.1])

print("Temperature 효과:")
for T in [0.5, 1.0, 2.0]:
    probs = softmax_with_temperature(logits, T)
    H = entropy(probs)
    print(f"  T={T}: probs={probs.round(3)}, entropy={H:.3f}")
```

## 1.3.5 Cross-Entropy

### 정의

두 확률분포 사이의 "거리". 정답 분포 $p$와 예측 분포 $q$의 차이를 측정한다.

$$
H(p, q) = -\sum_x p(x) \log q(x)
$$

**해석**: $p$를 따르는 데이터를 $q$로 인코딩할 때 필요한 평균 비트 수.

```python
def cross_entropy(p, q, eps=1e-10):
    """Cross-Entropy 계산"""
    q = np.clip(q, eps, 1)
    return -np.sum(p * np.log(q))

# 정답 분포
target = np.array([1, 0, 0, 0])  # one-hot: 첫 번째가 정답

# 좋은 예측
good_pred = np.array([0.9, 0.05, 0.03, 0.02])
print(f"좋은 예측 CE: {cross_entropy(target, good_pred):.4f}")  # 0.105

# 나쁜 예측
bad_pred = np.array([0.1, 0.6, 0.2, 0.1])
print(f"나쁜 예측 CE: {cross_entropy(target, bad_pred):.4f}")  # 2.303

# 최악의 예측 (정답에 확률 0)
worst_pred = np.array([0.0, 0.5, 0.3, 0.2])
print(f"최악 예측 CE: {cross_entropy(target, worst_pred):.4f}")  # 매우 큼
```

### Cross-Entropy Loss

분류 문제의 표준 Loss 함수.

$$
L = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)
$$

- $y$: 정답 (one-hot encoding)
- $\hat{y}$: 모델 예측 (softmax 출력)

```python
def cross_entropy_loss(logits, target_idx):
    """Cross-Entropy Loss (정수 타겟)"""
    probs = softmax_with_temperature(logits, 1.0)
    return -np.log(probs[target_idx] + 1e-10)

# 배치 처리
def batch_cross_entropy_loss(logits, targets):
    """배치 Cross-Entropy Loss"""
    batch_size = logits.shape[0]
    probs = np.array([softmax_with_temperature(l) for l in logits])
    log_probs = np.log(probs + 1e-10)
    return -np.mean(log_probs[np.arange(batch_size), targets])

# 예시
logits = np.array([[2.0, 1.0, 0.1],
                   [0.5, 2.5, 0.3]])
targets = np.array([0, 1])  # 첫 번째 샘플은 클래스 0, 두 번째는 클래스 1

loss = batch_cross_entropy_loss(logits, targets)
print(f"Batch CE Loss: {loss:.4f}")
```

## 1.3.6 KL Divergence

### 정의

두 확률분포 사이의 비대칭적 "거리".

$$
D_{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)
$$

**해석**: $p$를 $q$로 근사할 때 잃는 정보량.

**성질**:
- $D_{KL}(p \| q) \geq 0$ (항상 0 이상)
- $D_{KL}(p \| q) = 0$ iff $p = q$
- $D_{KL}(p \| q) \neq D_{KL}(q \| p)$ (비대칭!)

```python
def kl_divergence(p, q, eps=1e-10):
    """KL Divergence 계산"""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))

# 예시
p = np.array([0.4, 0.3, 0.2, 0.1])
q = np.array([0.25, 0.25, 0.25, 0.25])

kl_pq = kl_divergence(p, q)
kl_qp = kl_divergence(q, p)

print(f"KL(p||q): {kl_pq:.4f}")  # p를 q로 근사
print(f"KL(q||p): {kl_qp:.4f}")  # q를 p로 근사 (다른 값!)
```

### 딥러닝에서의 활용

1. **VAE (Variational Autoencoder)**: 잠재 공간 정규화
2. **Knowledge Distillation**: Teacher-Student 학습
3. **Policy Gradient (RL)**: 정책 업데이트 제한

```python
# Knowledge Distillation 예시
def distillation_loss(student_logits, teacher_probs, temperature=2.0):
    """Knowledge Distillation Loss"""
    student_probs = softmax_with_temperature(student_logits, temperature)
    return kl_divergence(teacher_probs, student_probs) * (temperature ** 2)

teacher_probs = np.array([0.7, 0.2, 0.1])
student_logits = np.array([1.5, 0.8, 0.3])

loss = distillation_loss(student_logits, teacher_probs)
print(f"Distillation Loss: {loss:.4f}")
```

## 1.3.7 최대우도추정 (Maximum Likelihood Estimation)

### 개념

데이터를 가장 잘 설명하는 파라미터를 찾는 방법.

$$
\theta^* = \arg\max_\theta P(D | \theta) = \arg\max_\theta \prod_{i=1}^{N} P(x_i | \theta)
$$

로그를 취하면:

$$
\theta^* = \arg\max_\theta \sum_{i=1}^{N} \log P(x_i | \theta)
$$

### Cross-Entropy와의 관계

Cross-Entropy 최소화 = 최대우도추정

$$
\min_\theta H(p_{\text{data}}, p_\theta) = \max_\theta \mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)]
$$

```python
# LLM의 학습: 다음 토큰 예측의 MLE
def language_model_loss(logits, targets):
    """
    Language Model의 Loss = Cross-Entropy
    = Negative Log Likelihood
    = MLE의 목적함수
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Cross-Entropy
    probs = np.array([softmax_with_temperature(l) for l in logits_flat])
    nll = -np.mean(np.log(probs[np.arange(len(targets_flat)), targets_flat] + 1e-10))

    return nll

# Perplexity = exp(Cross-Entropy)
def perplexity(loss):
    return np.exp(loss)
```

## 1.3.8 Sampling 기법

### Top-k Sampling

상위 k개의 토큰만 고려하고 나머지는 확률을 0으로.

```python
def top_k_sampling(logits, k=10):
    """Top-k Sampling"""
    indices = np.argsort(logits)[::-1][:k]
    top_k_logits = logits[indices]
    probs = softmax_with_temperature(top_k_logits)

    sampled_idx = np.random.choice(k, p=probs)
    return indices[sampled_idx]
```

### Top-p (Nucleus) Sampling

누적 확률이 p가 될 때까지의 토큰만 고려.

```python
def top_p_sampling(logits, p=0.9):
    """Top-p (Nucleus) Sampling"""
    probs = softmax_with_temperature(logits)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    cumsum = np.cumsum(sorted_probs)
    cutoff_idx = np.searchsorted(cumsum, p) + 1

    top_p_indices = sorted_indices[:cutoff_idx]
    top_p_probs = probs[top_p_indices]
    top_p_probs /= top_p_probs.sum()  # 재정규화

    sampled_idx = np.random.choice(len(top_p_indices), p=top_p_probs)
    return top_p_indices[sampled_idx]
```

## 1.3.9 실습 체크리스트

- [ ] 엔트로피 계산하고 분포별 비교하기
- [ ] Temperature 변화에 따른 분포 변화 시각화
- [ ] Cross-Entropy Loss 직접 구현하기
- [ ] KL Divergence의 비대칭성 확인하기
- [ ] Top-k, Top-p Sampling 구현하고 결과 비교

## 1.3.10 핵심 요약

| 개념 | 딥러닝에서의 역할 |
|------|------------------|
| 조건부 확률 | LLM의 다음 토큰 예측 |
| Categorical 분포 | 분류 출력, 토큰 확률 |
| 엔트로피 | 불확실성 측정, Temperature 조절 |
| Cross-Entropy | 분류 Loss 함수 |
| KL Divergence | Distillation, VAE |
| MLE | 학습의 이론적 기반 |

## 다음 단계

[Chapter 2: Transformer 아키텍처](../chapter02-transformer/01-self-attention.md)에서 Attention 메커니즘을 본격적으로 다룬다.
