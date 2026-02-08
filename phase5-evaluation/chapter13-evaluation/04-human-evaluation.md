# 13.4 Human Evaluation

자동 메트릭만으로는 OCR 품질을 완전히 평가할 수 없다. Human Evaluation은 자동 메트릭이 포착하지 못하는 의미적 정확성, 레이아웃 보존, 가독성 등을 평가한다. 핵심은 **평가자 간 일치도(IAA)**를 확보해서 평가의 신뢰성을 보장하는 것이다.

> **핵심 용어**
>
> | 용어 | 정의 |
> |------|------|
> | **IAA** (Inter-Annotator Agreement) | 평가자 간 일치도. 평가의 신뢰성 지표 |
> | **Cohen's Kappa** ($\kappa$) | 2명 평가자 간 우연 일치를 보정한 일치도 |
> | **Fleiss' Kappa** | 3명 이상 평가자에 대한 일치도 |
> | **Krippendorff's Alpha** ($\alpha$) | 다중 평가자, 다양한 척도에 범용 적용 가능한 일치도 |
> | **Likert Scale** | 1~5점 등 순서형 척도를 이용한 평가 방법 |
> | **Annotation Guide** | 평가 기준을 상세히 기술한 평가 지침서 |
> | **Adjudication** | 평가자 간 불일치를 해소하는 절차 |
> | **Sampling Strategy** | 평가 대상 선정 방법 (무작위, 층화, 오류 집중 등) |

---

## 수학적 원리

### Cohen's Kappa ($\kappa$)

2명의 평가자 간 일치도를 측정한다. 우연에 의한 일치를 보정한 값이다.

**관측된 일치율 (Observed Agreement):**

$$
p_o = \frac{\text{두 평가자가 같은 라벨을 부여한 샘플 수}}{N}
$$

**우연 일치율 (Expected Agreement):**

$$
p_e = \sum_{k=1}^{K} p_{1,k} \cdot p_{2,k}
$$

- $K$: 라벨(카테고리) 수
- $p_{1,k}$: 평가자 1이 라벨 $k$를 부여한 비율
- $p_{2,k}$: 평가자 2가 라벨 $k$를 부여한 비율

**Cohen's Kappa:**

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

**해석 기준:**

| $\kappa$ 범위 | 해석 |
|--------------|------|
| < 0 | 우연보다 못함 |
| 0.01 ~ 0.20 | 미미한 일치 |
| 0.21 ~ 0.40 | 약한 일치 |
| 0.41 ~ 0.60 | 중간 일치 |
| 0.61 ~ 0.80 | 상당한 일치 |
| 0.81 ~ 1.00 | 거의 완전한 일치 |

### Weighted Kappa

순서형 척도(Likert Scale 등)에서 "얼마나 떨어진 불일치인가"를 반영한다.

**가중치 행렬 $w_{ij}$:**

선형 가중:

$$
w_{ij} = 1 - \frac{|i - j|}{K - 1}
$$

이차 가중:

$$
w_{ij} = 1 - \frac{(i - j)^2}{(K - 1)^2}
$$

**Weighted Kappa:**

$$
\kappa_w = 1 - \frac{\sum_{i,j} w_{ij} \cdot O_{ij}}{\sum_{i,j} w_{ij} \cdot E_{ij}}
$$

- $O_{ij}$: 관측된 빈도 행렬의 $(i, j)$ 원소
- $E_{ij}$: 기대 빈도 행렬의 $(i, j)$ 원소

### Fleiss' Kappa

3명 이상의 평가자가 동일한 샘플 집합을 평가할 때 사용한다.

$N$개 샘플, $n$명 평가자, $K$개 카테고리에 대해:

**각 카테고리의 전체 비율:**

$$
\bar{p}_k = \frac{1}{Nn} \sum_{i=1}^{N} n_{ik}
$$

- $n_{ik}$: $i$번째 샘플에 카테고리 $k$를 부여한 평가자 수

**각 샘플의 일치도:**

$$
P_i = \frac{1}{n(n-1)} \sum_{k=1}^{K} n_{ik}(n_{ik} - 1)
$$

**전체 관측 일치율:**

$$
\bar{P} = \frac{1}{N} \sum_{i=1}^{N} P_i
$$

**기대 일치율:**

$$
\bar{P}_e = \sum_{k=1}^{K} \bar{p}_k^2
$$

**Fleiss' Kappa:**

$$
\kappa_F = \frac{\bar{P} - \bar{P}_e}{1 - \bar{P}_e}
$$

### Krippendorff's Alpha ($\alpha$)

가장 범용적인 IAA 지표다. 다음 특성을 모두 지원한다:
- 2명 이상의 평가자
- 명목/순서/구간/비율 척도
- 결측값 허용 (모든 평가자가 모든 샘플을 평가하지 않아도 됨)

**관측된 불일치 (Observed Disagreement):**

$$
D_o = \frac{1}{n'} \sum_{c,k} o_{ck} \cdot \delta^2(c, k)
$$

- $o_{ck}$: 동일 샘플에서 카테고리 $c$와 $k$가 동시에 부여된 횟수
- $\delta^2(c, k)$: 카테고리 $c$와 $k$ 사이의 거리 함수
- $n'$: 총 평가 쌍 수

**기대 불일치 (Expected Disagreement):**

$$
D_e = \frac{1}{n'(n'-1)} \sum_{c,k} n_c \cdot n_k \cdot \delta^2(c, k)
$$

- $n_c$: 전체에서 카테고리 $c$가 부여된 총 횟수

**Krippendorff's Alpha:**

$$
\alpha = 1 - \frac{D_o}{D_e}
$$

**거리 함수 $\delta^2(c, k)$:**

| 척도 | $\delta^2(c, k)$ |
|------|-----------------|
| 명목(Nominal) | $\begin{cases} 0 & c = k \\ 1 & c \neq k \end{cases}$ |
| 순서(Ordinal) | $(c - k)^2$ (순서 인덱스 기반) |
| 구간(Interval) | $(c - k)^2$ |
| 비율(Ratio) | $\left(\frac{c - k}{c + k}\right)^2$ |

**해석 기준:**
- $\alpha \geq 0.8$: 신뢰할 수 있는 수준
- $0.667 \leq \alpha < 0.8$: 잠정적으로 수용 가능
- $\alpha < 0.667$: 신뢰할 수 없음

---

## 평가 프로토콜 설계

### 샘플링 전략

```
전체 OCR 결과 풀
    │
    ├── 무작위 샘플링 (50%)
    │     평가 대상의 기본 선정. 전체 분포를 대표해야 함.
    │
    ├── 층화 샘플링 (30%)
    │     문서 유형별(입원확인서, 진단서, 수술기록 등) 비율 보장.
    │
    └── 오류 집중 샘플링 (20%)
          자동 메트릭에서 CER이 높은 샘플 우선 선정.
          경계 사례(borderline)와 모델 confidence가 낮은 샘플 포함.
```

**최소 샘플 수 산정:**

$$
n = \frac{z^2 \cdot p(1-p)}{e^2}
$$

- $z$: 신뢰수준 계수 (95% 신뢰구간이면 $z = 1.96$)
- $p$: 예상 정확도 (사전 추정, 보통 0.5로 설정하면 보수적)
- $e$: 허용 오차 (예: 0.05)

95% 신뢰구간, 5% 오차를 원하면:

$$
n = \frac{1.96^2 \times 0.5 \times 0.5}{0.05^2} = 384.16 \approx 385
$$

최소 385개 샘플이 필요하다.

### 평가 지침 (Annotation Guide) 구조

```
1. 평가 목적
   - OCR 결과의 정확성과 사용 적합성을 판단

2. 평가 척도 (5점 Likert Scale)
   - 5: 완벽 — 정답과 동일
   - 4: 우수 — 사소한 차이 (띄어쓰기, 마침표 등)
   - 3: 보통 — 의미 파악 가능하나 일부 오류
   - 2: 미흡 — 주요 정보 오류 존재
   - 1: 불량 — 사용 불가능한 수준

3. 필드별 평가 기준
   - 금액: 1원이라도 다르면 4점 이하
   - 진단코드: 1글자라도 다르면 3점 이하
   - 날짜: 형식 차이만 있으면 5점, 날짜 자체가 다르면 2점 이하
   - 텍스트: NED 기반 감점

4. 경계 사례 가이드
   - '0' vs 'O': 맥락으로 명확히 구분 가능하면 4점
   - 한자 vs 한글: 의미 동일하면 4점
   - 줄바꿈/공백 차이: 5점
```

### 불일치 해소 (Adjudication)

```
Step 1: 1차 평가 (최소 2명)
    │
    ├── 일치 → 확정
    │
    └── 불일치 → Step 2
          │
          Step 2: 3번째 평가자가 중재
              │
              ├── 다수결 → 확정
              │
              └── 전원 불일치 → Step 3
                    │
                    Step 3: 전문가 패널 논의 후 확정
```

---

## 코드 구현

### Cohen's Kappa

```python
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
from itertools import combinations


def cohens_kappa(
    rater1: List[int],
    rater2: List[int],
    labels: List[int] = None
) -> float:
    """
    Cohen's Kappa를 계산한다.

    Args:
        rater1: 평가자 1의 라벨 리스트
        rater2: 평가자 2의 라벨 리스트
        labels: 가능한 라벨 목록 (None이면 데이터에서 추출)

    Returns:
        kappa 값 (-1 ~ 1)
    """
    assert len(rater1) == len(rater2), "평가 수가 일치해야 한다"
    n = len(rater1)

    if labels is None:
        labels = sorted(set(rater1) | set(rater2))

    # 혼동 행렬 구성
    k = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    confusion = np.zeros((k, k), dtype=float)

    for r1, r2 in zip(rater1, rater2):
        i = label_to_idx[r1]
        j = label_to_idx[r2]
        confusion[i][j] += 1

    # 관측 일치율
    p_o = np.trace(confusion) / n

    # 기대 일치율
    row_sums = confusion.sum(axis=1) / n  # 평가자 1의 비율
    col_sums = confusion.sum(axis=0) / n  # 평가자 2의 비율
    p_e = np.sum(row_sums * col_sums)

    if p_e == 1.0:
        return 1.0

    kappa = (p_o - p_e) / (1 - p_e)
    return float(kappa)


def weighted_kappa(
    rater1: List[int],
    rater2: List[int],
    labels: List[int] = None,
    weight_type: str = "quadratic"
) -> float:
    """
    Weighted Kappa를 계산한다. 순서형 척도에 적합.

    Args:
        rater1, rater2: 평가 라벨
        labels: 순서가 있는 라벨 목록
        weight_type: 'linear' 또는 'quadratic'

    Returns:
        weighted kappa 값
    """
    assert len(rater1) == len(rater2)
    n = len(rater1)

    if labels is None:
        labels = sorted(set(rater1) | set(rater2))

    k = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    # 혼동 행렬
    confusion = np.zeros((k, k), dtype=float)
    for r1, r2 in zip(rater1, rater2):
        confusion[label_to_idx[r1]][label_to_idx[r2]] += 1

    confusion /= n

    # 가중치 행렬
    weights = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if weight_type == "linear":
                weights[i][j] = abs(i - j) / (k - 1) if k > 1 else 0
            elif weight_type == "quadratic":
                weights[i][j] = ((i - j) ** 2) / ((k - 1) ** 2) if k > 1 else 0

    # 기대 빈도
    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)
    expected = np.outer(row_sums, col_sums)

    # Weighted Kappa
    numerator = np.sum(weights * confusion)
    denominator = np.sum(weights * expected)

    if denominator == 0:
        return 1.0

    return float(1 - numerator / denominator)
```

### Fleiss' Kappa

```python
def fleiss_kappa(
    ratings: np.ndarray
) -> float:
    """
    Fleiss' Kappa를 계산한다.

    Args:
        ratings: (N, K) 행렬. ratings[i][k] = i번째 샘플에 카테고리 k를 부여한 평가자 수

    Returns:
        Fleiss' kappa 값
    """
    N, K = ratings.shape
    n = ratings.sum(axis=1)[0]  # 각 샘플의 평가자 수 (동일해야 함)

    assert np.all(ratings.sum(axis=1) == n), "모든 샘플의 평가자 수가 동일해야 한다"

    # 각 카테고리의 전체 비율
    p_bar = ratings.sum(axis=0) / (N * n)

    # 각 샘플의 일치도
    P_i = (np.sum(ratings ** 2, axis=1) - n) / (n * (n - 1))

    # 전체 관측 일치율
    P_bar = np.mean(P_i)

    # 기대 일치율
    P_e = np.sum(p_bar ** 2)

    if P_e == 1.0:
        return 1.0

    kappa = (P_bar - P_e) / (1 - P_e)
    return float(kappa)


def ratings_to_matrix(
    annotations: List[List[int]],
    num_categories: int
) -> np.ndarray:
    """
    평가자별 라벨 리스트를 Fleiss' Kappa 입력 행렬로 변환한다.

    Args:
        annotations: [평가자1의 라벨들, 평가자2의 라벨들, ...]
        num_categories: 카테고리 수

    Returns:
        (N, K) ratings 행렬
    """
    num_raters = len(annotations)
    num_samples = len(annotations[0])

    assert all(len(a) == num_samples for a in annotations), \
        "모든 평가자의 평가 수가 동일해야 한다"

    matrix = np.zeros((num_samples, num_categories), dtype=int)

    for rater_labels in annotations:
        for i, label in enumerate(rater_labels):
            matrix[i][label] += 1

    return matrix
```

### Krippendorff's Alpha

```python
def krippendorff_alpha(
    reliability_data: List[List[Optional[int]]],
    level_of_measurement: str = "nominal"
) -> float:
    """
    Krippendorff's Alpha를 계산한다.

    Args:
        reliability_data: [평가자1의 라벨들, 평가자2의 라벨들, ...]
                          None은 결측값.
        level_of_measurement: 'nominal', 'ordinal', 'interval', 'ratio'

    Returns:
        alpha 값
    """
    # 거리 함수
    def delta_nominal(c, k):
        return 0.0 if c == k else 1.0

    def delta_interval(c, k):
        return (c - k) ** 2

    def delta_ratio(c, k):
        if c + k == 0:
            return 0.0
        return ((c - k) / (c + k)) ** 2

    delta_funcs = {
        'nominal': delta_nominal,
        'ordinal': delta_interval,  # 간소화: interval과 동일 처리
        'interval': delta_interval,
        'ratio': delta_ratio,
    }
    delta = delta_funcs.get(level_of_measurement, delta_nominal)

    num_raters = len(reliability_data)
    num_items = len(reliability_data[0])

    # 각 아이템별 유효한 평가 수집
    # pairable values
    value_counts = Counter()
    coincidence_matrix = {}
    total_pairable = 0

    for i in range(num_items):
        # i번째 아이템에 대한 모든 평가자의 라벨
        values = [
            reliability_data[r][i]
            for r in range(num_raters)
            if reliability_data[r][i] is not None
        ]
        m = len(values)
        if m < 2:
            continue

        # 일치 행렬 업데이트
        for c in values:
            value_counts[c] += 1
            total_pairable += 1

        for idx_a in range(m):
            for idx_b in range(m):
                if idx_a == idx_b:
                    continue
                c = values[idx_a]
                k = values[idx_b]
                key = (c, k)
                if key not in coincidence_matrix:
                    coincidence_matrix[key] = 0
                coincidence_matrix[key] += 1.0 / (m - 1)

    if total_pairable == 0:
        return 0.0

    # 관측 불일치
    all_values = list(value_counts.keys())
    D_o = 0.0
    for (c, k), count in coincidence_matrix.items():
        D_o += count * delta(c, k)

    n_prime = sum(coincidence_matrix.values())
    if n_prime > 0:
        D_o /= n_prime

    # 기대 불일치
    D_e = 0.0
    total_n = sum(value_counts.values())
    for c in all_values:
        for k in all_values:
            if c == k:
                continue
            D_e += value_counts[c] * value_counts[k] * delta(c, k)

    if total_n > 1:
        D_e /= (total_n * (total_n - 1))

    if D_e == 0:
        return 1.0

    alpha = 1 - D_o / D_e
    return float(alpha)
```

### 평가 결과 분석 도구

```python
@dataclass
class HumanEvalSample:
    """단일 샘플의 인간 평가 결과"""
    sample_id: str
    ratings: Dict[str, int]  # {평가자ID: 점수}
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_rating(self) -> float:
        return np.mean(list(self.ratings.values()))

    @property
    def std_rating(self) -> float:
        return np.std(list(self.ratings.values()))

    @property
    def max_disagreement(self) -> int:
        vals = list(self.ratings.values())
        return max(vals) - min(vals) if vals else 0


class HumanEvaluationAnalyzer:
    """
    Human Evaluation 결과를 분석한다.

    기능:
    - IAA 계산 (Cohen's Kappa, Fleiss' Kappa, Krippendorff's Alpha)
    - 평가자별 편향 분석
    - 불일치 샘플 식별
    - 신뢰구간 추정
    """

    def __init__(self, scale: int = 5):
        """
        Args:
            scale: Likert Scale 범위 (예: 5이면 1~5)
        """
        self.scale = scale
        self.samples: List[HumanEvalSample] = []

    def add_sample(
        self,
        sample_id: str,
        ratings: Dict[str, int],
        metadata: Dict[str, Any] = None
    ):
        """평가 샘플을 추가한다."""
        self.samples.append(HumanEvalSample(
            sample_id=sample_id,
            ratings=ratings,
            metadata=metadata or {}
        ))

    def get_rater_ids(self) -> List[str]:
        """모든 평가자 ID를 반환한다."""
        raters = set()
        for s in self.samples:
            raters.update(s.ratings.keys())
        return sorted(raters)

    def compute_iaa(self) -> dict:
        """모든 IAA 지표를 계산한다."""
        raters = self.get_rater_ids()
        result = {}

        # Cohen's Kappa (2명 조합별)
        if len(raters) >= 2:
            kappas = {}
            for r1, r2 in combinations(raters, 2):
                labels_r1 = []
                labels_r2 = []
                for s in self.samples:
                    if r1 in s.ratings and r2 in s.ratings:
                        labels_r1.append(s.ratings[r1])
                        labels_r2.append(s.ratings[r2])

                if labels_r1:
                    k = cohens_kappa(labels_r1, labels_r2)
                    wk = weighted_kappa(
                        labels_r1, labels_r2,
                        labels=list(range(1, self.scale + 1)),
                        weight_type="quadratic"
                    )
                    kappas[f"{r1}-{r2}"] = {
                        'kappa': k,
                        'weighted_kappa': wk,
                        'n_samples': len(labels_r1)
                    }

            result['cohens_kappa'] = kappas
            if kappas:
                result['avg_kappa'] = np.mean([v['kappa'] for v in kappas.values()])
                result['avg_weighted_kappa'] = np.mean(
                    [v['weighted_kappa'] for v in kappas.values()]
                )

        # Fleiss' Kappa
        if len(raters) >= 2:
            # 모든 평가자가 평가한 샘플만 사용
            common_samples = [
                s for s in self.samples
                if all(r in s.ratings for r in raters)
            ]
            if common_samples:
                matrix = np.zeros((len(common_samples), self.scale), dtype=int)
                for i, s in enumerate(common_samples):
                    for r in raters:
                        rating = s.ratings[r]
                        matrix[i][rating - 1] += 1  # 1-indexed -> 0-indexed

                result['fleiss_kappa'] = fleiss_kappa(matrix)

        # Krippendorff's Alpha
        reliability_data = []
        for r in raters:
            rater_data = []
            for s in self.samples:
                rater_data.append(s.ratings.get(r))
            reliability_data.append(rater_data)

        result['krippendorff_alpha_nominal'] = krippendorff_alpha(
            reliability_data, 'nominal'
        )
        result['krippendorff_alpha_ordinal'] = krippendorff_alpha(
            reliability_data, 'ordinal'
        )

        return result

    def compute_rater_stats(self) -> Dict[str, dict]:
        """평가자별 통계를 계산한다."""
        raters = self.get_rater_ids()
        stats = {}

        for r in raters:
            ratings = [
                s.ratings[r] for s in self.samples if r in s.ratings
            ]
            if not ratings:
                continue

            stats[r] = {
                'mean': float(np.mean(ratings)),
                'std': float(np.std(ratings)),
                'median': float(np.median(ratings)),
                'count': len(ratings),
                'distribution': dict(Counter(ratings)),
            }

        return stats

    def find_disagreements(
        self,
        threshold: int = 2
    ) -> List[HumanEvalSample]:
        """
        평가자 간 불일치가 큰 샘플을 찾는다.

        Args:
            threshold: 최대-최소 차이 임계값

        Returns:
            불일치 샘플 리스트
        """
        return [
            s for s in self.samples
            if s.max_disagreement >= threshold
        ]

    def compute_confidence_interval(
        self,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        전체 평균 점수의 신뢰구간을 계산한다.

        Returns:
            (mean, lower, upper)
        """
        means = [s.mean_rating for s in self.samples]
        n = len(means)
        if n == 0:
            return 0.0, 0.0, 0.0

        mean = np.mean(means)
        std = np.std(means, ddof=1)
        se = std / np.sqrt(n)

        # z-score for confidence level
        from scipy import stats as scipy_stats
        z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)

        lower = mean - z * se
        upper = mean + z * se

        return float(mean), float(lower), float(upper)

    def compute_report(self) -> dict:
        """전체 분석 리포트를 생성한다."""
        iaa = self.compute_iaa()
        rater_stats = self.compute_rater_stats()
        disagreements = self.find_disagreements(threshold=2)

        means = [s.mean_rating for s in self.samples]

        report = {
            'num_samples': len(self.samples),
            'num_raters': len(self.get_rater_ids()),
            'overall_mean': float(np.mean(means)) if means else 0.0,
            'overall_std': float(np.std(means)) if means else 0.0,
            'iaa': iaa,
            'rater_stats': rater_stats,
            'num_disagreements': len(disagreements),
            'disagreement_rate': len(disagreements) / len(self.samples) if self.samples else 0.0,
            'score_distribution': dict(Counter(
                int(round(m)) for m in means
            )),
        }

        return report

    def print_report(self):
        """분석 결과를 출력한다."""
        report = self.compute_report()
        print("=" * 65)
        print("Human Evaluation 분석 리포트")
        print("=" * 65)
        print(f"  총 샘플 수:      {report['num_samples']}")
        print(f"  평가자 수:       {report['num_raters']}")
        print(f"  전체 평균 점수:  {report['overall_mean']:.3f}")
        print(f"  전체 표준편차:   {report['overall_std']:.3f}")
        print(f"  불일치 샘플 수:  {report['num_disagreements']} "
              f"({report['disagreement_rate']:.1%})")

        print("\n  점수 분포:")
        for score, count in sorted(report['score_distribution'].items()):
            bar = "#" * count
            print(f"    {score}점: {count:4d} {bar}")

        iaa = report['iaa']
        if 'avg_kappa' in iaa:
            print(f"\n  평균 Cohen's Kappa:          {iaa['avg_kappa']:.4f}")
        if 'avg_weighted_kappa' in iaa:
            print(f"  평균 Weighted Kappa:         {iaa['avg_weighted_kappa']:.4f}")
        if 'fleiss_kappa' in iaa:
            print(f"  Fleiss' Kappa:               {iaa['fleiss_kappa']:.4f}")
        if 'krippendorff_alpha_nominal' in iaa:
            print(f"  Krippendorff's Alpha (nom):  {iaa['krippendorff_alpha_nominal']:.4f}")
        if 'krippendorff_alpha_ordinal' in iaa:
            print(f"  Krippendorff's Alpha (ord):  {iaa['krippendorff_alpha_ordinal']:.4f}")

        print("\n  평가자별 통계:")
        for rater, stats in report['rater_stats'].items():
            print(f"    {rater}: 평균={stats['mean']:.2f}, "
                  f"표준편차={stats['std']:.2f}, "
                  f"평가 수={stats['count']}")
        print("=" * 65)
```

### 시각화

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_rater_comparison(
    analyzer: HumanEvaluationAnalyzer,
    save_path: str = None
):
    """평가자 간 점수 분포를 비교한다."""
    raters = analyzer.get_rater_ids()
    rater_stats = analyzer.compute_rater_stats()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 평가자별 점수 분포 (박스플롯)
    rater_ratings = []
    rater_labels = []
    for r in raters:
        ratings = [
            s.ratings[r] for s in analyzer.samples if r in s.ratings
        ]
        rater_ratings.append(ratings)
        rater_labels.append(r)

    axes[0].boxplot(rater_ratings, labels=rater_labels)
    axes[0].set_title('평가자별 점수 분포')
    axes[0].set_ylabel('점수')
    axes[0].set_xlabel('평가자')

    # 평가자별 평균 비교
    means = [rater_stats[r]['mean'] for r in raters]
    stds = [rater_stats[r]['std'] for r in raters]
    axes[1].bar(rater_labels, means, yerr=stds, capsize=5,
                color='steelblue', alpha=0.7)
    axes[1].axhline(np.mean(means), color='red', linestyle='--',
                    label=f'전체 평균: {np.mean(means):.2f}')
    axes[1].set_title('평가자별 평균 점수')
    axes[1].set_ylabel('평균 점수')
    axes[1].set_xlabel('평가자')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_heatmap(
    rater1_labels: List[int],
    rater2_labels: List[int],
    rater1_name: str = "평가자 1",
    rater2_name: str = "평가자 2",
    scale: int = 5,
    save_path: str = None
):
    """두 평가자 간 혼동 행렬을 히트맵으로 시각화한다."""
    labels = list(range(1, scale + 1))
    matrix = np.zeros((scale, scale), dtype=int)

    for r1, r2 in zip(rater1_labels, rater2_labels):
        matrix[r1 - 1][r2 - 1] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='Blues', interpolation='nearest')

    ax.set_xticks(range(scale))
    ax.set_yticks(range(scale))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel(rater2_name)
    ax.set_ylabel(rater1_name)
    ax.set_title('평가자 간 혼동 행렬')

    # 셀에 숫자 표시
    for i in range(scale):
        for j in range(scale):
            color = 'white' if matrix[i][j] > matrix.max() / 2 else 'black'
            ax.text(j, i, str(matrix[i][j]), ha='center', va='center', color=color)

    plt.colorbar(im)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

### 사용 예시

```python
if __name__ == "__main__":
    # 평가 데이터 생성 (시뮬레이션)
    np.random.seed(42)

    analyzer = HumanEvaluationAnalyzer(scale=5)

    rater_ids = ["rater_A", "rater_B", "rater_C"]
    num_samples = 100

    for i in range(num_samples):
        # 기본 점수 (3~5 분포)
        base_score = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])

        ratings = {}
        for r in rater_ids:
            # 평가자별 약간의 노이즈
            noise = np.random.choice([-1, 0, 0, 0, 1])
            score = np.clip(base_score + noise, 1, 5)
            ratings[r] = int(score)

        analyzer.add_sample(
            sample_id=f"sample_{i:03d}",
            ratings=ratings,
            metadata={'doc_type': np.random.choice(['진단서', '입원확인서', '수술기록'])}
        )

    # 분석 리포트
    analyzer.print_report()

    # 불일치 샘플
    disagreements = analyzer.find_disagreements(threshold=2)
    print(f"\n불일치 샘플 (차이 >= 2점): {len(disagreements)}개")
    for s in disagreements[:5]:
        print(f"  {s.sample_id}: {s.ratings} (차이={s.max_disagreement})")

    # IAA 상세
    iaa = analyzer.compute_iaa()
    print("\nCohen's Kappa (쌍별):")
    for pair, stats in iaa.get('cohens_kappa', {}).items():
        print(f"  {pair}: κ={stats['kappa']:.4f}, "
              f"κ_w={stats['weighted_kappa']:.4f}, "
              f"n={stats['n_samples']}")

    # 단독 함수 사용 예시
    rater1 = [5, 4, 3, 4, 5, 3, 4, 4, 5, 3]
    rater2 = [5, 4, 4, 4, 5, 2, 4, 3, 5, 3]

    print(f"\n단독 Cohen's Kappa: {cohens_kappa(rater1, rater2):.4f}")
    print(f"단독 Weighted Kappa: {weighted_kappa(rater1, rater2, weight_type='quadratic'):.4f}")

    # Fleiss' Kappa 단독 사용
    annotations = [
        [5, 4, 3, 4, 5, 3, 4, 4, 5, 3],  # 평가자 1
        [5, 4, 4, 4, 5, 2, 4, 3, 5, 3],  # 평가자 2
        [5, 3, 3, 4, 5, 3, 4, 4, 4, 3],  # 평가자 3
    ]
    matrix = ratings_to_matrix(annotations, num_categories=5)
    print(f"Fleiss' Kappa: {fleiss_kappa(matrix):.4f}")

    # Krippendorff's Alpha (결측값 포함)
    reliability_data = [
        [5, 4, 3,    4, 5, None, 4, 4, 5, 3],   # 평가자 1
        [5, 4, None, 4, 5, 2,    4, 3, 5, 3],   # 평가자 2
        [5, 3, 3,    4, 5, 3,    4, 4, 4, None], # 평가자 3
    ]
    alpha = krippendorff_alpha(reliability_data, 'ordinal')
    print(f"Krippendorff's Alpha (ordinal): {alpha:.4f}")
```

---

## Human Evaluation 모범 사례

| 항목 | 권장 사항 |
|------|----------|
| **평가자 수** | 최소 2명, 이상적으로 3명. 불일치 해소용 중재자 1명 추가 |
| **평가자 교육** | 본 평가 전 파일럿 세트(20~30개)로 연습 후 IAA 확인 |
| **목표 IAA** | Cohen's $\kappa \geq 0.7$, Krippendorff's $\alpha \geq 0.8$ |
| **샘플 수** | 최소 200개, 신뢰구간 5% 이내를 원하면 385개 이상 |
| **평가 순서** | 무작위 섞기(randomize). 순서 효과(order effect) 방지 |
| **세션 길이** | 1시간 이내. 피로에 의한 품질 저하 방지 |
| **이중 평가** | 전체 샘플의 20% 이상은 2명 이상이 평가 |
| **결과 보고** | IAA 값을 반드시 논문/보고서에 포함 |

---

## 용어 체크리스트

학습 후 아래 용어를 설명할 수 있는지 점검하라.

- [ ] **Cohen's Kappa**: $p_o$와 $p_e$가 각각 무엇이고, 왜 $p_e$로 보정해야 하는지 설명할 수 있는가?
- [ ] **Weighted Kappa**: 선형 가중과 이차 가중의 차이를 아는가? 순서형 척도에서 왜 필요한가?
- [ ] **Fleiss' Kappa**: Cohen's Kappa와의 차이점을 설명할 수 있는가?
- [ ] **Krippendorff's Alpha**: 다른 IAA 지표 대비 장점을 아는가? (결측값 허용, 척도 범용성)
- [ ] **Likert Scale**: OCR 평가에서 각 점수가 무엇을 의미하는지 구체적으로 정의할 수 있는가?
- [ ] **IAA 목표값**: $\kappa = 0.7$이 왜 실용적 기준인지 설명할 수 있는가?
- [ ] **Sampling Strategy**: 무작위/층화/오류 집중 샘플링의 장단점을 비교할 수 있는가?
- [ ] **Adjudication**: 불일치 해소 프로세스를 설계할 수 있는가?
