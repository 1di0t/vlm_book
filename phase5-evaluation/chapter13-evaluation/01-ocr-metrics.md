# 13.1 OCR 메트릭

OCR 시스템의 성능을 정량적으로 측정하는 메트릭들을 다룬다. 문자 단위, 단어 단위, 시퀀스 단위로 나누어 각각의 수학적 정의와 구현을 살펴본다.

> **핵심 용어**
>
> | 용어 | 정의 |
> |------|------|
> | **CER** (Character Error Rate) | 문자 단위 오류율. 치환·삭제·삽입 오류를 정답 문자 수로 나눈 값 |
> | **WER** (Word Error Rate) | 단어 단위 오류율. CER과 동일한 공식을 단어 단위로 적용 |
> | **Edit Distance** | 두 문자열을 같게 만들기 위한 최소 편집 연산 횟수 (Levenshtein Distance) |
> | **Accuracy** | 전체 샘플 중 올바르게 인식한 비율 |
> | **Precision** | 모델이 예측한 것 중 실제 정답인 비율 |
> | **Recall** | 실제 정답 중 모델이 올바르게 예측한 비율 |
> | **F1 Score** | Precision과 Recall의 조화평균 |
> | **BLEU Score** | n-gram 기반 생성 텍스트 품질 평가 지표 |
> | **NED** (Normalized Edit Distance) | 문자열 길이로 정규화한 Edit Distance |

---

## 수학적 원리

### Edit Distance (Levenshtein Distance)

두 문자열 $s_1$과 $s_2$ 사이의 최소 편집 거리를 구하는 DP 알고리즘이다.

문자열 $s_1 = a_1 a_2 \cdots a_m$, $s_2 = b_1 b_2 \cdots b_n$에 대해 $d(i, j)$를 $s_1$의 처음 $i$개 문자와 $s_2$의 처음 $j$개 문자 사이의 편집 거리라 정의한다.

**초기 조건:**

$$
d(i, 0) = i, \quad d(0, j) = j
$$

**점화식:**

$$
d(i, j) = \min \begin{cases}
d(i-1, j) + 1 & \text{(삭제: } a_i \text{를 제거)} \\
d(i, j-1) + 1 & \text{(삽입: } b_j \text{를 추가)} \\
d(i-1, j-1) + \mathbb{1}[a_i \neq b_j] & \text{(치환: } a_i \neq b_j \text{이면 1, 같으면 0)}
\end{cases}
$$

여기서 $\mathbb{1}[a_i \neq b_j]$는 지시 함수(indicator function)로, 두 문자가 다르면 1, 같으면 0이다.

**시간 복잡도:** $O(m \times n)$, **공간 복잡도:** $O(m \times n)$ (최적화 시 $O(\min(m, n))$)

### CER (Character Error Rate)

$$
\text{CER} = \frac{S + D + I}{N}
$$

- $S$: 치환(Substitution) 횟수 — 정답 문자를 다른 문자로 잘못 인식
- $D$: 삭제(Deletion) 횟수 — 정답에 있는 문자를 누락
- $I$: 삽입(Insertion) 횟수 — 정답에 없는 문자를 추가
- $N$: 정답 문자열의 총 문자 수

CER은 0 이상의 값을 가지며, 삽입 오류가 많으면 1을 초과할 수 있다.

### WER (Word Error Rate)

CER과 동일한 공식을 **단어 단위**로 적용한다.

$$
\text{WER} = \frac{S_w + D_w + I_w}{N_w}
$$

공백 기준으로 토큰화한 뒤 단어 수준에서 편집 거리를 계산한다.

### Normalized Edit Distance (NED)

Edit Distance를 문자열 길이로 정규화해서 0~1 범위로 만든다.

$$
\text{NED}(s_1, s_2) = \frac{d(s_1, s_2)}{\max(|s_1|, |s_2|)}
$$

$\text{NED} = 0$이면 두 문자열이 동일, $\text{NED} = 1$이면 완전히 다르다.

**정규화 정확도(Normalized Accuracy):**

$$
\text{NAcc} = 1 - \text{NED}
$$

### BLEU Score

n-gram precision 기반의 생성 텍스트 품질 평가 지표다. 기계번역에서 시작했지만 OCR 후처리 평가에도 사용된다.

**n-gram Precision:**

$$
p_n = \frac{\sum_{\text{n-gram} \in \hat{y}} \min\left(\text{Count}_{\hat{y}}(\text{n-gram}),\; \text{Count}_{y}(\text{n-gram})\right)}{\sum_{\text{n-gram} \in \hat{y}} \text{Count}_{\hat{y}}(\text{n-gram})}
$$

**Brevity Penalty (BP):**

$$
\text{BP} = \begin{cases}
1 & \text{if } c > r \\
e^{1 - r/c} & \text{if } c \leq r
\end{cases}
$$

- $c$: 후보(candidate) 문장의 길이
- $r$: 참조(reference) 문장의 길이

**BLEU Score:**

$$
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

보통 $N = 4$, $w_n = 1/N$으로 설정한다.

### F1 Score

**Precision:**

$$
P = \frac{TP}{TP + FP}
$$

**Recall:**

$$
R = \frac{TP}{TP + FN}
$$

**F1 Score:**

$$
F_1 = 2 \cdot \frac{P \cdot R}{P + R} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}
$$

**Micro F1:** 전체 샘플을 합산해서 계산. 클래스 불균형 시 다수 클래스에 편향된다.

$$
F_{1,\text{micro}} = \frac{2 \cdot \sum_i TP_i}{2 \cdot \sum_i TP_i + \sum_i FP_i + \sum_i FN_i}
$$

**Macro F1:** 클래스별 F1을 평균. 소수 클래스에도 동일 가중치를 부여한다.

$$
F_{1,\text{macro}} = \frac{1}{C} \sum_{i=1}^{C} F_{1,i}
$$

---

## 코드 구현

### Edit Distance (Levenshtein Distance)

```python
import numpy as np
from typing import List, Tuple


def edit_distance(s1: str, s2: str) -> int:
    """
    두 문자열 사이의 Levenshtein Distance를 DP로 계산한다.

    Args:
        s1: 원본 문자열 (정답)
        s2: 비교 문자열 (예측)

    Returns:
        최소 편집 거리 (int)
    """
    m, n = len(s1), len(s2)

    # DP 테이블 초기화
    dp = np.zeros((m + 1, n + 1), dtype=int)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # 삭제
                dp[i][j - 1] + 1,      # 삽입
                dp[i - 1][j - 1] + cost  # 치환 (or 일치)
            )

    return int(dp[m][n])


def edit_distance_with_ops(s1: str, s2: str) -> Tuple[int, List[str]]:
    """
    편집 거리와 함께 실제 편집 연산 목록을 반환한다 (역추적).

    Returns:
        (distance, operations): 편집 거리와 연산 리스트
    """
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    # 역추적으로 연산 목록 추출
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            ops.append(f"MATCH '{s1[i-1]}'")
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(f"SUB '{s1[i-1]}' -> '{s2[j-1]}'")
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(f"INS '{s2[j-1]}'")
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(f"DEL '{s1[i-1]}'")
            i -= 1

    ops.reverse()
    return int(dp[m][n]), ops
```

### CER & WER 계산

```python
def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate를 계산한다.

    Args:
        reference: 정답 문자열
        hypothesis: OCR 예측 문자열

    Returns:
        CER 값 (0 이상, 1 초과 가능)
    """
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else float('inf')

    dist = edit_distance(reference, hypothesis)
    return dist / len(reference)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate를 계산한다. 공백 기준 토큰화 후 단어 단위 편집 거리를 구한다.

    Args:
        reference: 정답 문자열
        hypothesis: OCR 예측 문자열

    Returns:
        WER 값 (0 이상)
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else float('inf')

    # 단어 단위 편집 거리 — 문자열 대신 리스트 비교
    m, n = len(ref_words), len(hyp_words)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return int(dp[m][n]) / len(ref_words)


def compute_ned(s1: str, s2: str) -> float:
    """
    Normalized Edit Distance를 계산한다.

    Returns:
        NED 값 (0~1 범위)
    """
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return edit_distance(s1, s2) / max_len
```

### BLEU Score 계산

```python
from collections import Counter
import math


def compute_ngrams(tokens: List[str], n: int) -> Counter:
    """n-gram 카운터를 생성한다."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(
    reference: str,
    hypothesis: str,
    max_n: int = 4,
    weights: List[float] = None
) -> float:
    """
    BLEU Score를 계산한다.

    Args:
        reference: 정답 문자열
        hypothesis: OCR 예측 문자열
        max_n: 최대 n-gram 차수 (기본 4)
        weights: 각 n-gram의 가중치 (기본 균등)

    Returns:
        BLEU score (0~1)
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n

    ref_tokens = reference.strip().split()
    hyp_tokens = hypothesis.strip().split()

    if len(hyp_tokens) == 0:
        return 0.0

    # Brevity Penalty
    c = len(hyp_tokens)
    r = len(ref_tokens)
    if c > r:
        bp = 1.0
    else:
        bp = math.exp(1 - r / c) if c > 0 else 0.0

    # n-gram precision
    log_precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = compute_ngrams(ref_tokens, n)
        hyp_ngrams = compute_ngrams(hyp_tokens, n)

        if len(hyp_ngrams) == 0:
            log_precisions.append(float('-inf'))
            continue

        # Clipped count
        clipped = 0
        total = 0
        for ngram, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))
            total += count

        precision = clipped / total if total > 0 else 0.0

        if precision > 0:
            log_precisions.append(math.log(precision))
        else:
            log_precisions.append(float('-inf'))

    # 가중 기하평균
    weighted_log = sum(w * lp for w, lp in zip(weights, log_precisions))

    if weighted_log == float('-inf'):
        return 0.0

    return bp * math.exp(weighted_log)
```

### F1 Score (문자 레벨)

```python
def compute_char_f1(reference: str, hypothesis: str) -> dict:
    """
    문자 레벨 F1 Score를 계산한다.
    reference와 hypothesis의 문자 집합 기반으로 Precision, Recall, F1을 구한다.

    Returns:
        {'precision': float, 'recall': float, 'f1': float}
    """
    ref_chars = Counter(reference)
    hyp_chars = Counter(hypothesis)

    # True Positives: 공통 문자 수 (최소값 기준)
    tp = sum((ref_chars & hyp_chars).values())
    fp = sum((hyp_chars - ref_chars).values())
    fn = sum((ref_chars - hyp_chars).values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_micro_macro_f1(
    references: List[str],
    hypotheses: List[str]
) -> dict:
    """
    여러 샘플에 대한 Micro/Macro F1을 계산한다.

    Returns:
        {'micro_f1': float, 'macro_f1': float}
    """
    total_tp, total_fp, total_fn = 0, 0, 0
    f1_scores = []

    for ref, hyp in zip(references, hypotheses):
        result = compute_char_f1(ref, hyp)
        f1_scores.append(result['f1'])

        ref_chars = Counter(ref)
        hyp_chars = Counter(hyp)
        tp = sum((ref_chars & hyp_chars).values())
        fp = sum((hyp_chars - ref_chars).values())
        fn = sum((ref_chars - hyp_chars).values())

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Micro F1
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if (micro_p + micro_r) > 0 else 0.0)

    # Macro F1
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1
    }
```

### 통합 OCR 평가기

```python
from dataclasses import dataclass, field


@dataclass
class OCRMetricResult:
    """단일 샘플의 OCR 메트릭 결과"""
    reference: str
    hypothesis: str
    cer: float = 0.0
    wer: float = 0.0
    ned: float = 0.0
    bleu: float = 0.0
    char_f1: float = 0.0
    edit_dist: int = 0


class OCRMetricEvaluator:
    """
    OCR 시스템의 성능을 종합적으로 평가하는 클래스.

    사용법:
        evaluator = OCRMetricEvaluator()
        evaluator.add_sample("정답 텍스트", "예측 텍스트")
        report = evaluator.compute_report()
    """

    def __init__(self):
        self.results: List[OCRMetricResult] = []

    def add_sample(self, reference: str, hypothesis: str) -> OCRMetricResult:
        """평가 샘플을 추가한다."""
        result = OCRMetricResult(
            reference=reference,
            hypothesis=hypothesis,
            cer=compute_cer(reference, hypothesis),
            wer=compute_wer(reference, hypothesis),
            ned=compute_ned(reference, hypothesis),
            bleu=compute_bleu(reference, hypothesis),
            char_f1=compute_char_f1(reference, hypothesis)['f1'],
            edit_dist=edit_distance(reference, hypothesis)
        )
        self.results.append(result)
        return result

    def add_batch(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> List[OCRMetricResult]:
        """여러 샘플을 일괄 추가한다."""
        assert len(references) == len(hypotheses), \
            "reference와 hypothesis 수가 일치해야 한다"
        return [
            self.add_sample(ref, hyp)
            for ref, hyp in zip(references, hypotheses)
        ]

    def compute_report(self) -> dict:
        """전체 평가 리포트를 생성한다."""
        if not self.results:
            return {}

        n = len(self.results)
        report = {
            'num_samples': n,
            'avg_cer': sum(r.cer for r in self.results) / n,
            'avg_wer': sum(r.wer for r in self.results) / n,
            'avg_ned': sum(r.ned for r in self.results) / n,
            'avg_bleu': sum(r.bleu for r in self.results) / n,
            'avg_char_f1': sum(r.char_f1 for r in self.results) / n,
            'total_edit_dist': sum(r.edit_dist for r in self.results),
            'perfect_match_rate': sum(
                1 for r in self.results if r.cer == 0.0
            ) / n,
        }

        # 분포 정보
        cers = [r.cer for r in self.results]
        report['cer_std'] = float(np.std(cers))
        report['cer_median'] = float(np.median(cers))
        report['cer_p95'] = float(np.percentile(cers, 95))
        report['cer_max'] = float(np.max(cers))

        return report

    def print_report(self):
        """평가 결과를 출력한다."""
        report = self.compute_report()
        print("=" * 60)
        print("OCR 평가 리포트")
        print("=" * 60)
        print(f"  총 샘플 수:          {report['num_samples']}")
        print(f"  평균 CER:            {report['avg_cer']:.4f}")
        print(f"  평균 WER:            {report['avg_wer']:.4f}")
        print(f"  평균 NED:            {report['avg_ned']:.4f}")
        print(f"  평균 BLEU:           {report['avg_bleu']:.4f}")
        print(f"  평균 Char F1:        {report['avg_char_f1']:.4f}")
        print(f"  Perfect Match Rate:  {report['perfect_match_rate']:.4f}")
        print(f"  CER 표준편차:        {report['cer_std']:.4f}")
        print(f"  CER 중위값:          {report['cer_median']:.4f}")
        print(f"  CER 95th percentile: {report['cer_p95']:.4f}")
        print("=" * 60)
```

### 시각화 코드

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_metric_distribution(evaluator: OCRMetricEvaluator, save_path: str = None):
    """
    CER, WER, NED, BLEU 분포를 히스토그램으로 시각화한다.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = {
        'CER': [r.cer for r in evaluator.results],
        'WER': [r.wer for r in evaluator.results],
        'NED': [r.ned for r in evaluator.results],
        'BLEU': [r.bleu for r in evaluator.results],
    }

    for ax, (name, values) in zip(axes.flat, metrics.items()):
        ax.hist(values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.mean(values), color='red', linestyle='--',
                   label=f'평균: {np.mean(values):.4f}')
        ax.axvline(np.median(values), color='green', linestyle=':',
                   label=f'중위값: {np.median(values):.4f}')
        ax.set_title(f'{name} 분포')
        ax.set_xlabel(name)
        ax.set_ylabel('빈도')
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_metric_comparison(evaluator: OCRMetricEvaluator, save_path: str = None):
    """
    샘플별 CER vs WER 산점도와 메트릭 상관관계를 시각화한다.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cers = [r.cer for r in evaluator.results]
    wers = [r.wer for r in evaluator.results]
    bleus = [r.bleu for r in evaluator.results]

    # CER vs WER
    axes[0].scatter(cers, wers, alpha=0.5, s=20, color='steelblue')
    axes[0].plot([0, max(cers)], [0, max(cers)], 'r--', alpha=0.5, label='y=x')
    axes[0].set_xlabel('CER')
    axes[0].set_ylabel('WER')
    axes[0].set_title('CER vs WER 산점도')
    axes[0].legend()

    # CER vs BLEU (역상관 관계)
    axes[1].scatter(cers, bleus, alpha=0.5, s=20, color='darkorange')
    axes[1].set_xlabel('CER')
    axes[1].set_ylabel('BLEU')
    axes[1].set_title('CER vs BLEU 산점도')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_error_analysis(
    reference: str,
    hypothesis: str,
    save_path: str = None
):
    """
    단일 샘플의 편집 연산을 시각적으로 보여준다.
    """
    _, ops = edit_distance_with_ops(reference, hypothesis)

    colors = {
        'MATCH': '#4CAF50',  # 녹색
        'SUB': '#FF9800',    # 주황
        'INS': '#2196F3',    # 파랑
        'DEL': '#F44336',    # 빨강
    }

    fig, ax = plt.subplots(figsize=(max(12, len(ops) * 0.5), 3))

    for i, op in enumerate(ops):
        op_type = op.split()[0]
        color = colors.get(op_type, 'gray')
        ax.barh(0, 1, left=i, color=color, edgecolor='white', height=0.5)
        ax.text(i + 0.5, 0, op_type[0], ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')

    # 범례
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['MATCH'], label='Match'),
        Patch(facecolor=colors['SUB'], label='Substitution'),
        Patch(facecolor=colors['INS'], label='Insertion'),
        Patch(facecolor=colors['DEL'], label='Deletion'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_xlim(0, len(ops))
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_title(f'편집 연산 시각화 (총 편집 거리: {sum(1 for o in ops if "MATCH" not in o)})')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

### 사용 예시

```python
if __name__ == "__main__":
    # 단일 샘플 테스트
    ref = "보험금 청구서 접수일: 2024년 3월 15일"
    hyp = "보험금 청구서 접수일: 2024년 3웛 15일"

    print(f"Reference: {ref}")
    print(f"Hypothesis: {hyp}")
    print(f"Edit Distance: {edit_distance(ref, hyp)}")
    print(f"CER: {compute_cer(ref, hyp):.4f}")
    print(f"WER: {compute_wer(ref, hyp):.4f}")
    print(f"NED: {compute_ned(ref, hyp):.4f}")
    print(f"BLEU: {compute_bleu(ref, hyp):.4f}")
    print(f"Char F1: {compute_char_f1(ref, hyp)}")

    # 편집 연산 추적
    dist, ops = edit_distance_with_ops(ref, hyp)
    print(f"\n편집 연산 (거리={dist}):")
    for op in ops:
        if "MATCH" not in op:
            print(f"  {op}")

    # 배치 평가
    references = [
        "진단명: 급성 충수염",
        "수술일: 2024-03-15",
        "총 진료비: 1,500,000원",
        "입원 기간: 5일",
    ]
    hypotheses = [
        "진단명: 급성 충수엽",
        "수술일: 2024-03-15",
        "총 진료비: 1,500,0OO원",
        "입원 기간: 5일",
    ]

    evaluator = OCRMetricEvaluator()
    evaluator.add_batch(references, hypotheses)
    evaluator.print_report()

    # 시각화
    # plot_metric_distribution(evaluator)
    # plot_error_analysis(ref, hyp)
```

---

## 메트릭 선택 가이드

| 시나리오 | 권장 메트릭 | 이유 |
|---------|-----------|------|
| 문자 단위 정확도 평가 | CER | 가장 세밀한 단위의 오류 측정 |
| 단어 의미 보존 평가 | WER | 단어 단위 오류가 의미 변화에 직결 |
| 서로 다른 길이의 문서 비교 | NED | 길이 정규화로 공정한 비교 가능 |
| 생성 품질 종합 평가 | BLEU | n-gram 기반 유창성 평가 |
| 문자 검출 성능 평가 | Char F1 | 누락과 오검출을 동시 고려 |
| 필드별 정확도 | Accuracy | 완전 일치 기반 (금액, 날짜 등) |
| 시스템 간 비교 | 복합 | CER + WER + F1 조합 사용 |

### 메트릭 간 관계

- **CER과 WER**: $\text{CER} \leq \text{WER}$ (일반적). 한 단어 내 문자 오류 하나가 단어 전체 오류로 카운트되므로 WER이 더 크다.
- **CER과 NED**: CER은 정답 길이 기준, NED는 최대 길이 기준으로 정규화. CER이 NED보다 크거나 같다.
- **CER과 BLEU**: 역상관 관계. CER이 낮으면 BLEU가 높다. 하지만 BLEU는 n-gram 순서를 보존해야 해서 더 엄격할 수 있다.

---

## 용어 체크리스트

학습 후 아래 용어를 설명할 수 있는지 점검하라.

- [ ] **Edit Distance**: DP 점화식을 직접 쓸 수 있는가?
- [ ] **CER**: S, D, I 각각이 무엇을 의미하는지 설명할 수 있는가?
- [ ] **WER**: CER과의 차이점, 일반적으로 WER >= CER인 이유를 아는가?
- [ ] **NED**: 왜 max(|s1|, |s2|)로 나누는지 설명할 수 있는가?
- [ ] **BLEU Score**: Brevity Penalty가 필요한 이유를 아는가?
- [ ] **Precision vs Recall**: OCR 맥락에서 각각 무엇을 의미하는가?
- [ ] **F1 Score**: Micro F1과 Macro F1의 차이를 설명할 수 있는가?
- [ ] **Perfect Match Rate**: CER=0인 샘플 비율이 왜 중요한가?
