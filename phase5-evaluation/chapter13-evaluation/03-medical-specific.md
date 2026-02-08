# 13.3 보험 청구 문서 특화 평가

보험 청구서 OCR은 범용 OCR과 다르다. 금액 1원, 진단코드 1글자 오류가 보험금 지급 거절이나 과오 청구로 직결된다. 이 장에서는 보험 청구 도메인에 특화된 평가 체계를 구축한다.

> **핵심 용어**
>
> | 용어 | 정의 |
> |------|------|
> | **Critical Field** | 오류 시 금전적·법적 영향이 큰 필드 (금액, 진단코드, 날짜 등) |
> | **Non-critical Field** | 오류 영향이 상대적으로 작은 필드 (환자명, 병원명 등) |
> | **Safety Metric** | Critical Field의 오류를 별도로 추적하는 안전 지표 |
> | **청구 오류율** | OCR 결과 기반 청구가 실패하거나 오류를 일으킬 확률 |
> | **KCD 코드** | 한국표준질병사인분류 코드. 알파벳 1자리 + 숫자 2~4자리 구조 |
> | **가중 정확도** | 필드 중요도에 따라 가중치를 다르게 부여한 종합 정확도 |
> | **청구 적합성** | OCR 결과가 보험 청구 양식 요건을 충족하는지 여부 |
> | **과소/과대 청구** | 금액 오인식으로 인한 실제 금액 대비 차이 |

---

## 수학적 원리

### 가중 평가 모델

보험 청구서의 필드별 중요도가 다르므로, 가중 정확도를 사용한다.

**가중 정확도:**

$$
\text{Score} = w_{\text{amount}} \cdot \text{Acc}_{\text{amount}} + w_{\text{code}} \cdot \text{Acc}_{\text{code}} + w_{\text{date}} \cdot \text{Acc}_{\text{date}} + w_{\text{other}} \cdot \text{Acc}_{\text{other}}
$$

가중치 조건:

$$
w_{\text{amount}} + w_{\text{code}} + w_{\text{date}} + w_{\text{other}} = 1
$$

**권장 가중치:**

| 필드 그룹 | 가중치 | 이유 |
|----------|--------|------|
| 금액 ($w_{\text{amount}}$) | 0.35 | 과소/과대 청구 직결 |
| 진단/수술코드 ($w_{\text{code}}$) | 0.30 | 보험 적용 여부 결정 |
| 날짜 ($w_{\text{date}}$) | 0.20 | 약관 적용 기간 판단 |
| 기타 ($w_{\text{other}}$) | 0.15 | 보조 정보 |

### Critical Field Error Cost

단순 정확도가 아닌, 오류 비용(cost)을 고려한 평가다.

**오류 비용 함수:**

$$
\text{Cost}(e) = \sum_{k \in \text{fields}} c_k \cdot \mathbb{1}[e_k \neq 0]
$$

- $c_k$: 필드 $k$의 오류 비용 (금전적 영향 크기)
- $e_k$: 필드 $k$의 오류 여부

**금액 필드의 오류 심각도:**

$$
\text{Severity}_{\text{amount}} = \frac{|\hat{a} - a|}{a}
$$

- $a$: 실제 금액
- $\hat{a}$: OCR 인식 금액

### 진단코드 (KCD) 오류 모델

KCD 코드 구조: 알파벳 1자리 + 숫자 2~4자리 (예: K35.80)

**코드 오류 분류:**

$$
\text{CodeError} = \begin{cases}
\text{Category Error} & \text{if 알파벳 부분 불일치 (예: K -> R)} \\
\text{Subcategory Error} & \text{if 숫자 주 카테고리 불일치 (예: K35 -> K36)} \\
\text{Detail Error} & \text{if 세부 코드 불일치 (예: K35.80 -> K35.81)} \\
\text{Format Error} & \text{if 코드 형식 자체가 틀림}
\end{cases}
$$

**코드 오류 심각도:**

$$
\text{Severity}_{\text{code}} = \begin{cases}
1.0 & \text{Category Error (완전히 다른 질환)} \\
0.7 & \text{Subcategory Error (같은 대분류, 다른 중분류)} \\
0.3 & \text{Detail Error (같은 중분류, 다른 세부)} \\
0.1 & \text{Format Error (마침표 누락 등)}
\end{cases}
$$

### 날짜 오류 모델

날짜 오류가 보험 약관 적용에 미치는 영향을 정량화한다.

$$
\text{DateError} = |\hat{d} - d| \text{ (일 단위 차이)}
$$

**날짜 오류 심각도:**

$$
\text{Severity}_{\text{date}} = \begin{cases}
0.0 & \text{if } \Delta = 0 \\
0.3 & \text{if } 0 < \Delta \leq 1 \\
0.7 & \text{if } 1 < \Delta \leq 7 \\
1.0 & \text{if } \Delta > 7
\end{cases}
$$

7일 초과 오류는 입퇴원 기간 산정에 심각한 영향을 준다.

### 종합 안전 점수

$$
\text{SafetyScore} = 1 - \frac{\sum_{k \in \text{critical}} \text{Severity}_k \cdot w_k}{\sum_{k \in \text{critical}} w_k}
$$

SafetyScore가 1이면 Critical Field 전부 정확, 0이면 전부 심각한 오류다.

### 청구 적합성 점수

$$
\text{ClaimValidity} = \prod_{k \in \text{required}} \mathbb{1}[\text{validate}_k(\hat{v}_k)]
$$

필수 필드 중 하나라도 유효하지 않으면 전체 청구가 부적합(0)이 된다.

---

## 코드 구현

### 보험 청구 필드 정의

```python
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np


class FieldCategory(Enum):
    """필드 중요도 카테고리"""
    AMOUNT = "amount"      # 금액 필드
    CODE = "code"          # 진단/수술 코드
    DATE = "date"          # 날짜 필드
    OTHER = "other"        # 기타 필드


class ErrorSeverity(Enum):
    """오류 심각도"""
    NONE = 0       # 오류 없음
    LOW = 1        # 경미한 오류
    MEDIUM = 2     # 중간 오류
    HIGH = 3       # 심각한 오류
    CRITICAL = 4   # 치명적 오류


@dataclass
class FieldSpec:
    """보험 청구서 필드 사양"""
    name: str
    category: FieldCategory
    required: bool = True
    weight: float = 1.0
    validation_pattern: Optional[str] = None
    description: str = ""


# 보험 청구서 필드 정의
INSURANCE_CLAIM_FIELDS = {
    # 금액 필드 (Critical)
    "total_amount": FieldSpec(
        name="총 진료비",
        category=FieldCategory.AMOUNT,
        required=True,
        weight=0.35,
        description="총 진료비 (1원 단위 정확도 필수)"
    ),
    "insured_amount": FieldSpec(
        name="보험 적용 금액",
        category=FieldCategory.AMOUNT,
        required=True,
        weight=0.35,
        description="보험 적용 금액"
    ),
    "patient_copay": FieldSpec(
        name="본인 부담금",
        category=FieldCategory.AMOUNT,
        required=True,
        weight=0.30,
        description="환자 본인 부담금"
    ),

    # 코드 필드 (Critical)
    "diagnosis_code": FieldSpec(
        name="진단코드 (KCD)",
        category=FieldCategory.CODE,
        required=True,
        weight=0.40,
        validation_pattern=r'^[A-Z]\d{2,4}(\.\d{1,2})?$',
        description="KCD 코드 (예: K35.80)"
    ),
    "procedure_code": FieldSpec(
        name="수술/의료행위코드",
        category=FieldCategory.CODE,
        required=False,
        weight=0.35,
        description="수술 및 의료 행위 코드"
    ),

    # 날짜 필드 (Critical)
    "admission_date": FieldSpec(
        name="입원일",
        category=FieldCategory.DATE,
        required=True,
        weight=0.30,
        description="입원 날짜"
    ),
    "discharge_date": FieldSpec(
        name="퇴원일",
        category=FieldCategory.DATE,
        required=True,
        weight=0.30,
        description="퇴원 날짜"
    ),
    "surgery_date": FieldSpec(
        name="수술일",
        category=FieldCategory.DATE,
        required=False,
        weight=0.25,
        description="수술 날짜"
    ),

    # 기타 필드 (Non-critical)
    "patient_name": FieldSpec(
        name="환자명",
        category=FieldCategory.OTHER,
        required=True,
        weight=0.15,
        description="환자 이름"
    ),
    "hospital_name": FieldSpec(
        name="병원명",
        category=FieldCategory.OTHER,
        required=True,
        weight=0.10,
        description="의료 기관명"
    ),
    "doctor_name": FieldSpec(
        name="담당의",
        category=FieldCategory.OTHER,
        required=False,
        weight=0.05,
        description="담당 의사 이름"
    ),
}
```

### 필드별 검증 함수

```python
class FieldValidator:
    """보험 청구 필드를 검증하고 오류 심각도를 판정한다."""

    @staticmethod
    def validate_amount(pred: Any, gold: Any) -> Tuple[bool, float, str]:
        """
        금액 필드를 검증한다. 1원 단위 정확도를 요구.

        Returns:
            (is_correct, severity, message)
        """
        def parse_amount(v: Any) -> Optional[float]:
            try:
                s = re.sub(r'[,\s원₩\.]', '', str(v))
                # O/0 혼동 보정 시도
                s = s.replace('O', '0').replace('o', '0')
                s = s.replace('l', '1').replace('I', '1')
                return float(s)
            except (ValueError, TypeError):
                return None

        pred_val = parse_amount(pred)
        gold_val = parse_amount(gold)

        if pred_val is None:
            return False, 1.0, f"금액 파싱 실패: '{pred}'"

        if gold_val is None or gold_val == 0:
            return pred_val == 0, 0.0 if pred_val == 0 else 1.0, ""

        if pred_val == gold_val:
            return True, 0.0, ""

        error_rate = abs(pred_val - gold_val) / abs(gold_val)
        diff = pred_val - gold_val

        if diff > 0:
            direction = "과대 청구"
        else:
            direction = "과소 청구"

        severity = min(1.0, error_rate)  # 오류율을 심각도로 사용

        return False, severity, f"{direction}: {diff:+,.0f}원 (오류율: {error_rate:.2%})"

    @staticmethod
    def validate_kcd_code(pred: str, gold: str) -> Tuple[bool, float, str]:
        """
        KCD 진단코드를 검증한다.
        카테고리/서브카테고리/세부코드 수준별로 오류 심각도를 다르게 판정.

        Returns:
            (is_correct, severity, message)
        """
        pred = str(pred).strip().upper()
        gold = str(gold).strip().upper()

        if pred == gold:
            return True, 0.0, ""

        # 형식 검증
        pattern = r'^[A-Z]\d{2,4}(\.\d{1,2})?$'
        if not re.match(pattern, pred):
            return False, 0.1, f"코드 형식 오류: '{pred}'"

        # 카테고리 비교 (첫 알파벳)
        if pred[0] != gold[0]:
            return False, 1.0, f"카테고리 오류: {gold[0]} -> {pred[0]}"

        # 서브카테고리 비교 (알파벳 + 2자리 숫자)
        pred_sub = pred[:3]
        gold_sub = gold[:3]
        if pred_sub != gold_sub:
            return False, 0.7, f"서브카테고리 오류: {gold_sub} -> {pred_sub}"

        # 세부 코드 비교
        return False, 0.3, f"세부코드 오류: {gold} -> {pred}"

    @staticmethod
    def validate_date(pred: str, gold: str) -> Tuple[bool, float, str]:
        """
        날짜 필드를 검증한다.

        Returns:
            (is_correct, severity, message)
        """
        def parse_date(s: str) -> Optional[datetime]:
            s = str(s).strip()
            formats = [
                '%Y-%m-%d', '%Y.%m.%d', '%Y/%m/%d',
                '%Y년 %m월 %d일', '%Y년%m월%d일',
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    continue
            # 정규식 파싱 시도
            m = re.search(r'(\d{4})[.\-/년\s]*(\d{1,2})[.\-/월\s]*(\d{1,2})', s)
            if m:
                try:
                    return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                except ValueError:
                    pass
            return None

        pred_date = parse_date(pred)
        gold_date = parse_date(gold)

        if pred_date is None:
            return False, 1.0, f"날짜 파싱 실패: '{pred}'"

        if gold_date is None:
            return False, 0.0, "정답 날짜 파싱 실패"

        if pred_date == gold_date:
            return True, 0.0, ""

        delta = abs((pred_date - gold_date).days)

        if delta <= 1:
            severity = 0.3
        elif delta <= 7:
            severity = 0.7
        else:
            severity = 1.0

        return False, severity, f"날짜 오류: {delta}일 차이 ({gold_date.date()} -> {pred_date.date()})"

    @staticmethod
    def validate_text(pred: str, gold: str) -> Tuple[bool, float, str]:
        """일반 텍스트 필드를 검증한다."""
        pred = str(pred).strip()
        gold = str(gold).strip()

        if pred == gold:
            return True, 0.0, ""

        # NED 기반 유사도
        max_len = max(len(pred), len(gold))
        if max_len == 0:
            return True, 0.0, ""

        # 간단한 Levenshtein
        m, n = len(pred), len(gold)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                cost = 0 if pred[i-1] == gold[j-1] else 1
                dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
                prev = temp

        ned = dp[n] / max_len
        severity = ned

        return False, severity, f"텍스트 불일치 (NED={ned:.3f})"
```

### MedicalOCREvaluator 클래스

```python
@dataclass
class FieldEvalResult:
    """단일 필드 평가 결과"""
    field_key: str
    field_name: str
    category: FieldCategory
    gold_value: Any
    pred_value: Any
    is_correct: bool
    severity: float
    weight: float
    message: str = ""


@dataclass
class ClaimEvalResult:
    """단일 청구서 평가 결과"""
    field_results: List[FieldEvalResult] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)

    @property
    def weighted_score(self) -> float:
        """가중 정확도 점수"""
        if not self.field_results:
            return 0.0
        total_weight = sum(fr.weight for fr in self.field_results)
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(
            fr.weight * (1.0 if fr.is_correct else 0.0)
            for fr in self.field_results
        )
        return weighted_sum / total_weight

    @property
    def safety_score(self) -> float:
        """Critical Field만의 안전 점수"""
        critical = [
            fr for fr in self.field_results
            if fr.category in (FieldCategory.AMOUNT, FieldCategory.CODE, FieldCategory.DATE)
        ]
        if not critical:
            return 1.0
        total_weight = sum(fr.weight for fr in critical)
        if total_weight == 0:
            return 1.0
        severity_sum = sum(fr.severity * fr.weight for fr in critical)
        return 1.0 - (severity_sum / total_weight)

    @property
    def claim_valid(self) -> bool:
        """청구 적합성 (필수 Critical 필드 전부 정확해야 True)"""
        if self.missing_required:
            return False
        for fr in self.field_results:
            if fr.category in (FieldCategory.AMOUNT, FieldCategory.CODE) and not fr.is_correct:
                return False
        return True

    @property
    def category_accuracy(self) -> Dict[str, float]:
        """카테고리별 정확도"""
        by_cat = {}
        for fr in self.field_results:
            cat = fr.category.value
            if cat not in by_cat:
                by_cat[cat] = {'correct': 0, 'total': 0}
            by_cat[cat]['total'] += 1
            by_cat[cat]['correct'] += int(fr.is_correct)

        return {
            cat: stats['correct'] / stats['total']
            for cat, stats in by_cat.items()
            if stats['total'] > 0
        }


class MedicalOCREvaluator:
    """
    보험 청구 문서 OCR 전용 평가기.

    특징:
    - 필드 중요도(Critical/Non-critical) 기반 가중 평가
    - KCD 코드 계층적 오류 판정
    - 금액 1원 단위 정확도 검증
    - 날짜 오류 심각도 단계적 판정
    - 청구 적합성 자동 검증
    """

    def __init__(
        self,
        field_specs: Dict[str, FieldSpec] = None,
        category_weights: Dict[str, float] = None
    ):
        self.field_specs = field_specs or INSURANCE_CLAIM_FIELDS
        self.category_weights = category_weights or {
            'amount': 0.35,
            'code': 0.30,
            'date': 0.20,
            'other': 0.15
        }
        self.validator = FieldValidator()
        self.results: List[ClaimEvalResult] = []

    def _validate_field(
        self,
        key: str,
        pred_value: Any,
        gold_value: Any,
        spec: FieldSpec
    ) -> FieldEvalResult:
        """필드 타입에 따른 검증을 수행한다."""
        if pred_value is None:
            return FieldEvalResult(
                field_key=key,
                field_name=spec.name,
                category=spec.category,
                gold_value=gold_value,
                pred_value=None,
                is_correct=False,
                severity=1.0,
                weight=spec.weight,
                message="필드 누락"
            )

        # 카테고리별 검증
        if spec.category == FieldCategory.AMOUNT:
            is_correct, severity, msg = self.validator.validate_amount(pred_value, gold_value)
        elif spec.category == FieldCategory.CODE:
            is_correct, severity, msg = self.validator.validate_kcd_code(pred_value, gold_value)
        elif spec.category == FieldCategory.DATE:
            is_correct, severity, msg = self.validator.validate_date(pred_value, gold_value)
        else:
            is_correct, severity, msg = self.validator.validate_text(pred_value, gold_value)

        # 패턴 검증 (있는 경우)
        if spec.validation_pattern and not is_correct:
            if not re.match(spec.validation_pattern, str(pred_value).strip()):
                severity = max(severity, 0.5)
                msg += f" | 패턴 불일치"

        return FieldEvalResult(
            field_key=key,
            field_name=spec.name,
            category=spec.category,
            gold_value=gold_value,
            pred_value=pred_value,
            is_correct=is_correct,
            severity=severity,
            weight=spec.weight,
            message=msg
        )

    def evaluate_single(
        self,
        prediction: Dict,
        ground_truth: Dict
    ) -> ClaimEvalResult:
        """
        단일 청구서를 평가한다.

        Args:
            prediction: OCR 예측 결과 (dict)
            ground_truth: 정답 (dict)

        Returns:
            ClaimEvalResult
        """
        result = ClaimEvalResult()

        for key, spec in self.field_specs.items():
            gold_value = ground_truth.get(key)
            pred_value = prediction.get(key)

            if gold_value is None:
                continue

            if pred_value is None and spec.required:
                result.missing_required.append(key)

            field_result = self._validate_field(key, pred_value, gold_value, spec)
            result.field_results.append(field_result)

        self.results.append(result)
        return result

    def evaluate_batch(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict]
    ) -> List[ClaimEvalResult]:
        """여러 청구서를 일괄 평가한다."""
        assert len(predictions) == len(ground_truths)
        return [
            self.evaluate_single(pred, gold)
            for pred, gold in zip(predictions, ground_truths)
        ]

    def compute_report(self) -> dict:
        """전체 평가 리포트를 생성한다."""
        if not self.results:
            return {}

        n = len(self.results)

        report = {
            'num_samples': n,
            'avg_weighted_score': sum(r.weighted_score for r in self.results) / n,
            'avg_safety_score': sum(r.safety_score for r in self.results) / n,
            'claim_validity_rate': sum(1 for r in self.results if r.claim_valid) / n,
            'missing_field_rate': sum(
                1 for r in self.results if r.missing_required
            ) / n,
        }

        # 카테고리별 정확도
        cat_totals = {}
        for result in self.results:
            for cat, acc in result.category_accuracy.items():
                if cat not in cat_totals:
                    cat_totals[cat] = []
                cat_totals[cat].append(acc)

        report['category_accuracy'] = {
            cat: np.mean(accs)
            for cat, accs in cat_totals.items()
        }

        # 필드별 정확도
        field_stats = {}
        for result in self.results:
            for fr in result.field_results:
                if fr.field_key not in field_stats:
                    field_stats[fr.field_key] = {
                        'correct': 0, 'total': 0,
                        'severity_sum': 0.0, 'name': fr.field_name
                    }
                field_stats[fr.field_key]['total'] += 1
                field_stats[fr.field_key]['correct'] += int(fr.is_correct)
                field_stats[fr.field_key]['severity_sum'] += fr.severity

        report['field_accuracy'] = {
            key: {
                'name': stats['name'],
                'accuracy': stats['correct'] / stats['total'],
                'avg_severity': stats['severity_sum'] / stats['total'],
                'count': stats['total']
            }
            for key, stats in field_stats.items()
        }

        # 오류 유형 분포
        error_messages = []
        for result in self.results:
            for fr in result.field_results:
                if not fr.is_correct and fr.message:
                    error_messages.append((fr.category.value, fr.message))

        report['error_count_by_category'] = {}
        for cat, _ in error_messages:
            report['error_count_by_category'][cat] = \
                report['error_count_by_category'].get(cat, 0) + 1

        return report

    def print_report(self):
        """평가 결과를 출력한다."""
        report = self.compute_report()
        print("=" * 70)
        print("보험 청구 문서 OCR 평가 리포트")
        print("=" * 70)
        print(f"  총 샘플 수:            {report['num_samples']}")
        print(f"  가중 정확도:           {report['avg_weighted_score']:.4f}")
        print(f"  안전 점수:             {report['avg_safety_score']:.4f}")
        print(f"  청구 적합률:           {report['claim_validity_rate']:.4f}")
        print(f"  필수 필드 누락률:      {report['missing_field_rate']:.4f}")

        print("\n  카테고리별 정확도:")
        for cat, acc in report['category_accuracy'].items():
            print(f"    {cat:10s}: {acc:.4f}")

        print("\n  필드별 정확도:")
        for key, stats in report['field_accuracy'].items():
            print(f"    {stats['name']:20s} ({key}): "
                  f"정확도={stats['accuracy']:.4f}, "
                  f"평균 심각도={stats['avg_severity']:.4f}")

        print("\n  카테고리별 오류 수:")
        for cat, count in report['error_count_by_category'].items():
            print(f"    {cat:10s}: {count}")
        print("=" * 70)

    def print_detail(self, claim_result: ClaimEvalResult):
        """단일 청구서의 상세 평가를 출력한다."""
        print("-" * 70)
        print(f"  가중 점수: {claim_result.weighted_score:.4f}")
        print(f"  안전 점수: {claim_result.safety_score:.4f}")
        print(f"  청구 적합: {'적합' if claim_result.claim_valid else '부적합'}")
        if claim_result.missing_required:
            print(f"  누락 필드: {', '.join(claim_result.missing_required)}")
        print()

        for fr in claim_result.field_results:
            status = "OK" if fr.is_correct else "FAIL"
            cat_mark = "*" if fr.category in (
                FieldCategory.AMOUNT, FieldCategory.CODE
            ) else " "
            print(f"  {cat_mark}[{status:4s}] {fr.field_name:15s}: "
                  f"정답={str(fr.gold_value):20s} "
                  f"예측={str(fr.pred_value):20s} "
                  f"심각도={fr.severity:.2f}")
            if fr.message:
                print(f"          {fr.message}")
        print("-" * 70)
```

### 청구 적합성 자동 검증

```python
class ClaimValidator:
    """
    OCR 결과가 보험 청구 가능한 형태인지 자동 검증한다.

    검증 항목:
    1. 필수 필드 존재 여부
    2. 금액 필드 형식 유효성
    3. 진단코드 형식 유효성
    4. 날짜 논리 검증 (입원일 <= 수술일 <= 퇴원일)
    5. 금액 일관성 (부분 합계 <= 총액)
    """

    def __init__(self, field_specs: Dict[str, FieldSpec] = None):
        self.field_specs = field_specs or INSURANCE_CLAIM_FIELDS

    def validate(self, claim_data: Dict) -> Tuple[bool, List[str]]:
        """
        청구 데이터를 검증한다.

        Returns:
            (is_valid, errors)
        """
        errors = []

        # 1. 필수 필드 존재
        for key, spec in self.field_specs.items():
            if spec.required and key not in claim_data:
                errors.append(f"필수 필드 누락: {spec.name} ({key})")

        # 2. 금액 유효성
        amounts = {}
        for key in ['total_amount', 'insured_amount', 'patient_copay']:
            if key in claim_data:
                try:
                    val = re.sub(r'[,\s원₩]', '', str(claim_data[key]))
                    amounts[key] = float(val)
                    if amounts[key] < 0:
                        errors.append(f"음수 금액: {key}={amounts[key]}")
                except ValueError:
                    errors.append(f"금액 형식 오류: {key}='{claim_data[key]}'")

        # 금액 일관성: 보험적용 + 본인부담 ≈ 총진료비
        if 'total_amount' in amounts and 'insured_amount' in amounts and 'patient_copay' in amounts:
            expected_total = amounts['insured_amount'] + amounts['patient_copay']
            if abs(expected_total - amounts['total_amount']) > 1:
                errors.append(
                    f"금액 불일치: 보험({amounts['insured_amount']:,.0f}) + "
                    f"본인부담({amounts['patient_copay']:,.0f}) = "
                    f"{expected_total:,.0f} != "
                    f"총액({amounts['total_amount']:,.0f})"
                )

        # 3. 진단코드 형식
        if 'diagnosis_code' in claim_data:
            code = str(claim_data['diagnosis_code']).strip()
            if not re.match(r'^[A-Z]\d{2,4}(\.\d{1,2})?$', code.upper()):
                errors.append(f"진단코드 형식 오류: '{code}'")

        # 4. 날짜 논리
        dates = {}
        for key in ['admission_date', 'discharge_date', 'surgery_date']:
            if key in claim_data:
                d = self._parse_date(str(claim_data[key]))
                if d:
                    dates[key] = d
                else:
                    errors.append(f"날짜 파싱 실패: {key}='{claim_data[key]}'")

        if 'admission_date' in dates and 'discharge_date' in dates:
            if dates['admission_date'] > dates['discharge_date']:
                errors.append(
                    f"입원일({dates['admission_date'].date()}) > "
                    f"퇴원일({dates['discharge_date'].date()})"
                )

        if 'surgery_date' in dates:
            if 'admission_date' in dates and dates['surgery_date'] < dates['admission_date']:
                errors.append(
                    f"수술일({dates['surgery_date'].date()}) < "
                    f"입원일({dates['admission_date'].date()})"
                )
            if 'discharge_date' in dates and dates['surgery_date'] > dates['discharge_date']:
                errors.append(
                    f"수술일({dates['surgery_date'].date()}) > "
                    f"퇴원일({dates['discharge_date'].date()})"
                )

        return len(errors) == 0, errors

    @staticmethod
    def _parse_date(s: str) -> Optional[datetime]:
        s = s.strip()
        m = re.search(r'(\d{4})[.\-/년\s]*(\d{1,2})[.\-/월\s]*(\d{1,2})', s)
        if m:
            try:
                return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
        return None
```

### 사용 예시

```python
if __name__ == "__main__":
    # 정답 데이터
    ground_truth = {
        "patient_name": "김철수",
        "hospital_name": "서울대학교병원",
        "diagnosis_code": "K35.80",
        "procedure_code": "Q2861",
        "total_amount": 1500000,
        "insured_amount": 1200000,
        "patient_copay": 300000,
        "admission_date": "2024-03-10",
        "discharge_date": "2024-03-15",
        "surgery_date": "2024-03-11",
    }

    # OCR 예측 (오류 포함)
    prediction = {
        "patient_name": "김철수",
        "hospital_name": "서울대학교병원",
        "diagnosis_code": "K35.8O",      # O(알파벳) vs 0(숫자)
        "procedure_code": "Q2861",
        "total_amount": "1,5OO,OOO",     # O/0 혼동
        "insured_amount": "1,200,000",
        "patient_copay": "300,000",
        "admission_date": "2024.03.10",
        "discharge_date": "2024-03-15",
        "surgery_date": "2024-03-11",
    }

    # 평가 실행
    evaluator = MedicalOCREvaluator()
    result = evaluator.evaluate_single(prediction, ground_truth)

    # 상세 결과
    evaluator.print_detail(result)

    # 청구 적합성 검증
    claim_validator = ClaimValidator()
    is_valid, claim_errors = claim_validator.validate(prediction)
    print(f"\n청구 적합성: {'적합' if is_valid else '부적합'}")
    for err in claim_errors:
        print(f"  - {err}")

    # 배치 평가 예시
    ground_truths = [ground_truth] * 5
    predictions_batch = []

    # 다양한 오류 시나리오
    for i in range(5):
        pred = ground_truth.copy()
        if i == 0:
            pred["total_amount"] = "1,500,OOO"  # O/0 혼동
        elif i == 1:
            pred["diagnosis_code"] = "R35.80"    # 카테고리 오류
        elif i == 2:
            pred["admission_date"] = "2024-04-10"  # 날짜 1달 오류
        elif i == 3:
            pred["patient_copay"] = 350000         # 금액 오류
        # i == 4는 정확한 예측

        predictions_batch.append(pred)

    evaluator2 = MedicalOCREvaluator()
    evaluator2.evaluate_batch(predictions_batch, ground_truths)
    evaluator2.print_report()
```

---

## 보험 청구 OCR 오류 패턴

실무에서 자주 발생하는 OCR 오류 패턴과 대응 방안이다.

| 오류 유형 | 예시 | 영향 | 대응 |
|----------|------|------|------|
| **0/O 혼동** | 1,500,0**O**0 -> 숫자 파싱 실패 | 금액 인식 불가 | 후처리에서 문맥 기반 0/O 보정 |
| **1/l/I 혼동** | K35.**l**0 -> 코드 오류 | 진단 코드 불일치 | 코드 DB 매칭으로 보정 |
| **콤마/마침표** | 1.500.000 vs 1,500,000 | 소수점/천단위 혼동 | 로케일 기반 금액 파서 |
| **날짜 형식** | 03/15/2024 vs 2024-03-15 | 월/일 뒤바뀜 | 다중 형식 파서 + 범위 검증 |
| **누락 필드** | 수술코드 미인식 | 보험금 산정 불가 | Confidence 기반 재처리 |
| **잘린 텍스트** | 1,500,0 (뒤가 잘림) | 금액 과소 인식 | 문서 영역 검출 개선 |

---

## 용어 체크리스트

학습 후 아래 용어를 설명할 수 있는지 점검하라.

- [ ] **Critical Field**: 보험 청구에서 Critical Field가 무엇이고, 왜 별도 관리가 필요한지 설명할 수 있는가?
- [ ] **Safety Metric**: 일반 정확도와 Safety Score의 차이를 아는가?
- [ ] **가중 정확도**: 필드별 가중치를 어떻게 설정하는지, 그 근거를 설명할 수 있는가?
- [ ] **KCD 코드**: 계층 구조(카테고리/서브카테고리/세부코드)를 이해하는가?
- [ ] **과소/과대 청구**: OCR 금액 오류가 어떤 법적 리스크를 만드는지 아는가?
- [ ] **청구 적합성**: 필드 정확도와 청구 적합성의 차이를 설명할 수 있는가?
- [ ] **날짜 논리 검증**: 입원일/수술일/퇴원일의 논리적 순서 검증이 왜 필요한지 아는가?
- [ ] **금액 일관성**: 부분 합산 검증이 OCR 오류 검출에 어떻게 활용되는지 아는가?
