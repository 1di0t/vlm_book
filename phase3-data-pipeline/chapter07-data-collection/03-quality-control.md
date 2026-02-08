# 7.3 데이터 품질 관리

---

## 핵심 용어 박스

| 용어 | 영문 | 설명 |
|------|------|------|
| 데이터 품질 | Data Quality | 데이터가 목적에 부합하는 정도를 측정하는 다차원 지표 |
| 이상치 탐지 | Outlier Detection | 데이터 분포에서 벗어난 비정상 샘플을 식별하는 기법 |
| 라벨 노이즈 | Label Noise | 어노테이션 오류로 인한 잘못된 라벨 |
| 데이터 드리프트 | Data Drift | 시간 경과에 따라 데이터 분포가 변화하는 현상 |
| 완전성 | Completeness | 누락 없이 데이터가 채워진 정도 |
| 일관성 | Consistency | 동일 항목이 데이터셋 내에서 모순 없이 표현되는 정도 |
| 정확성 | Accuracy | 데이터가 실제 값과 일치하는 정도 |
| 적시성 | Timeliness | 데이터가 현재 상태를 반영하는 정도 |

---

## 개요

OCR 모델의 성능은 학습 데이터의 품질에 직접적으로 비례한다. 의료 문서 OCR에서 데이터 품질 문제는 단순한 성능 저하를 넘어 진단코드 오인식, 금액 오류 같은 실질적 피해로 이어진다.

이 장에서는 데이터 품질의 4대 메트릭을 정의하고, 자동 검증 파이프라인을 설계하며, 라벨 노이즈 탐지/정제 기법을 다룬다.

---

## 데이터 품질 4대 메트릭

### 1. 완전성 (Completeness)

데이터 필드가 빠짐없이 채워져 있는지 측정.

$$
\text{Completeness}(D) = 1 - \frac{\text{null\_count}(D)}{\text{total\_cells}(D)}
$$

| 수준 | 범위 | 판정 |
|------|------|------|
| 우수 | $\geq 0.98$ | 학습 투입 가능 |
| 보통 | $0.90 \sim 0.98$ | 결측 보완 후 사용 |
| 불량 | $< 0.90$ | 재수집 또는 폐기 |

**의료 문서 체크 항목:**
- 환자명 필드가 비어 있는 어노테이션
- 진단코드 없이 상병명만 있는 경우
- 바운딩 박스는 있으나 전사(text)가 없는 경우

### 2. 일관성 (Consistency)

동일한 엔티티가 데이터셋 전체에서 동일하게 표현되는지 측정.

$$
\text{Consistency}(D) = 1 - \frac{|\text{conflicting\_records}(D)|}{|D|}
$$

**의료 문서 체크 항목:**
- 같은 진단코드에 다른 상병명이 매핑된 경우
- 날짜 포맷 혼재 (2024-01-15 vs 2024.01.15 vs 24/01/15)
- 금액 단위 혼재 (1000원 vs 1,000 vs 1000)
- 동일 환자명의 이름 표기 불일치

### 3. 정확성 (Accuracy)

어노테이션이 실제 문서 내용과 일치하는지 측정.

$$
\text{Accuracy}(D) = \frac{|\text{correct\_annotations}(D)|}{|D|}
$$

**의료 문서 체크 항목:**
- 진단코드가 KCD-8 코드표에 존재하는지
- 날짜가 유효한 날짜인지 (2024-02-30 같은 경우 탐지)
- 금액 합산이 맞는지 (항목별 합계 = 총 진료비)
- 바운딩 박스 좌표가 이미지 범위 내인지

### 4. 적시성 (Timeliness)

데이터가 현재 기준을 반영하는지 측정.

$$
\text{Timeliness}(D) = \frac{|\{d \in D : \text{age}(d) \leq T\}|}{|D|}
$$

여기서 $T$는 허용 최대 기간 (예: 최근 2년 이내).

**의료 문서 체크 항목:**
- KCD 코드 버전이 최신인지 (KCD-7 → KCD-8 전환 여부)
- EDI 코드가 현행 건강보험 급여 기준에 부합하는지
- 문서 양식이 현재 사용 중인 양식인지

---

## 수학적 원리

### 종합 품질 점수 (Composite Quality Score)

4대 메트릭의 가중 기하평균:

$$
Q(D) = \prod_{i=1}^{4} m_i(D)^{w_i}
$$

여기서:
- $m_i$: $i$번째 품질 메트릭 값 (0~1)
- $w_i$: 가중치 ($\sum w_i = 1$)

기하평균을 사용하는 이유: 하나라도 0에 가까운 메트릭이 있으면 전체 점수가 급격히 하락하여, 특정 차원의 품질 무시를 방지한다.

### 이상치 탐지: Z-Score

수치형 필드의 이상치를 Z-Score로 탐지:

$$
z_i = \frac{x_i - \mu}{\sigma}
$$

$|z_i| > 3$이면 이상치로 판정 (3-sigma rule).

### 이상치 탐지: IQR (Interquartile Range)

$$
\text{IQR} = Q_3 - Q_1
$$

$$
\text{Lower} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper} = Q_3 + 1.5 \times \text{IQR}
$$

$x < \text{Lower}$ 또는 $x > \text{Upper}$이면 이상치.

### 라벨 노이즈 추정: Confident Learning

Confident Learning (Northcutt et al., 2021)의 핵심 아이디어:

주어진 라벨 $\tilde{y}$와 모델 예측 확률 $\hat{p}(y|x)$에서, 노이즈 전이 행렬 $Q$를 추정:

$$
Q_{\tilde{y}, y^*} = \hat{P}(\tilde{y} = i, y^* = j)
$$

여기서:
- $\tilde{y}$: 주어진 (노이즈 포함) 라벨
- $y^*$: 실제 라벨
- $Q_{ij}$: 실제 라벨이 $j$인 샘플에 $i$ 라벨이 붙을 확률

자기 신뢰 임계값 $t_j$:

$$
t_j = \mathbb{E}_{x : \tilde{y}=j}[\hat{p}(\tilde{y} = j | x; \theta)]
$$

$\hat{p}(\tilde{y} = j | x) < t_j$인 샘플을 라벨 오류 후보로 식별.

### 데이터 드리프트 탐지: KL Divergence

학습 데이터 분포 $P$와 새 데이터 분포 $Q$ 간 분포 차이:

$$
D_{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}
$$

$D_{KL} > \epsilon$ (임계값)이면 드리프트 발생으로 판정.

대칭 버전 (Jensen-Shannon Divergence):

$$
D_{JS}(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M), \quad M = \frac{P + Q}{2}
$$

---

## 자동 검증 파이프라인 설계

### 파이프라인 아키텍처

```
데이터 입수
  ↓
[Stage 1] 스키마 검증 (Schema Validation)
  - 필수 필드 존재 확인
  - 데이터 타입 확인
  - 값 범위 검증
  ↓
[Stage 2] 포맷 정규화 (Format Normalization)
  - 날짜 포맷 통일 (→ YYYY-MM-DD)
  - 금액 포맷 통일 (→ 정수)
  - 코드 체계 매핑 (KCD, EDI)
  ↓
[Stage 3] 관계 검증 (Relational Validation)
  - 필드 간 논리적 관계 확인
  - 퇴원일 >= 입원일
  - 총액 = Σ(항목별 금액)
  ↓
[Stage 4] 통계적 이상치 탐지
  - Z-Score / IQR 기반 수치 이상치
  - 카테고리 분포 이상 탐지
  ↓
[Stage 5] 라벨 노이즈 탐지
  - Confident Learning 기반
  - Cross-validation 기반
  ↓
[Stage 6] 드리프트 모니터링
  - JS Divergence 기반
  - 주기적 분포 비교
  ↓
품질 보고서 생성 → 부적합 데이터 격리/재라벨링
```

---

## 라벨 노이즈 탐지 및 정제 기법

### 기법 1: Cross-Validation 기반 탐지

K-Fold Cross-Validation으로 각 샘플의 out-of-fold 예측 확률을 구하고, 주어진 라벨과 예측이 불일치하는 샘플을 노이즈 후보로 식별.

### 기법 2: Consensus 기반 정제

다수 어노테이터의 라벨이 있을 때, 다수결에 부합하지 않는 라벨을 노이즈로 판정.

### 기법 3: Confident Learning (cleanlab)

모델 예측 확률 + 자기 신뢰 임계값으로 노이즈 전이 행렬을 추정하고, 라벨 오류 확률이 높은 샘플을 자동 식별.

---

## 코드: DataQualityChecker 클래스

```python
"""
데이터 품질 관리 시스템
- 4대 품질 메트릭 (완전성, 일관성, 정확성, 적시성) 측정
- 이상치 탐지 (Z-Score, IQR)
- 라벨 노이즈 탐지
- 데이터 드리프트 탐지
- 자동 검증 파이프라인
"""

import re
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 1. 품질 메트릭 계산
# ============================================================

@dataclass
class QualityMetrics:
    """4대 품질 메트릭 결과"""
    completeness: float = 0.0
    consistency: float = 0.0
    accuracy: float = 0.0
    timeliness: float = 0.0
    composite_score: float = 0.0

    def compute_composite(
        self,
        weights: Optional[dict] = None,
    ) -> float:
        """가중 기하평균 종합 점수 계산"""
        if weights is None:
            weights = {
                "completeness": 0.25,
                "consistency": 0.25,
                "accuracy": 0.30,
                "timeliness": 0.20,
            }

        metrics = {
            "completeness": max(self.completeness, 1e-10),
            "consistency": max(self.consistency, 1e-10),
            "accuracy": max(self.accuracy, 1e-10),
            "timeliness": max(self.timeliness, 1e-10),
        }

        log_score = sum(
            weights[k] * np.log(metrics[k]) for k in weights
        )
        self.composite_score = np.exp(log_score)
        return self.composite_score


# ============================================================
# 2. 이상치 탐지
# ============================================================

class OutlierDetector:
    """이상치 탐지기"""

    @staticmethod
    def z_score_detect(
        values: np.ndarray,
        threshold: float = 3.0,
    ) -> np.ndarray:
        """
        Z-Score 기반 이상치 탐지.

        Args:
            values: 수치 배열
            threshold: Z-Score 임계값 (기본 3.0)

        Returns:
            이상치 인덱스 배열
        """
        if len(values) < 3:
            return np.array([], dtype=int)

        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return np.array([], dtype=int)

        z_scores = np.abs((values - mean) / std)
        return np.where(z_scores > threshold)[0]

    @staticmethod
    def iqr_detect(
        values: np.ndarray,
        multiplier: float = 1.5,
    ) -> np.ndarray:
        """
        IQR 기반 이상치 탐지.

        Args:
            values: 수치 배열
            multiplier: IQR 배수 (기본 1.5)

        Returns:
            이상치 인덱스 배열
        """
        if len(values) < 4:
            return np.array([], dtype=int)

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        return np.where((values < lower) | (values > upper))[0]


# ============================================================
# 3. 라벨 노이즈 탐지
# ============================================================

class LabelNoiseDetector:
    """라벨 노이즈 탐지기"""

    @staticmethod
    def confident_learning_detect(
        labels: np.ndarray,
        pred_probs: np.ndarray,
    ) -> np.ndarray:
        """
        Confident Learning 기반 라벨 노이즈 탐지.

        Args:
            labels: 주어진 라벨 (N,) — 정수 인덱스
            pred_probs: 모델 예측 확률 (N, K) — K는 클래스 수

        Returns:
            노이즈 의심 샘플 인덱스 배열
        """
        n_samples, n_classes = pred_probs.shape

        # 클래스별 자기 신뢰 임계값
        thresholds = np.zeros(n_classes)
        for c in range(n_classes):
            class_mask = labels == c
            if class_mask.sum() > 0:
                thresholds[c] = pred_probs[class_mask, c].mean()
            else:
                thresholds[c] = 0.5

        # 노이즈 후보: 주어진 라벨에 대한 예측 확률이 임계값 미만
        noise_indices = []
        for i in range(n_samples):
            given_label = labels[i]
            if pred_probs[i, given_label] < thresholds[given_label]:
                noise_indices.append(i)

        return np.array(noise_indices, dtype=int)

    @staticmethod
    def consensus_detect(
        multi_labels: np.ndarray,
    ) -> np.ndarray:
        """
        다수결 합의 기반 노이즈 탐지.

        Args:
            multi_labels: (N, M) — N개 샘플, M명 어노테이터의 라벨

        Returns:
            합의에 실패한 샘플 인덱스 배열
        """
        n_samples, n_annotators = multi_labels.shape
        disagreement_indices = []

        for i in range(n_samples):
            labels = multi_labels[i]
            counter = Counter(labels)
            most_common_count = counter.most_common(1)[0][1]

            # 과반수 합의 실패
            if most_common_count <= n_annotators // 2:
                disagreement_indices.append(i)

        return np.array(disagreement_indices, dtype=int)


# ============================================================
# 4. 데이터 드리프트 탐지
# ============================================================

class DriftDetector:
    """데이터 드리프트 탐지기"""

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        KL Divergence 계산.

        Args:
            p: 기준 분포
            q: 비교 분포

        Returns:
            D_KL(P || Q)
        """
        # 0 방지
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)

        # 정규화
        p = p / p.sum()
        q = q / q.sum()

        return float(np.sum(p * np.log(p / q)))

    @staticmethod
    def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Jensen-Shannon Divergence 계산 (대칭).

        Args:
            p: 기준 분포
            q: 비교 분포

        Returns:
            D_JS(P || Q)
        """
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)

        p = p / p.sum()
        q = q / q.sum()

        m = 0.5 * (p + q)

        return 0.5 * float(np.sum(p * np.log(p / m))) + \
               0.5 * float(np.sum(q * np.log(q / m)))

    @staticmethod
    def detect_drift(
        reference_dist: np.ndarray,
        current_dist: np.ndarray,
        threshold: float = 0.1,
    ) -> dict:
        """
        드리프트 탐지.

        Args:
            reference_dist: 기준(학습) 분포
            current_dist: 현재(운영) 분포
            threshold: JS Divergence 임계값

        Returns:
            {"drifted": bool, "js_divergence": float}
        """
        js = DriftDetector.js_divergence(reference_dist, current_dist)
        return {
            "drifted": js > threshold,
            "js_divergence": round(js, 6),
        }


# ============================================================
# 5. DataQualityChecker 통합 클래스
# ============================================================

@dataclass
class ValidationRule:
    """검증 규칙 정의"""
    name: str
    field: str
    rule_type: str           # "required", "regex", "range", "relation", "enum"
    params: dict = field(default_factory=dict)
    severity: str = "error"  # "error" 또는 "warning"


@dataclass
class ValidationIssue:
    """검증 이슈"""
    rule_name: str
    record_id: Any
    field: str
    value: Any
    message: str
    severity: str


class DataQualityChecker:
    """
    데이터 품질 검사 통합 클래스.
    스키마 검증, 포맷 정규화, 관계 검증, 이상치/노이즈/드리프트 탐지를 수행.
    """

    # 의료 문서 기본 검증 규칙
    DEFAULT_RULES = [
        # 필수 필드
        ValidationRule("환자명 필수", "patient_name", "required"),
        ValidationRule("상병명 필수", "disease_name", "required"),
        ValidationRule("진단코드 필수", "diagnosis_code", "required"),
        ValidationRule("의료기관명 필수", "hospital_name", "required"),

        # 정규식 패턴
        ValidationRule(
            "진단코드 형식", "diagnosis_code", "regex",
            {"pattern": r"^[A-Z]\d{2}(\.\d{1,2})?$"},
        ),
        ValidationRule(
            "주민번호 형식", "resident_number", "regex",
            {"pattern": r"^\d{6}-[1-4]\d{6}$"},
        ),
        ValidationRule(
            "날짜 형식", "admission_date", "regex",
            {"pattern": r"^\d{4}-\d{2}-\d{2}$"},
        ),

        # 값 범위
        ValidationRule(
            "금액 양수", "total_bill", "range",
            {"min": 0, "max": 1_000_000_000},
        ),
        ValidationRule(
            "입원일수 범위", "hospital_days", "range",
            {"min": 1, "max": 365},
        ),

        # 관계 규칙
        ValidationRule(
            "퇴원일 >= 입원일", "discharge_date", "relation",
            {"operator": ">=", "compare_field": "admission_date"},
        ),
    ]

    def __init__(
        self,
        rules: Optional[list[ValidationRule]] = None,
        kcd_codes: Optional[set[str]] = None,
    ):
        self.rules = rules or self.DEFAULT_RULES
        self.kcd_codes = kcd_codes or set()
        self.outlier_detector = OutlierDetector()
        self.noise_detector = LabelNoiseDetector()
        self.drift_detector = DriftDetector()

        logger.info("DataQualityChecker 초기화: %d개 규칙 로드", len(self.rules))

    def validate_records(
        self,
        records: list[dict],
    ) -> tuple[QualityMetrics, list[ValidationIssue]]:
        """
        레코드 목록 전체 검증.

        Args:
            records: [{"patient_name": "...", "diagnosis_code": "...", ...}, ...]

        Returns:
            (QualityMetrics, [ValidationIssue, ...])
        """
        issues: list[ValidationIssue] = []

        total_fields = 0
        null_fields = 0
        inconsistent_records = 0
        inaccurate_records = 0
        outdated_records = 0

        for idx, record in enumerate(records):
            record_id = record.get("id", idx)
            record_issues = self._validate_single_record(record, record_id)
            issues.extend(record_issues)

            # 완전성 계산
            for rule in self.rules:
                if rule.rule_type == "required":
                    total_fields += 1
                    val = record.get(rule.field)
                    if val is None or (isinstance(val, str) and not val.strip()):
                        null_fields += 1

            # 정확성: 에러 레벨 이슈 있으면 부정확
            has_error = any(
                i.severity == "error" for i in record_issues
            )
            if has_error:
                inaccurate_records += 1

        n = len(records) if records else 1

        metrics = QualityMetrics()
        metrics.completeness = 1.0 - (null_fields / max(total_fields, 1))
        metrics.consistency = 1.0  # 별도 일관성 검사 필요 시 업데이트
        metrics.accuracy = 1.0 - (inaccurate_records / n)
        metrics.timeliness = 1.0  # 별도 적시성 검사 필요 시 업데이트
        metrics.compute_composite()

        logger.info(
            "검증 완료: %d 레코드, %d 이슈, 종합점수=%.4f",
            len(records), len(issues), metrics.composite_score,
        )

        return metrics, issues

    def _validate_single_record(
        self,
        record: dict,
        record_id: Any,
    ) -> list[ValidationIssue]:
        """단일 레코드 검증"""
        issues = []

        for rule in self.rules:
            value = record.get(rule.field)

            if rule.rule_type == "required":
                if value is None or (isinstance(value, str) and not value.strip()):
                    issues.append(ValidationIssue(
                        rule_name=rule.name,
                        record_id=record_id,
                        field=rule.field,
                        value=value,
                        message=f"필수 필드 '{rule.field}' 누락",
                        severity=rule.severity,
                    ))

            elif rule.rule_type == "regex":
                if value is not None and isinstance(value, str):
                    pattern = rule.params.get("pattern", "")
                    if not re.match(pattern, value):
                        issues.append(ValidationIssue(
                            rule_name=rule.name,
                            record_id=record_id,
                            field=rule.field,
                            value=value,
                            message=f"'{rule.field}' 형식 불일치: {value}",
                            severity=rule.severity,
                        ))

            elif rule.rule_type == "range":
                if value is not None:
                    try:
                        num_val = float(value)
                        min_val = rule.params.get("min", float("-inf"))
                        max_val = rule.params.get("max", float("inf"))
                        if num_val < min_val or num_val > max_val:
                            issues.append(ValidationIssue(
                                rule_name=rule.name,
                                record_id=record_id,
                                field=rule.field,
                                value=value,
                                message=(
                                    f"'{rule.field}' 범위 초과: "
                                    f"{value} (허용: {min_val}~{max_val})"
                                ),
                                severity=rule.severity,
                            ))
                    except (ValueError, TypeError):
                        issues.append(ValidationIssue(
                            rule_name=rule.name,
                            record_id=record_id,
                            field=rule.field,
                            value=value,
                            message=f"'{rule.field}' 숫자 변환 실패: {value}",
                            severity="error",
                        ))

            elif rule.rule_type == "relation":
                compare_field = rule.params.get("compare_field")
                operator = rule.params.get("operator")
                compare_value = record.get(compare_field)

                if value is not None and compare_value is not None:
                    if operator == ">=" and str(value) < str(compare_value):
                        issues.append(ValidationIssue(
                            rule_name=rule.name,
                            record_id=record_id,
                            field=rule.field,
                            value=value,
                            message=(
                                f"관계 규칙 위반: "
                                f"{rule.field}({value}) {operator} "
                                f"{compare_field}({compare_value})"
                            ),
                            severity=rule.severity,
                        ))

        return issues

    def check_consistency(
        self,
        records: list[dict],
        code_field: str = "diagnosis_code",
        name_field: str = "disease_name",
    ) -> tuple[float, list[dict]]:
        """
        코드-이름 일관성 검사.
        동일 코드에 여러 이름이 매핑되면 비일관.

        Returns:
            (일관성 점수, 비일관 항목 리스트)
        """
        code_to_names: dict[str, set[str]] = {}

        for record in records:
            code = record.get(code_field)
            name = record.get(name_field)
            if code and name:
                code_to_names.setdefault(code, set()).add(name)

        inconsistent = []
        for code, names in code_to_names.items():
            if len(names) > 1:
                inconsistent.append({
                    "code": code,
                    "names": list(names),
                })

        total_codes = len(code_to_names) if code_to_names else 1
        consistency = 1.0 - len(inconsistent) / total_codes

        return consistency, inconsistent

    def detect_outliers(
        self,
        records: list[dict],
        numeric_field: str,
        method: str = "zscore",
    ) -> list[dict]:
        """
        수치형 필드 이상치 탐지.

        Args:
            records: 레코드 리스트
            numeric_field: 수치 필드명
            method: "zscore" 또는 "iqr"

        Returns:
            이상치 레코드 리스트
        """
        values = []
        valid_indices = []

        for i, record in enumerate(records):
            val = record.get(numeric_field)
            if val is not None:
                try:
                    values.append(float(val))
                    valid_indices.append(i)
                except (ValueError, TypeError):
                    continue

        if not values:
            return []

        arr = np.array(values)

        if method == "zscore":
            outlier_positions = self.outlier_detector.z_score_detect(arr)
        elif method == "iqr":
            outlier_positions = self.outlier_detector.iqr_detect(arr)
        else:
            raise ValueError(f"알 수 없는 method: {method}")

        outlier_records = []
        for pos in outlier_positions:
            original_idx = valid_indices[pos]
            outlier_records.append({
                "index": original_idx,
                "record": records[original_idx],
                "value": values[pos],
            })

        return outlier_records

    def monitor_drift(
        self,
        reference_records: list[dict],
        current_records: list[dict],
        category_field: str = "document_type",
        threshold: float = 0.1,
    ) -> dict:
        """
        카테고리 분포 기반 드리프트 모니터링.

        Args:
            reference_records: 기준(학습) 데이터
            current_records: 현재(운영) 데이터
            category_field: 카테고리 필드명
            threshold: JS Divergence 임계값

        Returns:
            드리프트 결과
        """
        ref_counter = Counter(
            r.get(category_field) for r in reference_records
            if r.get(category_field)
        )
        cur_counter = Counter(
            r.get(category_field) for r in current_records
            if r.get(category_field)
        )

        # 모든 카테고리 합집합
        all_categories = sorted(set(ref_counter.keys()) | set(cur_counter.keys()))

        ref_total = sum(ref_counter.values()) or 1
        cur_total = sum(cur_counter.values()) or 1

        ref_dist = np.array([ref_counter.get(c, 0) / ref_total for c in all_categories])
        cur_dist = np.array([cur_counter.get(c, 0) / cur_total for c in all_categories])

        result = self.drift_detector.detect_drift(ref_dist, cur_dist, threshold)
        result["categories"] = all_categories
        result["reference_dist"] = ref_dist.tolist()
        result["current_dist"] = cur_dist.tolist()

        return result

    def generate_report(
        self,
        metrics: QualityMetrics,
        issues: list[ValidationIssue],
    ) -> str:
        """품질 보고서 텍스트 생성"""
        lines = [
            "=" * 60,
            "데이터 품질 보고서",
            "=" * 60,
            f"생성 시각: {datetime.now().isoformat()}",
            "",
            "[품질 메트릭]",
            f"  완전성 (Completeness): {metrics.completeness:.4f}",
            f"  일관성 (Consistency):  {metrics.consistency:.4f}",
            f"  정확성 (Accuracy):     {metrics.accuracy:.4f}",
            f"  적시성 (Timeliness):   {metrics.timeliness:.4f}",
            f"  종합 점수:             {metrics.composite_score:.4f}",
            "",
            f"[이슈 요약] 총 {len(issues)}건",
        ]

        error_count = sum(1 for i in issues if i.severity == "error")
        warning_count = sum(1 for i in issues if i.severity == "warning")
        lines.append(f"  에러: {error_count}건, 경고: {warning_count}건")
        lines.append("")

        # 규칙별 이슈 집계
        rule_counts: dict[str, int] = {}
        for issue in issues:
            rule_counts[issue.rule_name] = rule_counts.get(issue.rule_name, 0) + 1

        lines.append("[규칙별 이슈 수]")
        for rule_name, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {rule_name}: {count}건")

        lines.append("")
        lines.append("[상세 이슈 (상위 20건)]")
        for issue in issues[:20]:
            lines.append(
                f"  [{issue.severity.upper()}] "
                f"레코드 {issue.record_id} / {issue.field}: "
                f"{issue.message}"
            )

        verdict = "PASS" if metrics.composite_score >= 0.9 else "FAIL"
        lines.extend(["", f"판정: {verdict}", "=" * 60])

        return "\n".join(lines)


# ============================================================
# 6. 사용 예시
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 테스트 데이터
    test_records = [
        {
            "id": 1,
            "patient_name": "홍길동",
            "disease_name": "급성 충수염",
            "diagnosis_code": "K35.9",
            "hospital_name": "서울대병원",
            "admission_date": "2024-01-15",
            "discharge_date": "2024-01-20",
            "total_bill": 1500000,
            "hospital_days": 6,
        },
        {
            "id": 2,
            "patient_name": "",            # 누락
            "disease_name": "골절",
            "diagnosis_code": "INVALID",    # 형식 오류
            "hospital_name": "강남세브란스",
            "admission_date": "2024-03-10",
            "discharge_date": "2024-03-05",  # 퇴원일 < 입원일
            "total_bill": -50000,            # 음수 금액
            "hospital_days": 3,
        },
        {
            "id": 3,
            "patient_name": "이영희",
            "disease_name": "폐렴",
            "diagnosis_code": "J18.9",
            "hospital_name": "삼성서울병원",
            "admission_date": "2024-05-01",
            "discharge_date": "2024-05-10",
            "total_bill": 3200000,
            "hospital_days": 10,
        },
    ]

    checker = DataQualityChecker()

    # 검증 실행
    metrics, issues = checker.validate_records(test_records)

    # 보고서 출력
    report = checker.generate_report(metrics, issues)
    print(report)

    # 이상치 탐지
    outliers = checker.detect_outliers(test_records, "total_bill", method="iqr")
    print(f"\n금액 이상치: {len(outliers)}건")
    for o in outliers:
        print(f"  레코드 {o['index']}: {o['value']}")

    # 일관성 검사
    consistency, inconsistent = checker.check_consistency(test_records)
    print(f"\n일관성: {consistency:.4f}")
    for item in inconsistent:
        print(f"  코드 {item['code']}: {item['names']}")

    # 드리프트 탐지
    ref_records = [{"document_type": "diagnosis"} for _ in range(50)] + \
                  [{"document_type": "bill"} for _ in range(50)]
    cur_records = [{"document_type": "diagnosis"} for _ in range(20)] + \
                  [{"document_type": "bill"} for _ in range(80)]
    drift_result = checker.monitor_drift(ref_records, cur_records)
    print(f"\n드리프트 탐지: {drift_result['drifted']}, JS={drift_result['js_divergence']}")
```

---

## 품질 관리 운영 가이드

### 검증 빈도

| 단계 | 검증 주기 | 검증 항목 |
|------|-----------|-----------|
| 어노테이션 진행 중 | 매일 | 완전성, 형식 오류 |
| 어노테이션 완료 후 | 1회 | 전체 4대 메트릭, 이상치, 라벨 노이즈 |
| 모델 학습 전 | 1회 | 최종 검증 + 노이즈 정제 |
| 운영 중 | 주간 | 드리프트 모니터링 |

### 이슈 대응 매트릭스

| 이슈 유형 | 심각도 | 대응 |
|-----------|--------|------|
| 필수 필드 누락 | 높음 | 어노테이터 재작업 |
| 형식 오류 | 중간 | 자동 정규화 시도, 실패 시 수동 수정 |
| 이상치 | 중간 | 수동 검토 후 판단 |
| 라벨 노이즈 | 높음 | 전문가 재라벨링 |
| 데이터 드리프트 | 높음 | 추가 데이터 수집 + 모델 재학습 |

---

## 용어 체크리스트

학습 후 아래 용어를 설명할 수 있는지 자가 점검해라.

- [ ] **Completeness** — 측정 수식과 의료 문서 적용 예시를 설명할 수 있다
- [ ] **Consistency** — 코드-이름 매핑 일관성 검사 방법을 설명할 수 있다
- [ ] **Accuracy** — 진단코드 유효성 검증이 왜 중요한지 설명할 수 있다
- [ ] **Timeliness** — KCD 코드 버전 변경이 적시성에 미치는 영향을 설명할 수 있다
- [ ] **Z-Score** — 수식과 3-sigma rule을 설명할 수 있다
- [ ] **IQR** — Q1, Q3 기반 이상치 판정 기준을 설명할 수 있다
- [ ] **Confident Learning** — 노이즈 전이 행렬과 자기 신뢰 임계값을 설명할 수 있다
- [ ] **KL Divergence** — 비대칭성과 JS Divergence와의 차이를 설명할 수 있다
- [ ] **Data Drift** — 원인과 탐지 방법을 설명할 수 있다
- [ ] **DataQualityChecker** — 파이프라인의 6단계를 순서대로 설명할 수 있다
