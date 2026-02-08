# 7.1 보험 청구 의료 문서 유형 분류

---

## 핵심 용어 박스

| 용어 | 영문 | 설명 |
|------|------|------|
| 진단서 | Medical Certificate / Diagnosis Certificate | 의사가 환자의 질병·상해를 진단하고 발급하는 공식 문서 |
| 수술기록 | Operative Record | 수술 과정, 소견, 절차를 기록한 의무기록 |
| 입퇴원확인서 | Admission/Discharge Certificate | 입원일·퇴원일·진단명 등을 증명하는 문서 |
| 진료비 세부내역서 | Itemized Medical Bill | 진료 항목별 비용을 세분화하여 기재한 명세서 |
| 보험 청구서 | Insurance Claim Form | 보험사에 보험금 지급을 요청하는 서식 |
| KCD | Korean Standard Classification of Diseases | 한국표준질병사인분류 (ICD 기반) |
| EDI | Electronic Data Interchange | 전자문서교환, 의료기관-보험사 간 전자청구 표준 |

---

## 개요

보험 청구 프로세스에서 OCR 시스템이 처리해야 하는 의료 문서는 크게 5종으로 나뉜다. 각 문서는 구조, 양식, 필수 필드가 다르기 때문에 OCR 파이프라인 설계 시 문서 유형별 특성을 정확히 파악해야 한다. 이 장에서는 각 문서의 상세 구조를 분석하고, OCR 난이도를 평가한 뒤, 문서 유형 분류기를 구현한다.

---

## 보험 청구 필수 문서 5종

### 1. 진단서 (Medical Certificate)

**구조적 특징:**
- 상단: 의료기관 정보 (기관명, 주소, 전화번호, 의료기관 코드)
- 환자 정보 블록: 성명, 주민등록번호, 주소
- 진단 블록: 상병명(한글), 진단코드(KCD), 발병일, 진단일
- 소견 블록: 의사 소견 (자유 텍스트, 손글씨 가능)
- 하단: 발급일, 의사 성명, 면허번호, 직인

**필수 필드:**
```
- 환자 성명
- 주민등록번호 (앞 6자리-뒤 7자리)
- 상병명 (한글 진단명)
- 진단코드 (KCD 코드, 예: J18.9)
- 진단일
- 의료기관명
- 의사 면허번호
```

**양식 특징:**
- 대부분 pre-printed 양식 위에 인쇄 또는 타이핑
- 소견란은 손글씨 비율 높음 (약 30~50%)
- 직인(도장)이 텍스트 위에 겹치는 경우 빈번

### 2. 수술기록 (Operative Record)

**구조적 특징:**
- 환자 기본정보 (성명, 등록번호, 병실)
- 수술 정보: 수술일, 수술명(한글/영문), 수술 코드, 집도의
- 수술 소견: 자유 텍스트 (대부분 타이핑, 일부 손글씨)
- 마취 정보: 마취 방법, 마취의
- 하단: 집도의 서명

**필수 필드:**
```
- 환자 성명
- 수술일
- 수술명 (한글)
- 수술 코드
- 집도의 성명
- 마취 방법
```

**양식 특징:**
- EMR(전자의무기록) 출력물이 대부분 → 구조화 수준 높음
- 표 형태로 정리된 경우 많음
- 의학 약어, 영문 혼용 빈번

### 3. 입퇴원확인서 (Admission/Discharge Certificate)

**구조적 특징:**
- 환자 정보: 성명, 주민등록번호, 주소
- 입퇴원 정보: 입원일, 퇴원일, 입원 일수
- 진단 정보: 상병명, 진단코드
- 병실 정보: 병동, 호실
- 하단: 발급일, 의료기관 직인

**필수 필드:**
```
- 환자 성명
- 주민등록번호
- 입원일 (YYYY-MM-DD)
- 퇴원일 (YYYY-MM-DD)
- 상병명
- 진단코드 (KCD)
- 의료기관명
```

**양식 특징:**
- 정형화 수준 높음 (표 기반 양식)
- 손글씨 거의 없음
- 기관별 양식 차이 존재하나 핵심 필드는 동일

### 4. 진료비 세부내역서 (Itemized Medical Bill)

**구조적 특징:**
- 환자 정보 헤더
- 진료 항목 테이블: 코드, 명칭, 단가, 횟수, 금액
- 분류 그룹: 진찰료, 입원료, 투약료, 주사료, 처치료, 검사료 등
- 합계 블록: 급여 합계, 비급여 합계, 총액, 본인부담금
- 하단: 발급일, 기관 정보

**필수 필드:**
```
- 환자 성명
- 진료 기간
- 항목별 코드 (EDI 코드)
- 항목별 금액
- 급여/비급여 구분
- 본인부담금 합계
- 총 진료비
```

**양식 특징:**
- 가장 구조화된 문서 (테이블 중심)
- 숫자 밀집도 높음
- 행 수가 매우 많을 수 있음 (수십~수백 행)
- 작은 폰트 사이즈 빈번

### 5. 보험 청구서 (Insurance Claim Form)

**구조적 특징:**
- 청구인 정보: 성명, 주민번호, 연락처, 계좌정보
- 피보험자 정보: 성명, 증권번호
- 사고/질병 정보: 사고일, 상병명, 원인
- 청구 내역: 청구 항목, 금액
- 첨부서류 체크리스트
- 하단: 청구인 서명, 날짜

**필수 필드:**
```
- 청구인 성명
- 주민등록번호
- 증권번호
- 사고일/발병일
- 청구 금액
- 입금 계좌정보
- 청구인 서명
```

**양식 특징:**
- 보험사별 양식 상이 (각 보험사 고유 서식)
- 수기 작성 비율 높음 (50~80%)
- 체크박스, 서명란 포함
- 볼펜 필기 + pre-printed 양식 혼합

---

## 문서 유형별 OCR 난이도 분석

### 난이도 평가 기준

| 평가 항목 | 설명 | 가중치 |
|-----------|------|--------|
| 구조화 정도 | 테이블/필드 경계 명확성 | 0.25 |
| 손글씨 비율 | 전체 텍스트 중 손글씨 비중 | 0.30 |
| 양식 복잡도 | pre-printed 레이아웃 복잡성 | 0.20 |
| 텍스트 밀도 | 단위 면적당 글자 수 | 0.15 |
| 겹침/노이즈 | 직인, 줄긋기 등 오버레이 | 0.10 |

### 난이도 매트릭스

| 문서 유형 | 구조화 | 손글씨 | 양식 복잡도 | 텍스트 밀도 | 겹침/노이즈 | **종합 난이도** |
|-----------|--------|--------|-------------|-------------|-------------|----------------|
| 진단서 | 중 (0.6) | 고 (0.8) | 중 (0.5) | 저 (0.3) | 고 (0.8) | **0.63** |
| 수술기록 | 고 (0.3) | 저 (0.2) | 저 (0.3) | 중 (0.5) | 저 (0.2) | **0.27** |
| 입퇴원확인서 | 고 (0.2) | 저 (0.1) | 저 (0.3) | 저 (0.2) | 저 (0.2) | **0.18** |
| 진료비 세부내역서 | 고 (0.2) | 저 (0.1) | 중 (0.5) | 고 (0.8) | 저 (0.2) | **0.33** |
| 보험 청구서 | 중 (0.5) | 고 (0.9) | 고 (0.7) | 중 (0.5) | 중 (0.5) | **0.67** |

> 종합 난이도 = Σ (항목 점수 × 가중치). 0에 가까울수록 쉬움, 1에 가까울수록 어려움.

### 난이도 순위 (어려운 순)

1. **보험 청구서** (0.67) — 손글씨 비율 최고, 보험사별 양식 다양
2. **진단서** (0.63) — 직인 겹침 + 소견란 손글씨
3. **진료비 세부내역서** (0.33) — 구조적이나 텍스트 밀도 높음
4. **수술기록** (0.27) — EMR 출력물이라 비교적 깨끗
5. **입퇴원확인서** (0.18) — 가장 정형화, 손글씨 거의 없음

---

## 수학적 원리

### 문서 유형 분류 확률 모델

문서 이미지 $x$가 주어졌을 때, 문서 유형 $c$의 사후 확률은 베이즈 정리로 구한다:

$$
P(c|x) = \frac{P(x|c) \cdot P(c)}{P(x)} = \frac{P(x|c) \cdot P(c)}{\sum_{k=1}^{K} P(x|c_k) \cdot P(c_k)}
$$

여기서:
- $P(c)$: 문서 유형 $c$의 사전 확률 (청구 건수 비율로 추정)
- $P(x|c)$: 유형 $c$에서 이미지 $x$가 관측될 우도 (likelihood)
- $K$: 전체 문서 유형 수 (여기서는 5)

### 분류 결정 규칙

최대 사후 확률(MAP) 기준:

$$
\hat{c} = \arg\max_{c \in \{1, ..., K\}} P(c|x)
$$

### OCR 난이도 점수 (Difficulty Score)

각 문서 유형 $d$의 난이도 점수 $D(d)$:

$$
D(d) = \sum_{i=1}^{N} w_i \cdot s_i(d)
$$

여기서:
- $w_i$: $i$번째 평가 항목의 가중치 ($\sum w_i = 1$)
- $s_i(d)$: 문서 유형 $d$의 $i$번째 항목 점수 ($0 \leq s_i \leq 1$)
- $N$: 평가 항목 수

### 문서 유사도 (Cosine Similarity)

두 문서 유형 간 특성 벡터의 유사도:

$$
\text{sim}(\mathbf{v}_a, \mathbf{v}_b) = \frac{\mathbf{v}_a \cdot \mathbf{v}_b}{\|\mathbf{v}_a\| \cdot \|\mathbf{v}_b\|}
$$

이 유사도가 높은 문서 유형 쌍은 분류기에서 혼동 가능성이 높으므로 추가 특징 추출이 필요하다.

---

## 핵심 추출 필드 정의

### 필드 분류 체계

| 카테고리 | 필드명 | 데이터 타입 | 정규식 패턴 | 필수 여부 |
|----------|--------|-------------|-------------|-----------|
| 환자정보 | 환자명 | 한글 텍스트 | `[가-힣]{2,5}` | 필수 |
| 환자정보 | 주민등록번호 | 숫자+하이픈 | `\d{6}-[1-4]\d{6}` | 필수 |
| 진단정보 | 상병명 | 한글 텍스트 | 자유 텍스트 | 필수 |
| 진단정보 | 진단코드 | 영문+숫자 | `[A-Z]\d{2}(\.\d{1,2})?` | 필수 |
| 일자정보 | 입원일 | 날짜 | `\d{4}[-./]\d{2}[-./]\d{2}` | 조건부 |
| 일자정보 | 퇴원일 | 날짜 | `\d{4}[-./]\d{2}[-./]\d{2}` | 조건부 |
| 수술정보 | 수술명 | 한글/영문 | 자유 텍스트 | 조건부 |
| 금액정보 | 총 진료비 | 숫자 | `[\d,]+원?` | 필수 |
| 금액정보 | 본인부담금 | 숫자 | `[\d,]+원?` | 필수 |
| 기관정보 | 의료기관명 | 한글 텍스트 | 자유 텍스트 | 필수 |

### 필드 간 관계 규칙 (Validation Rules)

```
1. 퇴원일 >= 입원일
2. 입원일수 = 퇴원일 - 입원일 + 1
3. 총 진료비 >= 본인부담금
4. 진단코드는 KCD-8 코드표에 존재해야 함
5. 의사 면허번호는 6자리 숫자
```

---

## 코드: 문서 유형 분류기

```python
"""
보험 청구 의료 문서 유형 분류기
- 이미지 특성 추출 → 규칙 기반 + ML 기반 분류
- 지원 문서 유형: 진단서, 수술기록, 입퇴원확인서, 진료비세부내역서, 보험청구서
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """보험 청구 의료 문서 유형"""
    DIAGNOSIS_CERT = "diagnosis_certificate"       # 진단서
    OPERATIVE_RECORD = "operative_record"           # 수술기록
    ADMISSION_CERT = "admission_certificate"        # 입퇴원확인서
    ITEMIZED_BILL = "itemized_bill"                 # 진료비 세부내역서
    CLAIM_FORM = "insurance_claim_form"             # 보험 청구서
    UNKNOWN = "unknown"


@dataclass
class DocumentFeatures:
    """문서 이미지에서 추출한 특성 벡터"""
    table_ratio: float = 0.0          # 테이블 영역 비율 (0~1)
    handwriting_ratio: float = 0.0    # 손글씨 비율 (0~1)
    text_density: float = 0.0         # 텍스트 밀도 (0~1)
    checkbox_count: int = 0           # 체크박스 수
    stamp_detected: bool = False      # 직인 감지 여부
    signature_detected: bool = False  # 서명 감지 여부
    line_count: int = 0               # 테이블 행 수
    numeric_ratio: float = 0.0        # 숫자 비율
    keyword_scores: dict = field(default_factory=dict)  # 키워드 매칭 점수


@dataclass
class ClassificationResult:
    """분류 결과"""
    doc_type: DocumentType
    confidence: float
    probabilities: dict
    features: DocumentFeatures


class KeywordPatterns:
    """문서 유형별 키워드 패턴"""

    PATTERNS = {
        DocumentType.DIAGNOSIS_CERT: {
            "primary": ["진단서", "진단명", "상병명", "소견"],
            "secondary": ["발병일", "진단일", "면허번호", "향후 치료"],
            "codes": [r"[A-Z]\d{2}\.\d"],
        },
        DocumentType.OPERATIVE_RECORD: {
            "primary": ["수술기록", "수술명", "집도의", "마취"],
            "secondary": ["수술 소견", "수술 전 진단", "수술 후 진단", "절개"],
            "codes": [r"[A-Z]\d{4}"],
        },
        DocumentType.ADMISSION_CERT: {
            "primary": ["입퇴원", "입원일", "퇴원일", "입원확인"],
            "secondary": ["병동", "호실", "입원일수", "퇴원확인"],
            "codes": [],
        },
        DocumentType.ITEMIZED_BILL: {
            "primary": ["세부내역", "진료비", "항목", "단가"],
            "secondary": ["급여", "비급여", "본인부담", "횟수", "금액"],
            "codes": [r"[A-Z]{2}\d{7}"],  # EDI 코드 패턴
        },
        DocumentType.CLAIM_FORM: {
            "primary": ["보험금 청구", "청구서", "증권번호", "피보험자"],
            "secondary": ["사고일", "계좌번호", "청구 금액", "수익자"],
            "codes": [],
        },
    }

    @classmethod
    def compute_keyword_score(cls, text: str, doc_type: DocumentType) -> float:
        """텍스트에서 특정 문서 유형 키워드 매칭 점수 계산"""
        if doc_type not in cls.PATTERNS:
            return 0.0

        patterns = cls.PATTERNS[doc_type]
        score = 0.0

        # primary 키워드: 가중치 2.0
        for kw in patterns["primary"]:
            if kw in text:
                score += 2.0

        # secondary 키워드: 가중치 1.0
        for kw in patterns["secondary"]:
            if kw in text:
                score += 1.0

        # 코드 패턴: 가중치 1.5
        for pattern in patterns["codes"]:
            matches = re.findall(pattern, text)
            score += len(matches) * 1.5

        # 정규화 (최대 점수 대비 비율)
        max_possible = (
            len(patterns["primary"]) * 2.0
            + len(patterns["secondary"]) * 1.0
            + 3.0  # 코드 패턴 최대 2개 가정
        )
        return min(score / max_possible, 1.0) if max_possible > 0 else 0.0


class LayoutAnalyzer:
    """문서 레이아웃 분석기 (이미지 기반)"""

    @staticmethod
    def estimate_table_ratio(image: np.ndarray) -> float:
        """
        이미지에서 테이블 영역 비율 추정.
        수평/수직 라인 검출 기반.
        """
        try:
            import cv2
        except ImportError:
            logger.warning("cv2 미설치 — table_ratio 기본값 반환")
            return 0.0

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 이진화
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        # 수평 라인 검출
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        # 수직 라인 검출
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        # 테이블 영역 = 수평 + 수직 라인 교차 영역
        table_mask = cv2.bitwise_or(h_lines, v_lines)
        table_pixels = np.count_nonzero(table_mask)
        total_pixels = table_mask.shape[0] * table_mask.shape[1]

        return table_pixels / total_pixels if total_pixels > 0 else 0.0

    @staticmethod
    def estimate_text_density(image: np.ndarray) -> float:
        """텍스트 밀도 추정"""
        try:
            import cv2
        except ImportError:
            return 0.0

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        text_pixels = np.count_nonzero(binary)
        total_pixels = binary.shape[0] * binary.shape[1]

        return text_pixels / total_pixels if total_pixels > 0 else 0.0

    @staticmethod
    def detect_stamp(image: np.ndarray) -> bool:
        """직인(빨간 도장) 감지"""
        try:
            import cv2
        except ImportError:
            return False

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 빨간색 범위 (HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        red_ratio = np.count_nonzero(red_mask) / (red_mask.shape[0] * red_mask.shape[1])

        # 빨간 영역이 0.5% 이상이면 직인 존재로 판단
        return red_ratio > 0.005


class DocumentClassifier:
    """
    보험 청구 의료 문서 유형 분류기.
    규칙 기반 + 특성 벡터 기반 앙상블.
    """

    # 문서 유형별 사전 확률 (실제 청구 비율 기반)
    PRIORS = {
        DocumentType.DIAGNOSIS_CERT: 0.25,
        DocumentType.OPERATIVE_RECORD: 0.10,
        DocumentType.ADMISSION_CERT: 0.20,
        DocumentType.ITEMIZED_BILL: 0.25,
        DocumentType.CLAIM_FORM: 0.20,
    }

    def __init__(self, use_layout: bool = True):
        self.use_layout = use_layout
        self.layout_analyzer = LayoutAnalyzer() if use_layout else None
        logger.info("DocumentClassifier 초기화 완료 (layout=%s)", use_layout)

    def extract_features(
        self,
        ocr_text: str,
        image: Optional[np.ndarray] = None,
    ) -> DocumentFeatures:
        """OCR 텍스트 + 이미지에서 특성 추출"""

        features = DocumentFeatures()

        # 키워드 점수 계산
        for doc_type in DocumentType:
            if doc_type == DocumentType.UNKNOWN:
                continue
            score = KeywordPatterns.compute_keyword_score(ocr_text, doc_type)
            features.keyword_scores[doc_type] = score

        # 숫자 비율
        digits = sum(c.isdigit() for c in ocr_text)
        total_chars = len(ocr_text) if ocr_text else 1
        features.numeric_ratio = digits / total_chars

        # 이미지 기반 특성 (이미지 제공 시)
        if image is not None and self.layout_analyzer is not None:
            features.table_ratio = self.layout_analyzer.estimate_table_ratio(image)
            features.text_density = self.layout_analyzer.estimate_text_density(image)
            features.stamp_detected = self.layout_analyzer.detect_stamp(image)

        return features

    def classify(
        self,
        ocr_text: str,
        image: Optional[np.ndarray] = None,
    ) -> ClassificationResult:
        """문서 유형 분류 실행"""

        features = self.extract_features(ocr_text, image)

        # 각 유형별 우도(likelihood) 계산
        likelihoods = {}
        for doc_type in DocumentType:
            if doc_type == DocumentType.UNKNOWN:
                continue
            likelihoods[doc_type] = self._compute_likelihood(features, doc_type)

        # 사후 확률 계산 (베이즈 정리)
        posteriors = {}
        evidence = sum(
            likelihoods[dt] * self.PRIORS[dt] for dt in likelihoods
        )

        if evidence == 0:
            # 모든 우도가 0이면 UNKNOWN
            return ClassificationResult(
                doc_type=DocumentType.UNKNOWN,
                confidence=0.0,
                probabilities={dt.value: 0.0 for dt in likelihoods},
                features=features,
            )

        for doc_type in likelihoods:
            posteriors[doc_type] = (
                likelihoods[doc_type] * self.PRIORS[doc_type] / evidence
            )

        # MAP 결정
        best_type = max(posteriors, key=posteriors.get)
        confidence = posteriors[best_type]

        prob_dict = {dt.value: round(p, 4) for dt, p in posteriors.items()}

        logger.info(
            "분류 결과: %s (confidence=%.4f)", best_type.value, confidence
        )

        return ClassificationResult(
            doc_type=best_type,
            confidence=confidence,
            probabilities=prob_dict,
            features=features,
        )

    def _compute_likelihood(
        self, features: DocumentFeatures, doc_type: DocumentType
    ) -> float:
        """특성 벡터로부터 우도 계산"""

        score = 0.0

        # 키워드 점수 (가중치 0.6)
        kw_score = features.keyword_scores.get(doc_type, 0.0)
        score += kw_score * 0.6

        # 문서 유형별 특화 규칙 (가중치 0.4)
        if doc_type == DocumentType.ITEMIZED_BILL:
            # 세부내역서: 숫자 비율 높고, 테이블 비율 높음
            score += features.numeric_ratio * 0.2
            score += features.table_ratio * 0.2

        elif doc_type == DocumentType.DIAGNOSIS_CERT:
            # 진단서: 직인 있고, 테이블 비율 중간
            if features.stamp_detected:
                score += 0.15
            score += (1.0 - features.numeric_ratio) * 0.1

        elif doc_type == DocumentType.CLAIM_FORM:
            # 청구서: 체크박스, 서명 감지
            if features.signature_detected:
                score += 0.15
            score += features.checkbox_count * 0.02

        elif doc_type == DocumentType.ADMISSION_CERT:
            # 입퇴원확인서: 구조화 높고, 텍스트 밀도 낮음
            score += (1.0 - features.text_density) * 0.15
            score += features.table_ratio * 0.1

        elif doc_type == DocumentType.OPERATIVE_RECORD:
            # 수술기록: 텍스트 밀도 높고, 영문 비율 높음
            score += features.text_density * 0.2

        return max(score, 1e-6)  # 0 방지


def classify_document_batch(
    documents: list[dict],
    classifier: Optional[DocumentClassifier] = None,
) -> list[ClassificationResult]:
    """
    배치 문서 분류.

    Args:
        documents: [{"ocr_text": str, "image_path": str}, ...]
        classifier: 분류기 인스턴스 (없으면 새로 생성)

    Returns:
        분류 결과 리스트
    """
    if classifier is None:
        classifier = DocumentClassifier(use_layout=False)

    results = []
    for doc in documents:
        ocr_text = doc.get("ocr_text", "")
        image = None

        image_path = doc.get("image_path")
        if image_path and Path(image_path).exists():
            try:
                import cv2
                image = cv2.imread(str(image_path))
            except ImportError:
                pass

        result = classifier.classify(ocr_text, image)
        results.append(result)

    return results


# --- 사용 예시 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 테스트 OCR 텍스트
    sample_texts = {
        "진단서": "진단서\n성명: 홍길동\n진단명: 급성 충수염\n진단코드: K35.9\n발병일: 2024-01-15",
        "수술기록": "수술기록지\n수술명: 복강경 충수절제술\n집도의: 김의사\n마취: 전신마취",
        "입퇴원확인서": "입퇴원확인서\n입원일: 2024-01-15\n퇴원일: 2024-01-20\n입원일수: 6일",
        "세부내역서": "진료비 세부내역서\n항목\t단가\t횟수\t금액\n진찰료\t15000\t1\t15000\n급여 합계: 250000",
        "청구서": "보험금 청구서\n증권번호: 12345678\n피보험자: 홍길동\n청구 금액: 500000원",
    }

    classifier = DocumentClassifier(use_layout=False)

    for label, text in sample_texts.items():
        result = classifier.classify(text)
        print(f"\n[{label}]")
        print(f"  분류 결과: {result.doc_type.value}")
        print(f"  신뢰도: {result.confidence:.4f}")
        print(f"  확률분포: {result.probabilities}")
```

---

## 문서 유형별 OCR 파이프라인 전략

| 문서 유형 | 전처리 중점 | OCR 엔진 | 후처리 |
|-----------|------------|----------|--------|
| 진단서 | 직인 제거, 손글씨 영역 분리 | 손글씨 특화 모델 | KCD 코드 검증 |
| 수술기록 | 표 영역 검출 | 범용 OCR | 의학 용어 사전 매칭 |
| 입퇴원확인서 | 양식 정합(template matching) | 범용 OCR | 날짜 포맷 정규화 |
| 진료비 세부내역서 | 테이블 구조 복원 | 테이블 OCR 특화 | EDI 코드 검증, 금액 합산 검증 |
| 보험 청구서 | 손글씨 영역 분리, 체크박스 인식 | 손글씨 특화 모델 | 계좌번호 포맷 검증 |

---

## 용어 체크리스트

학습 후 아래 용어를 설명할 수 있는지 자가 점검해라.

- [ ] **진단서** — 구조, 필수 필드, OCR 난이도를 설명할 수 있다
- [ ] **수술기록** — 구조적 특징과 EMR 출력물 특성을 설명할 수 있다
- [ ] **입퇴원확인서** — 정형화 수준이 높은 이유를 설명할 수 있다
- [ ] **진료비 세부내역서** — 테이블 구조와 EDI 코드 체계를 설명할 수 있다
- [ ] **보험 청구서** — OCR 난이도가 가장 높은 이유를 설명할 수 있다
- [ ] **KCD** — 한국표준질병사인분류의 코드 구조를 알고 있다
- [ ] **EDI** — 전자문서교환 방식과 코드 체계를 설명할 수 있다
- [ ] **MAP 분류** — 베이즈 정리 기반 최대 사후 확률 분류를 수식으로 설명할 수 있다
- [ ] **OCR 난이도 점수** — 가중합 기반 난이도 산출 방식을 이해한다
- [ ] **문서 유형 분류기** — 키워드 + 레이아웃 기반 분류 파이프라인을 설명할 수 있다
