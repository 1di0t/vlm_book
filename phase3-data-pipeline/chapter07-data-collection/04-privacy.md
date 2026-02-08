---
---

# 7.4 개인정보 처리

---

## 핵심 용어 박스

| 용어 | 영문 | 설명 |
|------|------|------|
| PII | Personally Identifiable Information | 개인을 식별할 수 있는 정보 (이름, 주민번호, 주소 등) |
| PHI | Protected Health Information | 보호 대상 건강정보 (진단명, 처방내역, 입원기록 등) |
| 비식별화 | De-identification | 개인정보를 식별할 수 없도록 처리하는 기술적 조치 |
| 가명처리 | Pseudonymization | 원래 값을 가명(대체값)으로 바꾸되 복원 키를 별도 보관 |
| k-anonymity | k-Anonymity | 각 레코드가 최소 k-1개의 동일 속성 레코드와 구별 불가한 상태 |
| 차분 프라이버시 | Differential Privacy | 개별 레코드의 존재 여부가 쿼리 결과에 미치는 영향을 수학적으로 제한 |
| 개인정보보호법 | Personal Information Protection Act | 대한민국 개인정보 보호 기본법 |

---

## 개요

보험 청구 의료 문서에는 환자의 민감한 개인정보와 건강정보가 집중적으로 포함된다. OCR 학습 데이터를 구축할 때 이 정보를 적법하게 처리하지 않으면 법적 제재(과태료, 형사처벌)는 물론 기업 신뢰 손실로 이어진다.

이 장에서는 보험 청구 문서에 포함된 개인정보 유형을 분류하고, 국내 법규 요건을 정리한 뒤, 비식별화 파이프라인을 코드로 구현한다.

---

## 보험 청구 문서의 개인정보 분류

### PII (개인식별정보)

| 정보 유형 | 예시 | 위치 (문서) | 위험도 |
|-----------|------|-------------|--------|
| 환자명 | 홍길동 | 진단서, 입퇴원확인서, 청구서 | 중 |
| 주민등록번호 | 900101-1234567 | 전 문서 | **최고** |
| 주소 | 서울시 강남구 역삼동 123 | 진단서, 청구서 | 중 |
| 전화번호 | 010-1234-5678 | 청구서 | 중 |
| 증권번호 | 12345678 | 청구서 | 높음 |
| 계좌번호 | 110-123-456789 | 청구서 | 높음 |
| 의사 면허번호 | 123456 | 진단서, 수술기록 | 중 |

### PHI (보호 대상 건강정보)

| 정보 유형 | 예시 | 위치 (문서) | 위험도 |
|-----------|------|-------------|--------|
| 상병명 | 급성 충수염, 우울증 | 진단서, 입퇴원확인서 | 높음 |
| 진단코드 | K35.9, F32.0 | 진단서, 입퇴원확인서 | 높음 |
| 수술명 | 복강경 충수절제술 | 수술기록 | 높음 |
| 입퇴원일 | 2024-01-15 ~ 2024-01-20 | 입퇴원확인서 | 중 |
| 처방내역 | 약품명, 용량 | 진료비 세부내역서 | 높음 |
| 검사결과 | 혈액검사 수치 | 진료비 세부내역서 | 중 |

### 준식별자 (Quasi-Identifier)

단독으로는 개인 식별이 불가하지만, 조합하면 식별 가능한 정보:

| 준식별자 | 조합 위험 |
|----------|-----------|
| 생년월일 (주민번호 앞 6자리) | 성별 + 지역 + 생년월일 → 고유 식별 가능 |
| 성별 | 다른 준식별자와 조합 |
| 거주 지역 (시/군/구) | 생년월일 + 성별과 조합 |
| 입원일 | 의료기관 + 입원일 → 식별 가능성 |

---

## 국내 법규 체계

### 1. 개인정보보호법

**핵심 조항:**

| 조항 | 내용 | OCR 데이터셋 적용 |
|------|------|-------------------|
| 제15조 | 개인정보 수집·이용 동의 | 학습 데이터 수집 시 정보주체 동의 필요 |
| 제17조 | 제3자 제공 동의 | 외부 어노테이터에게 데이터 전달 시 적용 |
| 제23조 | 민감정보 처리 제한 | 건강정보(PHI)는 민감정보 → 별도 동의 필요 |
| 제24조 | 고유식별정보 처리 제한 | 주민등록번호 수집 원칙적 금지 |
| 제28조의2 | 가명정보 처리 특례 | 통계·연구 목적으로 가명처리 후 동의 없이 이용 가능 |
| 제29조 | 안전조치 의무 | 기술적·관리적 보호조치 의무 |
| 제71조 | 벌칙 | 위반 시 5년 이하 징역 또는 5천만원 이하 벌금 |

**가명처리 특례 (제28조의2) 적용 조건:**
1. 통계 작성, 과학적 연구, 공익적 기록보존 목적
2. 적정한 가명처리 수행
3. 안전조치 의무 준수
4. 특정 개인 식별 목적으로 가명정보를 다른 정보와 결합 금지

### 2. 의료법

| 조항 | 내용 | 적용 |
|------|------|------|
| 제19조 | 비밀누설 금지 | 의료인이 환자 정보 누설 금지 |
| 제21조 | 기록 열람 | 환자 기록 열람/사본 발급 요건 |
| 제23조 | 전자의무기록 | EMR 관리 기준 |

### 3. 보험업법

| 조항 | 내용 | 적용 |
|------|------|------|
| 제102조 | 비밀유지 의무 | 보험업 종사자의 고객 정보 비밀 유지 |
| 제176조 | 과태료 | 위반 시 1억원 이하 과태료 |

### 법규 적용 시나리오

```
[시나리오 1] 자체 보유 보험 청구 문서로 OCR 모델 학습
→ 개인정보보호법 제28조의2 가명처리 특례 적용 가능
→ 가명처리 후 안전조치 하에 연구 목적 이용

[시나리오 2] 외부 어노테이터에게 데이터 전달
→ 제17조 제3자 제공 → 정보주체 동의 필요 또는 가명처리 후 제공
→ 어노테이터와 비밀유지계약(NDA) 필수

[시나리오 3] 학습된 모델을 외부에 서비스 제공
→ 모델 자체는 개인정보 미포함 (학습 데이터와 분리)
→ 단, 모델이 개인정보를 기억(memorization)하지 않음을 검증해야 함
```

---

## 수학적 원리

### k-anonymity

데이터셋 $D$에서 준식별자(Quasi-Identifier) 집합 $QI$에 대해, 모든 레코드가 동일한 $QI$ 값을 가진 레코드가 최소 $k$개 존재:

$$
\forall r \in D, \quad |\{r' \in D : r'[QI] = r[QI]\}| \geq k
$$

**예시 (k=3):**

| 나이 | 성별 | 지역 | 상병명 |
|------|------|------|--------|
| 30대 | 남 | 서울 | 충수염 |
| 30대 | 남 | 서울 | 골절 |
| 30대 | 남 | 서울 | 폐렴 |
| 40대 | 여 | 경기 | 당뇨 |
| 40대 | 여 | 경기 | 고혈압 |
| 40대 | 여 | 경기 | 당뇨 |

준식별자 `{나이, 성별, 지역}` 기준 각 그룹에 3개 이상 레코드 → 3-anonymity 만족.

**일반화(Generalization):**

$$
\text{Age}: 31 \rightarrow \text{30대}, \quad \text{Address}: \text{강남구} \rightarrow \text{서울}
$$

**억제(Suppression):**

$$
\text{특이한 레코드 삭제}: |group| < k \Rightarrow \text{해당 레코드 제거}
$$

### k-anonymity의 한계

- **동질성 공격 (Homogeneity Attack):** 같은 $QI$ 그룹의 민감 속성이 모두 동일하면 식별 가능
- **배경지식 공격 (Background Knowledge Attack):** 공격자가 부가 정보를 알고 있을 때

이를 보완하기 위한 확장:

**l-diversity:** 각 $QI$ 그룹 내에 민감 속성이 최소 $l$개의 서로 다른 값 보유

$$
\forall g \in \text{groups}(D, QI), \quad |\{v : v \in g[S]\}| \geq l
$$

**t-closeness:** 각 그룹 내 민감 속성 분포와 전체 분포 간 거리 $\leq t$

$$
\forall g, \quad \text{dist}(P_g(S), P_D(S)) \leq t
$$

### Differential Privacy (차분 프라이버시)

메커니즘 $M$이 $\epsilon$-differential privacy를 만족:

$$
\Pr[M(D) \in S] \leq e^{\epsilon} \cdot \Pr[M(D') \in S]
$$

여기서:
- $D$와 $D'$는 하나의 레코드만 다른 이웃 데이터셋
- $S$는 가능한 출력 집합
- $\epsilon$: 프라이버시 예산 (작을수록 강한 보호)

**라플라스 메커니즘 (Laplace Mechanism):**

쿼리 함수 $f$에 대해:

$$
M(D) = f(D) + \text{Lap}\left(\frac{\Delta f}{\epsilon}\right)
$$

여기서 $\Delta f$는 전역 민감도(global sensitivity):

$$
\Delta f = \max_{D, D' \text{ neighbors}} |f(D) - f(D')|
$$

**가우시안 메커니즘 (Gaussian Mechanism):**

$(\epsilon, \delta)$-differential privacy:

$$
M(D) = f(D) + \mathcal{N}\left(0, \frac{2 \ln(1.25/\delta) \cdot (\Delta f)^2}{\epsilon^2}\right)
$$

### 조합 정리 (Composition Theorem)

$k$번의 $\epsilon$-DP 메커니즘을 순차 적용하면:

**기본 조합:**
$$
\epsilon_{\text{total}} = k \cdot \epsilon
$$

**고급 조합 (Advanced Composition):**
$$
\epsilon_{\text{total}} = \sqrt{2k \ln(1/\delta)} \cdot \epsilon + k \cdot \epsilon(e^{\epsilon} - 1)
$$

---

## 비식별화 기법 분류

### 기법 매트릭스

| 기법 | 설명 | 적용 대상 | 가역성 |
|------|------|-----------|--------|
| 마스킹 (Masking) | 값의 일부를 `*`로 치환 | 주민번호, 전화번호 | 비가역 |
| 삭제 (Suppression) | 필드 자체를 제거 | 주소, 계좌번호 | 비가역 |
| 일반화 (Generalization) | 값을 상위 범주로 대체 | 나이, 지역 | 비가역 |
| 가명화 (Pseudonymization) | 일관된 가명으로 대체 | 환자명 | 가역 (키 보유 시) |
| 범주화 (Bucketing) | 수치를 구간으로 변환 | 금액, 나이 | 비가역 |
| 노이즈 추가 (Perturbation) | 랜덤 노이즈 추가 | 금액, 날짜 | 비가역 |
| 데이터 교환 (Swapping) | 레코드 간 값 교환 | 준식별자 | 비가역 |

### 필드별 비식별화 전략

| 필드 | PII/PHI | 비식별화 방법 | 예시 |
|------|---------|---------------|------|
| 환자명 | PII | 가명화 | 홍길동 → 가명_001 |
| 주민등록번호 | PII | 마스킹 + 삭제 | 900101-1****** → 삭제 |
| 주소 | PII | 일반화 | 서울시 강남구 역삼동 123 → 서울시 |
| 전화번호 | PII | 마스킹 | 010-1234-5678 → 010-****-**** |
| 증권번호 | PII | 삭제 | 완전 제거 |
| 계좌번호 | PII | 삭제 | 완전 제거 |
| 상병명 | PHI | 유지 (학습 필요) | 급성 충수염 (유지) |
| 진단코드 | PHI | 유지 (학습 필요) | K35.9 (유지) |
| 수술명 | PHI | 유지 (학습 필요) | 복강경 충수절제술 (유지) |
| 입퇴원일 | PHI | 일반화 | 2024-01-15 → 2024-01 |
| 금액 | - | 유지 또는 범주화 | 150만원 → 100~200만원 |

---

## 코드: 비식별화 파이프라인

```python
"""
보험 청구 의료 문서 비식별화 파이프라인
- 정규식 기반 PII/PHI 탐지
- 마스킹, 삭제, 일반화, 가명화 처리
- 이미지 내 PII 영역 마스킹
- k-anonymity 검증
"""

import re
import hashlib
import logging
import secrets
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 1. PII 유형 정의
# ============================================================

class PIIType(Enum):
    """개인정보 유형"""
    RESIDENT_NUMBER = "resident_number"    # 주민등록번호
    PATIENT_NAME = "patient_name"          # 환자명
    PHONE_NUMBER = "phone_number"          # 전화번호
    ADDRESS = "address"                    # 주소
    ACCOUNT_NUMBER = "account_number"      # 계좌번호
    POLICY_NUMBER = "policy_number"        # 증권번호
    LICENSE_NUMBER = "license_number"      # 의사 면허번호
    EMAIL = "email"                        # 이메일


class ActionType(Enum):
    """비식별화 처리 방식"""
    MASK = "mask"                # 마스킹
    DELETE = "delete"            # 삭제
    GENERALIZE = "generalize"   # 일반화
    PSEUDONYMIZE = "pseudonymize"  # 가명화


# ============================================================
# 2. PII 탐지기 (Regex 기반)
# ============================================================

@dataclass
class PIIMatch:
    """PII 탐지 결과"""
    pii_type: PIIType
    start: int
    end: int
    value: str
    confidence: float = 1.0


class PIIDetector:
    """정규식 기반 PII 탐지기"""

    # PII 유형별 정규식 패턴
    PATTERNS: dict[PIIType, list[re.Pattern]] = {
        PIIType.RESIDENT_NUMBER: [
            # 주민등록번호: 6자리-7자리
            re.compile(r'\b(\d{6})\s*[-–]\s*([1-4]\d{6})\b'),
            # 뒤 7자리 부분 마스킹된 형태도 탐지
            re.compile(r'\b(\d{6})\s*[-–]\s*([1-4]\d{0,6}[*]{0,6})\b'),
        ],
        PIIType.PHONE_NUMBER: [
            # 휴대폰: 010-1234-5678 (다양한 구분자)
            re.compile(r'\b(01[016789])\s*[-.)]\s*(\d{3,4})\s*[-.)]\s*(\d{4})\b'),
            # 일반전화: 02-1234-5678
            re.compile(r'\b(0[2-6][0-5]?)\s*[-.)]\s*(\d{3,4})\s*[-.)]\s*(\d{4})\b'),
        ],
        PIIType.EMAIL: [
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        ],
        PIIType.ACCOUNT_NUMBER: [
            # 주요 은행 계좌번호 패턴 (하이픈 포함)
            re.compile(r'\b(\d{3,4})\s*[-]\s*(\d{2,6})\s*[-]\s*(\d{4,8})\b'),
        ],
        PIIType.PATIENT_NAME: [
            # 한글 이름 (2~5자) — 문맥 키워드 동반 시
            re.compile(r'(?:성명|환자명|피보험자|청구인|수진자)\s*[:：]?\s*([가-힣]{2,5})'),
        ],
        PIIType.ADDRESS: [
            # 주소 패턴 (시/도 + 구/시/군)
            re.compile(
                r'(?:주소|거주지)\s*[:：]?\s*'
                r'([가-힣]+(?:시|도)\s+[가-힣]+(?:구|시|군|군)\s+[가-힣0-9\s-]+)'
            ),
        ],
        PIIType.POLICY_NUMBER: [
            # 증권번호 (문맥 키워드 동반)
            re.compile(r'(?:증권번호|보험증권)\s*[:：]?\s*(\d{6,15})'),
        ],
        PIIType.LICENSE_NUMBER: [
            # 의사 면허번호
            re.compile(r'(?:면허번호|의사면허)\s*[:：]?\s*(\d{5,8})'),
        ],
    }

    def detect(self, text: str) -> list[PIIMatch]:
        """
        텍스트에서 PII 탐지.

        Args:
            text: 입력 텍스트

        Returns:
            PIIMatch 리스트
        """
        matches = []

        for pii_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for m in pattern.finditer(text):
                    matches.append(PIIMatch(
                        pii_type=pii_type,
                        start=m.start(),
                        end=m.end(),
                        value=m.group(),
                    ))

        # 겹치는 매칭 제거 (더 긴 매칭 우선)
        matches = self._remove_overlaps(matches)

        logger.debug("PII 탐지: %d건 발견", len(matches))
        return matches

    @staticmethod
    def _remove_overlaps(matches: list[PIIMatch]) -> list[PIIMatch]:
        """겹치는 매칭 제거 (더 긴 매칭 우선)"""
        if not matches:
            return []

        # 길이 내림차순 정렬
        matches.sort(key=lambda m: -(m.end - m.start))
        result = []
        used_ranges = []

        for m in matches:
            overlaps = False
            for start, end in used_ranges:
                if m.start < end and m.end > start:
                    overlaps = True
                    break
            if not overlaps:
                result.append(m)
                used_ranges.append((m.start, m.end))

        # 위치순 정렬
        result.sort(key=lambda m: m.start)
        return result


# ============================================================
# 3. 비식별화 처리기
# ============================================================

class Pseudonymizer:
    """가명화 처리기 — 동일 원본에 동일 가명 매핑"""

    def __init__(self, salt: Optional[str] = None):
        self.salt = salt or secrets.token_hex(16)
        self._name_counter = 0
        self._name_map: dict[str, str] = {}

    def pseudonymize_name(self, name: str) -> str:
        """이름 가명화"""
        if name not in self._name_map:
            self._name_counter += 1
            self._name_map[name] = f"가명_{self._name_counter:04d}"
        return self._name_map[name]

    def hash_value(self, value: str) -> str:
        """SHA-256 해시 (소금 포함)"""
        salted = f"{self.salt}:{value}"
        return hashlib.sha256(salted.encode()).hexdigest()[:16]


class DeidentificationEngine:
    """비식별화 엔진"""

    # PII 유형별 기본 처리 방식
    DEFAULT_ACTIONS: dict[PIIType, ActionType] = {
        PIIType.RESIDENT_NUMBER: ActionType.MASK,
        PIIType.PATIENT_NAME: ActionType.PSEUDONYMIZE,
        PIIType.PHONE_NUMBER: ActionType.MASK,
        PIIType.ADDRESS: ActionType.GENERALIZE,
        PIIType.ACCOUNT_NUMBER: ActionType.DELETE,
        PIIType.POLICY_NUMBER: ActionType.DELETE,
        PIIType.LICENSE_NUMBER: ActionType.MASK,
        PIIType.EMAIL: ActionType.MASK,
    }

    def __init__(
        self,
        actions: Optional[dict[PIIType, ActionType]] = None,
        pseudonymizer: Optional[Pseudonymizer] = None,
    ):
        self.actions = actions or self.DEFAULT_ACTIONS
        self.pseudonymizer = pseudonymizer or Pseudonymizer()

    def deidentify_text(
        self,
        text: str,
        pii_matches: list[PIIMatch],
    ) -> tuple[str, list[dict]]:
        """
        텍스트 비식별화 처리.

        Args:
            text: 원본 텍스트
            pii_matches: PII 탐지 결과

        Returns:
            (비식별화된 텍스트, 처리 로그)
        """
        if not pii_matches:
            return text, []

        # 뒤에서부터 치환 (인덱스 유지)
        sorted_matches = sorted(pii_matches, key=lambda m: -m.start)
        result = text
        logs = []

        for match in sorted_matches:
            action = self.actions.get(match.pii_type, ActionType.MASK)
            replacement = self._apply_action(match, action)

            result = result[:match.start] + replacement + result[match.end:]

            logs.append({
                "pii_type": match.pii_type.value,
                "action": action.value,
                "original_length": len(match.value),
                "replacement": replacement,
                "position": (match.start, match.end),
            })

        logs.reverse()  # 원래 순서로 복원
        return result, logs

    def _apply_action(self, match: PIIMatch, action: ActionType) -> str:
        """PII 유형별 비식별화 처리 적용"""

        if action == ActionType.DELETE:
            return "[삭제됨]"

        elif action == ActionType.MASK:
            return self._mask_value(match)

        elif action == ActionType.PSEUDONYMIZE:
            return self._pseudonymize_value(match)

        elif action == ActionType.GENERALIZE:
            return self._generalize_value(match)

        return match.value

    def _mask_value(self, match: PIIMatch) -> str:
        """마스킹 처리"""
        value = match.value

        if match.pii_type == PIIType.RESIDENT_NUMBER:
            # 앞 6자리 유지, 뒤 7자리 마스킹
            parts = re.split(r'[-–]', value)
            if len(parts) == 2:
                return f"{parts[0].strip()}-*******"
            return "******-*******"

        elif match.pii_type == PIIType.PHONE_NUMBER:
            # 가운데 자리 마스킹
            digits = re.findall(r'\d+', value)
            if len(digits) == 3:
                return f"{digits[0]}-****-{digits[2]}"
            return "***-****-****"

        elif match.pii_type == PIIType.EMAIL:
            # @앞 3자리만 유지
            at_idx = value.find("@")
            if at_idx > 3:
                return value[:3] + "***" + value[at_idx:]
            return "***@***.***"

        elif match.pii_type == PIIType.LICENSE_NUMBER:
            # 앞 2자리만 유지
            digits = re.findall(r'\d+', value)
            if digits:
                num = digits[0]
                return value.replace(num, num[:2] + "*" * (len(num) - 2))
            return "**" + "*" * 4

        # 기본: 전체 마스킹
        return "*" * len(value)

    def _pseudonymize_value(self, match: PIIMatch) -> str:
        """가명화 처리"""
        if match.pii_type == PIIType.PATIENT_NAME:
            # 키워드 부분 유지, 이름만 가명화
            name_match = re.search(r'[가-힣]{2,5}$', match.value)
            if name_match:
                original_name = name_match.group()
                pseudonym = self.pseudonymizer.pseudonymize_name(original_name)
                return match.value[:name_match.start()] + pseudonym
            return self.pseudonymizer.pseudonymize_name(match.value)

        return self.pseudonymizer.hash_value(match.value)

    def _generalize_value(self, match: PIIMatch) -> str:
        """일반화 처리"""
        if match.pii_type == PIIType.ADDRESS:
            # 시/도 단위만 유지
            city_match = re.search(r'[가-힣]+(?:시|도)', match.value)
            if city_match:
                prefix_match = re.match(r'(?:주소|거주지)\s*[:：]?\s*', match.value)
                prefix = prefix_match.group() if prefix_match else ""
                return prefix + city_match.group()
            return "[지역 비공개]"

        return match.value


# ============================================================
# 4. 이미지 PII 마스킹
# ============================================================

class ImagePIIMasker:
    """
    이미지 내 PII 영역 마스킹.
    OCR 바운딩 박스 좌표를 활용하여 해당 영역을 블러/블랙아웃 처리.
    """

    @staticmethod
    def mask_regions(
        image: np.ndarray,
        regions: list[dict],
        method: str = "blackout",
    ) -> np.ndarray:
        """
        이미지 내 지정 영역 마스킹.

        Args:
            image: 입력 이미지 (numpy array)
            regions: [{"bbox": [x, y, w, h], "pii_type": "..."}, ...]
            method: "blackout" (검정 블록) 또는 "blur" (가우시안 블러)

        Returns:
            마스킹된 이미지
        """
        try:
            import cv2
        except ImportError:
            logger.error("cv2 미설치 — 이미지 마스킹 불가")
            return image

        result = image.copy()

        for region in regions:
            x, y, w, h = [int(v) for v in region["bbox"]]

            # 이미지 경계 클리핑
            x = max(0, x)
            y = max(0, y)
            x2 = min(result.shape[1], x + w)
            y2 = min(result.shape[0], y + h)

            if method == "blackout":
                result[y:y2, x:x2] = 0  # 검정색으로 채움

            elif method == "blur":
                roi = result[y:y2, x:x2]
                if roi.size > 0:
                    # 강한 가우시안 블러
                    ksize = max(31, min(w, h) // 2)
                    if ksize % 2 == 0:
                        ksize += 1
                    blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
                    result[y:y2, x:x2] = blurred

        logger.info("이미지 PII 마스킹 완료: %d 영역", len(regions))
        return result


# ============================================================
# 5. k-anonymity 검증
# ============================================================

class KAnonymityChecker:
    """k-anonymity 검증기"""

    @staticmethod
    def check(
        records: list[dict],
        quasi_identifiers: list[str],
        k: int = 5,
    ) -> dict:
        """
        k-anonymity 만족 여부 검증.

        Args:
            records: 레코드 리스트
            quasi_identifiers: 준식별자 필드명 리스트
            k: 최소 그룹 크기

        Returns:
            {
                "satisfied": bool,
                "min_group_size": int,
                "violating_groups": [...],
                "total_groups": int,
            }
        """
        # QI 값 기반 그룹핑
        groups: dict[tuple, list[int]] = {}

        for idx, record in enumerate(records):
            qi_values = tuple(
                record.get(qi, None) for qi in quasi_identifiers
            )
            groups.setdefault(qi_values, []).append(idx)

        # 위반 그룹 찾기
        violating_groups = []
        min_size = float("inf")

        for qi_values, indices in groups.items():
            size = len(indices)
            min_size = min(min_size, size)

            if size < k:
                violating_groups.append({
                    "qi_values": dict(zip(quasi_identifiers, qi_values)),
                    "group_size": size,
                    "record_indices": indices,
                })

        result = {
            "satisfied": len(violating_groups) == 0,
            "k": k,
            "min_group_size": int(min_size) if min_size != float("inf") else 0,
            "violating_groups": violating_groups,
            "total_groups": len(groups),
            "total_records": len(records),
        }

        logger.info(
            "k-anonymity (k=%d) 검증: %s (최소 그룹 크기=%d)",
            k,
            "만족" if result["satisfied"] else "불만족",
            result["min_group_size"],
        )

        return result

    @staticmethod
    def generalize_for_k_anonymity(
        records: list[dict],
        quasi_identifiers: list[str],
        generalization_rules: dict[str, Callable],
        k: int = 5,
        max_iterations: int = 10,
    ) -> list[dict]:
        """
        k-anonymity 만족을 위한 반복 일반화.

        Args:
            records: 레코드 리스트
            quasi_identifiers: 준식별자 필드명 리스트
            generalization_rules: {필드명: 일반화 함수}
            k: 목표 k값
            max_iterations: 최대 반복 횟수

        Returns:
            일반화된 레코드 리스트
        """
        import copy
        result = copy.deepcopy(records)

        for iteration in range(max_iterations):
            check = KAnonymityChecker.check(result, quasi_identifiers, k)

            if check["satisfied"]:
                logger.info("k-anonymity 달성: %d회 반복", iteration + 1)
                return result

            # 위반 그룹의 레코드에 일반화 적용
            violating_indices = set()
            for vg in check["violating_groups"]:
                violating_indices.update(vg["record_indices"])

            for idx in violating_indices:
                for qi in quasi_identifiers:
                    if qi in generalization_rules:
                        result[idx][qi] = generalization_rules[qi](
                            result[idx].get(qi)
                        )

        logger.warning("k-anonymity 미달성 (최대 반복 도달)")
        return result


# ============================================================
# 6. 통합 비식별화 파이프라인
# ============================================================

class DeidentificationPipeline:
    """
    비식별화 파이프라인 통합 클래스.
    탐지 → 처리 → 검증 → 보고.
    """

    def __init__(
        self,
        actions: Optional[dict[PIIType, ActionType]] = None,
        k_anonymity_k: int = 5,
    ):
        self.detector = PIIDetector()
        self.engine = DeidentificationEngine(actions=actions)
        self.image_masker = ImagePIIMasker()
        self.k_checker = KAnonymityChecker()
        self.k = k_anonymity_k

        logger.info("비식별화 파이프라인 초기화 (k=%d)", self.k)

    def process_text(self, text: str) -> dict:
        """
        텍스트 비식별화 처리.

        Returns:
            {
                "original_length": int,
                "deidentified_text": str,
                "pii_count": int,
                "pii_details": [...],
                "processing_log": [...],
            }
        """
        # 1. PII 탐지
        matches = self.detector.detect(text)

        # 2. 비식별화 적용
        deidentified, logs = self.engine.deidentify_text(text, matches)

        return {
            "original_length": len(text),
            "deidentified_text": deidentified,
            "pii_count": len(matches),
            "pii_details": [
                {
                    "type": m.pii_type.value,
                    "position": (m.start, m.end),
                }
                for m in matches
            ],
            "processing_log": logs,
        }

    def process_image(
        self,
        image: np.ndarray,
        ocr_results: list[dict],
        method: str = "blackout",
    ) -> tuple[np.ndarray, list[dict]]:
        """
        이미지 비식별화 처리.

        Args:
            image: 원본 이미지
            ocr_results: [{"bbox": [x,y,w,h], "text": "..."}, ...]
            method: 마스킹 방식

        Returns:
            (마스킹된 이미지, PII 영역 리스트)
        """
        pii_regions = []

        for ocr_item in ocr_results:
            text = ocr_item.get("text", "")
            matches = self.detector.detect(text)

            if matches:
                pii_regions.append({
                    "bbox": ocr_item["bbox"],
                    "pii_types": [m.pii_type.value for m in matches],
                    "text": text,
                })

        masked_image = self.image_masker.mask_regions(
            image, pii_regions, method=method,
        )

        return masked_image, pii_regions

    def process_dataset(
        self,
        records: list[dict],
        text_fields: list[str],
        quasi_identifiers: Optional[list[str]] = None,
    ) -> dict:
        """
        데이터셋 전체 비식별화.

        Args:
            records: 레코드 리스트
            text_fields: PII 탐지 대상 텍스트 필드 목록
            quasi_identifiers: k-anonymity 검증용 QI 필드 목록

        Returns:
            처리 결과 요약
        """
        total_pii = 0
        processed_records = []

        for record in records:
            processed = dict(record)

            for field_name in text_fields:
                text = record.get(field_name, "")
                if text:
                    result = self.process_text(str(text))
                    processed[field_name] = result["deidentified_text"]
                    total_pii += result["pii_count"]

            processed_records.append(processed)

        # k-anonymity 검증 (QI 지정 시)
        k_result = None
        if quasi_identifiers:
            k_result = self.k_checker.check(
                processed_records, quasi_identifiers, self.k,
            )

        summary = {
            "total_records": len(records),
            "total_pii_detected": total_pii,
            "processed_records": processed_records,
            "k_anonymity": k_result,
        }

        logger.info(
            "데이터셋 비식별화 완료: %d 레코드, %d PII 처리",
            len(records), total_pii,
        )

        return summary


# ============================================================
# 7. 사용 예시
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- 텍스트 PII 탐지 + 비식별화 ---
    sample_text = """
    진단서
    성명: 홍길동
    주민등록번호: 900101-1234567
    주소: 서울시 강남구 역삼동 123-45
    전화번호: 010-1234-5678

    상병명: 급성 충수염
    진단코드: K35.9

    위와 같이 진단합니다.

    서울대학교병원
    면허번호: 123456
    """

    pipeline = DeidentificationPipeline(k_anonymity_k=3)

    # 텍스트 비식별화
    result = pipeline.process_text(sample_text)
    print("=== 비식별화 결과 ===")
    print(result["deidentified_text"])
    print(f"\n탐지된 PII: {result['pii_count']}건")
    for detail in result["pii_details"]:
        print(f"  - {detail['type']}: 위치 {detail['position']}")

    # --- 데이터셋 비식별화 ---
    sample_records = [
        {
            "id": 1,
            "patient_info": "성명: 홍길동, 주민등록번호: 900101-1234567",
            "diagnosis": "급성 충수염 (K35.9)",
            "age_group": "30대",
            "gender": "남",
            "region": "서울",
        },
        {
            "id": 2,
            "patient_info": "성명: 김영희, 주민등록번호: 850505-2345678",
            "diagnosis": "골절 (S72.0)",
            "age_group": "30대",
            "gender": "여",
            "region": "서울",
        },
        {
            "id": 3,
            "patient_info": "성명: 박철수, 주민등록번호: 920303-1456789",
            "diagnosis": "폐렴 (J18.9)",
            "age_group": "30대",
            "gender": "남",
            "region": "서울",
        },
    ]

    dataset_result = pipeline.process_dataset(
        records=sample_records,
        text_fields=["patient_info"],
        quasi_identifiers=["age_group", "gender", "region"],
    )

    print("\n=== 데이터셋 비식별화 결과 ===")
    print(f"총 PII: {dataset_result['total_pii_detected']}건")

    if dataset_result["k_anonymity"]:
        k_res = dataset_result["k_anonymity"]
        print(f"k-anonymity (k={k_res['k']}): {'만족' if k_res['satisfied'] else '불만족'}")
        print(f"최소 그룹 크기: {k_res['min_group_size']}")

    for rec in dataset_result["processed_records"]:
        print(f"  ID {rec['id']}: {rec['patient_info']}")

    # --- k-anonymity 단독 검증 ---
    print("\n=== k-anonymity 검증 ===")
    test_data = [
        {"age": "30대", "gender": "남", "region": "서울", "disease": "충수염"},
        {"age": "30대", "gender": "남", "region": "서울", "disease": "골절"},
        {"age": "30대", "gender": "남", "region": "서울", "disease": "폐렴"},
        {"age": "40대", "gender": "여", "region": "경기", "disease": "당뇨"},
        {"age": "40대", "gender": "여", "region": "경기", "disease": "고혈압"},
    ]

    k_check = KAnonymityChecker.check(
        test_data,
        quasi_identifiers=["age", "gender", "region"],
        k=3,
    )
    print(f"3-anonymity 만족: {k_check['satisfied']}")
    print(f"최소 그룹 크기: {k_check['min_group_size']}")
    if k_check["violating_groups"]:
        for vg in k_check["violating_groups"]:
            print(f"  위반 그룹: {vg['qi_values']} (크기={vg['group_size']})")
```

---

## 비식별화 운영 체크리스트

### 처리 전

- [ ] 처리 대상 데이터의 개인정보 유형 식별 완료
- [ ] 법적 근거 확인 (동의, 가명처리 특례 등)
- [ ] 비식별화 방법 및 수준 결정
- [ ] 처리 담당자 지정 및 권한 부여

### 처리 중

- [ ] PII 탐지율 검증 (수동 샘플링 10% 이상)
- [ ] 비식별화 처리 로그 기록
- [ ] 가명화 매핑 키 안전 보관 (암호화 저장)
- [ ] 처리 결과 교차 검증

### 처리 후

- [ ] k-anonymity 검증 통과 (k >= 5)
- [ ] 재식별 위험성 평가 수행
- [ ] 비식별화 적정성 평가 보고서 작성
- [ ] 원본 데이터 안전 폐기 또는 격리 보관
- [ ] 접근 권한 최소화 (need-to-know 원칙)

---

## 용어 체크리스트

학습 후 아래 용어를 설명할 수 있는지 자가 점검해라.

- [ ] **PII** — 보험 청구 문서에서 PII에 해당하는 필드 7가지를 열거할 수 있다
- [ ] **PHI** — PII와 PHI의 차이를 설명할 수 있다
- [ ] **비식별화** — 마스킹, 삭제, 일반화, 가명화의 차이를 예시로 설명할 수 있다
- [ ] **가명처리** — 개인정보보호법 제28조의2 가명처리 특례 조건을 설명할 수 있다
- [ ] **k-anonymity** — 수학적 정의와 예시를 설명하고, 한계(동질성 공격)를 설명할 수 있다
- [ ] **l-diversity** — k-anonymity를 어떻게 보완하는지 설명할 수 있다
- [ ] **t-closeness** — l-diversity를 어떻게 보완하는지 설명할 수 있다
- [ ] **Differential Privacy** — 수식의 의미와 epsilon의 역할을 설명할 수 있다
- [ ] **라플라스 메커니즘** — 전역 민감도와 노이즈 크기의 관계를 설명할 수 있다
- [ ] **개인정보보호법** — OCR 학습 데이터 구축 시 적용 가능한 법적 근거를 설명할 수 있다
- [ ] **의료법/보험업법** — 의료 문서 데이터 처리 시 주의할 법적 제약을 설명할 수 있다
- [ ] **준식별자** — 조합에 의한 재식별 위험을 예시로 설명할 수 있다
