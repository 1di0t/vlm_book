# 7.2 어노테이션 전략

---

## 핵심 용어 박스

| 용어 | 영문 | 설명 |
|------|------|------|
| 바운딩 박스 | Bounding Box | 객체를 감싸는 최소 직사각형 영역 좌표 (x, y, w, h) |
| 폴리곤 | Polygon | 다각형 꼭짓점으로 정의하는 비정형 영역 |
| 세그멘테이션 마스크 | Segmentation Mask | 픽셀 단위로 영역을 표시하는 이진 마스크 |
| OCR 어노테이션 | OCR Annotation | 텍스트 영역 좌표 + 전사(transcription) 정보를 함께 기록 |
| IoU | Intersection over Union | 두 영역의 겹침 비율을 측정하는 지표 |
| NMS | Non-Maximum Suppression | IoU 기반으로 중복 검출 결과를 제거하는 알고리즘 |
| Cohen's Kappa | Cohen's Kappa Coefficient | 어노테이터 간 일치도를 측정하는 통계 지표 |
| COCO Format | Common Objects in Context Format | 객체 검출 데이터셋의 표준 JSON 포맷 |

---

## 개요

OCR 모델 학습에 필요한 어노테이션 데이터를 구축하는 것은 전체 파이프라인에서 가장 비용이 많이 드는 단계다. 특히 의료 문서는 도메인 전문 지식이 필요하므로 어노테이터 교육, 품질 관리, 일관성 유지가 핵심이다.

이 장에서는 어노테이션 유형, 수학적 품질 지표, 도구 비교, 의료 문서 특화 가이드라인, 그리고 COCO 포맷 변환 코드를 다룬다.

---

## 어노테이션 유형

### 1. Bounding Box (바운딩 박스)

가장 기본적인 어노테이션 방식. 직사각형 좌표로 텍스트 영역을 표현한다.

**포맷:**
```json
{
  "bbox": [x_min, y_min, width, height],
  "category": "patient_name",
  "text": "홍길동"
}
```

**장점:** 빠른 어노테이션 속도, 도구 지원 폭넓음
**단점:** 기울어진 텍스트나 곡선 영역 표현 불가

**적합한 문서:** 입퇴원확인서, 진료비 세부내역서 (정형화 양식)

### 2. Oriented Bounding Box (회전 바운딩 박스)

기울어진 텍스트를 표현하기 위한 회전 사각형.

**포맷:**
```json
{
  "bbox": [cx, cy, width, height, angle],
  "category": "diagnosis_code",
  "text": "K35.9"
}
```

**적합한 문서:** 손글씨 영역이 포함된 진단서, 보험 청구서

### 3. Polygon (폴리곤)

비정형 영역을 다각형 꼭짓점으로 표현.

**포맷:**
```json
{
  "segmentation": [[x1,y1, x2,y2, x3,y3, x4,y4, ...]],
  "category": "stamp",
  "text": ""
}
```

**적합한 문서:** 직인, 서명, 불규칙한 손글씨 영역

### 4. Segmentation Mask (세그멘테이션 마스크)

픽셀 단위 이진 마스크. 가장 정밀하지만 어노테이션 비용 최고.

**적합한 경우:** 직인과 텍스트 겹침 영역 분리, 배경 제거

### 5. OCR 어노테이션 (좌표 + 전사)

텍스트 영역 좌표와 해당 텍스트 내용을 동시에 기록. OCR 학습에 필수적인 형태.

**포맷:**
```json
{
  "bbox": [100, 200, 150, 30],
  "text": "급성 충수염",
  "language": "ko",
  "legibility": "clear",
  "category": "disease_name"
}
```

---

## 수학적 원리

### IoU (Intersection over Union)

두 영역 $A$와 $B$의 겹침 정도를 정량화:

$$
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$

- $\text{IoU} = 0$: 전혀 겹치지 않음
- $\text{IoU} = 1$: 완전히 일치
- 일반적으로 $\text{IoU} \geq 0.5$이면 "매칭"으로 판정

바운딩 박스 $(x_1^A, y_1^A, x_2^A, y_2^A)$과 $(x_1^B, y_1^B, x_2^B, y_2^B)$의 IoU 계산:

$$
x_1^{\cap} = \max(x_1^A, x_1^B), \quad y_1^{\cap} = \max(y_1^A, y_1^B)
$$

$$
x_2^{\cap} = \min(x_2^A, x_2^B), \quad y_2^{\cap} = \min(y_2^A, y_2^B)
$$

$$
\text{Area}_{\cap} = \max(0, x_2^{\cap} - x_1^{\cap}) \times \max(0, y_2^{\cap} - y_1^{\cap})
$$

$$
\text{Area}_{\cup} = \text{Area}_A + \text{Area}_B - \text{Area}_{\cap}
$$

$$
\text{IoU} = \frac{\text{Area}_{\cap}}{\text{Area}_{\cup}}
$$

### NMS (Non-Maximum Suppression)

IoU 기반 중복 검출 제거 알고리즘. 신뢰도 순으로 정렬한 뒤, 높은 IoU를 가진 중복 박스를 제거한다.

**알고리즘:**

1. 모든 박스를 신뢰도 기준 내림차순 정렬: $B = \{b_1, b_2, \ldots, b_n\}$, $s(b_1) \geq s(b_2) \geq \cdots$
2. 결과 집합 $D = \emptyset$
3. $B$가 비어있지 않은 동안:
   - $b^* = B[0]$ (최고 신뢰도 박스) → $D$에 추가
   - $B$에서 $b^*$ 제거
   - $B$에서 $\text{IoU}(b^*, b_i) \geq \tau$ 인 모든 $b_i$ 제거
4. $D$ 반환

여기서 $\tau$는 IoU 임계값 (보통 0.5).

### Cohen's Kappa (어노테이터 간 일치도)

두 어노테이터 간 일치도를 우연 일치를 보정하여 측정:

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

여기서:
- $p_o$: 관측된 일치 비율 (observed agreement)
- $p_e$: 우연에 의한 기대 일치 비율 (expected agreement)

$$
p_e = \sum_{k=1}^{K} p_{1k} \cdot p_{2k}
$$

- $p_{1k}$: 어노테이터 1이 카테고리 $k$를 선택한 비율
- $p_{2k}$: 어노테이터 2가 카테고리 $k$를 선택한 비율

**해석 기준:**

| $\kappa$ 범위 | 일치 수준 |
|----------------|-----------|
| $< 0$ | 우연보다 낮음 (disagreement) |
| $0.00 - 0.20$ | 약한 일치 |
| $0.21 - 0.40$ | 보통 일치 |
| $0.41 - 0.60$ | 중간 일치 |
| $0.61 - 0.80$ | 상당한 일치 |
| $0.81 - 1.00$ | 거의 완벽한 일치 |

의료 문서 어노테이션에서 $\kappa \geq 0.75$를 최소 기준으로 권장한다.

### Fleiss' Kappa (다수 어노테이터)

어노테이터가 3명 이상일 때 사용:

$$
\kappa_F = \frac{\bar{P} - \bar{P}_e}{1 - \bar{P}_e}
$$

$$
\bar{P} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{n(n-1)} \left( \sum_{j=1}^{K} n_{ij}^2 - n \right)
$$

$$
\bar{P}_e = \sum_{j=1}^{K} p_j^2, \quad p_j = \frac{1}{Nn} \sum_{i=1}^{N} n_{ij}
$$

---

## 어노테이션 도구 비교

### Label Studio

| 항목 | 내용 |
|------|------|
| 유형 | 오픈소스 (Apache 2.0) |
| 설치 | `pip install label-studio` |
| 지원 포맷 | 바운딩 박스, 폴리곤, OCR 전사, NER |
| 커스터마이징 | XML 기반 라벨링 인터페이스 정의 |
| 팀 협업 | 다중 어노테이터 지원, 리뷰 워크플로 |
| API | REST API 제공 (자동화 가능) |
| ML 통합 | ML Backend 연동 (pre-annotation) |

**의료 문서 적합도:** 높음 — 커스텀 라벨링 인터페이스로 도메인 특화 구성 가능

### CVAT (Computer Vision Annotation Tool)

| 항목 | 내용 |
|------|------|
| 유형 | 오픈소스 (MIT) |
| 설치 | Docker Compose |
| 지원 포맷 | 바운딩 박스, 폴리곤, 세그멘테이션, 키포인트 |
| 커스터마이징 | 라벨 속성 정의 가능 |
| 팀 협업 | 조직/프로젝트 기반 관리 |
| API | REST API, Python SDK |
| ML 통합 | Nuclio 기반 자동 어노테이션 |

**의료 문서 적합도:** 중상 — 이미지 중심, OCR 전사 기능은 제한적

### Prodigy

| 항목 | 내용 |
|------|------|
| 유형 | 상용 (라이선스 구매) |
| 설치 | `pip install prodigy` |
| 지원 포맷 | NER, 텍스트 분류, 이미지 분류 |
| 커스터마이징 | Python 기반 레시피(recipe) 시스템 |
| 팀 협업 | 제한적 (개인 도구 특화) |
| API | Python API |
| ML 통합 | spaCy 통합, Active Learning |

**의료 문서 적합도:** 중 — NER/텍스트 분류에 강점, 이미지 어노테이션은 약함

### 도구 선정 권장

| 시나리오 | 추천 도구 |
|----------|-----------|
| 텍스트 영역 + OCR 전사 | Label Studio |
| 정밀 세그멘테이션 | CVAT |
| 텍스트 NER (추출 필드 태깅) | Prodigy 또는 Label Studio |
| 대규모 팀 협업 | Label Studio Enterprise 또는 CVAT |

---

## 의료 문서 특화 어노테이션 가이드라인

### 라벨 체계 (Label Schema)

의료 문서 OCR에 사용할 라벨을 계층적으로 정의:

```
document_type/
├── patient_info/
│   ├── patient_name          # 환자명
│   ├── resident_number       # 주민등록번호
│   ├── address              # 주소
│   └── phone                # 전화번호
├── diagnosis_info/
│   ├── disease_name          # 상병명
│   ├── diagnosis_code        # 진단코드 (KCD)
│   ├── onset_date           # 발병일
│   └── diagnosis_date       # 진단일
├── treatment_info/
│   ├── surgery_name          # 수술명
│   ├── surgery_code         # 수술 코드
│   ├── surgery_date         # 수술일
│   └── anesthesia_type      # 마취 방법
├── hospitalization_info/
│   ├── admission_date       # 입원일
│   ├── discharge_date       # 퇴원일
│   └── hospital_days        # 입원 일수
├── billing_info/
│   ├── item_code            # 항목 코드 (EDI)
│   ├── item_name            # 항목명
│   ├── unit_price           # 단가
│   ├── quantity             # 횟수
│   ├── amount               # 금액
│   ├── total_bill           # 총 진료비
│   └── patient_payment      # 본인부담금
├── institution_info/
│   ├── hospital_name        # 의료기관명
│   ├── doctor_name          # 의사명
│   └── license_number       # 면허번호
└── non_text/
    ├── stamp                # 직인
    ├── signature            # 서명
    └── checkbox             # 체크박스
```

### 어노테이션 규칙

**규칙 1: 경계 설정**
- 바운딩 박스는 텍스트에 밀착 (여백 2px 이내)
- 직인과 겹치는 텍스트는 텍스트 영역과 직인 영역을 각각 어노테이션

**규칙 2: 전사 원칙**
- OCR 전사는 보이는 그대로 기록 (오타, 약어 포함)
- 판독 불가한 글자는 `[illegible]`로 표기
- 부분 판독은 `홍[illegible]동`처럼 표기

**규칙 3: 라벨 우선순위**
- 하나의 텍스트 영역이 여러 라벨에 해당하면 가장 구체적인 라벨 적용
- 예: "K35.9" → `diagnosis_code` (O), `text` (X)

**규칙 4: 테이블 처리**
- 테이블의 각 셀을 개별 바운딩 박스로 어노테이션
- 헤더 행은 `table_header` 라벨 추가
- 셀 병합 시 병합된 영역을 하나의 바운딩 박스로 처리

**규칙 5: 손글씨 처리**
- 손글씨 영역에 `handwritten: true` 속성 추가
- 필기체 판독이 어려우면 `legibility: "poor"` 속성 설정

---

## 코드: COCO Format 변환 및 어노테이션 품질 검증

```python
"""
의료 문서 어노테이션 유틸리티
- COCO Format 변환
- IoU 계산
- NMS 구현
- 어노테이터 간 일치도(Cohen's Kappa) 계산
- 어노테이션 품질 검증 스크립트
"""

import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 1. IoU 및 NMS 구현
# ============================================================

def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """
    두 바운딩 박스의 IoU 계산.

    Args:
        box_a: [x_min, y_min, x_max, y_max]
        box_b: [x_min, y_min, x_max, y_max]

    Returns:
        IoU 값 (0.0 ~ 1.0)
    """
    # 교집합 좌표
    x1_inter = max(box_a[0], box_b[0])
    y1_inter = max(box_a[1], box_b[1])
    x2_inter = min(box_a[2], box_b[2])
    y2_inter = min(box_a[3], box_b[3])

    # 교집합 면적
    inter_width = max(0.0, x2_inter - x1_inter)
    inter_height = max(0.0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # 각 박스 면적
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # 합집합 면적
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    두 박스 배열 간 IoU 행렬 계산 (벡터화).

    Args:
        boxes_a: shape (N, 4) — [x_min, y_min, x_max, y_max]
        boxes_b: shape (M, 4)

    Returns:
        shape (N, M) IoU 행렬
    """
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]

    # 교집합
    x1 = np.maximum(boxes_a[:, 0].reshape(N, 1), boxes_b[:, 0].reshape(1, M))
    y1 = np.maximum(boxes_a[:, 1].reshape(N, 1), boxes_b[:, 1].reshape(1, M))
    x2 = np.minimum(boxes_a[:, 2].reshape(N, 1), boxes_b[:, 2].reshape(1, M))
    y2 = np.minimum(boxes_a[:, 3].reshape(N, 1), boxes_b[:, 3].reshape(1, M))

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_a = (
        (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    ).reshape(N, 1)
    area_b = (
        (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    ).reshape(1, M)

    union = area_a + area_b - inter
    iou_matrix = np.where(union > 0, inter / union, 0.0)

    return iou_matrix


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> list[int]:
    """
    Non-Maximum Suppression.

    Args:
        boxes: shape (N, 4) — [x_min, y_min, x_max, y_max]
        scores: shape (N,) — 각 박스의 신뢰도
        iou_threshold: 중복 판정 IoU 임계값

    Returns:
        유지할 박스 인덱스 리스트
    """
    if len(boxes) == 0:
        return []

    # 신뢰도 내림차순 정렬
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        # 최고 신뢰도 박스 선택
        idx = order[0]
        keep.append(int(idx))

        if len(order) == 1:
            break

        # 나머지 박스와의 IoU 계산
        remaining = order[1:]
        ious = np.array([
            compute_iou(boxes[idx].tolist(), boxes[r].tolist())
            for r in remaining
        ])

        # IoU < threshold인 박스만 유지
        mask = ious < iou_threshold
        order = remaining[mask]

    return keep


# ============================================================
# 2. Cohen's Kappa 계산
# ============================================================

def cohens_kappa(
    annotator_1: list[str],
    annotator_2: list[str],
) -> float:
    """
    두 어노테이터의 라벨 일치도를 Cohen's Kappa로 계산.

    Args:
        annotator_1: 어노테이터 1의 라벨 리스트
        annotator_2: 어노테이터 2의 라벨 리스트

    Returns:
        kappa 값 (-1.0 ~ 1.0)
    """
    assert len(annotator_1) == len(annotator_2), "라벨 수가 일치해야 함"

    n = len(annotator_1)
    if n == 0:
        return 0.0

    # 카테고리 집합
    categories = sorted(set(annotator_1) | set(annotator_2))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    # 혼동 행렬 (confusion matrix)
    confusion = np.zeros((k, k), dtype=int)
    for a1, a2 in zip(annotator_1, annotator_2):
        confusion[cat_to_idx[a1]][cat_to_idx[a2]] += 1

    # 관측 일치율
    p_o = confusion.diagonal().sum() / n

    # 기대 일치율
    row_marginals = confusion.sum(axis=1) / n
    col_marginals = confusion.sum(axis=0) / n
    p_e = (row_marginals * col_marginals).sum()

    if p_e == 1.0:
        return 1.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return kappa


def fleiss_kappa(rating_matrix: np.ndarray) -> float:
    """
    Fleiss' Kappa 계산 (3명 이상 어노테이터).

    Args:
        rating_matrix: shape (N, K) — N개 항목, K개 카테고리.
                       rating_matrix[i][j] = 항목 i에 카테고리 j를 부여한 어노테이터 수

    Returns:
        Fleiss' kappa 값
    """
    N, K = rating_matrix.shape
    n = rating_matrix[0].sum()  # 각 항목을 평가한 어노테이터 수

    # 각 항목의 일치도
    P_i = (np.sum(rating_matrix ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()

    # 기대 일치율
    p_j = rating_matrix.sum(axis=0) / (N * n)
    P_e = (p_j ** 2).sum()

    if P_e == 1.0:
        return 1.0

    return (P_bar - P_e) / (1.0 - P_e)


# ============================================================
# 3. COCO Format 변환
# ============================================================

@dataclass
class COCOAnnotation:
    """COCO 포맷 단일 어노테이션"""
    id: int
    image_id: int
    category_id: int
    bbox: list[float]          # [x, y, width, height]
    area: float
    segmentation: list = field(default_factory=list)
    iscrowd: int = 0
    attributes: dict = field(default_factory=dict)  # 확장 속성 (text, legibility 등)


@dataclass
class COCODataset:
    """COCO 포맷 데이터셋"""
    images: list[dict] = field(default_factory=list)
    annotations: list[dict] = field(default_factory=list)
    categories: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "info": {
                "description": "Medical Document OCR Dataset",
                "version": "1.0",
                "year": 2024,
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [],
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("COCO 데이터셋 저장 완료: %s", path)


# 의료 문서 카테고리 정의
MEDICAL_CATEGORIES = [
    {"id": 1, "name": "patient_name", "supercategory": "patient_info"},
    {"id": 2, "name": "resident_number", "supercategory": "patient_info"},
    {"id": 3, "name": "disease_name", "supercategory": "diagnosis_info"},
    {"id": 4, "name": "diagnosis_code", "supercategory": "diagnosis_info"},
    {"id": 5, "name": "onset_date", "supercategory": "diagnosis_info"},
    {"id": 6, "name": "surgery_name", "supercategory": "treatment_info"},
    {"id": 7, "name": "surgery_code", "supercategory": "treatment_info"},
    {"id": 8, "name": "admission_date", "supercategory": "hospitalization_info"},
    {"id": 9, "name": "discharge_date", "supercategory": "hospitalization_info"},
    {"id": 10, "name": "item_code", "supercategory": "billing_info"},
    {"id": 11, "name": "item_name", "supercategory": "billing_info"},
    {"id": 12, "name": "amount", "supercategory": "billing_info"},
    {"id": 13, "name": "total_bill", "supercategory": "billing_info"},
    {"id": 14, "name": "hospital_name", "supercategory": "institution_info"},
    {"id": 15, "name": "doctor_name", "supercategory": "institution_info"},
    {"id": 16, "name": "stamp", "supercategory": "non_text"},
    {"id": 17, "name": "signature", "supercategory": "non_text"},
]


def convert_labelstudio_to_coco(
    ls_export: list[dict],
    image_dir: str,
    categories: Optional[list[dict]] = None,
) -> COCODataset:
    """
    Label Studio JSON export → COCO format 변환.

    Args:
        ls_export: Label Studio에서 export한 JSON 리스트
        image_dir: 이미지 디렉토리 경로
        categories: 카테고리 정의 (없으면 MEDICAL_CATEGORIES 사용)

    Returns:
        COCODataset 인스턴스
    """
    if categories is None:
        categories = MEDICAL_CATEGORIES

    cat_name_to_id = {c["name"]: c["id"] for c in categories}

    dataset = COCODataset(categories=categories)
    ann_id = 1

    for img_idx, task in enumerate(ls_export):
        image_id = img_idx + 1

        # 이미지 정보
        file_name = task.get("file_upload", f"image_{image_id}.png")
        img_width = task.get("meta", {}).get("width", 0)
        img_height = task.get("meta", {}).get("height", 0)

        dataset.images.append({
            "id": image_id,
            "file_name": file_name,
            "width": img_width,
            "height": img_height,
        })

        # 어노테이션 변환
        for annotation in task.get("annotations", []):
            for result in annotation.get("result", []):
                if result["type"] != "rectanglelabels":
                    continue

                value = result["value"]
                label = value["rectanglelabels"][0]

                if label not in cat_name_to_id:
                    logger.warning("알 수 없는 라벨: %s (건너뜀)", label)
                    continue

                # Label Studio 좌표 (%, 상대값) → COCO 좌표 (px, 절대값)
                x_pct = value["x"]
                y_pct = value["y"]
                w_pct = value["width"]
                h_pct = value["height"]

                x = x_pct / 100.0 * img_width
                y = y_pct / 100.0 * img_height
                w = w_pct / 100.0 * img_width
                h = h_pct / 100.0 * img_height

                ann_dict = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_name_to_id[label],
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "segmentation": [],
                    "iscrowd": 0,
                }

                # 텍스트 전사가 있으면 attributes에 추가
                text = value.get("text", "")
                if text:
                    ann_dict["attributes"] = {"text": text}

                dataset.annotations.append(ann_dict)
                ann_id += 1

    logger.info(
        "변환 완료: 이미지 %d장, 어노테이션 %d개",
        len(dataset.images),
        len(dataset.annotations),
    )
    return dataset


# ============================================================
# 4. 어노테이션 품질 검증
# ============================================================

@dataclass
class QualityReport:
    """어노테이션 품질 보고서"""
    total_images: int = 0
    total_annotations: int = 0
    avg_annotations_per_image: float = 0.0
    missing_required_fields: list = field(default_factory=list)
    overlapping_boxes: list = field(default_factory=list)
    out_of_bound_boxes: list = field(default_factory=list)
    empty_text_annotations: list = field(default_factory=list)
    kappa_score: Optional[float] = None
    category_distribution: dict = field(default_factory=dict)
    passed: bool = False


class AnnotationQualityChecker:
    """어노테이션 품질 검증기"""

    # 필수 카테고리 (이미지당 최소 1개 이상 존재해야 하는 라벨)
    REQUIRED_CATEGORIES = {"patient_name", "disease_name", "hospital_name"}

    # 최대 허용 겹침 IoU
    MAX_OVERLAP_IOU = 0.8

    def __init__(self, coco_path: str):
        with open(coco_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.images = {img["id"]: img for img in self.data["images"]}
        self.categories = {
            cat["id"]: cat["name"] for cat in self.data["categories"]
        }

        # 이미지별 어노테이션 그룹핑
        self.img_annotations: dict[int, list] = {}
        for ann in self.data["annotations"]:
            img_id = ann["image_id"]
            self.img_annotations.setdefault(img_id, []).append(ann)

    def check_all(self) -> QualityReport:
        """전체 품질 검증 실행"""
        report = QualityReport()
        report.total_images = len(self.images)
        report.total_annotations = len(self.data["annotations"])

        if report.total_images > 0:
            report.avg_annotations_per_image = (
                report.total_annotations / report.total_images
            )

        # 1. 필수 필드 존재 여부 검사
        report.missing_required_fields = self._check_required_fields()

        # 2. 바운딩 박스 경계 초과 검사
        report.out_of_bound_boxes = self._check_bounds()

        # 3. 과도한 겹침 검사
        report.overlapping_boxes = self._check_overlaps()

        # 4. 빈 텍스트 어노테이션 검사
        report.empty_text_annotations = self._check_empty_text()

        # 5. 카테고리 분포
        report.category_distribution = self._get_category_distribution()

        # 합격 판정
        errors = (
            len(report.missing_required_fields)
            + len(report.out_of_bound_boxes)
            + len(report.overlapping_boxes)
        )
        report.passed = errors == 0

        return report

    def _check_required_fields(self) -> list[dict]:
        """이미지별 필수 카테고리 존재 여부 검사"""
        issues = []
        req_cat_ids = {
            cat_id
            for cat_id, name in self.categories.items()
            if name in self.REQUIRED_CATEGORIES
        }

        for img_id in self.images:
            anns = self.img_annotations.get(img_id, [])
            present_cats = {ann["category_id"] for ann in anns}
            missing = req_cat_ids - present_cats

            if missing:
                missing_names = [self.categories[cid] for cid in missing]
                issues.append({
                    "image_id": img_id,
                    "missing": missing_names,
                })

        return issues

    def _check_bounds(self) -> list[dict]:
        """바운딩 박스가 이미지 경계를 초과하는지 검사"""
        issues = []
        for ann in self.data["annotations"]:
            img = self.images.get(ann["image_id"])
            if img is None:
                continue

            x, y, w, h = ann["bbox"]
            img_w = img.get("width", float("inf"))
            img_h = img.get("height", float("inf"))

            if x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
                issues.append({
                    "annotation_id": ann["id"],
                    "image_id": ann["image_id"],
                    "bbox": ann["bbox"],
                    "image_size": [img_w, img_h],
                })

        return issues

    def _check_overlaps(self) -> list[dict]:
        """같은 이미지 내 과도한 바운딩 박스 겹침 검사"""
        issues = []

        for img_id, anns in self.img_annotations.items():
            if len(anns) < 2:
                continue

            # [x_min, y_min, x_max, y_max] 변환
            boxes = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])

            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    iou = compute_iou(boxes[i], boxes[j])
                    if iou > self.MAX_OVERLAP_IOU:
                        issues.append({
                            "image_id": img_id,
                            "ann_id_1": anns[i]["id"],
                            "ann_id_2": anns[j]["id"],
                            "iou": round(iou, 4),
                        })

        return issues

    def _check_empty_text(self) -> list[dict]:
        """텍스트 속성이 비어 있는 어노테이션 검사"""
        issues = []
        text_categories = {
            cat_id
            for cat_id, name in self.categories.items()
            if name not in {"stamp", "signature", "checkbox"}
        }

        for ann in self.data["annotations"]:
            if ann["category_id"] not in text_categories:
                continue
            attrs = ann.get("attributes", {})
            text = attrs.get("text", "")
            if not text.strip():
                issues.append({
                    "annotation_id": ann["id"],
                    "image_id": ann["image_id"],
                    "category": self.categories.get(ann["category_id"]),
                })

        return issues

    def _get_category_distribution(self) -> dict:
        """카테고리별 어노테이션 수 분포"""
        dist = {}
        for ann in self.data["annotations"]:
            cat_name = self.categories.get(ann["category_id"], "unknown")
            dist[cat_name] = dist.get(cat_name, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))

    def print_report(self, report: QualityReport) -> None:
        """보고서 출력"""
        print("=" * 60)
        print("어노테이션 품질 보고서")
        print("=" * 60)
        print(f"전체 이미지: {report.total_images}")
        print(f"전체 어노테이션: {report.total_annotations}")
        print(f"이미지당 평균 어노테이션: {report.avg_annotations_per_image:.1f}")
        print()

        print("[필수 필드 누락]")
        if report.missing_required_fields:
            for item in report.missing_required_fields[:10]:
                print(f"  이미지 {item['image_id']}: {item['missing']}")
        else:
            print("  이상 없음")

        print()
        print("[경계 초과 박스]")
        if report.out_of_bound_boxes:
            for item in report.out_of_bound_boxes[:10]:
                print(f"  어노테이션 {item['annotation_id']}: bbox={item['bbox']}")
        else:
            print("  이상 없음")

        print()
        print("[과도한 겹침]")
        if report.overlapping_boxes:
            for item in report.overlapping_boxes[:10]:
                print(
                    f"  이미지 {item['image_id']}: "
                    f"ann {item['ann_id_1']} <-> {item['ann_id_2']} "
                    f"(IoU={item['iou']})"
                )
        else:
            print("  이상 없음")

        print()
        print("[카테고리 분포]")
        for cat, count in report.category_distribution.items():
            print(f"  {cat}: {count}")

        print()
        status = "PASS" if report.passed else "FAIL"
        print(f"판정: {status}")
        print("=" * 60)


# ============================================================
# 5. 사용 예시
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- IoU 테스트 ---
    box1 = [100, 100, 200, 200]  # x_min, y_min, x_max, y_max
    box2 = [150, 150, 250, 250]
    print(f"IoU: {compute_iou(box1, box2):.4f}")  # 약 0.1429

    # --- NMS 테스트 ---
    boxes = np.array([
        [100, 100, 200, 200],
        [110, 110, 210, 210],
        [300, 300, 400, 400],
    ], dtype=float)
    scores = np.array([0.9, 0.75, 0.8])
    kept = nms(boxes, scores, iou_threshold=0.5)
    print(f"NMS 결과 (유지 인덱스): {kept}")

    # --- Cohen's Kappa 테스트 ---
    ann1 = ["A", "A", "B", "B", "A", "B", "A", "A", "B", "B"]
    ann2 = ["A", "A", "B", "A", "A", "B", "A", "B", "B", "B"]
    kappa = cohens_kappa(ann1, ann2)
    print(f"Cohen's Kappa: {kappa:.4f}")

    # --- COCO 변환 테스트 ---
    sample_ls_export = [
        {
            "file_upload": "diagnosis_001.png",
            "meta": {"width": 2480, "height": 3508},
            "annotations": [
                {
                    "result": [
                        {
                            "type": "rectanglelabels",
                            "value": {
                                "x": 10.0, "y": 5.0,
                                "width": 20.0, "height": 3.0,
                                "rectanglelabels": ["patient_name"],
                                "text": "홍길동",
                            },
                        },
                        {
                            "type": "rectanglelabels",
                            "value": {
                                "x": 40.0, "y": 20.0,
                                "width": 30.0, "height": 4.0,
                                "rectanglelabels": ["disease_name"],
                                "text": "급성 충수염",
                            },
                        },
                    ]
                }
            ],
        }
    ]

    dataset = convert_labelstudio_to_coco(sample_ls_export, "/data/images")
    print(json.dumps(dataset.to_dict(), ensure_ascii=False, indent=2)[:500])
```

---

## 어노테이션 워크플로 요약

```
1. 가이드라인 문서 작성 (라벨 체계 + 규칙)
     ↓
2. 파일럿 어노테이션 (10~20장, 전체 어노테이터 참여)
     ↓
3. IAA(Inter-Annotator Agreement) 측정 → κ ≥ 0.75 미달 시 가이드라인 보완
     ↓
4. 본 어노테이션 (Label Studio 사용)
     ↓
5. 자동 품질 검증 (AnnotationQualityChecker)
     ↓
6. 리뷰/수정 사이클
     ↓
7. COCO format export → 모델 학습 파이프라인 연동
```

---

## 용어 체크리스트

학습 후 아래 용어를 설명할 수 있는지 자가 점검해라.

- [ ] **Bounding Box** — 좌표 표현 방식 (x, y, w, h)과 한계를 설명할 수 있다
- [ ] **Polygon** — 바운딩 박스 대비 장점과 적합한 사용 상황을 설명할 수 있다
- [ ] **Segmentation Mask** — 픽셀 단위 마스크의 장단점을 설명할 수 있다
- [ ] **IoU** — 수식을 유도하고, 바운딩 박스 간 IoU를 직접 계산할 수 있다
- [ ] **NMS** — 알고리즘 동작 과정을 단계별로 설명할 수 있다
- [ ] **Cohen's Kappa** — $p_o$와 $p_e$의 의미를 설명하고, 값을 직접 계산할 수 있다
- [ ] **Fleiss' Kappa** — Cohen's Kappa와의 차이를 설명할 수 있다
- [ ] **COCO Format** — JSON 구조(images, annotations, categories)를 이해한다
- [ ] **Label Studio** — 의료 문서 어노테이션에 적합한 이유를 설명할 수 있다
- [ ] **어노테이션 가이드라인** — 의료 문서 특화 규칙 5가지를 열거할 수 있다
