# 8.1 템플릿 기반 생성

의료 OCR 모델 학습에는 대량의 레이블된 문서 이미지가 필요하다. 하지만 실제 의료 문서는 개인정보 보호법(HIPAA, 개인정보보호법)으로 대규모 수집이 어렵다. 템플릿 기반 합성 데이터 생성은 실제 문서의 레이아웃과 폰트 특성을 모사하여 무한에 가까운 학습 데이터를 프로그래밍적으로 만들어내는 기법이다.

---

> **핵심 용어 박스**
>
> | 용어 | 정의 |
> |------|------|
> | **Template** | 문서의 고정 레이아웃 구조. 텍스트 필드, 라인, 박스 등의 위치와 크기를 정의한 청사진 |
> | **Rendering** | 템플릿 위에 실제 텍스트와 그래픽 요소를 그려 이미지로 변환하는 과정 |
> | **Font Variation** | 폰트 종류, 크기, 굵기, 기울기 등을 랜덤하게 변화시켜 다양성을 확보하는 기법 |
> | **Layout Randomization** | 텍스트 위치, 간격, 정렬 등을 확률적으로 변형하여 과적합을 방지하는 전략 |

---

## 8.1.1 왜 템플릿 기반 생성인가?

### 실제 데이터 수집의 한계

| 문제 | 설명 | 템플릿 생성의 해결 |
|------|------|-------------------|
| 개인정보 | 환자명, 주민번호 등 민감 정보 포함 | 가상 데이터로 대체 |
| 수량 부족 | 병원별 양식 차이, 수집 비용 | 무제한 생성 가능 |
| 레이블링 비용 | 전문가 수작업 어노테이션 필요 | 생성 시점에 자동 레이블 |
| 클래스 불균형 | 희귀 양식/질환 코드 부족 | 원하는 분포로 생성 |
| 법적 리스크 | 의료법, 개인정보보호법 | 합성 데이터는 규제 대상 아님 |

### 합성 데이터 파이프라인 개요

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│  템플릿 정의  │ →  │  텍스트 생성   │ →  │  렌더링 엔진  │ →  │  노이즈 추가  │
│  (JSON/YAML) │    │  (Faker 등)   │    │  (Pillow)    │    │  (8.2 참고)  │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
       │                  │                   │                    │
       ▼                  ▼                   ▼                    ▼
  레이아웃 좌표       필드별 텍스트        문서 이미지         최종 학습 이미지
  + 필드 타입         + 메타데이터        + 바운딩 박스        + GT 레이블
```

---

## 수학적 원리

### 아핀 변환 (Affine Transformation)

텍스트 렌더링 시 위치, 회전, 크기 조절은 아핀 변환으로 통합 표현된다.

2D 점 $(x, y)$에 대한 아핀 변환:

$$
\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}
$$

동차 좌표(homogeneous coordinates)를 사용하면 단일 행렬 곱으로 표현 가능하다:

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

각 변환별 행렬:

| 변환 | 행렬 $M$ | 파라미터 |
|------|---------|---------|
| 이동(Translation) | $\begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}$ | $t_x, t_y$: 이동량 |
| 회전(Rotation) | $\begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$ | $\theta$: 회전 각도 |
| 스케일(Scale) | $\begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{bmatrix}$ | $s_x, s_y$: 축별 배율 |
| 전단(Shear) | $\begin{bmatrix} 1 & k_x & 0 \\ k_y & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$ | $k_x, k_y$: 전단 계수 |

복합 변환은 행렬 곱의 연쇄로 구현한다:

$$
M_{\text{total}} = M_{\text{translate}} \cdot M_{\text{rotate}} \cdot M_{\text{scale}} \cdot M_{\text{shear}}
$$

### 폰트 렌더링 파라미터 샘플링

폰트의 다양성을 확보하기 위해 각 파라미터를 확률 분포에서 샘플링한다.

**폰트 크기** $s$:

$$
s \sim \mathcal{U}(s_{\min}, s_{\max})
$$

의료 문서의 경우 본문은 $s \in [10, 14]$pt, 제목은 $s \in [16, 24]$pt 범위가 일반적이다.

**기울기(slant) 각도** $\alpha$:

$$
\alpha \sim \mathcal{N}(0, \sigma_\alpha^2), \quad \sigma_\alpha \approx 2°
$$

실제 프린터 출력물의 미세한 기울기를 모사한다.

**텍스트 간격(spacing)** $\delta$:

$$
\delta = \delta_{\text{base}} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_\delta^2)
$$

**렌더링 품질 함수** — 생성된 이미지 $I_{\text{syn}}$와 실제 이미지 $I_{\text{real}}$의 유사도:

$$
\mathcal{Q}(I_{\text{syn}}, I_{\text{real}}) = 1 - \frac{\|F(I_{\text{syn}}) - F(I_{\text{real}})\|_2}{\|F(I_{\text{real}})\|_2}
$$

여기서 $F(\cdot)$는 사전학습된 CNN의 특징 추출 함수다. $\mathcal{Q}$가 높을수록 합성 데이터가 실제와 유사하다.

---

## 8.1.2 의료 문서 템플릿 설계

### 템플릿 구조 정의

의료 문서 템플릿은 JSON으로 정의한다. 각 필드는 위치, 크기, 데이터 타입, 폰트 속성을 포함한다.

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class FieldType(Enum):
    """문서 필드의 데이터 타입."""
    TEXT = "text"
    DATE = "date"
    NUMBER = "number"
    CHECKBOX = "checkbox"
    SIGNATURE = "signature"
    BARCODE = "barcode"


class FontStyle(Enum):
    """폰트 스타일."""
    REGULAR = "regular"
    BOLD = "bold"
    ITALIC = "italic"
    BOLD_ITALIC = "bold_italic"


@dataclass
class BoundingBox:
    """필드의 위치와 크기를 정의하는 바운딩 박스."""
    x: int
    y: int
    width: int
    height: int

    def to_xyxy(self) -> tuple:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def to_xywh(self) -> tuple:
        return (self.x, self.y, self.width, self.height)


@dataclass
class FontConfig:
    """폰트 설정."""
    family: str = "NanumGothic"
    size: int = 12
    style: FontStyle = FontStyle.REGULAR
    color: tuple = (0, 0, 0)

    # 랜덤 변형 범위
    size_variation: float = 0.1      # ±10%
    color_variation: int = 20        # ±20 (0~255 범위)
    spacing_variation: float = 0.05  # ±5%


@dataclass
class TemplateField:
    """템플릿의 개별 필드."""
    name: str
    field_type: FieldType
    bbox: BoundingBox
    font: FontConfig
    required: bool = True
    max_length: Optional[int] = None
    label: Optional[str] = None          # 필드 앞에 출력할 라벨 텍스트
    label_font: Optional[FontConfig] = None


@dataclass
class DocumentTemplate:
    """의료 문서 템플릿."""
    name: str
    width: int
    height: int
    dpi: int = 300
    background_color: tuple = (255, 255, 255)
    fields: list = field(default_factory=list)
    static_elements: list = field(default_factory=list)  # 로고, 고정 텍스트 등
    lines: list = field(default_factory=list)             # 구분선
```

### 의료 문서 양식 템플릿 정의

```python
import json


def create_diagnosis_template() -> DocumentTemplate:
    """진단서 템플릿을 생성한다.

    표준 의료 진단서 양식:
    - 상단: 의료기관명, 주소, 전화번호
    - 환자 정보: 성명, 주민등록번호, 주소
    - 진단 내용: 병명, 질병코드, 진단 소견
    - 하단: 발급일, 의사명, 직인
    """
    template = DocumentTemplate(
        name="진단서",
        width=2480,   # A4 at 300dpi
        height=3508,
        dpi=300,
    )

    title_font = FontConfig(family="NanumGothicBold", size=28, style=FontStyle.BOLD)
    label_font = FontConfig(family="NanumGothic", size=14, style=FontStyle.BOLD)
    value_font = FontConfig(family="NanumGothic", size=14)
    small_font = FontConfig(family="NanumGothic", size=11)

    # 제목
    template.fields.append(TemplateField(
        name="title",
        field_type=FieldType.TEXT,
        bbox=BoundingBox(x=840, y=200, width=800, height=60),
        font=title_font,
        label="진 단 서",
    ))

    # 환자 정보 영역
    patient_fields = [
        ("patient_name", "성    명", 400, 450, 300, 40),
        ("patient_id", "주민등록번호", 800, 450, 400, 40),
        ("patient_addr", "주    소", 400, 510, 800, 40),
        ("patient_phone", "전 화 번 호", 400, 570, 400, 40),
    ]

    for name, label, x, y, w, h in patient_fields:
        template.fields.append(TemplateField(
            name=name,
            field_type=FieldType.TEXT,
            bbox=BoundingBox(x=x, y=y, width=w, height=h),
            font=value_font,
            label=label,
            label_font=label_font,
        ))

    # 진단 내용 영역
    diagnosis_fields = [
        ("disease_name", "병        명", 400, 750, 800, 40),
        ("disease_code", "질 병 코 드", 400, 810, 300, 40),
        ("onset_date", "발 병 연 월 일", 400, 870, 300, 40),
        ("diagnosis_detail", "진 단 소 견", 400, 950, 1600, 400),
    ]

    for name, label, x, y, w, h in diagnosis_fields:
        ftype = FieldType.DATE if "date" in name else FieldType.TEXT
        template.fields.append(TemplateField(
            name=name,
            field_type=ftype,
            bbox=BoundingBox(x=x, y=y, width=w, height=h),
            font=value_font,
            label=label,
            label_font=label_font,
        ))

    # 하단 발급 정보
    footer_fields = [
        ("issue_date", "발 급 일", 400, 2800, 400, 40),
        ("hospital_name", "의료기관명", 400, 2880, 600, 40),
        ("doctor_name", "의 사 명", 400, 2960, 300, 40),
    ]

    for name, label, x, y, w, h in footer_fields:
        template.fields.append(TemplateField(
            name=name,
            field_type=FieldType.TEXT,
            bbox=BoundingBox(x=x, y=y, width=w, height=h),
            font=value_font,
            label=label,
            label_font=label_font,
        ))

    # 구분선 정의
    template.lines = [
        {"start": (200, 400), "end": (2280, 400), "width": 2},    # 제목 아래
        {"start": (200, 700), "end": (2280, 700), "width": 1},    # 환자정보 아래
        {"start": (200, 2750), "end": (2280, 2750), "width": 1},  # 진단내용 아래
    ]

    return template


def create_admission_template() -> DocumentTemplate:
    """입퇴원확인서 템플릿을 생성한다."""
    template = DocumentTemplate(
        name="입퇴원확인서",
        width=2480,
        height=3508,
        dpi=300,
    )

    value_font = FontConfig(family="NanumGothic", size=14)
    label_font = FontConfig(family="NanumGothic", size=14, style=FontStyle.BOLD)

    admission_fields = [
        ("patient_name", "환자성명", 400, 500, 300, 40),
        ("patient_id", "등록번호", 800, 500, 400, 40),
        ("admission_date", "입원일", 400, 600, 300, 40),
        ("discharge_date", "퇴원일", 800, 600, 300, 40),
        ("department", "진료과", 400, 700, 300, 40),
        ("ward", "병동/호실", 800, 700, 300, 40),
        ("diagnosis", "진단명", 400, 800, 800, 40),
        ("doctor_name", "담당의", 400, 900, 300, 40),
    ]

    for name, label, x, y, w, h in admission_fields:
        template.fields.append(TemplateField(
            name=name,
            field_type=FieldType.TEXT,
            bbox=BoundingBox(x=x, y=y, width=w, height=h),
            font=value_font,
            label=label,
            label_font=label_font,
        ))

    return template


def create_billing_template() -> DocumentTemplate:
    """진료비 세부내역서 템플릿을 생성한다."""
    template = DocumentTemplate(
        name="진료비세부내역서",
        width=2480,
        height=3508,
        dpi=300,
    )

    value_font = FontConfig(family="NanumGothic", size=11)
    label_font = FontConfig(family="NanumGothic", size=11, style=FontStyle.BOLD)

    # 상단 환자 정보
    header_fields = [
        ("patient_name", "환자명", 400, 400, 200, 35),
        ("patient_id", "등록번호", 700, 400, 300, 35),
        ("period", "진료기간", 400, 450, 500, 35),
    ]

    for name, label, x, y, w, h in header_fields:
        template.fields.append(TemplateField(
            name=name,
            field_type=FieldType.TEXT,
            bbox=BoundingBox(x=x, y=y, width=w, height=h),
            font=value_font,
            label=label,
            label_font=label_font,
        ))

    # 비용 항목 테이블 (행 반복 구조)
    table_y_start = 600
    row_height = 35
    categories = [
        "진찰료", "입원료", "투약및조제료", "주사료",
        "마취료", "이학요법료", "정신요법료", "처치및수술료",
        "검사료", "영상진단료", "방사선치료료", "혈액투석료",
        "식대", "전액본인부담", "선택진료료",
    ]

    for i, category in enumerate(categories):
        y = table_y_start + i * row_height
        # 항목명
        template.fields.append(TemplateField(
            name=f"item_{i}_name",
            field_type=FieldType.TEXT,
            bbox=BoundingBox(x=200, y=y, width=300, height=row_height),
            font=value_font,
            label=category,
            label_font=label_font,
        ))
        # 급여(본인부담)
        template.fields.append(TemplateField(
            name=f"item_{i}_copay",
            field_type=FieldType.NUMBER,
            bbox=BoundingBox(x=550, y=y, width=200, height=row_height),
            font=value_font,
        ))
        # 급여(공단부담)
        template.fields.append(TemplateField(
            name=f"item_{i}_insurance",
            field_type=FieldType.NUMBER,
            bbox=BoundingBox(x=800, y=y, width=200, height=row_height),
            font=value_font,
        ))
        # 비급여
        template.fields.append(TemplateField(
            name=f"item_{i}_uninsured",
            field_type=FieldType.NUMBER,
            bbox=BoundingBox(x=1050, y=y, width=200, height=row_height),
            font=value_font,
        ))

    return template
```

---

## 8.1.3 의료 데이터 생성기

### 가상 환자 데이터 생성

실제 환자 정보를 사용할 수 없으므로, 통계적으로 그럴듯한 가상 데이터를 생성한다.

```python
import random
import string
from datetime import datetime, timedelta


class MedicalDataGenerator:
    """의료 문서에 삽입할 가상 데이터를 생성한다.

    실제 의료 데이터의 분포를 모사하되, 개인을 특정할 수 없는
    완전한 합성 데이터를 생성한다.
    """

    # 한국 성씨 분포 (통계청 기준 상위 빈도)
    LAST_NAMES = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임",
                  "한", "오", "서", "신", "권", "황", "안", "송", "류", "홍"]
    LAST_NAME_WEIGHTS = [21.5, 14.7, 8.4, 4.7, 4.3, 2.3, 2.1, 2.0, 1.9, 1.7,
                         1.5, 1.5, 1.5, 1.4, 1.3, 1.2, 1.1, 1.1, 1.0, 0.9]

    FIRST_NAMES_M = ["민준", "서준", "도윤", "예준", "시우", "하준", "지호",
                     "주원", "지후", "준서", "건우", "현우", "선우", "우진"]
    FIRST_NAMES_F = ["서연", "서윤", "지우", "하은", "하린", "수아", "지아",
                     "다은", "유진", "민서", "채원", "지윤", "수빈", "예은"]

    HOSPITALS = [
        "서울대학교병원", "세브란스병원", "삼성서울병원", "서울아산병원",
        "고려대학교병원", "가톨릭대학교병원", "경희대학교병원",
        "한양대학교병원", "이화여자대학교부속목동병원", "중앙대학교병원",
    ]

    DEPARTMENTS = [
        "내과", "외과", "정형외과", "신경과", "안과", "이비인후과",
        "피부과", "비뇨의학과", "산부인과", "소아청소년과",
        "영상의학과", "마취통증의학과", "재활의학과", "가정의학과",
    ]

    # KCD 질병코드 (예시)
    DISEASE_CODES = {
        "J06.9": "급성 상기도 감염",
        "M54.5": "요통",
        "K29.7": "상세불명의 위염",
        "I10": "본태성 고혈압",
        "E11.9": "제2형 당뇨병",
        "J18.9": "상세불명의 폐렴",
        "S52.50": "아래팔뼈의 골절",
        "K80.2": "담석증",
        "N39.0": "요로감염",
        "G43.9": "상세불명의 편두통",
    }

    def generate_patient_name(self) -> str:
        last = random.choices(self.LAST_NAMES, weights=self.LAST_NAME_WEIGHTS, k=1)[0]
        gender = random.choice(["M", "F"])
        first = random.choice(self.FIRST_NAMES_M if gender == "M" else self.FIRST_NAMES_F)
        return f"{last}{first}"

    def generate_resident_id(self) -> str:
        """가상 주민등록번호를 생성한다 (형식만 모사, 실제 유효하지 않음)."""
        year = random.randint(40, 99)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        front = f"{year:02d}{month:02d}{day:02d}"
        gender_digit = random.choice([1, 2, 3, 4])
        back = f"{gender_digit}{random.randint(100000, 999999)}"
        return f"{front}-{back}"

    def generate_date(
        self,
        start_date: datetime = datetime(2020, 1, 1),
        end_date: datetime = datetime(2025, 12, 31),
    ) -> str:
        delta = end_date - start_date
        random_days = random.randint(0, delta.days)
        date = start_date + timedelta(days=random_days)
        return date.strftime("%Y년 %m월 %d일")

    def generate_diagnosis(self) -> dict:
        code, name = random.choice(list(self.DISEASE_CODES.items()))
        return {"code": code, "name": name}

    def generate_amount(self, min_val: int = 0, max_val: int = 500000) -> str:
        """의료비 금액을 생성한다. 1000원 단위."""
        amount = random.randint(min_val // 1000, max_val // 1000) * 1000
        return f"{amount:,}"

    def generate_phone(self) -> str:
        area = random.choice(["02", "031", "032", "051", "053", "062"])
        mid = random.randint(1000, 9999)
        last = random.randint(1000, 9999)
        return f"{area}-{mid}-{last}"

    def generate_address(self) -> str:
        cities = ["서울특별시", "부산광역시", "대구광역시", "인천광역시",
                  "광주광역시", "대전광역시", "울산광역시", "세종특별자치시"]
        districts = ["강남구", "서초구", "종로구", "중구", "마포구",
                     "영등포구", "송파구", "강서구", "노원구", "동작구"]
        city = random.choice(cities)
        district = random.choice(districts)
        road_num = random.randint(1, 300)
        return f"{city} {district} 테스트로 {road_num}"

    def fill_diagnosis_template(self) -> dict:
        """진단서 필드를 채울 데이터를 생성한다."""
        diagnosis = self.generate_diagnosis()
        return {
            "patient_name": self.generate_patient_name(),
            "patient_id": self.generate_resident_id(),
            "patient_addr": self.generate_address(),
            "patient_phone": self.generate_phone(),
            "disease_name": diagnosis["name"],
            "disease_code": diagnosis["code"],
            "onset_date": self.generate_date(),
            "diagnosis_detail": f"상기 환자는 {diagnosis['name']}으로 진단되어 "
                                f"치료가 필요한 상태임을 확인합니다.",
            "issue_date": self.generate_date(),
            "hospital_name": random.choice(self.HOSPITALS),
            "doctor_name": self.generate_patient_name(),
        }

    def fill_billing_template(self) -> dict:
        """진료비 세부내역서 필드를 채울 데이터를 생성한다."""
        data = {
            "patient_name": self.generate_patient_name(),
            "patient_id": self.generate_resident_id(),
            "period": f"{self.generate_date()} ~ {self.generate_date()}",
        }

        # 비용 항목 15개
        for i in range(15):
            # 일부 항목은 0원 (해당 없는 항목)
            has_value = random.random() > 0.3
            if has_value:
                data[f"item_{i}_copay"] = self.generate_amount(0, 200000)
                data[f"item_{i}_insurance"] = self.generate_amount(0, 500000)
                data[f"item_{i}_uninsured"] = self.generate_amount(0, 100000)
            else:
                data[f"item_{i}_copay"] = ""
                data[f"item_{i}_insurance"] = ""
                data[f"item_{i}_uninsured"] = ""

        return data
```

---

## 8.1.4 렌더링 엔진

### MedicalDocumentGenerator 클래스

```python
import os
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

logger = logging.getLogger(__name__)


class MedicalDocumentGenerator:
    """의료 문서 합성 이미지를 생성하는 엔진.

    템플릿 정의와 가상 데이터를 결합하여 실제 의료 문서와
    유사한 이미지를 렌더링한다. 동시에 OCR 학습에 필요한
    바운딩 박스와 텍스트 레이블을 자동 생성한다.
    """

    def __init__(
        self,
        font_dir: str,
        output_dir: str,
        dpi: int = 300,
        default_font_size: int = 14,
    ):
        self.font_dir = Path(font_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.default_font_size = default_font_size

        # 사용 가능한 폰트 로드
        self._fonts = self._load_fonts()
        self._data_gen = MedicalDataGenerator()

        logger.info(
            f"MedicalDocumentGenerator 초기화: "
            f"폰트 {len(self._fonts)}개 로드, 출력: {self.output_dir}"
        )

    def _load_fonts(self) -> dict:
        """폰트 디렉토리에서 사용 가능한 폰트를 로드한다."""
        fonts = {}
        font_extensions = (".ttf", ".otf", ".ttc")

        if not self.font_dir.exists():
            logger.warning(f"폰트 디렉토리 없음: {self.font_dir}, 기본 폰트 사용")
            return fonts

        for font_path in self.font_dir.iterdir():
            if font_path.suffix.lower() in font_extensions:
                fonts[font_path.stem] = str(font_path)
                logger.debug(f"폰트 로드: {font_path.stem}")

        return fonts

    def _get_font(self, config: FontConfig) -> ImageFont.FreeTypeFont:
        """FontConfig에 따라 PIL 폰트 객체를 반환한다."""
        # 폰트 크기에 랜덤 변형 적용
        size_var = config.size * config.size_variation
        actual_size = int(config.size + random.uniform(-size_var, size_var))
        actual_size = max(8, actual_size)  # 최소 크기 보장

        font_path = self._fonts.get(config.family)
        if font_path:
            try:
                return ImageFont.truetype(font_path, actual_size)
            except OSError:
                logger.warning(f"폰트 로드 실패: {config.family}")

        # 폴백: 기본 폰트
        try:
            return ImageFont.truetype("arial.ttf", actual_size)
        except OSError:
            return ImageFont.load_default()

    def _apply_text_color_variation(self, base_color: tuple, variation: int) -> tuple:
        """텍스트 색상에 미세한 변형을 적용한다."""
        varied = []
        for c in base_color:
            new_c = c + random.randint(-variation, variation)
            varied.append(max(0, min(255, new_c)))
        return tuple(varied)

    def _render_text_with_bbox(
        self,
        draw: ImageDraw.Draw,
        text: str,
        position: tuple,
        font: ImageFont.FreeTypeFont,
        color: tuple,
    ) -> dict:
        """텍스트를 렌더링하고 실제 바운딩 박스를 반환한다."""
        x, y = position
        bbox = draw.textbbox((x, y), text, font=font)

        draw.text((x, y), text, fill=color, font=font)

        return {
            "text": text,
            "bbox": {
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
            },
        }

    def _draw_table_grid(
        self,
        draw: ImageDraw.Draw,
        x: int, y: int,
        rows: int, cols: int,
        row_height: int, col_widths: list,
        line_color: tuple = (0, 0, 0),
        line_width: int = 1,
    ):
        """테이블 격자선을 그린다."""
        total_width = sum(col_widths)
        total_height = rows * row_height

        # 가로선
        for r in range(rows + 1):
            y_pos = y + r * row_height
            draw.line([(x, y_pos), (x + total_width, y_pos)],
                      fill=line_color, width=line_width)

        # 세로선
        x_pos = x
        for c in range(len(col_widths) + 1):
            draw.line([(x_pos, y), (x_pos, y + total_height)],
                      fill=line_color, width=line_width)
            if c < len(col_widths):
                x_pos += col_widths[c]

    def render_document(
        self,
        template: DocumentTemplate,
        data: dict,
        add_noise: bool = False,
    ) -> tuple:
        """템플릿과 데이터로 문서 이미지를 렌더링한다.

        Args:
            template: 문서 템플릿 정의
            data: 필드별 텍스트 데이터
            add_noise: 기본 노이즈 추가 여부

        Returns:
            (PIL.Image, list[dict]): 렌더링된 이미지와 어노테이션 목록
        """
        # 캔버스 생성
        img = Image.new("RGB", (template.width, template.height),
                        color=template.background_color)
        draw = ImageDraw.Draw(img)

        annotations = []

        # 구분선 렌더링
        for line_info in template.lines:
            draw.line(
                [line_info["start"], line_info["end"]],
                fill=(0, 0, 0),
                width=line_info.get("width", 1),
            )

        # 필드별 렌더링
        for field_def in template.fields:
            field_name = field_def.name
            text_value = data.get(field_name, "")

            if not text_value and field_def.label:
                text_value = ""

            font = self._get_font(field_def.font)
            color = self._apply_text_color_variation(
                field_def.font.color,
                field_def.font.color_variation,
            )

            x, y = field_def.bbox.x, field_def.bbox.y

            # 라벨 렌더링 (필드 앞의 고정 텍스트)
            if field_def.label and field_def.label_font:
                label_font = self._get_font(field_def.label_font)
                label_annotation = self._render_text_with_bbox(
                    draw, field_def.label, (x - 250, y), label_font, (0, 0, 0)
                )
                annotations.append({
                    "field": f"{field_name}_label",
                    "type": "label",
                    **label_annotation,
                })

            # 값 렌더링
            if text_value:
                value_annotation = self._render_text_with_bbox(
                    draw, text_value, (x, y), font, color
                )
                annotations.append({
                    "field": field_name,
                    "type": "value",
                    **value_annotation,
                })

        # 기본 노이즈 (선택적)
        if add_noise:
            img = self._add_basic_noise(img)

        return img, annotations

    def _add_basic_noise(self, img: Image.Image) -> Image.Image:
        """기본 수준의 노이즈를 추가한다. (상세한 노이즈는 8.2에서 다룸)"""
        img_array = np.array(img, dtype=np.float32)

        # 가우시안 노이즈
        noise = np.random.normal(0, 3, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def generate_batch(
        self,
        template: DocumentTemplate,
        num_samples: int,
        data_generator_func,
        prefix: str = "doc",
    ) -> list:
        """배치 단위로 합성 문서를 생성한다.

        Args:
            template: 문서 템플릿
            num_samples: 생성할 샘플 수
            data_generator_func: 데이터 생성 함수 (callable)
            prefix: 파일명 접두사

        Returns:
            생성된 파일 경로 목록
        """
        generated_files = []

        for i in range(num_samples):
            try:
                data = data_generator_func()
                img, annotations = self.render_document(
                    template, data, add_noise=True
                )

                # 이미지 저장
                img_filename = f"{prefix}_{i:06d}.png"
                img_path = self.output_dir / img_filename
                img.save(str(img_path), dpi=(self.dpi, self.dpi))

                # 어노테이션 저장 (JSON)
                anno_filename = f"{prefix}_{i:06d}.json"
                anno_path = self.output_dir / anno_filename
                with open(anno_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "image": img_filename,
                        "template": template.name,
                        "width": template.width,
                        "height": template.height,
                        "annotations": annotations,
                    }, f, ensure_ascii=False, indent=2)

                generated_files.append(str(img_path))

                if (i + 1) % 100 == 0:
                    logger.info(f"진행: {i + 1}/{num_samples} 생성 완료")

            except Exception as e:
                logger.error(f"문서 생성 실패 (인덱스 {i}): {e}")
                continue

        logger.info(
            f"배치 생성 완료: {len(generated_files)}/{num_samples} 성공"
        )
        return generated_files
```

---

## 8.1.5 레이아웃 랜덤화

### 위치 변형 전략

레이아웃 랜덤화는 모델이 고정 위치에 과적합하는 것을 방지한다.

```python
class LayoutRandomizer:
    """템플릿의 레이아웃에 랜덤 변형을 적용한다.

    텍스트 위치, 간격, 정렬에 확률적 변형을 가하여
    동일 템플릿에서도 다양한 외형의 문서를 생성한다.
    """

    def __init__(
        self,
        position_jitter: int = 5,
        spacing_variation: float = 0.1,
        alignment_prob: float = 0.1,
        margin_variation: int = 20,
    ):
        self.position_jitter = position_jitter
        self.spacing_variation = spacing_variation
        self.alignment_prob = alignment_prob
        self.margin_variation = margin_variation

    def randomize_field_position(self, bbox: BoundingBox) -> BoundingBox:
        """필드 위치에 미세한 변형을 적용한다.

        아핀 변환의 이동(translation) 성분에 노이즈를 추가하는 것과 동일:
        t_x' = t_x + ε_x,  ε_x ~ N(0, σ²)
        t_y' = t_y + ε_y,  ε_y ~ N(0, σ²)
        """
        jitter_x = int(np.random.normal(0, self.position_jitter))
        jitter_y = int(np.random.normal(0, self.position_jitter))

        return BoundingBox(
            x=bbox.x + jitter_x,
            y=bbox.y + jitter_y,
            width=bbox.width,
            height=bbox.height,
        )

    def randomize_line_spacing(self, base_spacing: int) -> int:
        """줄 간격에 변형을 적용한다."""
        variation = int(base_spacing * self.spacing_variation)
        return base_spacing + random.randint(-variation, variation)

    def randomize_margin(self, base_margin: int) -> int:
        """여백에 변형을 적용한다."""
        return base_margin + random.randint(
            -self.margin_variation, self.margin_variation
        )

    def apply_global_rotation(
        self,
        img: Image.Image,
        max_angle: float = 1.5,
    ) -> Image.Image:
        """전체 문서에 미세한 회전을 적용한다.

        실제 스캐너에서 발생하는 미세 기울어짐을 모사한다.
        """
        angle = np.random.normal(0, max_angle / 3)
        angle = np.clip(angle, -max_angle, max_angle)

        return img.rotate(
            angle,
            resample=Image.BICUBIC,
            expand=False,
            fillcolor=(255, 255, 255),
        )

    def apply_to_template(
        self,
        template: DocumentTemplate,
    ) -> DocumentTemplate:
        """템플릿 전체에 랜덤 변형을 적용한 복사본을 반환한다."""
        import copy
        new_template = copy.deepcopy(template)

        for field_def in new_template.fields:
            field_def.bbox = self.randomize_field_position(field_def.bbox)

        # 전체 여백 변형
        margin_shift_x = random.randint(
            -self.margin_variation, self.margin_variation
        )
        margin_shift_y = random.randint(
            -self.margin_variation, self.margin_variation
        )

        for field_def in new_template.fields:
            field_def.bbox.x += margin_shift_x
            field_def.bbox.y += margin_shift_y

        return new_template
```

---

## 8.1.6 전체 파이프라인 실행

```python
def run_template_generation_pipeline(
    font_dir: str,
    output_dir: str,
    num_diagnosis: int = 1000,
    num_admission: int = 500,
    num_billing: int = 500,
):
    """템플릿 기반 합성 데이터 생성 파이프라인을 실행한다.

    Args:
        font_dir: 폰트 파일이 있는 디렉토리
        output_dir: 생성된 이미지/어노테이션 저장 디렉토리
        num_diagnosis: 진단서 생성 수
        num_admission: 입퇴원확인서 생성 수
        num_billing: 진료비 세부내역서 생성 수
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    generator = MedicalDocumentGenerator(
        font_dir=font_dir,
        output_dir=output_dir,
    )
    randomizer = LayoutRandomizer()
    data_gen = MedicalDataGenerator()

    # 1. 진단서 생성
    logger.info("=== 진단서 생성 시작 ===")
    diag_template = create_diagnosis_template()
    generator.generate_batch(
        template=diag_template,
        num_samples=num_diagnosis,
        data_generator_func=data_gen.fill_diagnosis_template,
        prefix="diagnosis",
    )

    # 2. 입퇴원확인서 생성
    logger.info("=== 입퇴원확인서 생성 시작 ===")
    admit_template = create_admission_template()
    generator.generate_batch(
        template=admit_template,
        num_samples=num_admission,
        data_generator_func=lambda: {
            "patient_name": data_gen.generate_patient_name(),
            "patient_id": data_gen.generate_resident_id(),
            "admission_date": data_gen.generate_date(),
            "discharge_date": data_gen.generate_date(),
            "department": random.choice(data_gen.DEPARTMENTS),
            "ward": f"{random.randint(1,10)}병동 {random.randint(100,999)}호",
            "diagnosis": data_gen.generate_diagnosis()["name"],
            "doctor_name": data_gen.generate_patient_name(),
        },
        prefix="admission",
    )

    # 3. 진료비 세부내역서 생성
    logger.info("=== 진료비 세부내역서 생성 시작 ===")
    bill_template = create_billing_template()
    generator.generate_batch(
        template=bill_template,
        num_samples=num_billing,
        data_generator_func=data_gen.fill_billing_template,
        prefix="billing",
    )

    logger.info("=== 전체 파이프라인 완료 ===")


# 실행 예시
if __name__ == "__main__":
    run_template_generation_pipeline(
        font_dir="./fonts",
        output_dir="./output/synthetic_medical_docs",
        num_diagnosis=1000,
        num_admission=500,
        num_billing=500,
    )
```

---

## 8.1.7 품질 검증

생성된 합성 데이터의 품질을 정량적으로 평가하는 방법이다.

```python
class SyntheticDataValidator:
    """합성 데이터의 품질을 검증한다."""

    def validate_annotation(self, img_path: str, anno_path: str) -> dict:
        """이미지와 어노테이션의 정합성을 검증한다."""
        img = Image.open(img_path)
        with open(anno_path, "r", encoding="utf-8") as f:
            anno = json.load(f)

        issues = []

        # 1. 이미지 크기 일치 확인
        if img.size != (anno["width"], anno["height"]):
            issues.append(f"크기 불일치: 이미지={img.size}, 어노테이션="
                          f"({anno['width']}, {anno['height']})")

        # 2. 바운딩 박스 범위 검증
        for entry in anno["annotations"]:
            bbox = entry["bbox"]
            if bbox["x1"] < 0 or bbox["y1"] < 0:
                issues.append(f"음수 좌표: {entry['field']}")
            if bbox["x2"] > anno["width"] or bbox["y2"] > anno["height"]:
                issues.append(f"이미지 범위 초과: {entry['field']}")

        # 3. 필수 필드 존재 확인
        field_names = {e["field"] for e in anno["annotations"]}
        if "patient_name" not in field_names and "patient_name_label" not in field_names:
            issues.append("환자명 필드 누락")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "num_annotations": len(anno["annotations"]),
        }

    def compute_diversity_score(self, annotations_dir: str) -> dict:
        """생성된 데이터셋의 다양성 점수를 계산한다.

        텍스트 유니크 비율, 바운딩 박스 분포 분산 등을 측정한다.
        """
        all_texts = []
        all_bboxes = []

        anno_dir = Path(annotations_dir)
        for anno_file in anno_dir.glob("*.json"):
            with open(anno_file, "r", encoding="utf-8") as f:
                anno = json.load(f)

            for entry in anno["annotations"]:
                if entry["type"] == "value":
                    all_texts.append(entry["text"])
                    bbox = entry["bbox"]
                    all_bboxes.append([
                        bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                    ])

        unique_ratio = len(set(all_texts)) / max(len(all_texts), 1)

        bbox_array = np.array(all_bboxes) if all_bboxes else np.zeros((1, 4))
        bbox_variance = bbox_array.var(axis=0).mean()

        return {
            "total_annotations": len(all_texts),
            "unique_text_ratio": unique_ratio,
            "bbox_position_variance": float(bbox_variance),
        }
```

---

## 용어 체크리스트

학습 후 아래 항목을 설명할 수 있는지 점검하라.

- [ ] **Template**: 문서의 레이아웃 구조를 정의하는 청사진이 뭔지, JSON/YAML로 어떻게 표현하는지 설명할 수 있는가?
- [ ] **아핀 변환**: 이동, 회전, 스케일, 전단을 하나의 행렬로 표현하는 방법을 수식으로 쓸 수 있는가?
- [ ] **동차 좌표**: 아핀 변환을 단일 행렬 곱으로 표현하기 위해 좌표를 $(x, y, 1)$로 확장하는 이유를 설명할 수 있는가?
- [ ] **Font Variation**: 폰트 크기, 색상, 간격을 확률 분포에서 샘플링하는 이유와 적절한 분포 선택 기준을 아는가?
- [ ] **Rendering**: Pillow를 이용한 텍스트 렌더링 과정에서 `textbbox`로 실제 바운딩 박스를 구하는 방법을 아는가?
- [ ] **Layout Randomization**: 위치 지터(jitter), 여백 변형, 미세 회전이 모델 일반화에 기여하는 원리를 설명할 수 있는가?
- [ ] **의료 문서 양식**: 진단서, 입퇴원확인서, 진료비 세부내역서의 표준 레이아웃 구조를 알고 있는가?
- [ ] **품질 검증**: 합성 데이터의 어노테이션 정합성과 다양성 점수를 어떻게 측정하는지 설명할 수 있는가?
- [ ] **배치 생성**: 대량의 합성 데이터를 효율적으로 생성하고 저장하는 파이프라인 구조를 이해하는가?
