# 8.3 Pre-printed 라인 간섭 시뮬레이션

의료 문서 대부분은 pre-printed form — 미리 인쇄된 양식 위에 텍스트를 기입하는 형태다. 양식의 구분선, 표 격자선, 배경 로고 등이 실제 텍스트와 겹치면서 OCR 인식률을 떨어뜨린다. 이 장에서는 pre-printed 배경과 텍스트의 간섭을 시뮬레이션하고, 두 레이어를 분리하는 기법을 다룬다.

---

> **핵심 용어 박스**
>
> | 용어 | 정의 |
> |------|------|
> | **Pre-printed Form** | 양식 선, 표, 로고 등이 미리 인쇄된 문서. 위에 텍스트를 수기 또는 프린터로 기입 |
> | **Line Detection** | 이미지에서 직선/곡선을 검출하는 기법. Hough Transform이 대표적 |
> | **Background Subtraction** | 배경(양식)과 전경(텍스트)을 분리하는 기법 |
> | **Otsu's Method** | 클래스 간 분산을 최대화하는 최적 이진화 임계값 자동 결정 알고리즘 |

---

## 8.3.1 Pre-printed Form의 OCR 난제

### 간섭 유형 분류

```
Pre-printed Form 간섭 유형
├── 구분선 간섭
│   ├── 수평선이 텍스트를 관통
│   ├── 수직선이 글자 사이를 지남
│   └── 테이블 격자가 텍스트 영역과 겹침
├── 배경 패턴 간섭
│   ├── 워터마크/로고
│   ├── 컬러 배경 (연한 색상)
│   └── 보안 패턴 (미세 문양)
└── 잉크 혼합 간섭
    ├── 양식 잉크 색상 ≈ 텍스트 잉크 색상
    ├── 열전사 프린터의 번짐
    └── 복사 시 양식선 강조
```

| 간섭 유형 | OCR 영향 | 발생 빈도 |
|----------|---------|----------|
| 수평선 관통 | 글자 분할 오류, 획 인식 실패 | 매우 높음 |
| 수직선 간섭 | 문자 경계 인식 오류 | 높음 |
| 테이블 격자 | 셀 내 텍스트 추출 어려움 | 높음 |
| 워터마크 | 대비 저하, 배경 노이즈 | 보통 |
| 잉크 혼합 | 전경/배경 구분 불가 | 낮음 (고난이도) |

---

## 수학적 원리

### Hough Transform (직선 검출)

이미지의 각 에지 점 $(x, y)$를 파라미터 공간 $(\rho, \theta)$로 변환하여 직선을 검출한다.

직선의 정규 형태:

$$
\rho = x \cos\theta + y \sin\theta
$$

여기서:
- $\rho$: 원점에서 직선까지의 수직 거리
- $\theta$: 수직선과 x축이 이루는 각도
- $(x, y)$: 이미지 상의 점

**알고리즘 절차**:

1. 에지 검출 (Canny 등)으로 에지 맵 $E(x, y)$ 생성
2. 파라미터 공간의 누적 배열(accumulator) $A(\rho, \theta)$ 초기화
3. 각 에지 점 $(x_i, y_i)$에 대해 모든 $\theta \in [0, \pi)$에 대해:

$$
\rho_i = x_i \cos\theta + y_i \sin\theta
$$

$$
A(\rho_i, \theta) \leftarrow A(\rho_i, \theta) + 1
$$

4. $A(\rho, \theta)$에서 피크를 탐색하여 직선 파라미터 추출

**양자화**:
- $\theta$: $[0, \pi)$ 범위를 $N_\theta$개 빈으로 양자화
- $\rho$: $[-D, D]$ 범위를 $N_\rho$개 빈으로 양자화 ($D$는 이미지 대각선 길이)

$$
D = \sqrt{W^2 + H^2}
$$

피크 검출 임계값 $T$: $A(\rho, \theta) > T$인 점만 직선으로 판별한다.

### Background Subtraction (배경 분리)

전경(텍스트)과 배경(양식선)을 분리한다:

$$
F(x, y) = |I_{\text{current}}(x, y) - I_{\text{background}}(x, y)|
$$

이진 전경 마스크:

$$
M(x, y) = \begin{cases} 1 & \text{if } F(x, y) > \tau \\ 0 & \text{otherwise} \end{cases}
$$

여기서 $\tau$는 전경 판별 임계값이다.

**적응적 배경 추정** — 배경 참조가 없는 경우, 형태학적 연산으로 배경을 추정한다:

$$
I_{\text{bg}} = \text{close}(\text{open}(I, K), K)
$$

$K$는 구조 요소(structuring element)로, 텍스트보다 큰 크기로 설정하면 텍스트가 제거되고 배경만 남는다.

### Otsu's Method (최적 이진화)

히스토그램 기반으로 최적 임계값 $t^*$를 자동 결정한다.

전체 픽셀을 임계값 $t$로 두 클래스 $C_0$ (배경)과 $C_1$ (전경)으로 분할할 때, 클래스 간 분산(between-class variance):

$$
\sigma_B^2(t) = \omega_0(t) \omega_1(t) [\mu_0(t) - \mu_1(t)]^2
$$

여기서:
- $\omega_0(t) = \sum_{i=0}^{t} p_i$: 클래스 0의 확률 (가중치)
- $\omega_1(t) = \sum_{i=t+1}^{L-1} p_i$: 클래스 1의 확률
- $\mu_0(t) = \frac{1}{\omega_0(t)} \sum_{i=0}^{t} i \cdot p_i$: 클래스 0의 평균
- $\mu_1(t) = \frac{1}{\omega_1(t)} \sum_{i=t+1}^{L-1} i \cdot p_i$: 클래스 1의 평균
- $p_i$: 히스토그램에서 강도 $i$의 확률

최적 임계값:

$$
t^* = \arg\max_t \sigma_B^2(t)
$$

$\sigma_B^2(t)$가 최대인 $t$를 선택하면 두 클래스의 분리가 최대화된다.

---

## 8.3.2 양식 배경 생성기

### FormBackgroundGenerator 클래스

```python
import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class LineStyle(Enum):
    """양식 선 스타일."""
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    DOUBLE = "double"


@dataclass
class FormLine:
    """양식 구분선 정의."""
    start: tuple  # (x, y)
    end: tuple    # (x, y)
    thickness: int = 1
    color: tuple = (0, 0, 0)
    style: LineStyle = LineStyle.SOLID


@dataclass
class FormTable:
    """양식 테이블 정의."""
    x: int
    y: int
    rows: int
    cols: int
    row_height: int
    col_widths: list
    line_thickness: int = 1
    line_color: tuple = (0, 0, 0)
    header_color: tuple = (230, 230, 230)  # 헤더 배경색


@dataclass
class FormConfig:
    """양식 배경 전체 설정."""
    width: int = 2480
    height: int = 3508
    background_color: tuple = (255, 255, 255)
    line_color: tuple = (150, 150, 150)       # 양식선 기본 색상 (회색)
    line_thickness: int = 1
    lines: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    watermark_text: Optional[str] = None
    watermark_opacity: float = 0.1


class FormBackgroundGenerator:
    """Pre-printed 양식 배경 이미지를 생성한다.

    의료 문서의 양식선, 테이블, 워터마크, 보안 패턴 등을
    프로그래밍적으로 생성하여 합성 데이터의 배경으로 사용한다.
    """

    def __init__(self, config: Optional[FormConfig] = None):
        self.config = config or FormConfig()

    def _draw_solid_line(
        self,
        img: np.ndarray,
        start: tuple,
        end: tuple,
        color: tuple,
        thickness: int,
    ):
        """실선을 그린다."""
        cv2.line(img, start, end, color, thickness)

    def _draw_dashed_line(
        self,
        img: np.ndarray,
        start: tuple,
        end: tuple,
        color: tuple,
        thickness: int,
        dash_length: int = 10,
        gap_length: int = 5,
    ):
        """점선을 그린다."""
        x1, y1 = start
        x2, y2 = end
        total_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if total_length == 0:
            return

        dx = (x2 - x1) / total_length
        dy = (y2 - y1) / total_length

        current = 0
        drawing = True

        while current < total_length:
            if drawing:
                seg_end = min(current + dash_length, total_length)
                pt1 = (int(x1 + current * dx), int(y1 + current * dy))
                pt2 = (int(x1 + seg_end * dx), int(y1 + seg_end * dy))
                cv2.line(img, pt1, pt2, color, thickness)
                current = seg_end + gap_length
            else:
                current += gap_length
            drawing = not drawing

    def _draw_dotted_line(
        self,
        img: np.ndarray,
        start: tuple,
        end: tuple,
        color: tuple,
        dot_spacing: int = 8,
        dot_radius: int = 1,
    ):
        """점선(dot)을 그린다."""
        x1, y1 = start
        x2, y2 = end
        total_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if total_length == 0:
            return

        dx = (x2 - x1) / total_length
        dy = (y2 - y1) / total_length

        for d in range(0, int(total_length), dot_spacing):
            pt = (int(x1 + d * dx), int(y1 + d * dy))
            cv2.circle(img, pt, dot_radius, color, -1)

    def draw_line(
        self,
        img: np.ndarray,
        form_line: FormLine,
    ):
        """양식 선을 스타일에 따라 그린다."""
        if form_line.style == LineStyle.SOLID:
            self._draw_solid_line(
                img, form_line.start, form_line.end,
                form_line.color, form_line.thickness,
            )
        elif form_line.style == LineStyle.DASHED:
            self._draw_dashed_line(
                img, form_line.start, form_line.end,
                form_line.color, form_line.thickness,
            )
        elif form_line.style == LineStyle.DOTTED:
            self._draw_dotted_line(
                img, form_line.start, form_line.end,
                form_line.color,
            )
        elif form_line.style == LineStyle.DOUBLE:
            offset = form_line.thickness + 1
            # 위 선
            self._draw_solid_line(
                img,
                (form_line.start[0], form_line.start[1] - offset),
                (form_line.end[0], form_line.end[1] - offset),
                form_line.color, form_line.thickness,
            )
            # 아래 선
            self._draw_solid_line(
                img,
                (form_line.start[0], form_line.start[1] + offset),
                (form_line.end[0], form_line.end[1] + offset),
                form_line.color, form_line.thickness,
            )

    def draw_table(self, img: np.ndarray, table: FormTable):
        """테이블 격자를 그린다."""
        x, y = table.x, table.y
        total_width = sum(table.col_widths)
        total_height = table.rows * table.row_height

        # 헤더 배경
        if table.header_color:
            cv2.rectangle(
                img,
                (x, y),
                (x + total_width, y + table.row_height),
                table.header_color,
                -1,  # 채움
            )

        # 가로선
        for r in range(table.rows + 1):
            y_pos = y + r * table.row_height
            thickness = table.line_thickness * 2 if r == 0 else table.line_thickness
            cv2.line(
                img,
                (x, y_pos),
                (x + total_width, y_pos),
                table.line_color,
                thickness,
            )

        # 세로선
        x_pos = x
        for c in range(len(table.col_widths) + 1):
            thickness = table.line_thickness * 2 if c == 0 or c == len(table.col_widths) else table.line_thickness
            cv2.line(
                img,
                (x_pos, y),
                (x_pos, y + total_height),
                table.line_color,
                thickness,
            )
            if c < len(table.col_widths):
                x_pos += table.col_widths[c]

    def add_watermark(
        self,
        img: np.ndarray,
        text: str,
        opacity: float = 0.1,
        angle: float = -30,
        font_scale: float = 3.0,
    ) -> np.ndarray:
        """대각선 워터마크를 추가한다."""
        overlay = img.copy()
        h, w = img.shape[:2]

        # 텍스트 크기 계산
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, 3)[0]

        # 중앙에 배치
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2

        # 회전 적용
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 텍스트를 별도 레이어에 렌더링
        text_layer = np.zeros_like(img)
        cv2.putText(
            text_layer, text,
            (text_x, text_y),
            font, font_scale, (128, 128, 128), 3,
        )

        # 회전
        text_layer = cv2.warpAffine(text_layer, M, (w, h))

        # 알파 블렌딩
        mask = text_layer > 0
        img_float = img.astype(np.float32)
        img_float[mask] = img_float[mask] * (1 - opacity) + text_layer[mask].astype(np.float32) * opacity

        return img_float.astype(np.uint8)

    def add_security_pattern(
        self,
        img: np.ndarray,
        pattern_type: str = "crosshatch",
        intensity: float = 0.03,
    ) -> np.ndarray:
        """보안 패턴을 추가한다. 복사 방지 미세 문양."""
        h, w = img.shape[:2]

        if pattern_type == "crosshatch":
            # 미세 교차 해칭
            spacing = 20
            pattern = np.zeros((h, w), dtype=np.float32)

            for i in range(0, h + w, spacing):
                # 대각선 /
                pt1 = (max(0, i - h), min(i, h - 1))
                pt2 = (min(i, w - 1), max(0, i - w))
                cv2.line(pattern, pt1, pt2, 1.0, 1)

                # 대각선 \
                pt1 = (max(0, w - i), min(i, h - 1))
                pt2 = (min(w - 1, w - i + h), max(0, i - w))
                cv2.line(pattern, pt1, pt2, 1.0, 1)

            pattern = np.stack([pattern] * 3, axis=-1) if len(img.shape) == 3 else pattern
            result = img.astype(np.float32) - pattern * intensity * 255
            return np.clip(result, 0, 255).astype(np.uint8)

        elif pattern_type == "microdot":
            # 미세 점 패턴
            spacing = 15
            for y_pos in range(0, h, spacing):
                for x_pos in range(0, w, spacing):
                    if np.random.random() < 0.3:
                        color_val = int(255 * (1 - intensity))
                        cv2.circle(img, (x_pos, y_pos), 1, (color_val,) * 3, -1)

        return img

    def generate_medical_form_background(
        self,
        form_type: str = "diagnosis",
    ) -> np.ndarray:
        """의료 문서 유형별 양식 배경을 생성한다."""
        img = np.full(
            (self.config.height, self.config.width, 3),
            self.config.background_color,
            dtype=np.uint8,
        )

        line_color = self.config.line_color

        if form_type == "diagnosis":
            # 진단서 양식
            # 외곽 테두리 (이중선)
            cv2.rectangle(img, (150, 150), (2330, 3358), line_color, 3)
            cv2.rectangle(img, (160, 160), (2320, 3348), line_color, 1)

            # 제목 영역 구분선
            self.draw_line(img, FormLine(
                start=(200, 350), end=(2280, 350),
                thickness=2, color=line_color, style=LineStyle.DOUBLE,
            ))

            # 환자 정보 영역
            lines_y = [430, 490, 550, 610, 680]
            for ly in lines_y:
                self.draw_line(img, FormLine(
                    start=(200, ly), end=(2280, ly),
                    thickness=1, color=line_color,
                ))

            # 진단 내용 영역
            self.draw_line(img, FormLine(
                start=(200, 730), end=(2280, 730),
                thickness=2, color=line_color,
            ))

            # 진단 소견 밑줄 (여러 줄)
            for i in range(8):
                ly = 950 + i * 50
                self.draw_line(img, FormLine(
                    start=(300, ly), end=(2200, ly),
                    thickness=1, color=line_color, style=LineStyle.DOTTED,
                ))

            # 하단 발급 정보 구분선
            self.draw_line(img, FormLine(
                start=(200, 2700), end=(2280, 2700),
                thickness=2, color=line_color,
            ))

        elif form_type == "billing":
            # 진료비 세부내역서 — 복잡한 테이블 구조
            cv2.rectangle(img, (100, 100), (2380, 3408), line_color, 2)

            # 상단 환자 정보
            self.draw_table(img, FormTable(
                x=150, y=300, rows=3, cols=4,
                row_height=40,
                col_widths=[200, 350, 200, 500],
                line_color=line_color,
            ))

            # 비용 항목 테이블
            self.draw_table(img, FormTable(
                x=150, y=550, rows=18, cols=5,
                row_height=35,
                col_widths=[300, 250, 250, 250, 200],
                line_color=line_color,
                header_color=(220, 220, 240),
            ))

            # 하단 합계
            self.draw_table(img, FormTable(
                x=150, y=1250, rows=3, cols=5,
                row_height=40,
                col_widths=[300, 250, 250, 250, 200],
                line_color=line_color,
                header_color=(240, 240, 220),
            ))

        elif form_type == "admission":
            # 입퇴원확인서
            cv2.rectangle(img, (150, 150), (2330, 3358), line_color, 2)

            # 테이블 형식의 환자 정보
            self.draw_table(img, FormTable(
                x=200, y=400, rows=8, cols=3,
                row_height=50,
                col_widths=[300, 400, 400],
                line_color=line_color,
            ))

        return img
```

---

## 8.3.3 텍스트-배경 합성 파이프라인

### 양식 위에 텍스트를 합성하는 엔진

```python
from PIL import Image, ImageDraw, ImageFont
import random


class TextFormCompositor:
    """Pre-printed 양식 배경 위에 텍스트를 합성한다.

    양식선과 텍스트의 간섭을 현실적으로 시뮬레이션하며,
    분리 가능한 레이어 구조로 GT 생성을 지원한다.
    """

    def __init__(
        self,
        font_dir: str,
        form_generator: FormBackgroundGenerator,
    ):
        self.font_dir = Path(font_dir)
        self.form_generator = form_generator
        self._fonts = self._scan_fonts()

    def _scan_fonts(self) -> list:
        """사용 가능한 폰트 파일을 스캔한다."""
        fonts = []
        if self.font_dir.exists():
            for ext in ("*.ttf", "*.otf", "*.ttc"):
                fonts.extend(self.font_dir.glob(ext))
        return fonts

    def _get_random_font(self, size: int) -> ImageFont.FreeTypeFont:
        """랜덤 폰트를 반환한다."""
        if self._fonts:
            font_path = random.choice(self._fonts)
            try:
                return ImageFont.truetype(str(font_path), size)
            except OSError:
                pass
        try:
            return ImageFont.truetype("arial.ttf", size)
        except OSError:
            return ImageFont.load_default()

    def composite_text_on_form(
        self,
        form_background: np.ndarray,
        text_fields: list,
        ink_color: tuple = (0, 0, 0),
        ink_variation: int = 20,
    ) -> tuple:
        """양식 배경 위에 텍스트를 합성한다.

        Args:
            form_background: 양식 배경 이미지 (numpy array)
            text_fields: 텍스트 필드 목록
                [{"text": str, "x": int, "y": int, "font_size": int}, ...]
            ink_color: 기본 잉크 색상
            ink_variation: 잉크 색상 변형 범위

        Returns:
            (합성 이미지, 텍스트 전용 마스크, 어노테이션 목록)
        """
        # numpy → PIL 변환
        form_pil = Image.fromarray(cv2.cvtColor(form_background, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(form_pil)

        # 텍스트 전용 마스크 (배경 분리 GT용)
        text_mask = Image.new("L", form_pil.size, 0)
        mask_draw = ImageDraw.Draw(text_mask)

        annotations = []

        for field_info in text_fields:
            text = field_info["text"]
            x = field_info["x"]
            y = field_info["y"]
            font_size = field_info.get("font_size", 14)

            font = self._get_random_font(font_size)

            # 잉크 색상 변형
            color = tuple(
                max(0, min(255, c + random.randint(-ink_variation, ink_variation)))
                for c in ink_color
            )

            # 텍스트 렌더링
            bbox = draw.textbbox((x, y), text, font=font)
            draw.text((x, y), text, fill=color, font=font)

            # 마스크에도 렌더링
            mask_draw.text((x, y), text, fill=255, font=font)

            annotations.append({
                "text": text,
                "bbox": {
                    "x1": bbox[0], "y1": bbox[1],
                    "x2": bbox[2], "y2": bbox[3],
                },
                "font_size": font_size,
            })

        # PIL → numpy 변환
        composite = cv2.cvtColor(np.array(form_pil), cv2.COLOR_RGB2BGR)
        text_mask_np = np.array(text_mask)

        return composite, text_mask_np, annotations

    def simulate_ink_bleeding(
        self,
        image: np.ndarray,
        text_mask: np.ndarray,
        bleed_amount: float = 0.5,
    ) -> np.ndarray:
        """잉크 번짐 효과를 시뮬레이션한다.

        텍스트 영역을 약간 팽창(dilate)시키고 블러를 적용하여
        잉크가 양식선 위로 번지는 현상을 재현한다.
        """
        # 텍스트 마스크 팽창
        kernel_size = max(1, int(bleed_amount * 3))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        dilated_mask = cv2.dilate(text_mask, kernel, iterations=1)

        # 번짐 영역에 블러 적용
        blurred = cv2.GaussianBlur(image, (3, 3), bleed_amount)

        # 번짐 영역만 블러된 이미지로 교체
        bleed_region = dilated_mask > text_mask
        result = image.copy()
        result[bleed_region] = blurred[bleed_region]

        return result

    def simulate_line_text_overlap(
        self,
        form_background: np.ndarray,
        text_image: np.ndarray,
        text_mask: np.ndarray,
        overlap_mode: str = "darken",
    ) -> np.ndarray:
        """양식선과 텍스트의 겹침을 시뮬레이션한다.

        Args:
            overlap_mode:
                - "darken": 더 어두운 값 채택 (min)
                - "multiply": 곱셈 블렌딩
                - "overlay": 오버레이 블렌딩
        """
        form_f = form_background.astype(np.float32) / 255.0
        text_f = text_image.astype(np.float32) / 255.0
        mask_f = text_mask.astype(np.float32) / 255.0

        if len(mask_f.shape) == 2:
            mask_f = np.stack([mask_f] * 3, axis=-1)

        if overlap_mode == "darken":
            # min 연산: 양식선과 텍스트 중 더 어두운 값
            result = np.minimum(form_f, text_f)

        elif overlap_mode == "multiply":
            # 곱셈 블렌딩: 양식선과 텍스트가 겹치면 더 어두워짐
            result = form_f * text_f

        elif overlap_mode == "overlay":
            # 텍스트 영역만 합성, 나머지는 배경
            result = form_f * (1 - mask_f) + text_f * mask_f

        else:
            result = text_f

        return (result * 255).astype(np.uint8)
```

---

## 8.3.4 배경 분리 (전처리)

### OCR 전에 양식선을 제거하는 전처리 기법

```python
class FormLineRemover:
    """양식 배경에서 구분선을 검출하고 제거한다.

    Hough Transform으로 직선을 검출한 뒤, 형태학적 연산으로
    텍스트를 보존하면서 양식선만 제거한다.
    """

    def __init__(
        self,
        hough_threshold: int = 100,
        min_line_length: int = 100,
        max_line_gap: int = 10,
    ):
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    def detect_lines(
        self,
        image: np.ndarray,
        orientation: str = "both",
    ) -> list:
        """Hough Transform으로 직선을 검출한다.

        ρ = x cos θ + y sin θ
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 에지 검출
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # 확률적 Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        if lines is None:
            return []

        detected = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            if orientation == "horizontal" and angle > 10:
                continue
            if orientation == "vertical" and abs(angle - 90) > 10:
                continue

            detected.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "angle": angle,
                "length": np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2),
            })

        logger.info(f"직선 {len(detected)}개 검출 (방향: {orientation})")
        return detected

    def remove_horizontal_lines(
        self,
        image: np.ndarray,
        kernel_length_ratio: float = 0.3,
    ) -> np.ndarray:
        """수평선을 형태학적 연산으로 제거한다."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 수평선 검출용 커널
        h, w = binary.shape
        horizontal_kernel_length = int(w * kernel_length_ratio)
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (horizontal_kernel_length, 1)
        )

        # 수평선 추출
        horizontal_lines = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
        )

        # 원본에서 수평선 제거 (흰색으로 대체)
        result = gray.copy()
        result[horizontal_lines > 0] = 255

        if len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        return result

    def remove_vertical_lines(
        self,
        image: np.ndarray,
        kernel_length_ratio: float = 0.3,
    ) -> np.ndarray:
        """수직선을 형태학적 연산으로 제거한다."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h, w = binary.shape
        vertical_kernel_length = int(h * kernel_length_ratio)
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, vertical_kernel_length)
        )

        vertical_lines = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2
        )

        result = gray.copy()
        result[vertical_lines > 0] = 255

        if len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        return result

    def remove_all_form_lines(self, image: np.ndarray) -> np.ndarray:
        """수평선과 수직선을 모두 제거한다."""
        result = self.remove_horizontal_lines(image)
        result = self.remove_vertical_lines(result)
        return result

    def extract_text_foreground(
        self,
        image: np.ndarray,
        struct_element_size: int = 15,
    ) -> np.ndarray:
        """형태학적 배경 추정으로 텍스트 전경을 추출한다.

        I_bg = close(open(I, K), K)  (배경 추정)
        F = |I - I_bg|              (전경 추출)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 구조 요소: 텍스트보다 큰 크기
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (struct_element_size, struct_element_size)
        )

        # 배경 추정: opening → closing
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        background = cv2.morphologyEx(background, cv2.MORPH_CLOSE, kernel)

        # 전경 = |원본 - 배경|
        foreground = cv2.absdiff(gray, background)

        # 이진화
        _, foreground_binary = cv2.threshold(
            foreground, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return foreground_binary
```

---

## 8.3.5 전체 합성 파이프라인

```python
class PreprintedFormPipeline:
    """Pre-printed 양식 + 텍스트 합성 전체 파이프라인.

    1. 양식 배경 생성
    2. 텍스트 데이터 생성
    3. 텍스트-배경 합성
    4. 노이즈/열화 적용
    5. 어노테이션 + 분리 마스크 저장
    """

    def __init__(
        self,
        font_dir: str,
        output_dir: str,
        form_type: str = "diagnosis",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.form_config = FormConfig()
        self.form_generator = FormBackgroundGenerator(self.form_config)
        self.compositor = TextFormCompositor(font_dir, self.form_generator)
        self.form_type = form_type

        # 서브 디렉토리
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "masks").mkdir(exist_ok=True)
        (self.output_dir / "backgrounds").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)

    def generate_sample(self, sample_id: int, data: dict) -> dict:
        """단일 합성 샘플을 생성한다.

        Returns:
            생성된 파일 경로 딕셔너리
        """
        # 1. 양식 배경 생성
        form_bg = self.form_generator.generate_medical_form_background(
            form_type=self.form_type
        )

        # 워터마크 추가 (확률적)
        if np.random.random() < 0.3:
            form_bg = self.form_generator.add_watermark(
                form_bg,
                text="SAMPLE",
                opacity=np.random.uniform(0.05, 0.15),
            )

        # 보안 패턴 추가 (확률적)
        if np.random.random() < 0.2:
            pattern_type = random.choice(["crosshatch", "microdot"])
            form_bg = self.form_generator.add_security_pattern(
                form_bg, pattern_type=pattern_type,
            )

        # 2. 텍스트 필드 구성
        text_fields = []
        for key, value in data.items():
            if not value or key.startswith("item_"):
                continue
            # 간단한 위치 매핑 (실제로는 템플릿에서 가져옴)
            field_x = 400 + random.randint(-5, 5)
            field_y = 400 + len(text_fields) * 60 + random.randint(-3, 3)
            text_fields.append({
                "text": f"{key}: {value}",
                "x": field_x,
                "y": field_y,
                "font_size": random.choice([12, 13, 14]),
            })

        # 3. 텍스트-배경 합성
        composite, text_mask, annotations = self.compositor.composite_text_on_form(
            form_bg, text_fields,
        )

        # 4. 잉크 번짐 (확률적)
        if np.random.random() < 0.2:
            composite = self.compositor.simulate_ink_bleeding(
                composite, text_mask, bleed_amount=np.random.uniform(0.3, 0.8),
            )

        # 5. 파일 저장
        prefix = f"{self.form_type}_{sample_id:06d}"

        img_path = self.output_dir / "images" / f"{prefix}.png"
        cv2.imwrite(str(img_path), composite)

        mask_path = self.output_dir / "masks" / f"{prefix}_mask.png"
        cv2.imwrite(str(mask_path), text_mask)

        bg_path = self.output_dir / "backgrounds" / f"{prefix}_bg.png"
        cv2.imwrite(str(bg_path), form_bg)

        import json
        anno_path = self.output_dir / "annotations" / f"{prefix}.json"
        with open(anno_path, "w", encoding="utf-8") as f:
            json.dump({
                "image": str(img_path.name),
                "mask": str(mask_path.name),
                "background": str(bg_path.name),
                "form_type": self.form_type,
                "annotations": annotations,
            }, f, ensure_ascii=False, indent=2)

        return {
            "image": str(img_path),
            "mask": str(mask_path),
            "background": str(bg_path),
            "annotation": str(anno_path),
        }

    def run(self, num_samples: int = 100) -> list:
        """파이프라인을 실행한다."""
        from chapter08_01 import MedicalDataGenerator  # noqa: 실제 경로에 맞게 수정

        data_gen = MedicalDataGenerator()
        results = []

        for i in range(num_samples):
            try:
                if self.form_type == "diagnosis":
                    data = data_gen.fill_diagnosis_template()
                elif self.form_type == "billing":
                    data = data_gen.fill_billing_template()
                else:
                    data = {
                        "patient_name": data_gen.generate_patient_name(),
                        "admission_date": data_gen.generate_date(),
                    }

                result = self.generate_sample(i, data)
                results.append(result)

                if (i + 1) % 50 == 0:
                    logger.info(f"진행: {i + 1}/{num_samples}")

            except Exception as e:
                logger.error(f"샘플 {i} 생성 실패: {e}")

        logger.info(f"파이프라인 완료: {len(results)}/{num_samples} 성공")
        return results


# 실행 예시
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pipeline = PreprintedFormPipeline(
        font_dir="./fonts",
        output_dir="./output/preprinted_forms",
        form_type="diagnosis",
    )
    pipeline.run(num_samples=200)
```

---

## 용어 체크리스트

학습 후 아래 항목을 설명할 수 있는지 점검하라.

- [ ] **Pre-printed Form**: 양식선과 텍스트가 겹칠 때 OCR 인식률이 떨어지는 원인을 구체적으로 설명할 수 있는가?
- [ ] **Hough Transform**: 이미지 좌표 $(x, y)$를 파라미터 공간 $(\rho, \theta)$로 변환하는 원리를 수식으로 유도할 수 있는가?
- [ ] **누적 배열 (Accumulator)**: Hough Transform에서 피크 검출로 직선 파라미터를 추출하는 과정을 이해하는가?
- [ ] **Background Subtraction**: $F(x,y) = |I_{\text{current}} - I_{\text{background}}| > \tau$ 수식의 각 항과 임계값 $\tau$의 역할을 아는가?
- [ ] **형태학적 배경 추정**: opening-closing 연산으로 배경을 추정하는 원리를 설명할 수 있는가?
- [ ] **Otsu's Method**: 클래스 간 분산 $\sigma^2_B(t)$를 최대화하는 과정을 수식으로 유도할 수 있는가?
- [ ] **잉크 번짐**: 팽창(dilation) + 블러로 잉크 번짐을 시뮬레이션하는 원리를 이해하는가?
- [ ] **다크닝 합성**: 양식선과 텍스트를 `min` 연산으로 합성하는 이유를 설명할 수 있는가?
- [ ] **텍스트 마스크**: 텍스트 전용 마스크가 학습 데이터에서 어떤 역할을 하는지 아는가?
- [ ] **수평/수직선 제거**: 형태학적 커널의 방향과 크기가 선 검출에 미치는 영향을 설명할 수 있는가?
