---
---

# 6.2 OLMoCR 분석

> **OLMoCR**은 Allen AI(AI2)가 2024년에 공개한 문서 OCR 특화 Vision-Language Model이다.
> "Open Language Model for OCR"의 약자로, 오픈소스 철학 위에 문서 이해 성능을 극대화한 모델이다.
> Qwen2-VL을 backbone으로 사용하면서, 문서 OCR에 특화된 학습 전략으로 차별화된다.

---

## 핵심 용어

| 용어 | 정의 | 왜 중요한가 |
|------|------|-------------|
| **Document OCR** | 문서 이미지에서 텍스트와 구조를 추출하는 기술 | 전통 OCR은 텍스트만 뽑지만, Document OCR은 레이아웃·표·수식까지 이해한다 |
| **Page-level Understanding** | 문서 한 페이지 전체를 하나의 단위로 이해하는 접근법 | 영역별 분리 처리 대신 페이지 전체의 맥락을 활용하여 정확도를 높인다 |
| **Layout-aware** | 문서의 물리적 레이아웃(배치, 열, 표 등)을 인식하는 능력 | 같은 텍스트라도 위치에 따라 의미가 다르다 (제목 vs 본문 vs 각주) |
| **Anchor Text** | PDF에서 프로그래밍적으로 추출한 텍스트 좌표 정보 | 학습 데이터 생성 시 이미지-텍스트 정렬의 기준점 역할 |
| **ChatQA-style Prompt** | 대화형 질의응답 형식의 프롬프트 설계 | 다양한 문서 이해 태스크를 통일된 인터페이스로 수행 |

---

## OLMoCR의 설계 철학

### 1. 왜 OLMoCR인가

기존 문서 OCR 모델들의 한계:

| 문제점 | 설명 |
|--------|------|
| **비공개 모델 의존** | GPT-4V, Gemini 등 상용 API에 의존하면 비용·프라이버시 문제 |
| **파이프라인 복잡도** | Detection → Recognition → Layout Analysis → Structuring의 다단계 |
| **도메인 편향** | 학술 논문에 편향된 학습 데이터로 범용성 부족 |
| **구조 손실** | OCR 결과에서 표, 수식, 레이아웃 구조가 손실 |

OLMoCR의 해결 방향:

1. **오픈소스**: 모델 가중치, 학습 코드, 데이터 파이프라인 전부 공개
2. **End-to-end**: 이미지 → 구조화된 텍스트를 단일 모델로
3. **범용 문서**: 논문, 보고서, 영수증, 의료 문서 등 다양한 문서 유형
4. **효율적 학습**: 기존 VLM 위에 fine-tuning하여 학습 비용 절감

### 2. 아키텍처

OLMoCR은 **기존 VLM 위에 fine-tuning하는 전략**을 택한다. 자체 아키텍처를 설계하는 대신 검증된 Qwen2-VL을 backbone으로 사용한다.

```
                    OLMoCR 아키텍처
    ┌──────────────────────────────────────────┐
    │                                          │
    │  ┌────────────────────────────────────┐   │
    │  │  Qwen2-VL (7B) - Backbone         │   │
    │  │  ┌──────────┐   ┌──────────────┐  │   │
    │  │  │ ViT-G/14 │──▶│ Qwen2-7B LLM │  │   │
    │  │  │ (동결)    │   │ (LoRA 학습)   │  │   │
    │  │  └──────────┘   └──────────────┘  │   │
    │  └────────────────────────────────────┘   │
    │                                          │
    │  ┌────────────────────────────────────┐   │
    │  │  Document OCR 특화 학습 데이터       │   │
    │  │  - PDF → 이미지 + 텍스트 쌍          │   │
    │  │  - 합성 문서 데이터                   │   │
    │  │  - 구조화 출력 (Markdown, JSON)      │   │
    │  └────────────────────────────────────┘   │
    └──────────────────────────────────────────┘
```

### Qwen2-VL과의 관계

| 구성요소 | Qwen2-VL (Base) | OLMoCR |
|----------|-----------------|--------|
| Vision Encoder | ViT-G/14 | ViT-G/14 (동결) |
| LLM | Qwen2-7B | Qwen2-7B + LoRA |
| 학습 목표 | 범용 VLM | 문서 OCR 특화 |
| 출력 형식 | 자유 형식 | 구조화된 Markdown/JSON |
| 해상도 처리 | Dynamic Resolution | Dynamic Resolution 유지 |

---

## 수학적 원리

### 1. LoRA Fine-tuning

OLMoCR은 전체 파라미터를 학습하지 않고 LoRA(Low-Rank Adaptation)를 사용한다.

원본 가중치 $W_0 \in \mathbb{R}^{d \times k}$에 저랭크 분해를 추가:

$$
W = W_0 + \Delta W = W_0 + BA
$$

여기서 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$이고 rank $r \ll \min(d, k)$이다.

학습 가능 파라미터 수:

$$
\text{LoRA 파라미터} = r \times (d + k) \quad \text{vs} \quad \text{전체} = d \times k
$$

예를 들어 $d = k = 4096$, $r = 16$이면:
- 전체 파라미터: $4096 \times 4096 = 16.7M$
- LoRA 파라미터: $16 \times (4096 + 4096) = 131K$ (약 **0.8%**)

Forward pass:

$$
h = (W_0 + BA)x = W_0 x + BAx
$$

$W_0$은 동결하고 $B$, $A$만 학습하므로 메모리와 계산 비용이 대폭 절감된다.

### 2. 학습 손실 함수

OLMoCR의 학습 손실은 표준 autoregressive language modeling loss에 기반한다:

$$
\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P(y_t \mid y_{<t}, I, x)
$$

여기서:
- $I$: 입력 문서 이미지
- $x$: 프롬프트 (지시문)
- $y_t$: $t$번째 출력 토큰
- $T$: 출력 시퀀스 길이

다만, 프롬프트 부분의 토큰에 대해서는 loss를 계산하지 않는다 (label masking):

$$
\mathcal{L} = -\frac{1}{T_{\text{answer}}} \sum_{t \in \text{answer tokens}} \log P(y_t \mid y_{<t}, I, x)
$$

### 3. Anchor Text 기반 데이터 생성

PDF에서 프로그래밍적으로 추출한 텍스트 좌표를 "앵커"로 활용한다.

PDF 텍스트 요소 $a_i = (\text{text}_i, x_i, y_i, w_i, h_i)$가 주어지면:

1. **이미지 렌더링**: PDF 페이지를 고해상도 이미지로 변환
2. **텍스트 정렬**: 앵커의 좌표와 이미지 내 실제 텍스트 위치를 대응
3. **구조 추론**: 앵커들의 공간 관계로 표, 단, 제목 등 구조를 추론

공간 관계 판단 기준:

$$
\text{same\_line}(a_i, a_j) = \begin{cases} 1 & \text{if } |y_i - y_j| < \epsilon_y \\ 0 & \text{otherwise} \end{cases}
$$

$$
\text{same\_column}(a_i, a_j) = \begin{cases} 1 & \text{if } |x_i - x_j| < \epsilon_x \\ 0 & \text{otherwise} \end{cases}
$$

---

## 학습 데이터 구성 전략

OLMoCR의 학습 데이터 파이프라인은 3단계로 구성된다.

### 단계 1: PDF 수집 및 전처리

```python
import fitz  # PyMuPDF
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple
import io
import json


@dataclass
class AnchorText:
    """PDF에서 추출한 텍스트 앵커."""
    text: str
    x0: float      # 좌상단 x
    y0: float      # 좌상단 y
    x1: float      # 우하단 x
    y1: float      # 우하단 y
    font_size: float
    font_name: str
    is_bold: bool


def extract_anchors_from_pdf(pdf_path: str, page_num: int = 0) -> Tuple[Image.Image, List[AnchorText]]:
    """
    PDF 한 페이지에서 이미지와 텍스트 앵커를 추출한다.

    Args:
        pdf_path: PDF 파일 경로
        page_num: 페이지 번호 (0-indexed)

    Returns:
        page_image: 렌더링된 페이지 이미지
        anchors: 텍스트 앵커 리스트
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # 고해상도로 페이지 렌더링 (300 DPI)
    mat = fitz.Matrix(300 / 72, 300 / 72)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    page_image = Image.open(io.BytesIO(img_data)).convert("RGB")

    # 텍스트 블록 추출
    anchors = []
    blocks = page.get_text("dict")["blocks"]

    for block in blocks:
        if block["type"] != 0:  # 텍스트 블록만
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                anchor = AnchorText(
                    text=span["text"].strip(),
                    x0=span["bbox"][0],
                    y0=span["bbox"][1],
                    x1=span["bbox"][2],
                    y1=span["bbox"][3],
                    font_size=span["size"],
                    font_name=span["font"],
                    is_bold="Bold" in span["font"] or "bold" in span["font"],
                )
                if anchor.text:  # 빈 텍스트 제외
                    anchors.append(anchor)

    doc.close()
    return page_image, anchors


def anchors_to_reading_order(anchors: List[AnchorText], line_threshold: float = 5.0) -> List[AnchorText]:
    """
    앵커들을 읽기 순서(위→아래, 왼→오른)로 정렬한다.

    Args:
        anchors: 텍스트 앵커 리스트
        line_threshold: 같은 줄로 판단하는 y좌표 차이 임계값

    Returns:
        정렬된 앵커 리스트
    """
    if not anchors:
        return anchors

    # y좌표 기준으로 그룹핑 (같은 줄)
    sorted_by_y = sorted(anchors, key=lambda a: a.y0)

    lines = []
    current_line = [sorted_by_y[0]]

    for anchor in sorted_by_y[1:]:
        if abs(anchor.y0 - current_line[0].y0) < line_threshold:
            current_line.append(anchor)
        else:
            # 현재 줄을 x좌표로 정렬 후 저장
            current_line.sort(key=lambda a: a.x0)
            lines.append(current_line)
            current_line = [anchor]

    current_line.sort(key=lambda a: a.x0)
    lines.append(current_line)

    # 전체 줄을 y좌표 순으로 펼침
    ordered = []
    for line in lines:
        ordered.extend(line)

    return ordered
```

### 단계 2: 학습 데이터 포맷 생성

```python
from typing import Dict, Any


def create_ocr_training_sample(
    image_path: str,
    anchors: List[AnchorText],
    task_type: str = "full_ocr",
) -> Dict[str, Any]:
    """
    OCR 학습 데이터 샘플을 생성한다.

    다양한 태스크 유형에 맞는 프롬프트-응답 쌍을 만든다.

    Args:
        image_path: 문서 이미지 경로
        anchors: 정렬된 텍스트 앵커 리스트
        task_type: "full_ocr" | "table_extract" | "key_value" | "summary"

    Returns:
        학습 데이터 딕셔너리
    """
    prompts = {
        "full_ocr": "이 문서의 모든 텍스트를 읽기 순서대로 추출해라. "
                     "표가 있으면 마크다운 표로, 수식은 LaTeX로 변환해라.",
        "table_extract": "이 문서에서 표를 찾아 마크다운 형식으로 변환해라.",
        "key_value": "이 문서에서 키-값 쌍을 추출해라. JSON 형식으로 출력해라.",
        "summary": "이 문서의 내용을 요약해라.",
    }

    # 앵커 텍스트를 구조화된 출력으로 변환
    if task_type == "full_ocr":
        answer = reconstruct_document_text(anchors)
    elif task_type == "key_value":
        answer = extract_key_values_from_anchors(anchors)
    else:
        answer = " ".join(a.text for a in anchors)

    return {
        "image": image_path,
        "conversations": [
            {"role": "user", "content": prompts[task_type]},
            {"role": "assistant", "content": answer},
        ],
    }


def reconstruct_document_text(anchors: List[AnchorText]) -> str:
    """
    앵커 텍스트를 문서 구조를 보존하면서 재구성한다.

    제목은 ##, 본문은 일반 텍스트, 줄바꿈 등을 적용한다.
    """
    lines = []
    prev_y = None
    prev_size = None

    for anchor in anchors:
        # 새로운 줄 판단
        if prev_y is not None and abs(anchor.y0 - prev_y) > 3.0:
            # 큰 간격이면 빈 줄 추가
            if abs(anchor.y0 - prev_y) > 20.0:
                lines.append("")

            # 폰트 크기로 제목 판단
            if anchor.font_size > 14 and anchor.is_bold:
                lines.append(f"## {anchor.text}")
            elif anchor.font_size > 12 and anchor.is_bold:
                lines.append(f"### {anchor.text}")
            else:
                lines.append(anchor.text)
        else:
            # 같은 줄에 이어 붙이기
            if lines:
                lines[-1] += " " + anchor.text
            else:
                lines.append(anchor.text)

        prev_y = anchor.y0
        prev_size = anchor.font_size

    return "\n".join(lines)


def extract_key_values_from_anchors(anchors: List[AnchorText]) -> str:
    """
    앵커에서 키-값 쌍을 추출한다.

    "키: 값" 또는 "키 ─ 값" 패턴을 탐지한다.
    """
    import re

    kv_pairs = {}

    for anchor in anchors:
        # "키: 값" 패턴
        match = re.match(r'^(.+?)\s*[:：]\s*(.+)$', anchor.text)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            kv_pairs[key] = value

    return json.dumps(kv_pairs, ensure_ascii=False, indent=2)
```

### 단계 3: 데이터 품질 필터링

```python
import re
from typing import Optional


class DocumentQualityFilter:
    """
    문서 OCR 학습 데이터의 품질을 검증하는 필터.

    낮은 품질의 데이터를 걸러내어 학습 효율을 높인다.
    """

    def __init__(
        self,
        min_text_length: int = 50,
        max_text_length: int = 10000,
        min_anchors: int = 5,
        max_repeat_ratio: float = 0.3,
        min_unique_chars: int = 20,
    ):
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.min_anchors = min_anchors
        self.max_repeat_ratio = max_repeat_ratio
        self.min_unique_chars = min_unique_chars

    def check(self, anchors: List[AnchorText]) -> tuple[bool, Optional[str]]:
        """
        데이터 품질을 검증한다.

        Args:
            anchors: 텍스트 앵커 리스트

        Returns:
            (통과 여부, 실패 사유)
        """
        # 앵커 수 검사
        if len(anchors) < self.min_anchors:
            return False, f"앵커 수 부족: {len(anchors)} < {self.min_anchors}"

        # 전체 텍스트 길이
        full_text = " ".join(a.text for a in anchors)
        if len(full_text) < self.min_text_length:
            return False, f"텍스트 너무 짧음: {len(full_text)}"
        if len(full_text) > self.max_text_length:
            return False, f"텍스트 너무 김: {len(full_text)}"

        # 반복 텍스트 비율 검사
        texts = [a.text for a in anchors]
        unique_texts = set(texts)
        repeat_ratio = 1.0 - len(unique_texts) / len(texts)
        if repeat_ratio > self.max_repeat_ratio:
            return False, f"반복 비율 초과: {repeat_ratio:.2f}"

        # 고유 문자 수 검사
        unique_chars = len(set(full_text))
        if unique_chars < self.min_unique_chars:
            return False, f"고유 문자 부족: {unique_chars}"

        # 깨진 인코딩 검사
        garbled_pattern = re.compile(r'[\\x80-\\xff]{3,}')
        if garbled_pattern.search(full_text):
            return False, "깨진 인코딩 탐지"

        return True, None


# ---- 사용 예시 ----
quality_filter = DocumentQualityFilter()

# 품질 검사 실행
# passed, reason = quality_filter.check(anchors)
# if passed:
#     sample = create_ocr_training_sample(image_path, anchors)
#     training_data.append(sample)
# else:
#     print(f"필터링됨: {reason}")
```

---

## Qwen-VL과의 차이점

OLMoCR은 Qwen2-VL과 아키텍처를 공유하지만 목적과 학습이 다르다.

### 차이점 비교

| 측면 | Qwen2-VL | OLMoCR |
|------|----------|--------|
| **목적** | 범용 VLM (대화, 추론, 분석) | 문서 OCR 특화 |
| **학습 데이터** | 웹 이미지-텍스트 쌍 + 다양한 태스크 | PDF 문서 + OCR 데이터 집중 |
| **출력 형식** | 자유 형식 자연어 | 구조화된 Markdown, JSON |
| **학습 방식** | 처음부터 사전학습 | Qwen2-VL 위에 LoRA fine-tuning |
| **Vision Encoder** | 학습 포함 | 동결 (freeze) |
| **특화 능력** | 범용적 이미지 이해 | 표, 수식, 레이아웃 구조 추출 |
| **학습 비용** | 수천 GPU-hours | 수십 GPU-hours |

### OLMoCR의 학습 전략 핵심

1. **PDF-native 데이터**: 웹 크롤링 대신 PDF에서 직접 학습 데이터 생성
2. **Anchor 기반 정답 생성**: OCR 엔진 대신 PDF 내장 텍스트를 ground truth로 사용
3. **다양한 태스크 혼합**: full OCR, 표 추출, QA, 요약 등을 혼합 학습
4. **구조 보존 출력**: 단순 텍스트 나열이 아닌 Markdown 구조로 출력하도록 학습

---

## 모델 추론 파이프라인

```python
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from typing import Dict, Any, List


class OLMoCRPipeline:
    """
    OLMoCR 추론 파이프라인.

    문서 이미지를 입력받아 구조화된 텍스트를 출력한다.
    """

    # 태스크별 시스템 프롬프트
    TASK_PROMPTS = {
        "full_ocr": (
            "아래 문서 이미지의 모든 텍스트를 읽기 순서대로 추출해라.\n"
            "규칙:\n"
            "1. 제목은 ## 마크다운 헤더로\n"
            "2. 표는 마크다운 표로\n"
            "3. 수식은 LaTeX로\n"
            "4. 줄바꿈과 단락 구분을 유지해라"
        ),
        "table_extract": (
            "이 문서에서 모든 표를 찾아 마크다운 형식으로 변환해라.\n"
            "각 표에 대해:\n"
            "1. 표의 제목이 있으면 먼저 적어라\n"
            "2. 마크다운 표 형식으로 출력해라\n"
            "3. 셀 병합이 있으면 적절히 처리해라"
        ),
        "key_value": (
            "이 문서에서 모든 키-값 쌍을 추출하여 JSON으로 출력해라.\n"
            "양식의 필드명이 키, 기재된 내용이 값이다."
        ),
    }

    def __init__(
        self,
        model_name: str = "allenai/olmocr-7b",
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        task: str = "full_ocr",
        custom_prompt: str = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """
        문서 이미지에 대해 OCR을 수행한다.

        Args:
            image: PIL 이미지
            task: "full_ocr" | "table_extract" | "key_value"
            custom_prompt: 사용자 정의 프롬프트 (None이면 task 기본 프롬프트 사용)
            max_new_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도 (0.0 = greedy)

        Returns:
            추출된 텍스트
        """
        prompt = custom_prompt or self.TASK_PROMPTS.get(task, self.TASK_PROMPTS["full_ocr"])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.model.device)

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_config["temperature"] = temperature

        output_ids = self.model.generate(**inputs, **generation_config)

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        result = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return result.strip()

    def batch_process(
        self,
        images: List[Image.Image],
        task: str = "full_ocr",
        max_new_tokens: int = 4096,
    ) -> List[str]:
        """
        여러 이미지를 순차 처리한다.

        메모리 제약 때문에 배치 처리 대신 순차 처리를 기본으로 한다.
        """
        results = []
        for i, image in enumerate(images):
            try:
                result = self(image, task=task, max_new_tokens=max_new_tokens)
                results.append(result)
            except Exception as e:
                results.append(f"[ERROR] 페이지 {i}: {str(e)}")
        return results


# ---- 사용 예시 ----
# pipeline = OLMoCRPipeline()
# image = Image.open("document.png").convert("RGB")
# text = pipeline(image, task="full_ocr")
# print(text)
```

---

## 문서 유형별 처리 전략

OLMoCR이 처리하는 다양한 문서 유형과 각 유형에 맞는 전략:

### 1. 학술 논문

```python
ACADEMIC_PROMPT = """
이 학술 논문 페이지를 분석해라:
1. 섹션 제목과 번호를 보존해라
2. 수식은 LaTeX로 변환해라 (인라인: $...$, 블록: $$...$$)
3. 그림/표 캡션을 [Figure N: 캡션] 형식으로 표시해라
4. 참고문헌 번호를 유지해라
"""
```

### 2. 비즈니스 문서 (보고서, 계약서)

```python
BUSINESS_PROMPT = """
이 비즈니스 문서의 내용을 추출해라:
1. 모든 텍스트를 읽기 순서대로
2. 표와 차트 데이터를 마크다운 표로
3. 서명/도장 영역은 [서명] 또는 [도장]으로 표시
4. 금액은 원래 형식 유지 (예: ₩1,000,000)
"""
```

### 3. 양식 문서 (신청서, 청구서)

```python
FORM_PROMPT = """
이 양식 문서에서 모든 필드를 JSON으로 추출해라:
- 필드명 (인쇄된 레이블)이 키
- 기재된 내용 (수기 또는 타이핑)이 값
- 체크박스는 true/false로
- 빈 필드는 null로
"""
```

---

## OLMoCR 학습 데이터 통계

Allen AI가 공개한 학습 데이터 구성:

| 데이터 소스 | 문서 수 | 페이지 수 | 설명 |
|------------|---------|----------|------|
| Common Crawl PDFs | ~2M | ~50M | 웹에서 수집한 PDF 문서 |
| arXiv | ~300K | ~5M | 학술 논문 |
| USPTO | ~200K | ~10M | 미국 특허 문서 |
| Government docs | ~100K | ~3M | 정부 공개 문서 |
| Synthetic | ~500K | ~500K | 합성 문서 데이터 |

### 합성 데이터 생성 전략

```python
from PIL import Image, ImageDraw, ImageFont
import random
from typing import List, Tuple


class SyntheticDocumentGenerator:
    """
    합성 문서 이미지를 생성하는 클래스.

    실제 PDF가 부족한 경우 학습 데이터를 보충한다.
    """

    def __init__(
        self,
        width: int = 2480,    # A4 at 300 DPI
        height: int = 3508,
        margin: int = 200,
        font_path: str = None,
    ):
        self.width = width
        self.height = height
        self.margin = margin
        self.font_path = font_path

    def generate_table_document(
        self,
        rows: int = 5,
        cols: int = 4,
        cell_data: List[List[str]] = None,
    ) -> Tuple[Image.Image, str]:
        """
        표가 포함된 합성 문서를 생성한다.

        Args:
            rows: 행 수
            cols: 열 수
            cell_data: 셀 데이터 (None이면 랜덤 생성)

        Returns:
            (이미지, 마크다운 정답)
        """
        img = Image.new("RGB", (self.width, self.height), "white")
        draw = ImageDraw.Draw(img)

        # 표 크기 계산
        table_width = self.width - 2 * self.margin
        table_height = rows * 60
        cell_w = table_width // cols
        cell_h = 60

        start_x = self.margin
        start_y = self.margin + 100  # 제목 공간

        # 제목 그리기
        draw.text((self.margin, self.margin), "Sample Table Document", fill="black")

        # 표 그리기
        for r in range(rows + 1):
            y = start_y + r * cell_h
            draw.line([(start_x, y), (start_x + table_width, y)], fill="black", width=2)

        for c in range(cols + 1):
            x = start_x + c * cell_w
            draw.line([(x, start_y), (x, start_y + table_height)], fill="black", width=2)

        # 셀 데이터 채우기
        if cell_data is None:
            cell_data = [
                [f"Cell({r},{c})" for c in range(cols)]
                for r in range(rows)
            ]

        markdown_rows = []
        for r, row_data in enumerate(cell_data):
            for c, text in enumerate(row_data):
                x = start_x + c * cell_w + 10
                y = start_y + r * cell_h + 20
                draw.text((x, y), text, fill="black")

            markdown_rows.append("| " + " | ".join(row_data) + " |")
            if r == 0:
                markdown_rows.append("|" + "---|" * cols)

        markdown = "\n".join(markdown_rows)

        return img, markdown

    def generate_form_document(
        self,
        fields: dict,
    ) -> Tuple[Image.Image, str]:
        """
        양식 문서를 생성한다.

        Args:
            fields: {"필드명": "값"} 딕셔너리

        Returns:
            (이미지, JSON 정답)
        """
        img = Image.new("RGB", (self.width, self.height), "white")
        draw = ImageDraw.Draw(img)

        y_pos = self.margin
        for field_name, field_value in fields.items():
            # 필드명 (인쇄체 스타일)
            draw.text((self.margin, y_pos), f"{field_name}:", fill="gray")
            # 필드값 (수기 스타일 - 파란색)
            draw.text((self.margin + 300, y_pos), field_value, fill="blue")
            # 밑줄
            draw.line(
                [(self.margin + 300, y_pos + 30), (self.width - self.margin, y_pos + 30)],
                fill="gray", width=1
            )
            y_pos += 60

        answer_json = json.dumps(fields, ensure_ascii=False, indent=2)
        return img, answer_json
```

---

## 성능 벤치마크

| 벤치마크 | OLMoCR-7B | Qwen2-VL-7B | GPT-4o | Marker |
|----------|-----------|-------------|--------|--------|
| DocVQA (ANLS) | 91.2 | 94.5 | 92.8 | - |
| ChartQA | 80.1 | 87.3 | 85.7 | - |
| PDF Text Extraction | **95.8** | 89.3 | 91.0 | 88.7 |
| Table Recognition F1 | **93.4** | 90.1 | 91.5 | 85.2 |
| Layout Accuracy | **94.1** | 88.7 | 90.3 | 82.4 |

주목할 점: OLMoCR은 DocVQA 같은 범용 VQA에서는 Qwen2-VL에 밀리지만, **PDF 텍스트 추출, 표 인식, 레이아웃 정확도** 등 문서 OCR 핵심 지표에서 우위를 보인다. 이게 특화 학습의 효과다.

---

## 한계점과 미래 방향

| 한계 | 상세 | 개선 방향 |
|------|------|----------|
| 손글씨 인식 | PDF 기반 학습이라 손글씨 데이터 부족 | 손글씨 합성 데이터 추가 |
| 비영어 문서 | 영어 문서 중심 학습 | 다국어 PDF 데이터 확대 |
| 스캔 품질 | 고품질 PDF 기반이라 저품질 스캔에 취약 | 이미지 증강(노이즈, 블러, 왜곡) |
| 긴 문서 | 페이지 단위 처리, 페이지 간 문맥 부재 | 멀티페이지 처리 메커니즘 |
| 도메인 특화 | 범용 문서 위주, 의료/법률 문서 취약 | 도메인별 fine-tuning |

---

## 용어 체크리스트

아래 용어들을 설명할 수 있으면 이 챕터를 이해한 거다.

- [ ] **Document OCR**: 전통 OCR과 Document OCR의 차이를 설명할 수 있는가?
- [ ] **Page-level Understanding**: 영역별 처리 대비 페이지 레벨 이해의 장점을 설명할 수 있는가?
- [ ] **Layout-aware**: 레이아웃 인식이 OCR 정확도에 미치는 영향을 설명할 수 있는가?
- [ ] **Anchor Text**: PDF에서 앵커 텍스트를 추출하는 방법과 학습 데이터 생성에서의 역할을 설명할 수 있는가?
- [ ] **LoRA Fine-tuning**: OLMoCR이 전체 학습 대신 LoRA를 택한 이유와 파라미터 효율을 계산할 수 있는가?
- [ ] **OLMoCR vs Qwen2-VL**: 두 모델의 목적, 학습 데이터, 출력 형식 차이를 비교할 수 있는가?
- [ ] **합성 데이터**: 문서 OCR에서 합성 데이터가 필요한 이유와 생성 전략을 설명할 수 있는가?
- [ ] **품질 필터링**: 학습 데이터 품질 필터의 주요 기준을 나열할 수 있는가?
