# Chapter 2.3: OCR 특화 VLM 분석

## 개요

문서 이해(Document Understanding)에 특화된 VLM 모델들을 분석한다. OCR-free 접근법으로 텍스트 추출과 구조 이해를 동시에 수행하는 최신 모델들을 다룬다.

---

## 1. 문서 이해 VLM 개요

### 1.1 전통적 OCR vs VLM 기반 OCR

| 특성 | 전통 OCR (Tesseract 등) | VLM 기반 OCR |
|------|------------------------|--------------|
| 파이프라인 | 다단계 (Detection → Recognition) | End-to-end |
| 레이아웃 이해 | 별도 처리 필요 | 통합 처리 |
| 테이블/그래프 | 취약 | 우수 |
| 컨텍스트 활용 | 없음 | 가능 |
| 다국어 | 언어별 모델 필요 | 통합 처리 |

### 1.2 주요 문서 이해 VLM

| 모델 | 출처 | 핵심 특징 | 성능 (DocVQA) |
|------|------|----------|---------------|
| **Nougat** | Meta AI | 학술 문서 특화, Markdown 출력 | - |
| **Pix2Struct** | Google | Screenshot parsing pretrain | 76.6% |
| **DocOwl** | Alibaba | OCR-free 문서 이해 | 62.2% |
| **Qwen2-VL** | Alibaba | Dynamic resolution | 94.5% |

---

## 2. Nougat: 학술 문서 이해

> **논문**: Blecher et al. (2023). "Nougat: Neural Optical Understanding for Academic Documents"
> - arXiv: https://arxiv.org/abs/2308.13418
> - GitHub: https://github.com/facebookresearch/nougat

### 2.1 아키텍처

```
PDF Page Image → Swin Transformer (Encoder) → mBART (Decoder) → Markdown
```

**구성요소:**
- **Encoder**: Swin Transformer (Visual feature 추출)
- **Decoder**: mBART 기반 텍스트 생성
- **총 파라미터**: 350M

```python
# Nougat 사용 예시 (HuggingFace)
from transformers import NougatProcessor, VisionEncoderDecoderModel
from PIL import Image

# 모델 로드
processor = NougatProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

# 이미지 처리
image = Image.open("paper_page.png")
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# 생성
outputs = model.generate(
    pixel_values,
    max_length=4096,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True
)

# 디코딩
markdown_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
print(markdown_text)
```

### 2.2 핵심 특징

1. **수식 처리**: LaTeX 수식을 정확하게 추출
2. **구조 보존**: 섹션, 표, 그림 캡션 구조 유지
3. **학습 데이터**: arXiv 논문 (LaTeX 소스 + PDF 쌍)

### 2.3 한계
- 스캔 품질이 낮은 문서에 취약
- 복잡한 테이블 구조 어려움
- 손글씨 인식 불가

---

## 3. Pix2Struct: Screenshot Parsing

> **논문**: Lee et al. (2022). "Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding"
> - arXiv: https://arxiv.org/abs/2210.03347
> - ICML 2023

### 3.1 핵심 아이디어

**Pretraining Task**: 웹페이지 스크린샷 → 간소화된 HTML 파싱

```
웹페이지 (렌더링된 이미지) → Pix2Struct → HTML 구조
```

**이점:**
- OCR, 캡셔닝, 레이아웃 이해를 동시 학습
- 웹의 다양한 시각적 요소 학습

### 3.2 아키텍처 특징

**Variable Resolution Input:**
- 고정 해상도 대신 가변 해상도 사용
- 종횡비 왜곡 방지

```python
class VariableResolutionViT(nn.Module):
    """
    Pix2Struct의 가변 해상도 ViT
    """
    def __init__(self, patch_size=16, hidden_size=768, max_patches=2048):
        super().__init__()
        self.patch_size = patch_size
        self.max_patches = max_patches

        self.patch_embed = nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.row_embed = nn.Embedding(max_patches, hidden_size // 2)
        self.col_embed = nn.Embedding(max_patches, hidden_size // 2)

    def forward(self, pixel_values):
        """
        pixel_values: (batch, channels, height, width) - 가변 크기
        """
        # Patch embedding
        patches = self.patch_embed(pixel_values)  # (B, D, H', W')

        # 2D positional embedding
        h, w = patches.shape[2], patches.shape[3]
        row_ids = torch.arange(h, device=patches.device)
        col_ids = torch.arange(w, device=patches.device)

        row_emb = self.row_embed(row_ids).unsqueeze(1)  # (H', 1, D/2)
        col_emb = self.col_embed(col_ids).unsqueeze(0)  # (1, W', D/2)

        pos_emb = torch.cat([
            row_emb.expand(-1, w, -1),
            col_emb.expand(h, -1, -1)
        ], dim=-1)  # (H', W', D)

        patches = patches.permute(0, 2, 3, 1)  # (B, H', W', D)
        patches = patches + pos_emb

        # Flatten
        patches = patches.flatten(1, 2)  # (B, N, D)

        return patches
```

### 3.3 Fine-tuning 성능

| 데이터셋 | Donut | Pix2Struct-Large |
|----------|-------|-----------------|
| DocVQA | 67.5% | 76.6% |
| ChartQA | 41.8% | 56.0% |
| InfographicVQA | 28.4% | 40.0% |

---

## 4. mPLUG-DocOwl: 범용 문서 이해

> **논문**: Ye et al. (2023). "mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding"
> - arXiv: https://arxiv.org/abs/2307.02499
> - GitHub: https://github.com/X-PLUG/mPLUG-DocOwl

### 4.1 아키텍처

mPLUG-Owl 기반에 문서 특화 학습:

```
Image → ViT → Visual Abstractor → LLM (LLaMA)
```

### 4.2 학습 데이터 구성

**Instruction Tuning Dataset:**
- 웹페이지
- 표 (테이블)
- 차트/그래프
- 자연 이미지
- 문서 스캔

```python
# DocOwl 스타일 instruction 예시
instruction_templates = [
    "이 문서에서 {field}를 추출해주세요.",
    "이 표의 내용을 JSON으로 변환해주세요.",
    "이 영수증의 총액은 얼마입니까?",
    "이 계약서의 당사자는 누구입니까?",
]
```

### 4.3 DocOwl 1.5 개선

- 고해상도 이미지 압축 기법
- DocVQA 82.2% 달성
- EMNLP 2024 발표

---

## 5. Qwen2-VL의 문서 이해 능력

### 5.1 Dynamic Resolution의 이점

문서 이미지는 다양한 종횡비를 가짐:
- A4 문서: 세로로 긴 형태
- 영수증: 매우 긴 세로
- 양식: 가로로 넓은 경우도 있음

```python
def compute_dynamic_resolution(image_size, patch_size=14, max_patches=16384):
    """
    Qwen2-VL 스타일의 동적 해상도 계산
    """
    h, w = image_size
    aspect_ratio = w / h

    # 종횡비 유지하면서 패치 수 최적화
    total_patches = (h // patch_size) * (w // patch_size)

    if total_patches > max_patches:
        # 리사이즈 필요
        scale = (max_patches / total_patches) ** 0.5
        new_h = int(h * scale) // patch_size * patch_size
        new_w = int(w * scale) // patch_size * patch_size
        return new_h, new_w

    # 패치 크기의 배수로 조정
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    return new_h, new_w
```

### 5.2 문서 OCR 성능

| 벤치마크 | Qwen2-VL-72B | GPT-4V |
|---------|--------------|--------|
| DocVQA | 94.5% | 88.4% |
| ChartQA | 83.0% | 78.5% |
| TextVQA | 84.3% | - |

---

## 6. 의료 문서 OCR 고려사항

### 6.1 의료 문서 특성

| 유형 | 특징 | 난이도 |
|------|------|--------|
| 처방전 | 손글씨, 약어 | 높음 |
| 진단서 | 양식 + 입력 텍스트 | 중간 |
| 영수증 | 구조화된 테이블 | 낮음-중간 |
| 검사결과지 | 표 + 참조값 | 중간 |

### 6.2 Pre-printed Line 문제

양식의 밑줄, 체크박스 등이 텍스트와 간섭:

```python
def preprocess_medical_form(image):
    """
    의료 양식 전처리: Pre-printed 라인 제거
    """
    import cv2
    import numpy as np

    # Grayscale 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 수평/수직 라인 검출
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)

    # 라인 제거 (인페인팅)
    lines_mask = cv2.add(horizontal_lines, vertical_lines)
    _, lines_mask = cv2.threshold(lines_mask, 127, 255, cv2.THRESH_BINARY)

    cleaned = cv2.inpaint(image, lines_mask, 3, cv2.INPAINT_TELEA)

    return cleaned
```

### 6.3 필수 필드 추출 전략

```python
MEDICAL_DOCUMENT_SCHEMA = {
    "진단서": {
        "required": ["환자명", "생년월일", "진단명", "진단일", "의사명", "병원명"],
        "optional": ["병명코드", "소견", "치료내용"]
    },
    "처방전": {
        "required": ["환자명", "약품명", "용량", "투약일수"],
        "optional": ["복용법", "주의사항"]
    },
    "영수증": {
        "required": ["총액", "진료일", "환자명"],
        "optional": ["항목별 금액", "할인", "본인부담금"]
    }
}

def extract_medical_fields(vlm_output, document_type):
    """
    VLM 출력에서 필수 필드 추출 및 검증
    """
    schema = MEDICAL_DOCUMENT_SCHEMA.get(document_type, {})
    required = schema.get("required", [])

    extracted = {}
    missing = []

    for field in required:
        value = extract_field_from_text(vlm_output, field)
        if value:
            extracted[field] = value
        else:
            missing.append(field)

    return {
        "extracted": extracted,
        "missing": missing,
        "complete": len(missing) == 0
    }
```

---

## 7. OCR VLM 선택 가이드

### 7.1 문서 유형별 권장 모델

| 문서 유형 | 1순위 | 2순위 | 이유 |
|----------|-------|-------|------|
| 학술 논문 (PDF) | Nougat | Qwen2-VL | 수식/LaTeX 처리 |
| 웹페이지 캡처 | Pix2Struct | Qwen2-VL | Screenshot pretrain |
| 일반 문서 | Qwen2-VL | DocOwl | Dynamic resolution |
| 테이블 중심 | Qwen2-VL | Pix2Struct | 구조 이해 |
| 의료 문서 | Qwen2-VL (fine-tuned) | - | 커스텀 학습 필요 |

### 7.2 Fine-tuning 권장 설정

**의료 문서 OCR Fine-tuning:**

```yaml
# Qwen2-VL Fine-tuning config
model:
  base: "Qwen/Qwen2-VL-7B-Instruct"
  vision_encoder: "freeze"  # 또는 low_lr
  lm: "lora"

lora:
  r: 64
  alpha: 128
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

training:
  learning_rate: 1e-4
  warmup_ratio: 0.03
  num_epochs: 3
  per_device_batch_size: 1
  gradient_accumulation_steps: 16
  max_seq_length: 8192  # 긴 문서 처리

data:
  image_resolution: "dynamic"
  max_image_tokens: 4096
```

---

## 핵심 참고 자료

### 논문
- **Nougat** (Blecher et al., 2023)
  - https://arxiv.org/abs/2308.13418
  - GitHub: https://github.com/facebookresearch/nougat

- **Pix2Struct** (Lee et al., 2022)
  - https://arxiv.org/abs/2210.03347
  - GitHub: https://github.com/google-research/pix2struct

- **DocOwl** (Ye et al., 2023)
  - https://arxiv.org/abs/2307.02499
  - GitHub: https://github.com/X-PLUG/mPLUG-DocOwl

- **Qwen2-VL** (Wang et al., 2024)
  - https://arxiv.org/abs/2409.12191

### 벤치마크
- **DocVQA**: https://www.docvqa.org/
- **ChartQA**: https://github.com/vis-nlp/ChartQA
- **InfographicVQA**: https://rrc.cvc.uab.es/?ch=17

---

## 핵심 요약

| 모델 | 최적 사용처 | 출력 형식 | 핵심 강점 |
|------|------------|----------|----------|
| Nougat | 학술 논문 | Markdown | 수식 처리 |
| Pix2Struct | 웹/UI | 구조화 텍스트 | 범용성 |
| DocOwl | 일반 문서 | 자연어 | Instruction following |
| Qwen2-VL | 고해상도 문서 | 자연어/JSON | Dynamic resolution |
