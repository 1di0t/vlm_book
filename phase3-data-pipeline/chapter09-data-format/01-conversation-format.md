---
---

# 9.1 Conversation Format 변환

VLM 파인튜닝의 첫 관문은 데이터 포맷이다. 이미지-텍스트 쌍을 모델이 이해하는 대화 형식으로 변환해야 학습이 가능하다. 포맷이 틀리면 모델은 아무것도 배우지 못한다. 이 절에서는 주요 VLM 모델별 Conversation Format을 파헤치고, 의료 문서 OCR에 특화된 변환 파이프라인을 구축한다.

---

## 핵심 용어

| 용어 | 정의 | 예시/비고 |
|------|------|-----------|
| **Chat Template** | 모델이 대화를 파싱하기 위한 구조화된 템플릿 | Qwen2-VL ChatML, LLaVA format |
| **System Role** | 모델의 행동 지침을 정의하는 시스템 메시지 | "너는 의료 문서 OCR 전문가다" |
| **User/Assistant Role** | 사용자 입력과 모델 응답을 구분하는 역할 태그 | user → 질문, assistant → 답변 |
| **Special Token** | 역할 경계·이미지 위치 등을 표시하는 제어 토큰 | `<image>`, `<\|im_start\|>`, `<\|im_end\|>` |
| **Image Token** | 이미지가 삽입될 위치를 나타내는 플레이스홀더 토큰 | `<image>`, `<img>`, `<|image|>` |
| **Multi-turn** | 여러 차례의 질의-응답이 이어지는 대화 구조 | OCR → 검증 → 수정 흐름 |
| **Loss Mask** | 학습 시 손실 계산에서 제외할 토큰 범위 지정 | system/user 토큰은 mask 처리 |

---

## 9.1.1 VLM 대화 포맷 개요

### 왜 포맷이 중요한가?

VLM은 사전학습 단계에서 특정 대화 템플릿을 학습한다. 파인튜닝할 때 이 템플릿을 정확히 따라야 모델이 역할 구분과 이미지 위치를 올바르게 인식한다.

```
잘못된 포맷 → 모델이 역할 경계를 못 찾음 → 엉뚱한 출력
올바른 포맷 → 역할 경계 + 이미지 위치 정확 → 의도된 학습
```

### 주요 모델별 포맷 비교

| 모델 | 템플릿 형식 | 이미지 토큰 | 특수 토큰 |
|------|------------|------------|-----------|
| Qwen2-VL | ChatML | `<image>` | `<\|im_start\|>`, `<\|im_end\|>` |
| LLaVA-1.5 | Vicuna | `<image>` | `USER:`, `ASSISTANT:` |
| LLaVA-NeXT | Llama3 Chat | `<image>` | `<\|begin_of_text\|>`, `<\|eot_id\|>` |
| InternVL2 | InternLM2 | `<img>...</img>` | `<\|im_start\|>`, `<\|im_end\|>` |
| Phi-3-Vision | Phi-3 | `<image_1>` | `<\|user\|>`, `<\|assistant\|>` |

---

## 9.1.2 Qwen2-VL 형식 (ChatML)

Qwen2-VL은 ChatML(Chat Markup Language)을 사용한다. 구조가 명확하고 멀티턴 확장이 쉽다.

### 기본 구조

```
<|im_start|>system
{시스템 메시지}<|im_end|>
<|im_start|>user
<image>
{사용자 질문}<|im_end|>
<|im_start|>assistant
{모델 응답}<|im_end|>
```

### 멀티턴 예시 (의료 문서 OCR)

```
<|im_start|>system
너는 의료 문서 OCR 전문가다. 이미지에서 텍스트를 추출하고 구조화된 JSON으로 응답해라.<|im_end|>
<|im_start|>user
<image>
이 진단서에서 환자 정보를 추출해줘.<|im_end|>
<|im_start|>assistant
{
  "환자명": "홍길동",
  "생년월일": "1985-03-15",
  "진단코드": "J06.9",
  "진단명": "급성 상기도 감염"
}<|im_end|>
<|im_start|>user
진단코드 J06.9의 정확도는 어떻게 판단한거야?<|im_end|>
<|im_start|>assistant
진단서 중앙부에 "상병코드: J06.9"가 명확하게 인쇄되어 있다. OCR 신뢰도 0.97로, 활자체이며 번짐이나 가림이 없어 고신뢰도로 판단했다.<|im_end|>
```

### Qwen2-VL JSON 데이터 구조

```json
{
  "id": "medical_001",
  "conversations": [
    {
      "role": "system",
      "content": "너는 의료 문서 OCR 전문가다."
    },
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "data/medical/diag_001.png"},
        {"type": "text", "text": "이 진단서에서 환자 정보를 추출해줘."}
      ]
    },
    {
      "role": "assistant",
      "content": "{\"환자명\": \"홍길동\", \"진단코드\": \"J06.9\"}"
    }
  ]
}
```

---

## 9.1.3 LLaVA 형식

LLaVA는 Vicuna 스타일과 Llama3 스타일 두 가지를 사용한다.

### LLaVA-1.5 (Vicuna 스타일)

```
A chat between a curious user and an AI assistant.

USER: <image>
이 처방전에서 약품명과 용량을 추출해줘.
ASSISTANT: 1. 아목시실린 500mg - 1일 3회
2. 이부프로펜 200mg - 1일 2회 (식후)
```

### LLaVA-NeXT (Llama3 스타일)

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

너는 의료 문서 OCR 전문가다.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

<image>
이 처방전에서 약품명과 용량을 추출해줘.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"medications": [{"name": "아목시실린", "dose": "500mg"}]}<|eot_id|>
```

### LLaVA JSON 데이터 구조

```json
{
  "id": "medical_002",
  "image": "data/medical/prescription_001.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n이 처방전에서 약품명과 용량을 추출해줘."
    },
    {
      "from": "gpt",
      "value": "{\"medications\": [{\"name\": \"아목시실린\", \"dose\": \"500mg\"}]}"
    }
  ]
}
```

---

## 9.1.4 범용 Conversation Format

모델마다 포맷이 다르면 데이터 관리가 지옥이 된다. 내부적으로는 범용 포맷을 하나 정의하고, 모델별 변환은 마지막 단계에서 수행하는 게 맞다.

### 범용 스키마 정의

```json
{
  "id": "unique_sample_id",
  "metadata": {
    "source": "hospital_A",
    "document_type": "진단서",
    "language": "ko",
    "image_resolution": [2480, 3508],
    "created_at": "2025-01-15"
  },
  "images": [
    {
      "path": "data/medical/diag_001.png",
      "position": 0
    }
  ],
  "conversations": [
    {
      "role": "system",
      "content": "너는 의료 문서 OCR 전문가다."
    },
    {
      "role": "user",
      "content": "이 진단서에서 진단코드를 추출해줘.",
      "image_indices": [0]
    },
    {
      "role": "assistant",
      "content": "{\"diagnosis_code\": \"J06.9\", \"diagnosis_name\": \"급성 상기도 감염\"}"
    }
  ]
}
```

### 범용 → 모델별 변환 흐름

```
Raw Annotation (COCO, 자체 포맷 등)
         ↓
    범용 Conversation Format (내부 표준)
         ↓
    ┌────┼────┬────────────┐
    ↓    ↓    ↓            ↓
 Qwen2  LLaVA InternVL  Phi-3
```

---

## 수학적 원리

### Loss Masking

VLM 학습에서 모든 토큰에 loss를 걸면 안 된다. System/User 역할의 토큰은 "이미 주어진 조건"이므로 loss에서 제외하고, Assistant 토큰에만 loss를 적용한다.

전체 시퀀스 $\mathbf{x} = (x_1, x_2, \ldots, x_T)$에 대해 마스크 $\mathbf{m} = (m_1, m_2, \ldots, m_T)$를 정의한다:

$$
m_t = \begin{cases} 1 & \text{if } x_t \in \text{Assistant tokens} \\ 0 & \text{otherwise} \end{cases}
$$

마스크된 Cross-Entropy Loss:

$$
\mathcal{L} = -\frac{1}{\sum_{t=1}^{T} m_t} \sum_{t=1}^{T} m_t \cdot \log P(x_t \mid x_{<t})
$$

마스크 없이 학습하면 모델이 시스템 프롬프트를 "외워서 생성"하려는 경향이 생긴다. 의료 문서처럼 응답 정확도가 중요한 도메인에서는 치명적이다.

### 토큰 효율성

이미지 하나를 $N_{\text{img}}$개 토큰으로 표현할 때, 전체 시퀀스 대비 이미지 토큰 비율:

$$
r_{\text{img}} = \frac{N_{\text{img}}}{N_{\text{img}} + N_{\text{text}}}
$$

Qwen2-VL의 경우 Dynamic Resolution으로 이미지 토큰 수가 가변적이다:

$$
N_{\text{img}} = \left\lceil \frac{H}{P} \right\rceil \times \left\lceil \frac{W}{P} \right\rceil
$$

여기서 $P$는 패치 크기(14px). 고해상도 의료 문서(A4, 2480×3508)의 경우:

$$
N_{\text{img}} = \left\lceil \frac{2480}{14} \right\rceil \times \left\lceil \frac{3508}{14} \right\rceil = 178 \times 251 = 44{,}678 \text{ tokens}
$$

이건 너무 크므로 실무에서는 리사이즈나 타일링으로 줄인다.

### 대화 턴 수와 정보 밀도

$K$개 턴의 대화에서 턴당 평균 정보량을 $I_k$ (bits)라 하면, 총 정보량:

$$
I_{\text{total}} = \sum_{k=1}^{K} I_k
$$

각 턴의 정보 밀도:

$$
\rho_k = \frac{I_k}{N_k}
$$

여기서 $N_k$는 $k$번째 턴의 토큰 수. 정보 밀도가 높은 간결한 응답이 학습 효율이 좋다.

---

## 9.1.5 Special Token의 역할

### 토큰별 기능 정리

| Special Token | 역할 | 처리 방식 |
|---------------|------|-----------|
| `<image>` | 이미지 삽입 위치 표시 | Vision Encoder 출력으로 대체 |
| `<\|im_start\|>` | 역할 블록 시작 | 토크나이저에 추가, loss mask 경계 |
| `<\|im_end\|>` | 역할 블록 종료 | EOS와 유사, 생성 중단 신호 |
| `<pad>` | 배치 패딩용 | Attention mask = 0, loss mask = 0 |
| `<unk>` | 미등록 토큰 대체 | 발생 시 데이터 정제 필요 |

### 이미지 토큰 처리 흐름

```
텍스트: "이 <image> 에서 텍스트를 추출해줘"
         ↓ Tokenize
토큰 ID: [1234, 5, <IMG_TOKEN_ID>, 7, 2345, 89, 456, 23]
         ↓ Embedding
         [text_emb, text_emb, ■■■, text_emb, ...]
                               ↑
                    Vision Encoder 출력 (N_img × D)으로 대체
         ↓
최종 시퀀스: [text_emb, text_emb, img_emb_1, ..., img_emb_N, text_emb, ...]
```

---

## 9.1.6 의료 문서 OCR용 대화 포맷 설계

### 태스크별 대화 패턴

**1. 전체 텍스트 추출 (Full OCR)**
```json
{
  "role": "user",
  "content": "이 문서의 모든 텍스트를 추출해줘."
}
→
{
  "role": "assistant",
  "content": "서울대학교병원 진단서\n\n환자명: 홍길동\n생년월일: 1985-03-15\n..."
}
```

**2. 구조화 추출 (Structured Extraction)**
```json
{
  "role": "user",
  "content": "이 진단서에서 진단코드를 추출해줘."
}
→
{
  "role": "assistant",
  "content": "{\"diagnosis_code\": \"J06.9\", \"diagnosis_name\": \"급성 상기도 감염\", \"confidence\": 0.97}"
}
```

**3. 질의응답 (VQA)**
```json
{
  "role": "user",
  "content": "이 처방전에서 하루 복용 횟수가 가장 많은 약은?"
}
→
{
  "role": "assistant",
  "content": "아목시실린이 1일 3회로 가장 많다."
}
```

**4. 검증 및 수정 (Multi-turn Verification)**
```json
[
  {"role": "user", "content": "이 문서를 OCR 해줘."},
  {"role": "assistant", "content": "환자명: 홈길동 ..."},
  {"role": "user", "content": "'홈길동'이 맞아? 다시 확인해봐."},
  {"role": "assistant", "content": "확인 결과 '홍길동'이 맞다. '홈'→'홍' 수정한다."}
]
```

---

## 9.1.7 코드: DataFormatter 클래스

### 범용 포맷 변환 파이프라인

```python
"""
Chapter 9.1 - Conversation Format 변환 파이프라인
Raw annotation → 범용 format → 모델별 format 변환
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. 데이터 모델 정의
# ──────────────────────────────────────────────

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ImageInfo:
    """이미지 메타데이터"""
    path: str
    position: int = 0  # 대화 내 이미지 순서
    width: int = 0
    height: int = 0


@dataclass
class Message:
    """단일 대화 턴"""
    role: Role
    content: str
    image_indices: list[int] = field(default_factory=list)


@dataclass
class ConversationSample:
    """범용 Conversation Format - 내부 표준"""
    id: str
    images: list[ImageInfo]
    conversations: list[Message]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """데이터 유효성 검증"""
        if not self.id:
            raise ValueError("Sample ID가 비어있다.")
        if not self.conversations:
            raise ValueError(f"[{self.id}] 대화가 비어있다.")

        has_assistant = any(
            m.role == Role.ASSISTANT for m in self.conversations
        )
        if not has_assistant:
            raise ValueError(f"[{self.id}] Assistant 응답이 없다.")

        # 이미지 인덱스 범위 검증
        for msg in self.conversations:
            for idx in msg.image_indices:
                if idx >= len(self.images):
                    raise ValueError(
                        f"[{self.id}] 이미지 인덱스 {idx} 범위 초과 "
                        f"(총 {len(self.images)}개)"
                    )
        return True

    def to_dict(self) -> dict:
        """직렬화"""
        data = asdict(self)
        for msg in data["conversations"]:
            msg["role"] = msg["role"].value if isinstance(msg["role"], Role) else msg["role"]
        return data


# ──────────────────────────────────────────────
# 2. Raw Annotation → 범용 포맷 변환기
# ──────────────────────────────────────────────

class RawAnnotationConverter:
    """다양한 원본 어노테이션을 범용 ConversationSample로 변환"""

    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt

    def from_ocr_annotation(
        self,
        image_path: str,
        ocr_text: str,
        doc_type: str = "general",
        questions: list[dict] | None = None,
    ) -> ConversationSample:
        """
        OCR 어노테이션 → ConversationSample

        Args:
            image_path: 이미지 경로
            ocr_text: 정답 OCR 텍스트
            doc_type: 문서 유형
            questions: 추가 Q&A 쌍 리스트 [{"q": "...", "a": "..."}]
        """
        sample_id = self._generate_id(image_path)
        images = [ImageInfo(path=image_path, position=0)]

        conversations: list[Message] = []

        # System prompt
        if self.system_prompt:
            conversations.append(
                Message(role=Role.SYSTEM, content=self.system_prompt)
            )

        # 기본 OCR 턴
        conversations.append(
            Message(
                role=Role.USER,
                content="이 문서의 모든 텍스트를 추출해줘.",
                image_indices=[0],
            )
        )
        conversations.append(
            Message(role=Role.ASSISTANT, content=ocr_text)
        )

        # 추가 Q&A 턴
        if questions:
            for qa in questions:
                conversations.append(
                    Message(role=Role.USER, content=qa["q"])
                )
                conversations.append(
                    Message(role=Role.ASSISTANT, content=qa["a"])
                )

        return ConversationSample(
            id=sample_id,
            images=images,
            conversations=conversations,
            metadata={
                "source": "ocr_annotation",
                "document_type": doc_type,
            },
        )

    def from_coco_format(
        self,
        coco_annotation: dict,
        image_dir: str,
    ) -> list[ConversationSample]:
        """COCO 형식 어노테이션 → ConversationSample 리스트"""
        samples = []
        image_map = {
            img["id"]: img for img in coco_annotation["images"]
        }

        for ann in coco_annotation["annotations"]:
            img_info = image_map[ann["image_id"]]
            image_path = str(Path(image_dir) / img_info["file_name"])

            sample = ConversationSample(
                id=f"coco_{ann['id']}",
                images=[ImageInfo(
                    path=image_path,
                    width=img_info.get("width", 0),
                    height=img_info.get("height", 0),
                )],
                conversations=[
                    Message(
                        role=Role.USER,
                        content="이 이미지를 설명해줘.",
                        image_indices=[0],
                    ),
                    Message(
                        role=Role.ASSISTANT,
                        content=ann["caption"],
                    ),
                ],
                metadata={"source": "coco"},
            )
            samples.append(sample)

        return samples

    def from_structured_extraction(
        self,
        image_path: str,
        fields: dict[str, str],
        extraction_prompt: str = "이 문서에서 주요 정보를 구조화해서 추출해줘.",
    ) -> ConversationSample:
        """
        구조화 추출 어노테이션 → ConversationSample

        Args:
            image_path: 이미지 경로
            fields: {"필드명": "값"} 딕셔너리
            extraction_prompt: 추출 요청 프롬프트
        """
        sample_id = self._generate_id(image_path + "_struct")
        response_json = json.dumps(fields, ensure_ascii=False, indent=2)

        return ConversationSample(
            id=sample_id,
            images=[ImageInfo(path=image_path, position=0)],
            conversations=[
                Message(
                    role=Role.SYSTEM,
                    content=self.system_prompt,
                ) if self.system_prompt else None,
                Message(
                    role=Role.USER,
                    content=extraction_prompt,
                    image_indices=[0],
                ),
                Message(
                    role=Role.ASSISTANT,
                    content=response_json,
                ),
            ],
            metadata={"source": "structured_extraction"},
        )

    @staticmethod
    def _generate_id(seed: str) -> str:
        return hashlib.md5(seed.encode()).hexdigest()[:12]


# ──────────────────────────────────────────────
# 3. 범용 포맷 → 모델별 포맷 변환기 (Strategy Pattern)
# ──────────────────────────────────────────────

class ModelFormatConverter(ABC):
    """모델별 포맷 변환기 추상 클래스"""

    @abstractmethod
    def convert(self, sample: ConversationSample) -> dict:
        """범용 포맷 → 모델별 포맷 변환"""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        ...


class Qwen2VLConverter(ModelFormatConverter):
    """Qwen2-VL ChatML 형식 변환기"""

    IMAGE_TOKEN = "<image>"
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    def get_model_name(self) -> str:
        return "qwen2-vl"

    def convert(self, sample: ConversationSample) -> dict:
        messages = []
        for msg in sample.conversations:
            if msg is None:
                continue

            if msg.role == Role.SYSTEM:
                messages.append({
                    "role": "system",
                    "content": msg.content,
                })
            elif msg.role == Role.USER:
                content_parts = []
                # 이미지 토큰 삽입
                for img_idx in msg.image_indices:
                    img = sample.images[img_idx]
                    content_parts.append({
                        "type": "image",
                        "image": img.path,
                    })
                content_parts.append({
                    "type": "text",
                    "text": msg.content,
                })
                messages.append({
                    "role": "user",
                    "content": content_parts,
                })
            elif msg.role == Role.ASSISTANT:
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                })

        return {
            "id": sample.id,
            "conversations": messages,
        }

    def to_chat_string(self, sample: ConversationSample) -> str:
        """디버그용: 실제 ChatML 문자열 생성"""
        lines = []
        for msg in sample.conversations:
            if msg is None:
                continue
            lines.append(f"{self.IM_START}{msg.role.value}")
            if msg.image_indices:
                lines.append(self.IMAGE_TOKEN)
            lines.append(msg.content + self.IM_END)
        return "\n".join(lines)


class LLaVAConverter(ModelFormatConverter):
    """LLaVA 형식 변환기"""

    IMAGE_TOKEN = "<image>"

    def get_model_name(self) -> str:
        return "llava"

    def convert(self, sample: ConversationSample) -> dict:
        conversations = []
        first_user = True

        for msg in sample.conversations:
            if msg is None or msg.role == Role.SYSTEM:
                continue  # LLaVA-1.5는 system role 미지원

            if msg.role == Role.USER:
                text = msg.content
                # 첫 번째 user 메시지에 이미지 토큰 삽입
                if first_user and msg.image_indices:
                    text = f"{self.IMAGE_TOKEN}\n{text}"
                    first_user = False
                conversations.append({
                    "from": "human",
                    "value": text,
                })
            elif msg.role == Role.ASSISTANT:
                conversations.append({
                    "from": "gpt",
                    "value": msg.content,
                })

        result = {
            "id": sample.id,
            "conversations": conversations,
        }

        if sample.images:
            result["image"] = sample.images[0].path

        return result


class LLaVANextConverter(ModelFormatConverter):
    """LLaVA-NeXT (Llama3) 형식 변환기"""

    def get_model_name(self) -> str:
        return "llava-next"

    def convert(self, sample: ConversationSample) -> dict:
        messages = []
        for msg in sample.conversations:
            if msg is None:
                continue

            if msg.role == Role.SYSTEM:
                messages.append({
                    "role": "system",
                    "content": msg.content,
                })
            elif msg.role == Role.USER:
                content = msg.content
                if msg.image_indices:
                    content = f"<image>\n{content}"
                messages.append({
                    "role": "user",
                    "content": content,
                })
            elif msg.role == Role.ASSISTANT:
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                })

        return {
            "id": sample.id,
            "image": sample.images[0].path if sample.images else None,
            "messages": messages,
        }

    def to_chat_string(self, sample: ConversationSample) -> str:
        """Llama3 Chat 형식 문자열"""
        parts = ["<|begin_of_text|>"]
        for msg in sample.conversations:
            if msg is None:
                continue
            parts.append(f"<|start_header_id|>{msg.role.value}<|end_header_id|>\n")
            if msg.image_indices:
                parts.append("<image>")
            parts.append(f"{msg.content}<|eot_id|>")
        return "".join(parts)


# ──────────────────────────────────────────────
# 4. Loss Mask 생성기
# ──────────────────────────────────────────────

class LossMaskGenerator:
    """
    대화 포맷에서 Loss Mask를 생성한다.
    Assistant 토큰에만 loss를 적용하고, System/User 토큰은 마스킹한다.
    """

    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: HuggingFace 토크나이저 (AutoTokenizer)
        """
        self.tokenizer = tokenizer

    def generate_mask(
        self,
        sample: ConversationSample,
        converter: ModelFormatConverter,
    ) -> list[int]:
        """
        대화 샘플에 대한 loss mask 생성

        Returns:
            mask: 각 토큰에 대해 1(loss 계산) 또는 0(무시)
        """
        if isinstance(converter, Qwen2VLConverter):
            return self._generate_chatml_mask(sample)
        elif isinstance(converter, LLaVAConverter):
            return self._generate_llava_mask(sample)
        else:
            logger.warning(
                f"지원하지 않는 변환기: {converter.get_model_name()}. "
                "기본 마스크 생성."
            )
            return self._generate_default_mask(sample)

    def _generate_chatml_mask(
        self,
        sample: ConversationSample,
    ) -> list[int]:
        """ChatML 형식 loss mask"""
        full_text = ""
        mask_ranges: list[tuple[int, int, bool]] = []

        for msg in sample.conversations:
            if msg is None:
                continue

            header = f"<|im_start|>{msg.role.value}\n"
            start = len(full_text)
            full_text += header

            if msg.image_indices:
                full_text += "<image>\n"
            full_text += msg.content + "<|im_end|>\n"
            end = len(full_text)

            is_assistant = msg.role == Role.ASSISTANT
            mask_ranges.append((start, end, is_assistant))

        tokens = self.tokenizer.encode(full_text)
        mask = [0] * len(tokens)

        # 토큰-문자 매핑으로 assistant 범위에만 1 설정
        char_to_token = self._build_char_to_token_map(full_text, tokens)

        for start, end, is_assistant in mask_ranges:
            if is_assistant:
                for char_pos in range(start, end):
                    if char_pos in char_to_token:
                        token_idx = char_to_token[char_pos]
                        if token_idx < len(mask):
                            mask[token_idx] = 1

        return mask

    def _generate_llava_mask(
        self,
        sample: ConversationSample,
    ) -> list[int]:
        """LLaVA 형식 loss mask (간소화)"""
        # LLaVA는 "ASSISTANT:" 이후 토큰에만 loss
        full_text = ""
        for msg in sample.conversations:
            if msg is None or msg.role == Role.SYSTEM:
                continue
            if msg.role == Role.USER:
                full_text += f"USER: {msg.content}\n"
            elif msg.role == Role.ASSISTANT:
                full_text += f"ASSISTANT: {msg.content}\n"

        tokens = self.tokenizer.encode(full_text)
        mask = [0] * len(tokens)

        # 단순 전략: "ASSISTANT:" 토큰 이후부터 다음 "USER:" 까지 1
        assistant_token = self.tokenizer.encode("ASSISTANT:")
        in_assistant = False
        for i, token_id in enumerate(tokens):
            if self._matches_subsequence(tokens, i, assistant_token):
                in_assistant = True
                continue
            if in_assistant:
                mask[i] = 1

        return mask

    def _generate_default_mask(
        self,
        sample: ConversationSample,
    ) -> list[int]:
        """기본 마스크: assistant 내용만 토큰화해서 전체 mask 생성"""
        all_text = " ".join(
            m.content for m in sample.conversations if m is not None
        )
        tokens = self.tokenizer.encode(all_text)
        # 보수적으로 전체 1 (모든 토큰에 loss)
        return [1] * len(tokens)

    def _build_char_to_token_map(
        self,
        text: str,
        token_ids: list[int],
    ) -> dict[int, int]:
        """문자 위치 → 토큰 인덱스 매핑 구축"""
        mapping = {}
        current_char = 0
        for token_idx, token_id in enumerate(token_ids):
            token_str = self.tokenizer.decode([token_id])
            for _ in token_str:
                if current_char < len(text):
                    mapping[current_char] = token_idx
                    current_char += 1
        return mapping

    @staticmethod
    def _matches_subsequence(
        tokens: list[int],
        start: int,
        pattern: list[int],
    ) -> bool:
        if start + len(pattern) > len(tokens):
            return False
        return tokens[start:start + len(pattern)] == pattern


# ──────────────────────────────────────────────
# 5. 데이터 변환 파이프라인
# ──────────────────────────────────────────────

class DataFormatterPipeline:
    """
    End-to-End 데이터 포맷 변환 파이프라인

    Raw annotation → 범용 포맷 → 검증 → 모델별 포맷 → 저장
    """

    def __init__(
        self,
        system_prompt: str = "",
        target_model: str = "qwen2-vl",
    ):
        self.raw_converter = RawAnnotationConverter(system_prompt)
        self.model_converter = self._get_model_converter(target_model)
        self.stats = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "errors": [],
        }

    def _get_model_converter(self, model_name: str) -> ModelFormatConverter:
        converters = {
            "qwen2-vl": Qwen2VLConverter,
            "llava": LLaVAConverter,
            "llava-next": LLaVANextConverter,
        }
        if model_name not in converters:
            raise ValueError(
                f"지원하지 않는 모델: {model_name}. "
                f"가능한 값: {list(converters.keys())}"
            )
        return converters[model_name]()

    def process_ocr_batch(
        self,
        annotations: list[dict],
        output_path: str,
    ) -> None:
        """
        OCR 어노테이션 배치를 변환하여 JSONL로 저장

        Args:
            annotations: [{"image": "path", "text": "ocr결과", "type": "진단서", ...}]
            output_path: 출력 JSONL 경로
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        results = []
        for ann in annotations:
            self.stats["total"] += 1
            try:
                sample = self.raw_converter.from_ocr_annotation(
                    image_path=ann["image"],
                    ocr_text=ann["text"],
                    doc_type=ann.get("type", "general"),
                    questions=ann.get("questions"),
                )
                sample.validate()
                converted = self.model_converter.convert(sample)
                results.append(converted)
                self.stats["valid"] += 1

            except (ValueError, KeyError) as e:
                self.stats["invalid"] += 1
                self.stats["errors"].append(str(e))
                logger.warning(f"변환 실패: {e}")

        # JSONL 형식으로 저장
        with open(output, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(
            f"변환 완료: {self.stats['valid']}/{self.stats['total']} "
            f"({self.stats['invalid']} 실패) → {output_path}"
        )

    def process_structured_batch(
        self,
        annotations: list[dict],
        output_path: str,
    ) -> None:
        """
        구조화 추출 어노테이션 배치 변환

        Args:
            annotations: [{"image": "path", "fields": {"key": "val"}}]
            output_path: 출력 JSONL 경로
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        results = []
        for ann in annotations:
            self.stats["total"] += 1
            try:
                sample = self.raw_converter.from_structured_extraction(
                    image_path=ann["image"],
                    fields=ann["fields"],
                    extraction_prompt=ann.get(
                        "prompt",
                        "이 문서에서 주요 정보를 구조화해서 추출해줘.",
                    ),
                )
                sample.conversations = [
                    m for m in sample.conversations if m is not None
                ]
                sample.validate()
                converted = self.model_converter.convert(sample)
                results.append(converted)
                self.stats["valid"] += 1

            except (ValueError, KeyError) as e:
                self.stats["invalid"] += 1
                self.stats["errors"].append(str(e))
                logger.warning(f"변환 실패: {e}")

        with open(output, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(
            f"구조화 변환 완료: {self.stats['valid']}/{self.stats['total']} → {output_path}"
        )

    def get_stats(self) -> dict:
        return self.stats.copy()


# ──────────────────────────────────────────────
# 6. 사용 예시
# ──────────────────────────────────────────────

def example_usage():
    """파이프라인 사용 예시"""

    # 1. 파이프라인 초기화
    pipeline = DataFormatterPipeline(
        system_prompt="너는 의료 문서 OCR 전문가다. 정확하게 텍스트를 추출하고 구조화해라.",
        target_model="qwen2-vl",
    )

    # 2. OCR 어노테이션 배치
    ocr_annotations = [
        {
            "image": "data/medical/diag_001.png",
            "text": "서울대학교병원 진단서\n환자명: 홍길동\n진단코드: J06.9",
            "type": "진단서",
            "questions": [
                {
                    "q": "진단코드가 뭐야?",
                    "a": "J06.9 (급성 상기도 감염)",
                },
            ],
        },
        {
            "image": "data/medical/prescription_001.png",
            "text": "처방전\n아목시실린 500mg 1일 3회",
            "type": "처방전",
        },
    ]

    pipeline.process_ocr_batch(
        annotations=ocr_annotations,
        output_path="output/qwen2vl_ocr_train.jsonl",
    )

    # 3. 구조화 추출 배치
    struct_annotations = [
        {
            "image": "data/medical/diag_001.png",
            "fields": {
                "환자명": "홍길동",
                "생년월일": "1985-03-15",
                "진단코드": "J06.9",
                "진단명": "급성 상기도 감염",
            },
        },
    ]

    pipeline.process_structured_batch(
        annotations=struct_annotations,
        output_path="output/qwen2vl_struct_train.jsonl",
    )

    # 4. 통계 확인
    stats = pipeline.get_stats()
    print(f"처리 결과: {stats}")

    # 5. 모델별 변환 비교
    converter_raw = RawAnnotationConverter(
        system_prompt="너는 의료 문서 OCR 전문가다."
    )
    sample = converter_raw.from_ocr_annotation(
        image_path="data/test.png",
        ocr_text="테스트 OCR 결과",
    )

    for ModelConv in [Qwen2VLConverter, LLaVAConverter, LLaVANextConverter]:
        conv = ModelConv()
        result = conv.convert(sample)
        print(f"\n--- {conv.get_model_name()} ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    example_usage()
```

---

## 9.1.8 데이터 검증 및 품질 관리

### 자동 검증 체크리스트

```python
class ConversationValidator:
    """대화 포맷 데이터 품질 검증기"""

    def __init__(self, max_turns: int = 20, max_tokens: int = 8192):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.issues: list[str] = []

    def validate_sample(self, sample: ConversationSample) -> bool:
        """단일 샘플 종합 검증"""
        self.issues = []

        self._check_structure(sample)
        self._check_role_alternation(sample)
        self._check_content_quality(sample)
        self._check_image_consistency(sample)

        return len(self.issues) == 0

    def _check_structure(self, sample: ConversationSample) -> None:
        """구조적 무결성 검증"""
        if len(sample.conversations) > self.max_turns:
            self.issues.append(
                f"턴 수 초과: {len(sample.conversations)} > {self.max_turns}"
            )

        # 첫 번째 메시지가 system 또는 user인지
        first_msg = next(
            (m for m in sample.conversations if m is not None), None
        )
        if first_msg and first_msg.role == Role.ASSISTANT:
            self.issues.append("대화가 Assistant로 시작한다.")

    def _check_role_alternation(self, sample: ConversationSample) -> None:
        """역할 교대 규칙 검증 (system 제외)"""
        non_system = [
            m for m in sample.conversations
            if m is not None and m.role != Role.SYSTEM
        ]
        for i in range(1, len(non_system)):
            if non_system[i].role == non_system[i - 1].role:
                self.issues.append(
                    f"턴 {i}: 연속 {non_system[i].role.value} 역할"
                )

    def _check_content_quality(self, sample: ConversationSample) -> None:
        """내용 품질 검증"""
        for i, msg in enumerate(sample.conversations):
            if msg is None:
                self.issues.append(f"턴 {i}: None 메시지")
                continue
            if not msg.content or not msg.content.strip():
                self.issues.append(f"턴 {i}: 빈 내용 ({msg.role.value})")
            if len(msg.content) > 50000:
                self.issues.append(
                    f"턴 {i}: 내용 과도하게 김 ({len(msg.content)} chars)"
                )

    def _check_image_consistency(self, sample: ConversationSample) -> None:
        """이미지 참조 일관성 검증"""
        referenced_images = set()
        for msg in sample.conversations:
            if msg is not None:
                referenced_images.update(msg.image_indices)

        for i, img in enumerate(sample.images):
            if i not in referenced_images:
                self.issues.append(
                    f"이미지 {i} ({img.path})이 어떤 메시지에서도 참조되지 않는다."
                )

    def get_issues(self) -> list[str]:
        return self.issues.copy()
```

---

## 9.1.9 실전 팁

### 1. JSON 응답 포맷 가이드

의료 문서 OCR에서 구조화 응답을 요구할 때, 학습 데이터의 JSON 형식을 통일해야 한다. 모델이 들쭉날쭉한 JSON을 보면 출력도 불안정해진다.

```json
// 좋은 예: 일관된 스키마
{"patient_name": "홍길동", "diagnosis_code": "J06.9", "confidence": 0.97}

// 나쁜 예: 스키마 불일치
{"환자": "홍길동", "코드": "J06.9"}       // 샘플 A
{"patient_name": "이영희", "diag": "K21"} // 샘플 B → 키가 다름
```

### 2. 이미지 토큰 위치

이미지 토큰은 질문 **앞**에 넣는 게 대부분의 VLM에서 성능이 더 좋다. 이유는 attention이 이미지를 먼저 처리한 후 질문을 이미지에 맞춰 해석하기 때문이다.

```
# 권장
<image>\n이 진단서에서 진단코드를 추출해줘.

# 비권장
이 진단서에서 진단코드를 추출해줘.\n<image>
```

### 3. 멀티턴에서의 이미지 재참조

같은 이미지에 대해 여러 질문을 하는 멀티턴 구조에서, 2번째 턴부터는 이미지 토큰을 다시 넣지 않는 게 보통이다. 이미지는 이미 KV Cache에 있으므로 중복은 비효율적이다.

---

## 용어 체크리스트

학습 후 아래 질문에 답할 수 있는지 스스로 점검해라.

| # | 체크 항목 | 핵심 키워드 |
|---|----------|------------|
| 1 | Chat Template이 뭐고 왜 모델마다 다른가? | 사전학습 포맷, 역할 구분 |
| 2 | Qwen2-VL의 ChatML 형식을 직접 작성할 수 있는가? | `<\|im_start\|>`, `<\|im_end\|>` |
| 3 | LLaVA와 LLaVA-NeXT의 포맷 차이를 설명할 수 있는가? | Vicuna vs Llama3 Chat |
| 4 | Special Token이 왜 필요한가? | 역할 경계, 이미지 위치, 생성 중단 |
| 5 | Loss Mask를 왜 적용하는가? 안 하면 어떻게 되는가? | System/User 마스킹, 외워서 생성 문제 |
| 6 | 범용 포맷을 도입하는 이유는? | 모델 전환 용이, 데이터 일원 관리 |
| 7 | 의료 문서 OCR의 4가지 대화 패턴을 나열할 수 있는가? | Full OCR, 구조화 추출, VQA, 검증 |
| 8 | 이미지 토큰 위치가 성능에 미치는 영향은? | 질문 앞 배치 권장, attention 순서 |
| 9 | 멀티턴에서 이미지 토큰을 반복하지 않는 이유는? | KV Cache, 중복 비효율 |
| 10 | DataFormatter 파이프라인의 전체 흐름을 설명할 수 있는가? | Raw → 범용 → 검증 → 모델별 → JSONL |
