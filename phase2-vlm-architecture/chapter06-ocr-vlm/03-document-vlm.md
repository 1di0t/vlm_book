---
---

# 6.3 문서 이해 VLM

> 문서 이해(Document Understanding)는 단순히 텍스트를 읽는 것이 아니라,
> 문서의 **구조**(표, 레이아웃, 계층)까지 파악하여 정보를 추출하는 기술이다.
> 이 챕터에서는 DocTR, LayoutLM, Donut, Pix2Struct 네 모델의 아키텍처를 비교 분석한다.

---

## 핵심 용어

| 용어 | 정의 | 왜 중요한가 |
|------|------|-------------|
| **Document Understanding** | 문서 이미지에서 텍스트, 구조, 의미를 통합적으로 이해하는 기술 | 단순 OCR을 넘어 문서의 논리적 구조까지 파악해야 실용적 |
| **Table Recognition** | 문서 내 표의 행/열 구조를 인식하고 셀 내용을 추출하는 기술 | 비즈니스 문서의 핵심 정보는 대부분 표에 담겨 있다 |
| **Key-Value Extraction** | 양식 문서에서 필드명(키)과 기재 내용(값)의 쌍을 추출 | 보험 청구, 신청서 등 양식 문서 자동화의 기본 |
| **Layout Analysis** | 문서의 물리적 배치(제목, 본문, 각주, 표, 그림)를 분류하는 기술 | 같은 텍스트도 위치에 따라 의미가 다르다 |
| **OCR-free** | 별도 OCR 엔진 없이 이미지에서 직접 텍스트를 인식하는 방식 | 파이프라인 단순화, 에러 전파 방지 |

---

## 수학적 원리

### 1. 표 인식: 그래프 기반 모델링

표는 본질적으로 **그래프 구조**다. 각 셀이 노드이고, 인접 관계가 엣지다.

#### Adjacency Matrix

$N$개의 셀이 있을 때, 인접 행렬 $A \in \{0, 1\}^{N \times N}$을 정의한다:

$$
A_{ij} = \begin{cases}
1 & \text{if cell } i \text{ and cell } j \text{ are adjacent} \\
0 & \text{otherwise}
\end{cases}
$$

인접 관계는 4방향으로 나눌 수 있다:

$$
A = A_{\text{left}} + A_{\text{right}} + A_{\text{up}} + A_{\text{down}}
$$

각 방향별 인접:

$$
A_{\text{right}, ij} = \begin{cases}
1 & \text{if cell } j \text{ is the right neighbor of cell } i \\
0 & \text{otherwise}
\end{cases}
$$

#### 셀 영역 표현

각 셀 $c_i$는 바운딩 박스로 표현된다:

$$
c_i = (x_i^{\min}, y_i^{\min}, x_i^{\max}, y_i^{\max})
$$

두 셀의 수평 인접 조건:

$$
A_{\text{right}, ij} = 1 \iff \begin{cases}
x_j^{\min} \approx x_i^{\max} & \text{(수평 인접)} \\
|y_i^{\min} - y_j^{\min}| < \epsilon_y & \text{(같은 행)}
\end{cases}
$$

#### Graph Neural Network으로 표 구조 추론

표 인식을 GNN 문제로 공식화하면:

노드 특성: $h_i^{(0)} = \text{Encoder}(\text{cell}_i)$ (셀의 시각적/텍스트 특성)

메시지 패싱:

$$
h_i^{(\ell+1)} = \text{Update}\left(h_i^{(\ell)}, \text{Aggregate}\left(\{h_j^{(\ell)} : j \in \mathcal{N}(i)\}\right)\right)
$$

엣지 분류 (인접 여부 예측):

$$
P(\text{adjacent} \mid i, j) = \sigma\left(\text{MLP}\left([h_i^{(L)}; h_j^{(L)}; |h_i^{(L)} - h_j^{(L)}|]\right)\right)
$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class TableCellEncoder(nn.Module):
    """
    표 셀을 인코딩하는 모듈.

    셀의 시각적 특성(위치, 크기)과 텍스트 특성을 결합한다.
    """

    def __init__(self, text_dim: int = 256, spatial_dim: int = 64, hidden_dim: int = 256):
        super().__init__()

        # 공간 특성 인코딩 (바운딩 박스 4좌표 → spatial_dim)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, spatial_dim),
            nn.ReLU(),
            nn.Linear(spatial_dim, spatial_dim),
        )

        # 텍스트 + 공간 특성 결합
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + spatial_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, text_features: torch.Tensor, bboxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            text_features: (N, text_dim) — 셀 텍스트의 임베딩
            bboxes: (N, 4) — 정규화된 바운딩 박스 [x_min, y_min, x_max, y_max]

        Returns:
            cell_features: (N, hidden_dim)
        """
        spatial = self.spatial_encoder(bboxes)
        combined = torch.cat([text_features, spatial], dim=-1)
        return self.fusion(combined)


class TableGraphNetwork(nn.Module):
    """
    GNN 기반 표 구조 추론 모듈.

    셀 간의 인접 관계를 예측하여 표 구조를 복원한다.
    """

    def __init__(self, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()

        self.layers = nn.ModuleList([
            GraphConvLayer(hidden_dim) for _ in range(num_layers)
        ])

        # 엣지 분류기: 두 셀이 같은 행/열에 속하는지 예측
        self.row_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

        self.col_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        cell_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cell_features: (N, hidden_dim)
            edge_index: (2, E) — 후보 엣지 인덱스

        Returns:
            row_logits: (E, 1) — 같은 행 확률
            col_logits: (E, 1) — 같은 열 확률
        """
        h = cell_features

        for layer in self.layers:
            h = layer(h, edge_index)

        # 엣지 특성 계산
        src, dst = edge_index[0], edge_index[1]
        edge_features = torch.cat([
            h[src],
            h[dst],
            torch.abs(h[src] - h[dst]),  # 차이 특성
        ], dim=-1)

        row_logits = self.row_classifier(edge_features)
        col_logits = self.col_classifier(edge_features)

        return row_logits, col_logits


class GraphConvLayer(nn.Module):
    """
    간단한 Graph Convolution Layer.

    이웃 노드의 정보를 집계하여 노드 특성을 업데이트한다.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, hidden_dim)
            edge_index: (2, E)

        Returns:
            updated: (N, hidden_dim)
        """
        src, dst = edge_index[0], edge_index[1]
        N = x.size(0)

        # 이웃 메시지 집계 (평균)
        messages = x[src]  # (E, hidden_dim)
        agg = torch.zeros_like(x)  # (N, hidden_dim)
        count = torch.zeros(N, 1, device=x.device)

        agg.index_add_(0, dst, messages)
        count.index_add_(0, dst, torch.ones(src.size(0), 1, device=x.device))
        count = count.clamp(min=1)
        agg = agg / count

        # 업데이트
        updated = self.linear(torch.cat([x, agg], dim=-1))
        updated = self.norm(updated + x)  # Residual connection
        updated = F.relu(updated)

        return updated


# ---- 사용 예시 ----
def demo_table_graph():
    N = 12  # 3행 × 4열 = 12 셀
    hidden_dim = 256

    # 셀 특성 (텍스트 임베딩 + 위치)
    text_features = torch.randn(N, 256)
    bboxes = torch.rand(N, 4)  # 정규화된 좌표

    encoder = TableCellEncoder()
    cell_features = encoder(text_features, bboxes)

    # 후보 엣지: 모든 셀 쌍 (실제로는 거리 기반 필터링)
    edges = []
    for i in range(N):
        for j in range(N):
            if i != j:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t()  # (2, E)

    gnn = TableGraphNetwork(hidden_dim=hidden_dim)
    row_logits, col_logits = gnn(cell_features, edge_index)

    print(f"셀 수: {N}")
    print(f"엣지 수: {edge_index.shape[1]}")
    print(f"행 분류 logits: {row_logits.shape}")
    print(f"열 분류 logits: {col_logits.shape}")


demo_table_graph()
```

### 2. 좌표 정규화

문서 이미지의 크기는 제각각이다. 모델이 위치 정보를 일관되게 처리하려면 좌표를 정규화해야 한다.

원본 이미지 크기가 $(W, H)$일 때, 좌표 $(x, y)$를 정규화:

$$
\hat{x} = \frac{x}{W}, \quad \hat{y} = \frac{y}{H}
$$

정규화된 좌표 $(\hat{x}, \hat{y}) \in [0, 1]^2$.

바운딩 박스 정규화:

$$
\hat{b} = \left(\frac{x_{\min}}{W}, \frac{y_{\min}}{H}, \frac{x_{\max}}{W}, \frac{y_{\max}}{H}\right)
$$

**왜 정규화가 필요한가?**

1. 같은 문서를 300 DPI와 150 DPI로 스캔하면 좌표가 2배 차이
2. A4와 Letter 크기의 문서가 섞여 있으면 절대 좌표는 무의미
3. 정규화하면 "페이지 상단 1/3 위치"같은 상대적 위치를 학습 가능

추가로 **이산화(discretization)**를 적용하기도 한다:

$$
\bar{x} = \lfloor \hat{x} \times L \rfloor, \quad L = 1000
$$

이렇게 하면 좌표를 0~999 정수 토큰으로 변환하여 LLM의 어휘로 표현할 수 있다.

```python
import torch
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class BoundingBox:
    """문서 내 요소의 바운딩 박스."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str = ""
    text: str = ""


def normalize_bboxes(
    bboxes: List[BoundingBox],
    image_width: int,
    image_height: int,
    discretize: bool = False,
    num_bins: int = 1000,
) -> torch.Tensor:
    """
    바운딩 박스 좌표를 정규화한다.

    Args:
        bboxes: 바운딩 박스 리스트
        image_width: 이미지 너비 (픽셀)
        image_height: 이미지 높이 (픽셀)
        discretize: True이면 0~num_bins 정수로 이산화
        num_bins: 이산화 구간 수

    Returns:
        normalized: (N, 4) 텐서 — [x_min, y_min, x_max, y_max]
    """
    coords = torch.tensor([
        [b.x_min, b.y_min, b.x_max, b.y_max]
        for b in bboxes
    ], dtype=torch.float32)

    # 정규화: [0, 1]
    coords[:, [0, 2]] /= image_width
    coords[:, [1, 3]] /= image_height

    # 범위 클리핑
    coords = coords.clamp(0.0, 1.0)

    if discretize:
        coords = (coords * num_bins).long().clamp(0, num_bins - 1)

    return coords


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    두 바운딩 박스 세트 간 IoU를 계산한다.

    Args:
        box1: (N, 4) — [x_min, y_min, x_max, y_max]
        box2: (M, 4)

    Returns:
        iou: (N, M) IoU 행렬
    """
    # 교집합
    x_min = torch.max(box1[:, None, 0], box2[None, :, 0])
    y_min = torch.max(box1[:, None, 1], box2[None, :, 1])
    x_max = torch.min(box1[:, None, 2], box2[None, :, 2])
    y_max = torch.min(box1[:, None, 3], box2[None, :, 3])

    intersection = (x_max - x_min).clamp(min=0) * (y_max - y_min).clamp(min=0)

    # 합집합
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - intersection

    return intersection / union.clamp(min=1e-6)


# ---- 사용 예시 ----
bboxes = [
    BoundingBox(100, 50, 500, 80, "title", "진단서"),
    BoundingBox(100, 100, 300, 120, "field_name", "성명"),
    BoundingBox(310, 100, 500, 120, "field_value", "홍길동"),
    BoundingBox(100, 140, 300, 160, "field_name", "진단코드"),
    BoundingBox(310, 140, 500, 160, "field_value", "S72.0"),
]

normalized = normalize_bboxes(bboxes, image_width=600, image_height=800)
discretized = normalize_bboxes(bboxes, image_width=600, image_height=800, discretize=True)

print("정규화 좌표:")
for i, b in enumerate(bboxes):
    print(f"  [{b.label}] {b.text}: {normalized[i].tolist()}")

print("\n이산화 좌표 (0-999):")
for i, b in enumerate(bboxes):
    print(f"  [{b.label}] {b.text}: {discretized[i].tolist()}")
```

### 3. 레이아웃 분석 손실 함수

레이아웃 분석은 각 영역을 분류하는 문제다. 일반적으로 Focal Loss를 사용한다. 클래스 불균형이 심하기 때문이다 (본문 영역 >> 표 영역 >> 수식 영역).

$$
\text{FocalLoss}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

여기서:
- $p_t$: 정답 클래스에 대한 예측 확률
- $\alpha_t$: 클래스별 가중치 (희귀 클래스에 높은 가중치)
- $\gamma$: focusing parameter (보통 2.0), 쉬운 샘플의 영향을 줄임

```python
import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal Loss 계산.

    Args:
        logits: (N, C) — 클래스별 로짓
        targets: (N,) — 정답 클래스 인덱스
        alpha: (C,) — 클래스별 가중치. None이면 균등
        gamma: focusing parameter

    Returns:
        loss: 스칼라
    """
    probs = F.softmax(logits, dim=-1)
    targets_one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).float()

    # 정답 클래스의 확률
    pt = (probs * targets_one_hot).sum(dim=-1)  # (N,)

    # Focal 가중치
    focal_weight = (1 - pt) ** gamma

    # 클래스 가중치
    if alpha is not None:
        alpha_t = alpha[targets]
        focal_weight = focal_weight * alpha_t

    # Cross-entropy
    ce_loss = -torch.log(pt + 1e-8)

    loss = (focal_weight * ce_loss).mean()
    return loss


# 문서 레이아웃 클래스
LAYOUT_CLASSES = {
    0: "text",       # 본문 텍스트
    1: "title",      # 제목
    2: "table",      # 표
    3: "figure",     # 그림
    4: "formula",    # 수식
    5: "list",       # 리스트
    6: "header",     # 머리글
    7: "footer",     # 바닥글
}

# 클래스 불균형 가중치 (희귀 클래스에 높은 가중치)
alpha = torch.tensor([0.5, 1.0, 2.0, 2.0, 3.0, 1.5, 1.5, 1.5])
```

---

## 모델 비교: DocTR, LayoutLM, Donut, Pix2Struct

### 아키텍처 비교표

| 특성 | DocTR | LayoutLM v3 | Donut | Pix2Struct |
|------|-------|-------------|-------|------------|
| **접근법** | Detection + Recognition | Pre-training + Fine-tuning | OCR-free Encoder-Decoder | OCR-free Encoder-Decoder |
| **Vision Encoder** | ResNet/VGG | ViT (Linear Patch) | Swin Transformer | ViT |
| **텍스트 입력** | OCR 결과 필요 | OCR 결과 + 위치 | 불필요 (OCR-free) | 불필요 (OCR-free) |
| **LM Decoder** | CTC / Attention | BERT/RoBERTa | BART | T5 |
| **레이아웃 인식** | Detection 기반 | 좌표 임베딩 내장 | 암묵적 학습 | Screenshot parsing |
| **사전학습** | ImageNet | Masked Language/Image/Layout | Reading order prediction | Screenshot parsing |
| **파라미터 수** | ~30M | ~133M (base) | ~200M | ~300M |
| **추론 속도** | 빠름 | 보통 | 느림 | 느림 |

### 1. DocTR

전통적인 2단계 접근: Detection → Recognition.

```
이미지 → [텍스트 탐지 (Detection)]
              │
              ▼
         바운딩 박스들
              │
              ▼
     [텍스트 인식 (Recognition)]
              │
              ▼
       텍스트 + 좌표
```

**장점**: 가볍고 빠르다. CPU에서도 실시간 처리 가능.
**단점**: 구조 이해 없음. 표, 수식은 별도 후처리 필요.

```python
import torch
import torch.nn as nn
from typing import List, Tuple


class TextDetectionHead(nn.Module):
    """
    DocTR 스타일 텍스트 탐지 헤드.

    Feature map에서 텍스트 영역의 히트맵과 기하학 정보를 예측한다.
    """

    def __init__(self, in_channels: int = 512):
        super().__init__()

        # 텍스트 존재 확률 히트맵
        self.prob_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

        # 바운딩 박스 회귀 (4좌표)
        self.bbox_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 4, 1),
            nn.Sigmoid(),  # 0~1 정규화 좌표
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, C, H, W) — backbone feature map

        Returns:
            prob_map: (B, 1, H, W) — 텍스트 존재 확률
            bbox_map: (B, 4, H, W) — 바운딩 박스 오프셋
        """
        prob_map = self.prob_head(features)
        bbox_map = self.bbox_head(features)
        return prob_map, bbox_map


class CTCRecognitionHead(nn.Module):
    """
    CTC 기반 텍스트 인식 헤드.

    잘라낸 텍스트 이미지에서 문자 시퀀스를 인식한다.
    """

    def __init__(
        self,
        in_channels: int = 512,
        hidden_size: int = 256,
        num_classes: int = 7000,  # 한국어 포함 문자 수
    ):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, 1, W) — 잘라낸 텍스트 영역의 feature

        Returns:
            logits: (B, W, num_classes) — CTC 로짓
        """
        # (B, C, 1, W) → (B, W, C)
        features = features.squeeze(2).permute(0, 2, 1)

        output, _ = self.rnn(features)
        logits = self.classifier(output)

        return logits
```

### 2. LayoutLM v3

마이크로소프트가 개발한 문서 이해 사전학습 모델. **텍스트 + 이미지 + 레이아웃**을 통합한다.

핵심 아이디어: OCR 결과의 텍스트와 위치 좌표를 토큰 임베딩에 포함.

$$
h_i = \text{TextEmbed}(w_i) + \text{PosEmbed}(\hat{b}_i) + \text{ImageEmbed}(\text{patch}_i)
$$

여기서:
- $w_i$: $i$번째 텍스트 토큰
- $\hat{b}_i = (\hat{x}_{\min}, \hat{y}_{\min}, \hat{x}_{\max}, \hat{y}_{\max}, \hat{w}, \hat{h})$: 정규화된 바운딩 박스 6차원
- $\text{patch}_i$: 해당 위치의 이미지 패치

```python
import torch
import torch.nn as nn


class LayoutLMv3Embedding(nn.Module):
    """
    LayoutLM v3 스타일 임베딩 레이어.

    텍스트, 위치(2D), 이미지 패치를 결합한다.
    """

    def __init__(
        self,
        vocab_size: int = 50265,
        hidden_size: int = 768,
        max_2d_position: int = 1024,  # 이산화된 좌표 최대값
        patch_size: int = 16,
        image_size: int = 224,
    ):
        super().__init__()

        # 텍스트 임베딩
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)

        # 2D 위치 임베딩 (x, y, w, h 각각)
        self.x_position_embeddings = nn.Embedding(max_2d_position, hidden_size)
        self.y_position_embeddings = nn.Embedding(max_2d_position, hidden_size)
        self.h_position_embeddings = nn.Embedding(max_2d_position, hidden_size)
        self.w_position_embeddings = nn.Embedding(max_2d_position, hidden_size)

        # 이미지 패치 임베딩
        self.patch_embeddings = nn.Conv2d(
            3, hidden_size,
            kernel_size=patch_size, stride=patch_size
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, S) 텍스트 토큰 ID
            bbox: (B, S, 4) 이산화된 바운딩 박스 [x_min, y_min, x_max, y_max]
            pixel_values: (B, 3, H, W) 문서 이미지

        Returns:
            embeddings: (B, S + num_patches, hidden_size)
        """
        # 텍스트 임베딩
        text_emb = self.word_embeddings(input_ids)

        # 2D 위치 임베딩
        x_min, y_min, x_max, y_max = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
        w = (x_max - x_min).clamp(min=0)
        h = (y_max - y_min).clamp(min=0)

        pos_emb = (
            self.x_position_embeddings(x_min)
            + self.y_position_embeddings(y_min)
            + self.x_position_embeddings(x_max)
            + self.y_position_embeddings(y_max)
            + self.w_position_embeddings(w)
            + self.h_position_embeddings(h)
        )

        text_out = text_emb + pos_emb

        # 이미지 패치 임베딩
        if pixel_values is not None:
            patch_emb = self.patch_embeddings(pixel_values)  # (B, D, H', W')
            patch_emb = patch_emb.flatten(2).transpose(1, 2)  # (B, num_patches, D)
            text_out = torch.cat([text_out, patch_emb], dim=1)

        return self.dropout(self.norm(text_out))
```

### 3. Donut

네이버가 개발한 **OCR-free** 문서 이해 모델. OCR 엔진 없이 이미지에서 직접 텍스트와 구조를 추출한다.

```
이미지 → [Swin Transformer Encoder] → [BART Decoder] → JSON/텍스트
```

핵심: **Teacher Forcing**으로 구조화된 출력(JSON)을 생성하도록 학습한다.

출력 예시:
```json
{
  "document_type": "receipt",
  "store_name": "스타벅스",
  "total": "5,500원",
  "items": [
    {"name": "아메리카노", "price": "4,500원"},
    {"name": "시럽 추가", "price": "1,000원"}
  ]
}
```

```python
import torch
import torch.nn as nn


class DonutDecoder(nn.Module):
    """
    Donut 스타일 디코더.

    비전 인코더의 출력을 받아 구조화된 텍스트(JSON 등)를 생성한다.
    Cross-attention으로 이미지 특성을 참조한다.
    """

    def __init__(
        self,
        vocab_size: int = 57522,
        hidden_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 16,
        max_length: int = 2048,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        encoder_output: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: (B, N, D) — 비전 인코더 출력
            target_ids: (B, T) — 타겟 토큰 ID (teacher forcing)

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = target_ids.shape

        # 타겟 임베딩
        positions = torch.arange(T, device=target_ids.device).unsqueeze(0)
        tgt = self.embedding(target_ids) + self.position_embedding(positions)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)

        # Decoder
        output = self.decoder(
            tgt=tgt,
            memory=encoder_output,
            tgt_mask=causal_mask,
        )

        logits = self.output_head(output)
        return logits
```

### 4. Pix2Struct

Google이 개발한 모델. **Screenshot Parsing**이라는 독특한 사전학습을 사용한다.

사전학습 방식: 웹페이지 스크린샷 → HTML 코드를 생성하도록 학습.

이걸로 모델이 "이미지의 구조를 코드(텍스트)로 표현하는 법"을 배운다.

**Variable Resolution 입력**: Pix2Struct은 이미지를 고정 크기로 리사이즈하지 않고, 패치 수를 고정한다. Qwen2.5-VL의 Dynamic Resolution과 유사한 아이디어인데, Pix2Struct이 더 먼저 나왔다.

패치 수 고정: 최대 $N_{\max} = 2048$ 패치.

$$
n_h = \left\lfloor \sqrt{N_{\max} \cdot \frac{H}{W}} \right\rfloor, \quad n_w = \left\lfloor \frac{N_{\max}}{n_h} \right\rfloor
$$

```python
import math


def pix2struct_resolution(
    height: int,
    width: int,
    max_patches: int = 2048,
    patch_size: int = 16,
) -> tuple[int, int]:
    """
    Pix2Struct의 Variable Resolution 계산.

    총 패치 수를 max_patches로 고정하면서 원본 비율을 유지한다.

    Args:
        height: 원본 이미지 높이
        width: 원본 이미지 너비
        max_patches: 최대 패치 수
        patch_size: 패치 크기 (픽셀)

    Returns:
        (new_height, new_width): 리사이즈할 크기
    """
    aspect_ratio = height / width

    n_h = math.floor(math.sqrt(max_patches * aspect_ratio))
    n_w = math.floor(max_patches / n_h)

    # 패치 수 제한
    while n_h * n_w > max_patches:
        if n_h > n_w:
            n_h -= 1
        else:
            n_w -= 1

    new_height = n_h * patch_size
    new_width = n_w * patch_size

    return new_height, new_width
```

---

## 장단점 종합 비교

| 기준 | DocTR | LayoutLM v3 | Donut | Pix2Struct |
|------|-------|-------------|-------|------------|
| **설치 난이도** | 낮음 | 보통 | 보통 | 보통 |
| **추론 속도** | 매우 빠름 | 빠름 | 느림 | 느림 |
| **GPU 요구** | CPU 가능 | 중간 | 높음 | 높음 |
| **텍스트 정확도** | 높음 | 높음 | 보통 | 보통 |
| **구조 이해** | 없음 | 우수 | 우수 | 우수 |
| **표 인식** | 별도 필요 | 우수 | 좋음 | 좋음 |
| **수식 인식** | 없음 | 제한적 | 좋음 | 좋음 |
| **다국어** | 좋음 | 좋음 | 좋음 | 좋음 |
| **Fine-tuning 난이도** | 낮음 | 보통 | 높음 | 높음 |
| **적합한 용도** | 빠른 텍스트 추출 | 양식 이해, KV 추출 | 범용 문서 파싱 | 차트/문서 이해 |

### 선택 가이드

```
문서 OCR 모델 선택 의사결정 트리:

Q1: OCR만 필요한가? (구조 이해 불필요)
    ├─ YES → DocTR (가볍고 빠름)
    └─ NO → Q2

Q2: OCR 엔진 결과를 활용할 수 있는가?
    ├─ YES → LayoutLM v3 (텍스트+위치 통합)
    └─ NO → Q3 (OCR-free 필요)

Q3: 출력이 구조화된 JSON이어야 하는가?
    ├─ YES → Donut (JSON 구조 출력에 강함)
    └─ NO → Pix2Struct (범용 이미지→텍스트)
```

---

## 통합 파이프라인 예시

실무에서는 단일 모델만 쓰지 않고, 여러 모델을 조합하는 경우가 많다.

```python
import torch
from PIL import Image
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DocumentAnalysisResult:
    """문서 분석 결과."""
    raw_text: str = ""
    tables: list = field(default_factory=list)
    key_values: dict = field(default_factory=dict)
    layout_regions: list = field(default_factory=list)
    confidence: float = 0.0


class DocumentAnalysisPipeline:
    """
    문서 분석 통합 파이프라인.

    여러 모델을 조합하여 최적의 결과를 뽑아낸다.

    전략:
    1. DocTR로 빠른 텍스트 추출 + 위치 정보
    2. LayoutLM으로 레이아웃 분석 + KV 추출
    3. 표가 감지되면 별도 표 인식 모듈
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        # 실제 구현에서는 여기서 각 모델을 로드한다
        # self.ocr_model = DocTR(...)
        # self.layout_model = LayoutLM(...)
        # self.table_model = TableRecognitionModel(...)

    def analyze(self, image: Image.Image) -> DocumentAnalysisResult:
        """
        문서 이미지를 종합 분석한다.

        Args:
            image: PIL 이미지

        Returns:
            분석 결과
        """
        result = DocumentAnalysisResult()

        # Step 1: 텍스트 탐지 + 인식 (DocTR)
        text_regions = self._detect_and_recognize(image)
        result.raw_text = self._merge_text_regions(text_regions)

        # Step 2: 레이아웃 분석 (LayoutLM)
        layout = self._analyze_layout(image, text_regions)
        result.layout_regions = layout

        # Step 3: 표 추출
        table_regions = [r for r in layout if r["type"] == "table"]
        for table_region in table_regions:
            table = self._extract_table(image, table_region)
            result.tables.append(table)

        # Step 4: Key-Value 추출
        form_regions = [r for r in layout if r["type"] in ("form_field", "key_value")]
        result.key_values = self._extract_key_values(text_regions, form_regions)

        # 신뢰도 계산
        result.confidence = self._calculate_confidence(text_regions)

        return result

    def _detect_and_recognize(self, image: Image.Image) -> list:
        """텍스트 탐지 + 인식 (placeholder)."""
        # 실제로는 DocTR 또는 유사 모델 사용
        return []

    def _analyze_layout(self, image: Image.Image, text_regions: list) -> list:
        """레이아웃 분석 (placeholder)."""
        return []

    def _extract_table(self, image: Image.Image, region: dict) -> dict:
        """표 추출 (placeholder)."""
        return {}

    def _extract_key_values(self, text_regions: list, form_regions: list) -> dict:
        """키-값 추출 (placeholder)."""
        return {}

    def _merge_text_regions(self, regions: list) -> str:
        """텍스트 영역들을 읽기 순서로 병합."""
        # y좌표 → x좌표 순으로 정렬
        sorted_regions = sorted(regions, key=lambda r: (r.get("y", 0), r.get("x", 0)))
        return "\n".join(r.get("text", "") for r in sorted_regions)

    def _calculate_confidence(self, text_regions: list) -> float:
        """전체 신뢰도 계산."""
        if not text_regions:
            return 0.0
        confidences = [r.get("confidence", 0.0) for r in text_regions]
        return sum(confidences) / len(confidences)
```

---

## 용어 체크리스트

아래 용어들을 설명할 수 있으면 이 챕터를 이해한 거다.

- [ ] **Document Understanding**: 단순 OCR과 문서 이해의 차이를 설명할 수 있는가?
- [ ] **Table Recognition**: 표를 그래프로 모델링하는 방법과 adjacency matrix의 의미를 설명할 수 있는가?
- [ ] **Key-Value Extraction**: 양식 문서에서 KV 쌍을 추출하는 과정을 설명할 수 있는가?
- [ ] **좌표 정규화**: $(x, y) \to (\hat{x}, \hat{y})$ 정규화와 이산화가 왜 필요한지 설명할 수 있는가?
- [ ] **IoU**: 두 바운딩 박스 간 IoU 계산 방법을 설명할 수 있는가?
- [ ] **Focal Loss**: 클래스 불균형 상황에서 Focal Loss가 일반 CE보다 나은 이유를 설명할 수 있는가?
- [ ] **DocTR vs LayoutLM**: 두 모델의 접근법 차이(OCR 의존 vs OCR-free)를 설명할 수 있는가?
- [ ] **Donut vs Pix2Struct**: 두 OCR-free 모델의 사전학습 전략 차이를 설명할 수 있는가?
- [ ] **Variable Resolution**: 고정 해상도 대비 가변 해상도의 장점을 설명할 수 있는가?
- [ ] **모델 선택 기준**: 주어진 요구사항에 맞는 문서 이해 모델을 선택할 수 있는가?
