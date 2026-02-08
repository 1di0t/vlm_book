---
---

# 5.2 Projection/Connector 방식

Projection 방식은 VLM 아키텍처 중 가장 단순하면서도 강력한 접근법이다. Vision Encoder의 출력을 Linear Projection이나 MLP로 LLM의 임베딩 공간에 매핑하고, 변환된 visual token을 텍스트 토큰과 함께 LLM에 입력한다. LLaVA가 이 방식의 대표 모델이다.

---

## 핵심 용어

| 용어 | 정의 | 관련 모델 |
|------|------|-----------|
| **Linear Projection** | 단일 선형 변환 $W \cdot z + b$로 vision 차원을 LLM 차원에 매핑 | LLaVA (v1) |
| **MLP Connector** | 2-Layer MLP로 비선형 변환을 추가하여 더 풍부한 feature alignment 수행 | LLaVA-1.5 |
| **Visual Token** | Vision Encoder의 패치별 출력을 LLM이 이해 가능한 토큰으로 변환한 결과 | 전체 Projection 계열 |
| **Feature Alignment** | 서로 다른 모달리티의 feature 공간을 동일한 의미 공간으로 정렬하는 과정 | 전체 VLM |
| **Resampler** | Pooling이나 attention으로 visual token 수를 줄이는 모듈 | Qwen-VL, InternVL |

---

## 5.2.1 Projection의 기본 원리

### 왜 Projection이 필요한가

Vision Encoder(ViT)와 LLM은 서로 다른 차원의 feature 공간을 사용한다.

```
ViT-L/14:  출력 차원 D_vision = 1024,  토큰 수 N_v = 256
LLaMA-7B:  임베딩 차원 D_LLM = 4096
```

이 두 공간을 연결하려면 차원 변환이 필요하다. 가장 직관적인 방법이 선형 프로젝션이다.

### 아키텍처 개요

```
이미지 → Vision Encoder (Frozen) → z_vision (N_v × D_vision)
                                         ↓
                                    Projection Layer (Trainable)
                                         ↓
                                    z_LLM (N_v × D_LLM)
                                         ↓
         텍스트 토큰 임베딩 ─────→ [visual tokens ; text tokens]
                                         ↓
                                    LLM (Frozen or Fine-tuned)
                                         ↓
                                      텍스트 출력
```

Cross-Attention 방식(5.1절)과 비교하면 아키텍처가 훨씬 단순하다. LLM 내부를 수정할 필요 없이, 입력 앞에 visual token을 붙이기만 하면 된다.

---

## 수학적 원리

### Linear Projection

가장 단순한 형태. 단일 행렬 곱셈으로 차원을 변환한다.

$$
z_{\text{LLM}} = W \cdot z_{\text{vision}} + b
$$

$$
W \in \mathbb{R}^{D_{\text{LLM}} \times D_{\text{vision}}}, \quad b \in \mathbb{R}^{D_{\text{LLM}}}
$$

각 visual token $z_i^{\text{vision}} \in \mathbb{R}^{D_{\text{vision}}}$에 대해:

$$
z_i^{\text{LLM}} = W z_i^{\text{vision}} + b, \quad i = 1, \ldots, N_v
$$

파라미터 수: $D_{\text{LLM}} \times D_{\text{vision}} + D_{\text{LLM}} = 4096 \times 1024 + 4096 \approx 4.2\text{M}$

이건 전체 모델(7B) 대비 약 0.06%에 불과하다.

### 2-Layer MLP Connector

LLaVA-1.5에서 도입한 방식. 비선형 활성 함수를 추가하여 더 복잡한 공간 매핑을 학습한다.

$$
h = \text{GELU}(W_1 \cdot z_{\text{vision}} + b_1)
$$

$$
z_{\text{LLM}} = W_2 \cdot h + b_2
$$

풀어쓰면:

$$
z_{\text{LLM}} = W_2 \cdot \text{GELU}(W_1 \cdot z_{\text{vision}} + b_1) + b_2
$$

여기서:
- $W_1 \in \mathbb{R}^{D_{\text{hidden}} \times D_{\text{vision}}}$: 첫 번째 선형 변환
- $W_2 \in \mathbb{R}^{D_{\text{LLM}} \times D_{\text{hidden}}}$: 두 번째 선형 변환
- $D_{\text{hidden}}$: 보통 $D_{\text{LLM}}$과 동일하게 설정
- GELU: Gaussian Error Linear Unit 활성 함수

$$
\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)
$$

파라미터 수 (hidden = D_LLM일 때):

$$
(D_{\text{LLM}} \times D_{\text{vision}} + D_{\text{LLM}}) + (D_{\text{LLM}} \times D_{\text{LLM}} + D_{\text{LLM}})
$$

$$
= 4096 \times 1024 + 4096 + 4096 \times 4096 + 4096 \approx 21\text{M}
$$

### 차원 정렬의 기하학적 해석

선형 프로젝션은 고차원 공간에서의 **아핀 변환**이다. Vision feature space $\mathcal{V}$와 LLM embedding space $\mathcal{L}$이 있을 때:

$$
f: \mathcal{V} \rightarrow \mathcal{L}, \quad f(z) = Wz + b
$$

이 변환이 의미론적으로 올바르게 작동하려면:
- 비슷한 시각적 의미를 가진 visual feature는 LLM 공간에서도 가까워야 한다
- 시각 정보가 LLM의 언어적 의미 공간과 정렬되어야 한다

MLP는 비선형 매핑을 허용하므로 더 복잡한 공간 정렬이 가능하다:

$$
f(z) = W_2 \cdot \sigma(W_1 z + b_1) + b_2
$$

Universal Approximation Theorem에 의해, 충분히 넓은 hidden layer를 가진 2-layer MLP는 임의의 연속 함수를 근사할 수 있다.

### Token 수 조절

Vision Encoder 출력의 토큰 수 $N_v$가 크면 LLM의 컨텍스트 길이를 많이 차지한다. Pooling이나 Resampler로 토큰 수를 줄일 수 있다.

**Average Pooling (2×2)**:

$$
N_v' = \frac{N_v}{k^2}, \quad k=2 \Rightarrow 576 \rightarrow 144
$$

**Adaptive Pooling**:

$$
z_{\text{pooled}} = \text{AdaptiveAvgPool2d}(z_{\text{2d}}, (h', w'))
$$

여기서 $z_{\text{2d}}$는 패치 토큰을 2D 그리드로 재배열한 것이다. ViT-L/14 (336px)의 경우 $24 \times 24 = 576$개 패치를 $12 \times 12 = 144$개로 줄일 수 있다.

---

## 5.2.2 LLaVA 스타일 프로젝터 구현

### Linear Projection

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LinearProjection(nn.Module):
    """
    LLaVA v1 스타일 Linear Projection.
    가장 단순한 vision-language connector.

    z_LLM = W · z_vision + b
    """
    def __init__(
        self,
        dim_vision: int = 1024,
        dim_llm: int = 4096,
    ):
        super().__init__()
        self.proj = nn.Linear(dim_vision, dim_llm)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (batch, N_v, D_vision)

        Returns:
            visual_tokens: (batch, N_v, D_LLM)
        """
        return self.proj(visual_features)
```

### MLP Connector (LLaVA-1.5)

```python
class MLPConnector(nn.Module):
    """
    LLaVA-1.5 스타일 2-Layer MLP Connector.

    z_LLM = W2 · GELU(W1 · z_vision + b1) + b2

    Linear Projection 대비 약 5배 많은 파라미터를 사용하지만,
    비선형 매핑으로 더 풍부한 feature alignment을 학습한다.
    LLaVA-1.5 논문에서 이 간단한 변경만으로 성능이 크게 향상됨을 보였다.
    """
    def __init__(
        self,
        dim_vision: int = 1024,
        dim_llm: int = 4096,
        dim_hidden: Optional[int] = None,
    ):
        super().__init__()
        dim_hidden = dim_hidden or dim_llm

        self.mlp = nn.Sequential(
            nn.Linear(dim_vision, dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim_llm),
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (batch, N_v, D_vision)

        Returns:
            visual_tokens: (batch, N_v, D_LLM)
        """
        return self.mlp(visual_features)
```

### Token Downsampler 포함 Connector

```python
class DownsamplingConnector(nn.Module):
    """
    Visual token 수를 줄이면서 차원 변환하는 Connector.

    1. 2D spatial pooling으로 토큰 수 감소 (N_v → N_v')
    2. MLP로 차원 변환 (D_vision → D_LLM)

    고해상도 이미지에서 visual token이 너무 많을 때 사용.
    예: ViT-L/14@336 → 576 patches → pooling → 144 patches
    """
    def __init__(
        self,
        dim_vision: int = 1024,
        dim_llm: int = 4096,
        grid_size: int = 24,       # 원래 패치 그리드 크기 (24x24=576)
        target_size: int = 12,     # 타겟 그리드 크기 (12x12=144)
    ):
        super().__init__()
        self.grid_size = grid_size
        self.target_size = target_size

        self.pool = nn.AdaptiveAvgPool2d((target_size, target_size))
        self.connector = MLPConnector(dim_vision, dim_llm)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (batch, N_v, D_vision)
                N_v = grid_size * grid_size (CLS 토큰 제외)

        Returns:
            visual_tokens: (batch, target_size², D_LLM)
        """
        batch_size, num_tokens, dim = visual_features.shape

        # 1D → 2D 재배열
        features_2d = visual_features.view(
            batch_size, self.grid_size, self.grid_size, dim
        )
        features_2d = features_2d.permute(0, 3, 1, 2)  # (B, D, H, W)

        # Spatial pooling
        pooled = self.pool(features_2d)  # (B, D, target_H, target_W)

        # 2D → 1D 재배열
        pooled = pooled.permute(0, 2, 3, 1).contiguous()  # (B, H', W', D)
        pooled = pooled.view(batch_size, -1, dim)  # (B, N_v', D)

        # MLP Connector
        visual_tokens = self.connector(pooled)

        return visual_tokens
```

---

## 5.2.3 LLaVA 전체 파이프라인

```python
class LLaVAModel(nn.Module):
    """
    LLaVA (Large Language and Vision Assistant) 간략 구현.

    아키텍처:
        이미지 → CLIP ViT (Frozen) → visual features
                                        ↓
                                   MLP Connector (Trainable)
                                        ↓
                                   visual tokens
                                        ↓
            텍스트 → Tokenizer → text tokens → [visual; text] → LLM → 출력

    학습 전략:
        Stage 1 (Pretrain): Connector만 학습, LLM frozen
        Stage 2 (Finetune): Connector + LLM 전체 학습
    """
    def __init__(
        self,
        vision_encoder: nn.Module,
        llm: nn.Module,
        dim_vision: int = 1024,
        dim_llm: int = 4096,
        connector_type: str = "mlp",  # "linear" or "mlp"
        freeze_vision: bool = True,
        freeze_llm: bool = False,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.llm = llm

        # Connector 선택
        if connector_type == "linear":
            self.connector = LinearProjection(dim_vision, dim_llm)
        elif connector_type == "mlp":
            self.connector = MLPConnector(dim_vision, dim_llm)
        else:
            raise ValueError(f"Unknown connector type: {connector_type}")

        # Vision Encoder는 항상 freeze
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # LLM freeze 여부 (Stage 1 vs Stage 2)
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        이미지를 visual token으로 변환.

        Args:
            images: (batch, 3, H, W)

        Returns:
            visual_tokens: (batch, N_v, D_LLM)
        """
        with torch.no_grad():
            visual_features = self.vision_encoder(images)
            # ViT 출력: (batch, N_v+1, D_vision) - CLS 포함
            # CLS 토큰 제거 (패치 토큰만 사용)
            if visual_features.size(1) > 1:
                visual_features = visual_features[:, 1:, :]

        visual_tokens = self.connector(visual_features)
        return visual_tokens

    def prepare_inputs(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        image_token_id: int = -200,
    ) -> torch.Tensor:
        """
        Visual tokens과 text tokens를 결합하여 LLM 입력을 준비.

        input_ids에서 image_token_id 위치를 visual tokens로 대체한다.

        Args:
            images: (batch, 3, H, W)
            input_ids: (batch, seq_len) - <image> 토큰 포함
            image_token_id: <image> placeholder 토큰 ID

        Returns:
            inputs_embeds: (batch, N_v + seq_len - 1, D_LLM)
        """
        # Visual tokens
        visual_tokens = self.encode_image(images)
        # (batch, N_v, D_LLM)

        # Text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        # (batch, seq_len, D_LLM)

        # <image> 토큰 위치 찾기
        batch_size = input_ids.size(0)
        new_embeds_list = []

        for b in range(batch_size):
            img_positions = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]

            if len(img_positions) == 0:
                # 이미지 토큰이 없으면 앞에 붙이기
                new_embeds = torch.cat([
                    visual_tokens[b],
                    text_embeds[b],
                ], dim=0)
            else:
                # <image> 토큰 위치에 visual tokens 삽입
                pos = img_positions[0].item()
                new_embeds = torch.cat([
                    text_embeds[b, :pos],          # <image> 이전
                    visual_tokens[b],               # visual tokens
                    text_embeds[b, pos + 1:],       # <image> 이후
                ], dim=0)

            new_embeds_list.append(new_embeds)

        # 패딩하여 배치로 만들기
        max_len = max(e.size(0) for e in new_embeds_list)
        inputs_embeds = torch.zeros(
            batch_size, max_len, visual_tokens.size(-1),
            device=visual_tokens.device, dtype=visual_tokens.dtype,
        )
        for b, embed in enumerate(new_embeds_list):
            inputs_embeds[b, :embed.size(0)] = embed

        return inputs_embeds

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            images: (batch, 3, H, W)
            input_ids: (batch, seq_len)
            labels: (batch, seq_len) - 학습 시 정답 토큰
            attention_mask: (batch, total_len)
        """
        inputs_embeds = self.prepare_inputs(images, input_ids)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs
```

---

## 5.2.4 학습 전략: 2-Stage Training

### Stage 1: Feature Alignment Pre-training

```python
def stage1_pretrain(
    model: LLaVAModel,
    dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 1,
    lr: float = 1e-3,
):
    """
    Stage 1: Connector만 학습.
    Vision Encoder와 LLM은 frozen.
    Image-Caption 데이터로 feature alignment을 학습한다.

    데이터: CC3M 등 이미지-캡션 데이터 595K
    목표: visual feature를 LLM이 이해 가능한 표현으로 정렬
    """
    # Vision Encoder, LLM freeze
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.llm.parameters():
        param.requires_grad = False
    # Connector만 학습
    for param in model.connector.parameters():
        param.requires_grad = True

    # Connector 파라미터만 optimizer에 전달
    optimizer = torch.optim.AdamW(
        model.connector.parameters(),
        lr=lr,
        weight_decay=0.0,
    )

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            images = batch["images"]
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]

            outputs = model(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Stage 1 Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
```

### Stage 2: Visual Instruction Tuning

```python
def stage2_finetune(
    model: LLaVAModel,
    dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 1,
    lr: float = 2e-5,
):
    """
    Stage 2: Connector + LLM 전체 학습.
    Vision Encoder만 frozen.
    Visual instruction 데이터로 멀티모달 지시 따르기를 학습한다.

    데이터: LLaVA-Instruct-158K (GPT-4 생성)
    목표: 이미지에 대한 질의응답, 설명 생성 등
    """
    # Vision Encoder만 freeze
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    # Connector + LLM 학습
    for param in model.connector.parameters():
        param.requires_grad = True
    for param in model.llm.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.0,
    )

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(dataloader),
        eta_min=0,
    )

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            images = batch["images"]
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]

            outputs = model(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Stage 2 Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
```

---

## 5.2.5 Connector 변형들

### C-Abstractor (Honeybee)

Convolution 기반으로 spatial 정보를 보존하면서 토큰 수를 줄인다.

```python
class CAbstractor(nn.Module):
    """
    Honeybee의 C-Abstractor.
    Convolution으로 spatial structure를 보존하면서 visual token 수를 줄인다.
    Adaptive pooling만 쓰면 공간 정보가 손실되는 문제를 완화한다.
    """
    def __init__(
        self,
        dim_vision: int = 1024,
        dim_llm: int = 4096,
        grid_size: int = 24,
        target_size: int = 12,
        num_conv_layers: int = 2,
    ):
        super().__init__()
        self.grid_size = grid_size

        # Convolution layers (stride=2로 다운샘플링)
        layers = []
        in_channels = dim_vision
        for i in range(num_conv_layers):
            stride = 2 if (grid_size // (2 ** (i + 1))) >= target_size else 1
            layers.extend([
                nn.Conv2d(in_channels, dim_vision, 3, stride=stride, padding=1),
                nn.BatchNorm2d(dim_vision),
                nn.GELU(),
            ])
        self.conv_layers = nn.Sequential(*layers)

        # Adaptive pooling으로 정확한 타겟 크기 보장
        self.pool = nn.AdaptiveAvgPool2d((target_size, target_size))

        # MLP Projection
        self.proj = nn.Sequential(
            nn.Linear(dim_vision, dim_llm),
            nn.GELU(),
            nn.Linear(dim_llm, dim_llm),
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (batch, N_v, D_vision)

        Returns:
            visual_tokens: (batch, target_size², D_LLM)
        """
        batch_size, num_tokens, dim = visual_features.shape

        # 1D → 2D
        x = visual_features.view(batch_size, self.grid_size, self.grid_size, dim)
        x = x.permute(0, 3, 1, 2)  # (B, D, H, W)

        # Conv downsampling
        x = self.conv_layers(x)

        # Adaptive pooling
        x = self.pool(x)

        # 2D → 1D
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, -1, dim)

        # MLP projection
        visual_tokens = self.proj(x)

        return visual_tokens
```

### Dynamic Resolution Connector

```python
class DynamicResolutionConnector(nn.Module):
    """
    다양한 해상도의 이미지를 처리하는 Connector.
    이미지 크기에 따라 visual token 수가 달라지므로,
    동적 처리가 필요하다.

    고해상도 이미지(문서, 표 등)에서 중요하다.
    """
    def __init__(
        self,
        dim_vision: int = 1024,
        dim_llm: int = 4096,
        max_tokens: int = 576,
        min_tokens: int = 64,
    ):
        super().__init__()
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

        self.connector = MLPConnector(dim_vision, dim_llm)
        self.token_selector = nn.Sequential(
            nn.Linear(dim_vision, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        target_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: (batch, N_v, D_vision)
            target_tokens: 원하는 visual token 수 (None이면 자동 선택)

        Returns:
            visual_tokens: (batch, N_selected, D_LLM)
        """
        batch_size, num_tokens, dim = visual_features.shape

        if target_tokens is None:
            target_tokens = min(num_tokens, self.max_tokens)

        if num_tokens <= target_tokens:
            # 토큰 수가 적으면 그대로 projection
            return self.connector(visual_features)

        # Importance score 기반 토큰 선택
        scores = self.token_selector(visual_features).squeeze(-1)
        # (batch, N_v)

        # Top-k 선택
        _, indices = scores.topk(target_tokens, dim=1)
        indices = indices.sort(dim=1).values  # 순서 유지

        # 선택된 토큰만 추출
        selected = torch.gather(
            visual_features,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, dim),
        )

        return self.connector(selected)
```

---

## 5.2.6 Linear vs MLP: 실험 비교

```python
def compare_connectors():
    """Linear Projection vs MLP Connector 비교 실험"""
    torch.manual_seed(42)

    dim_vision = 1024
    dim_llm = 4096
    batch_size = 4
    num_patches = 576  # ViT-L/14@336

    visual_features = torch.randn(batch_size, num_patches, dim_vision)

    # 1. Linear Projection
    linear_proj = LinearProjection(dim_vision, dim_llm)
    linear_params = sum(p.numel() for p in linear_proj.parameters())

    # 2. MLP Connector
    mlp_conn = MLPConnector(dim_vision, dim_llm)
    mlp_params = sum(p.numel() for p in mlp_conn.parameters())

    # 3. Downsampling Connector
    down_conn = DownsamplingConnector(dim_vision, dim_llm, grid_size=24, target_size=12)
    down_params = sum(p.numel() for p in down_conn.parameters())

    print("=" * 60)
    print("Connector 비교")
    print("=" * 60)

    # Forward pass + 시간 측정
    import time

    for name, module, params in [
        ("Linear Projection", linear_proj, linear_params),
        ("MLP Connector", mlp_conn, mlp_params),
        ("Downsampling Connector", down_conn, down_params),
    ]:
        start = time.time()
        with torch.no_grad():
            output = module(visual_features)
        elapsed = time.time() - start

        print(f"\n{name}:")
        print(f"  파라미터 수: {params:,}")
        print(f"  입력 shape:  {visual_features.shape}")
        print(f"  출력 shape:  {output.shape}")
        print(f"  추론 시간:   {elapsed*1000:.2f}ms")
        print(f"  출력 토큰 수: {output.size(1)}")


if __name__ == "__main__":
    compare_connectors()
```

### 성능 비교 요약

| Connector | 파라미터 수 | Visual Token 수 | LLaVA 벤치마크 성능 |
|-----------|-----------|----------------|-------------------|
| Linear Projection | ~4.2M | 576 (그대로) | 기준선 |
| MLP (2-layer) | ~21M | 576 (그대로) | +2~5% 향상 |
| MLP + Pooling | ~21M | 144 (4x 감소) | -1~2% 소폭 하락, 속도 4x 향상 |

LLaVA-1.5 논문의 핵심 발견: **Linear → MLP로 바꾸는 것만으로 전 벤치마크에서 성능 향상**. 추가 데이터나 모델 크기 증가 없이도 효과적이다.

---

## 5.2.7 OCR에서의 Projection 방식

### 문서 이미지의 특수성

문서 OCR에서는 일반 이미지 이해와 다른 요구사항이 있다.

```
일반 이미지: 개, 고양이 등 전체적 의미 이해
문서 이미지: 작은 글자 하나하나의 세밀한 인식 필요
```

따라서 문서 OCR에서는:
1. **높은 해상도** 필요 → visual token 수가 많아짐
2. **공간 정보 보존** 필수 → 무분별한 pooling은 위험
3. **세밀한 패치-텍스트 대응** 필요 → 강한 feature alignment 필요

```python
class OCRProjectionConnector(nn.Module):
    """
    OCR 특화 Projection Connector.

    일반 자연 이미지보다 고해상도 처리가 필요하므로:
    1. 해상도별 다른 pooling 전략 사용
    2. 위치 정보를 명시적으로 추가
    3. 텍스트 영역에 더 많은 토큰 할당
    """
    def __init__(
        self,
        dim_vision: int = 1024,
        dim_llm: int = 4096,
        max_visual_tokens: int = 1024,
    ):
        super().__init__()
        self.max_visual_tokens = max_visual_tokens

        # 2D positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_visual_tokens, dim_vision) * 0.02
        )

        # MLP Connector
        self.connector = nn.Sequential(
            nn.LayerNorm(dim_vision),
            nn.Linear(dim_vision, dim_llm),
            nn.GELU(),
            nn.Linear(dim_llm, dim_llm),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: (batch, N_v, D_vision)

        Returns:
            visual_tokens: (batch, N_v, D_LLM)
        """
        batch_size, num_tokens, dim = visual_features.shape

        # 위치 임베딩 추가 (토큰 수에 맞게 보간)
        if num_tokens <= self.max_visual_tokens:
            pos = self.pos_embed[:, :num_tokens, :]
        else:
            # 토큰이 많으면 위치 임베딩을 보간
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=num_tokens,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)

        visual_features = visual_features + pos

        # Projection
        visual_tokens = self.connector(visual_features)

        return visual_tokens
```

---

## 5.2.8 실습: Projection 성능 테스트

```python
def test_projection_pipeline():
    """Projection 방식 전체 파이프라인 테스트"""
    torch.manual_seed(42)

    batch_size = 2
    num_patches = 576
    dim_vision = 1024
    dim_llm = 4096

    # 더미 데이터
    visual_features = torch.randn(batch_size, num_patches, dim_vision)

    # 1. Linear Projection 테스트
    linear = LinearProjection(dim_vision, dim_llm)
    out_linear = linear(visual_features)
    print(f"Linear: {visual_features.shape} → {out_linear.shape}")

    # 2. MLP Connector 테스트
    mlp = MLPConnector(dim_vision, dim_llm)
    out_mlp = mlp(visual_features)
    print(f"MLP:    {visual_features.shape} → {out_mlp.shape}")

    # 3. Downsampling 테스트
    down = DownsamplingConnector(dim_vision, dim_llm, grid_size=24, target_size=12)
    out_down = down(visual_features)
    print(f"Down:   {visual_features.shape} → {out_down.shape}")

    # Gradient 흐름 확인
    loss = out_mlp.sum()
    loss.backward()
    print(f"\nMLP grad norm: {mlp.mlp[0].weight.grad.norm():.4f}")

    # 파라미터 비교
    print(f"\nLinear params:       {sum(p.numel() for p in linear.parameters()):>12,}")
    print(f"MLP params:          {sum(p.numel() for p in mlp.parameters()):>12,}")
    print(f"Downsampling params: {sum(p.numel() for p in down.parameters()):>12,}")


if __name__ == "__main__":
    test_projection_pipeline()
```

---

## 용어 체크리스트

학습 후 아래 질문에 답할 수 있는지 확인하라.

| 체크 | 용어 | 자가 점검 질문 |
|------|------|----------------|
| ☐ | Linear Projection | 단일 행렬 곱으로 차원 변환할 때 파라미터 수는 얼마인가? ($D_{\text{LLM}} \times D_{\text{vision}}$) |
| ☐ | MLP Connector | 2-Layer MLP가 Linear보다 나은 이유를 비선형성 관점에서 설명할 수 있는가? |
| ☐ | Visual Token | ViT의 패치 토큰이 LLM 입력으로 변환되는 전체 흐름을 그릴 수 있는가? |
| ☐ | Feature Alignment | Vision space와 Language space의 정렬이란 무엇이고, 왜 학습이 필요한가? |
| ☐ | Token Downsampling | Pooling으로 visual token을 줄이면 어떤 정보가 손실될 수 있는가? OCR에서 문제가 되는가? |
| ☐ | 2-Stage Training | Stage 1(alignment)과 Stage 2(instruction tuning)의 학습 대상과 목적 차이를 설명할 수 있는가? |
| ☐ | LLaVA vs LLaVA-1.5 | 두 버전의 Connector 차이가 성능에 미치는 영향을 아는가? |
| ☐ | C-Abstractor | Convolution 기반 downsampling이 average pooling보다 공간 정보를 잘 보존하는 이유는? |

---

## 다음 단계

[5.3 Early Fusion 방식](03-early-fusion.md)에서 Vision과 Language를 처음부터 통합하는 아키텍처를 다룬다.
