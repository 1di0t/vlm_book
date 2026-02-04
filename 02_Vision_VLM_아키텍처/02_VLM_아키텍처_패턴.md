# Chapter 2.2: VLM 아키텍처 패턴

## 개요

Vision-Language Model(VLM)은 이미지와 텍스트를 함께 이해하는 멀티모달 모델이다. 본 챕터에서는 주요 VLM 아키텍처 패턴을 분석한다.

---

## 1. VLM 아키텍처 3대 패턴

### 패턴 비교 요약

| 패턴 | 대표 모델 | Vision-LLM 연결 방식 | 장점 | 단점 |
|------|----------|---------------------|------|------|
| **Cross-Attention** | Flamingo, Kosmos | LLM에 cross-attention 레이어 추가 | 토큰 수 조절 가능 | 구조 복잡 |
| **Projection/Connector** | LLaVA, Qwen-VL | MLP로 embedding space 매핑 | 단순, 효과적 | Vision encoder 종속 |
| **Early Fusion** | Fuyu | 이미지 패치를 직접 토큰화 | End-to-end 학습 | 학습 비용 높음 |

---

## 2. Pattern 1: Cross-Attention 방식

### 2.1 Flamingo

> **논문**: Alayrac et al. (2022). "Flamingo: a Visual Language Model for Few-Shot Learning"
> - arXiv: https://arxiv.org/abs/2204.14198
> - NeurIPS 2022

**핵심 구성요소:**

1. **Vision Encoder**: 사전학습된 NFNet
2. **Perceiver Resampler**: 가변 길이 visual features → 고정 길이 visual tokens (64개)
3. **Gated Cross-Attention**: LLM 레이어 사이에 visual information 주입

```python
class PerceiverResampler(nn.Module):
    """
    가변 개수의 visual features를 고정 개수의 latent tokens으로 압축
    """
    def __init__(self, dim, num_latents=64, num_layers=6):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(dim, num_heads=8, batch_first=True),
                'cross_norm': nn.LayerNorm(dim),
                'self_attn': nn.MultiheadAttention(dim, num_heads=8, batch_first=True),
                'self_norm': nn.LayerNorm(dim),
                'ffn': nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                ),
                'ffn_norm': nn.LayerNorm(dim)
            })
            for _ in range(num_layers)
        ])

    def forward(self, visual_features):
        """
        visual_features: (batch, num_visual_tokens, dim) - 가변 길이
        returns: (batch, num_latents, dim) - 고정 길이
        """
        batch_size = visual_features.size(0)
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        for layer in self.layers:
            # Cross-attention: latents가 visual_features를 참조
            latents = latents + layer['cross_attn'](
                layer['cross_norm'](latents),
                visual_features,
                visual_features
            )[0]

            # Self-attention
            latents = latents + layer['self_attn'](
                layer['self_norm'](latents),
                layer['self_norm'](latents),
                layer['self_norm'](latents)
            )[0]

            # FFN
            latents = latents + layer['ffn'](layer['ffn_norm'](latents))

        return latents


class GatedCrossAttentionBlock(nn.Module):
    """
    LLM 레이어 사이에 삽입되는 Gated Cross-Attention
    """
    def __init__(self, dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(dim)

        # Gating parameter - 초기값 0으로 시작
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, text_hidden, visual_tokens):
        """
        text_hidden: (batch, seq_len, dim) - LLM hidden states
        visual_tokens: (batch, num_visual_tokens, dim) - Perceiver 출력
        """
        # Cross-attention: text가 visual tokens를 참조
        attn_out = self.cross_attn(
            self.norm(text_hidden),
            visual_tokens,
            visual_tokens
        )[0]

        # Gated residual (tanh(gate)로 0에서 시작)
        return text_hidden + torch.tanh(self.gate) * attn_out
```

**Flamingo의 Attention Masking:**
- 각 텍스트 토큰은 가장 최근 선행 이미지의 visual tokens만 참조
- 여러 이미지가 있을 때 적절한 이미지-텍스트 연결 유지

### 2.2 InternVL

> **논문**: Chen et al. (2024). "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks"
> - arXiv: https://arxiv.org/abs/2312.14238
> - CVPR 2024 Oral

**특징:**
- 대규모 Vision Encoder (InternViT-6B)
- Cross-attention 레이어로 LLM (QLLaMA-8B)과 연결
- 96개의 learnable queries

---

## 3. Pattern 2: Projection/Connector 방식

### 3.1 LLaVA

> **논문**: Liu et al. (2023). "Visual Instruction Tuning"
> - arXiv: https://arxiv.org/abs/2304.08485
> - NeurIPS 2023 Oral

**아키텍처:**
```
Image → CLIP ViT → MLP Projector → LLM (Vicuna)
```

```python
class LLaVAProjector(nn.Module):
    """
    LLaVA-1.5에서 사용하는 2-layer MLP Projector
    """
    def __init__(self, vision_dim, llm_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, visual_features):
        """
        visual_features: (batch, num_patches, vision_dim)
        returns: (batch, num_patches, llm_dim)
        """
        return self.projector(visual_features)


class LLaVA(nn.Module):
    def __init__(self, vision_encoder, llm, projector):
        super().__init__()
        self.vision_encoder = vision_encoder  # CLIP ViT
        self.projector = projector
        self.llm = llm

        # Vision encoder는 보통 freeze
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def forward(self, images, input_ids, attention_mask):
        # 1. Visual features 추출
        with torch.no_grad():
            visual_features = self.vision_encoder(images)  # (B, N, vision_dim)

        # 2. LLM embedding space로 projection
        visual_embeddings = self.projector(visual_features)  # (B, N, llm_dim)

        # 3. Text embeddings
        text_embeddings = self.llm.get_input_embeddings()(input_ids)

        # 4. Visual + Text embeddings 결합
        # [visual_tokens] + [text_tokens]
        combined_embeddings = torch.cat([visual_embeddings, text_embeddings], dim=1)

        # 5. LLM forward
        outputs = self.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=self._create_attention_mask(visual_embeddings, attention_mask)
        )

        return outputs

    def _create_attention_mask(self, visual_embeds, text_mask):
        """Visual tokens + Text tokens에 대한 attention mask"""
        batch_size = visual_embeds.size(0)
        visual_length = visual_embeds.size(1)

        visual_mask = torch.ones(batch_size, visual_length, device=visual_embeds.device)
        return torch.cat([visual_mask, text_mask], dim=1)
```

### 3.2 LLaVA-1.5 개선사항

> **논문**: Liu et al. (2023). "Improved Baselines with Visual Instruction Tuning"
> - arXiv: https://arxiv.org/abs/2310.03744

| 개선 | LLaVA | LLaVA-1.5 |
|------|-------|-----------|
| Vision Encoder | CLIP ViT-L/14 (224px) | CLIP ViT-L/14 (336px) |
| Projector | Linear | 2-layer MLP |
| 학습 데이터 | 150K | 1.2M |
| VQA 데이터 | 없음 | 추가 |

### 3.3 Qwen2-VL

> **논문**: Wang et al. (2024). "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution"
> - arXiv: https://arxiv.org/abs/2409.12191

**핵심 특징:**

1. **Naive Dynamic Resolution**
   - 고정 해상도 대신 원본 해상도에 가깝게 처리
   - 긴 문서, 다양한 종횡비 이미지에 유리

2. **M-RoPE (Multimodal RoPE)**
   - 텍스트, 이미지, 비디오에 대한 통합 positional encoding
   - 2D 이미지 위치 정보 보존

```python
class Qwen2VLVisionEncoder(nn.Module):
    """
    Qwen2-VL의 Dynamic Resolution Vision Encoder
    """
    def __init__(self, hidden_size=1280, patch_size=14, max_patches=16384):
        super().__init__()
        self.patch_size = patch_size
        self.max_patches = max_patches

        self.patch_embed = nn.Conv2d(
            3, hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.transformer = VisionTransformer(hidden_size, num_layers=32)

    def forward(self, pixel_values, image_grid_thw):
        """
        pixel_values: (batch, 3, H, W) - 원본 해상도에 가까운 크기
        image_grid_thw: (batch, 3) - [temporal, height_patches, width_patches]
        """
        # Patch embedding
        patches = self.patch_embed(pixel_values)  # (B, D, H', W')
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, D)

        # 2D position encoding (M-RoPE)
        pos_embed = self.compute_2d_rope(image_grid_thw)

        # Transformer
        visual_features = self.transformer(patches, pos_embed)

        return visual_features

    def compute_2d_rope(self, grid_thw):
        """
        이미지 패치의 2D 위치에 대한 RoPE 계산
        """
        # ... 2D RoPE 구현
        pass
```

---

## 4. Pattern 3: Early Fusion 방식

### 4.1 Fuyu

> **출처**: Adept AI (2023). "Fuyu-8B: A Multimodal Architecture for AI Agents"
> - Blog: https://www.adept.ai/blog/fuyu-8b/
> - HuggingFace: https://huggingface.co/adept/fuyu-8b

**핵심 특징:**
- **No Vision Encoder**: 이미지 패치를 직접 linear projection
- 텍스트 토큰과 이미지 토큰을 동일하게 처리
- 임의의 해상도 지원

```python
class FuyuImageEncoder(nn.Module):
    """
    Fuyu 스타일의 단순 이미지 인코딩
    Vision Encoder 없이 직접 패치를 embedding
    """
    def __init__(self, patch_size=30, hidden_size=4096, image_newline_token_id=None):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        # 단순 linear projection
        self.image_projection = nn.Linear(
            patch_size * patch_size * 3,
            hidden_size
        )

        # 이미지 줄바꿈 토큰
        self.image_newline_embedding = nn.Embedding(1, hidden_size)

    def forward(self, pixel_values):
        """
        pixel_values: (batch, channels, height, width)
        returns: (batch, num_patches + newlines, hidden_size)
        """
        B, C, H, W = pixel_values.shape

        # 패치로 분할
        patches = pixel_values.unfold(2, self.patch_size, self.patch_size)
        patches = patches.unfold(3, self.patch_size, self.patch_size)
        # (B, C, H', W', patch_size, patch_size)

        h_patches = patches.size(2)
        w_patches = patches.size(3)

        # Flatten patches
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, h_patches, w_patches, -1)  # (B, H', W', P*P*C)

        # Linear projection
        patch_embeddings = self.image_projection(patches)  # (B, H', W', D)

        # 각 행 끝에 newline 토큰 추가
        outputs = []
        for h in range(h_patches):
            row = patch_embeddings[:, h, :, :]  # (B, W', D)
            newline = self.image_newline_embedding.weight.expand(B, 1, -1)
            row_with_newline = torch.cat([row, newline], dim=1)
            outputs.append(row_with_newline)

        image_embeddings = torch.cat(outputs, dim=1)
        return image_embeddings


class Fuyu(nn.Module):
    """
    Fuyu: 단순한 VLM 아키텍처
    """
    def __init__(self, llm, image_encoder):
        super().__init__()
        self.llm = llm
        self.image_encoder = image_encoder

    def forward(self, pixel_values, input_ids, attention_mask):
        # 1. 이미지를 직접 embedding
        image_embeds = self.image_encoder(pixel_values)

        # 2. 텍스트 embedding
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # 3. 결합 [image] + [text]
        combined = torch.cat([image_embeds, text_embeds], dim=1)

        # 4. LLM 처리 (이미지 토큰도 텍스트 토큰처럼)
        outputs = self.llm(inputs_embeds=combined)

        return outputs
```

**Fuyu의 장점:**
- 아키텍처 단순성
- 임의 해상도 지원
- End-to-end 학습
- 빠른 추론 (100ms 미만 for large images)

**단점:**
- 대규모 데이터 필요
- 사전학습된 Vision Encoder의 이점 활용 불가

---

## 5. 아키텍처 선택 가이드

### 5.1 데이터 규모별 권장

| 데이터 규모 | 권장 패턴 | 이유 |
|------------|----------|------|
| < 100K | Projection (LLaVA) | 사전학습 모델 최대 활용 |
| 100K - 10M | Projection / Cross-Attention | 균형 잡힌 선택 |
| > 10M | Early Fusion 고려 가능 | End-to-end 학습 가능 |

### 5.2 사용 사례별 권장

| 사용 사례 | 권장 모델 | 이유 |
|----------|----------|------|
| 일반 VQA | LLaVA-1.5 | 검증된 성능, 학습 용이 |
| 문서 이해 | Qwen2-VL | Dynamic resolution |
| Few-shot 학습 | Flamingo 스타일 | In-context learning |
| 실시간 응용 | Fuyu | 빠른 추론 |

### 5.3 리소스별 권장

| GPU 메모리 | 권장 접근법 |
|-----------|------------|
| 24GB | LLaVA-7B + LoRA |
| 40GB | LLaVA-13B 또는 Qwen2-VL-7B |
| 80GB+ | Full fine-tuning 가능 |

---

## 핵심 참고 자료

### 논문
- **Flamingo** (Alayrac et al., 2022)
  - https://arxiv.org/abs/2204.14198

- **LLaVA** (Liu et al., 2023)
  - https://arxiv.org/abs/2304.08485
  - GitHub: https://github.com/haotian-liu/LLaVA

- **Qwen2-VL** (Wang et al., 2024)
  - https://arxiv.org/abs/2409.12191
  - GitHub: https://github.com/QwenLM/Qwen2-VL

- **InternVL** (Chen et al., 2024)
  - CVPR 2024
  - GitHub: https://github.com/OpenGVLab/InternVL

### 튜토리얼
- **Aman's AI Journal: VLM Architectures**
  - https://aman.ai/primers/ai/VLM/

---

## 핵심 요약

| 패턴 | 핵심 컴포넌트 | 학습 전략 | 적합한 상황 |
|------|-------------|----------|------------|
| Cross-Attention | Perceiver, Gated Attention | Vision frozen, attention 학습 | Few-shot, 다중 이미지 |
| Projection | MLP Projector | Vision frozen, projector+LLM 학습 | 일반 VQA, 제한된 데이터 |
| Early Fusion | Linear Patch Embedding | End-to-end 학습 | 대규모 데이터, 특수 도메인 |
