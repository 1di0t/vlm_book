---
---

# 9.3 Multi-resolution 처리

의료 문서는 크기와 비율이 천차만별이다. A4 진단서(2480×3508), 영수증(400×1200), 보험 청구서(1700×2200), 약봉지 사진(3000×4000). 이걸 하나의 해상도로 리사이즈하면 작은 글씨가 뭉개지거나 패딩 낭비가 심해진다. VLM에서 가변 해상도를 효율적으로 처리하는 것은 OCR 성능의 핵심이다. 이 절에서는 Aspect Ratio Bucketing, Dynamic Padding, 해상도 보간의 수학적 원리를 다루고, 실전 코드를 구현한다.

---

## 핵심 용어

| 용어 | 정의 | 관련 기법 |
|------|------|-----------|
| **Dynamic Batching** | 배치 내 샘플들의 크기에 따라 배치 구성을 동적으로 조절하는 기법 | 가변 시퀀스 길이, 가변 해상도 |
| **Padding** | 배치 내 크기를 통일하기 위해 빈 영역을 채우는 것 | zero-padding, attention mask |
| **Aspect Ratio Bucketing** | 유사한 종횡비의 이미지를 같은 버킷에 모아 배치하는 전략 | SDXL, Qwen2-VL |
| **Variable Resolution** | 고정 크기 대신 이미지 원본 비율을 유지하는 처리 방식 | NaViT, Qwen2-VL Dynamic Resolution |
| **Bilinear Interpolation** | 4개 인접 픽셀의 가중 평균으로 새 픽셀값을 계산하는 보간법 | 이미지 리사이즈 |
| **Bicubic Interpolation** | 16개 인접 픽셀 기반의 3차 다항식 보간법 | 고품질 리사이즈 |
| **Attention Mask** | 패딩 영역을 attention 계산에서 제외하는 마스크 | Transformer |
| **Tile-based Processing** | 고해상도 이미지를 타일로 분할하여 처리하는 방식 | LLaVA-NeXT, InternVL2 |

---

## 9.3.1 왜 Multi-resolution이 중요한가?

### 고정 해상도의 한계

대부분의 Vision Encoder(ViT)는 고정 입력 크기(예: 224×224, 336×336)로 학습됐다. 모든 이미지를 이 크기로 리사이즈하면:

```
원본 A4 진단서 (2480 × 3508)
    ↓ Resize to 336×336
    ↓ 종횡비 왜곡: 0.71 → 1.0
    ↓ 해상도 손실: 99.6% 픽셀 제거
    → 작은 글씨 판독 불가
```

### 해상도별 정보 손실

| 원본 해상도 | 타겟 336×336 | 픽셀 유지율 | 텍스트 판독 |
|------------|-------------|-----------|-----------|
| 640×480 | 336×336 | 36.7% | 대체로 가능 |
| 1920×1080 | 336×336 | 5.4% | 작은 글씨 뭉개짐 |
| 2480×3508 | 336×336 | 1.3% | 거의 불가 |

### VLM별 Multi-resolution 전략

| 모델 | 전략 | 최대 해상도 | 방식 |
|------|------|-----------|------|
| Qwen2-VL | Dynamic Resolution | 가변 | 원본 비율 유지, 패치 수 동적 조절 |
| LLaVA-NeXT | AnyRes | 672×672 (타일) | 이미지를 타일로 분할 후 개별 처리 |
| InternVL2 | Dynamic Image Size | 448×448 (타일) | 1~12개 타일 동적 분할 |
| Phi-3-Vision | HD Transform | 1344×1344 | 패딩 최소화 리사이즈 |

---

## 수학적 원리

### Aspect Ratio Bucketing

이미지 집합 $\{I_1, I_2, \ldots, I_N\}$에서 각 이미지의 종횡비:

$$
r_i = \frac{W_i}{H_i}
$$

$K$개의 버킷 $\{B_1, B_2, \ldots, B_K\}$을 정의하고, 각 버킷의 대표 종횡비를 $\hat{r}_k$로 설정한다. 이미지 $I_i$는 가장 가까운 버킷에 할당된다:

$$
k^* = \arg\min_{k \in \{1, \ldots, K\}} \left| r_i - \hat{r}_k \right|
$$

버킷 내에서 모든 이미지를 동일 크기로 리사이즈하므로 패딩이 필요 없다. 총 픽셀 수 제약 $T$를 두고 버킷 크기를 결정한다:

$$
\hat{W}_k \times \hat{H}_k \leq T, \quad \frac{\hat{W}_k}{\hat{H}_k} = \hat{r}_k
$$

이를 풀면:

$$
\hat{H}_k = \left\lfloor \sqrt{\frac{T}{\hat{r}_k}} \right\rfloor, \quad \hat{W}_k = \left\lfloor \hat{r}_k \cdot \hat{H}_k \right\rfloor
$$

패치 크기 $P$의 배수로 정렬:

$$
\hat{H}_k \leftarrow P \cdot \left\lfloor \frac{\hat{H}_k}{P} \right\rfloor, \quad \hat{W}_k \leftarrow P \cdot \left\lfloor \frac{\hat{W}_k}{P} \right\rfloor
$$

### 패딩 효율

버킷 없이 단일 크기 $(W_{\max}, H_{\max})$로 패딩할 때의 낭비:

$$
\eta_{\text{no\_bucket}} = 1 - \frac{\sum_{i=1}^{N} W_i \cdot H_i}{N \cdot W_{\max} \cdot H_{\max}}
$$

버킷 $K$개를 사용할 때의 낭비:

$$
\eta_{\text{bucket}} = 1 - \frac{\sum_{k=1}^{K} \sum_{i \in B_k} W_i \cdot H_i}{\sum_{k=1}^{K} |B_k| \cdot \hat{W}_k \cdot \hat{H}_k}
$$

일반적으로 $\eta_{\text{bucket}} \ll \eta_{\text{no\_bucket}}$이다. 의료 문서처럼 종횡비 분포가 넓은 데이터셋에서는 버킷 수 $K=8\sim16$으로 패딩 낭비를 80% 이상 줄일 수 있다.

### Bilinear Interpolation

이미지를 리사이즈할 때 새 좌표 $(x, y)$가 원본의 정수 그리드 위에 정확히 놓이지 않으면 보간이 필요하다. Bilinear Interpolation은 4개 인접 픽셀을 사용한다:

$$
f(x, y) = (1-a)(1-b) \cdot f_{00} + a(1-b) \cdot f_{10} + (1-a)b \cdot f_{01} + ab \cdot f_{11}
$$

여기서:
- $f_{00}, f_{10}, f_{01}, f_{11}$: 인접 4개 픽셀값
- $a = x - \lfloor x \rfloor$: x 방향 소수부
- $b = y - \lfloor y \rfloor$: y 방향 소수부

```
f_00 ──── f_10
  │  (x,y)  │
  │    ·    │
f_01 ──── f_11
```

행렬 형태로:

$$
f(x, y) = \begin{bmatrix} 1-a & a \end{bmatrix} \begin{bmatrix} f_{00} & f_{01} \\ f_{10} & f_{11} \end{bmatrix} \begin{bmatrix} 1-b \\ b \end{bmatrix}
$$

### Bicubic Interpolation

Bicubic은 16개 인접 픽셀(4×4)을 사용하며 3차 다항식으로 보간한다:

$$
f(x, y) = \sum_{m=-1}^{2} \sum_{n=-1}^{2} f_{m,n} \cdot w(x - x_m) \cdot w(y - y_n)
$$

커널 함수 $w(t)$ (Keys 보간):

$$
w(t) = \begin{cases}
(a+2)|t|^3 - (a+3)|t|^2 + 1 & \text{if } |t| \leq 1 \\
a|t|^3 - 5a|t|^2 + 8a|t| - 4a & \text{if } 1 < |t| < 2 \\
0 & \text{otherwise}
\end{cases}
$$

$a = -0.5$가 일반적이다. Bicubic은 Bilinear보다 선명하지만 연산량이 4배 많다. 의료 문서 OCR에서는 Bicubic을 권장한다 — 텍스트 경계가 더 선명하게 유지된다.

### 토큰 수 vs 해상도 트레이드오프

패치 크기 $P$로 이미지를 토큰화할 때 토큰 수:

$$
N_{\text{tokens}} = \left\lceil \frac{W}{P} \right\rceil \times \left\lceil \frac{H}{P} \right\rceil
$$

Attention의 연산 복잡도는 $O(N^2)$이므로:

$$
\text{Compute} \propto N_{\text{tokens}}^2 = \left(\frac{WH}{P^2}\right)^2
$$

해상도를 2배 올리면 토큰 수 4배, 연산량 16배가 된다. 이것이 무제한 고해상도가 불가능한 근본적 이유다.

---

## 9.3.2 Aspect Ratio Bucketing 구현

```python
"""
Chapter 9.3 - Multi-resolution 처리
AspectRatioBucketSampler, DynamicPaddingCollator 구현
"""

import math
import logging
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. Aspect Ratio Bucket 정의 및 할당
# ──────────────────────────────────────────────

class AspectRatioBucket:
    """
    종횡비 버킷 관리자

    이미지들을 종횡비 기준으로 그룹화하여
    배치 내 패딩 낭비를 최소화한다.
    """

    # 기본 버킷 종횡비 (가로:세로)
    DEFAULT_RATIOS = [
        (1, 1),     # 1.0   정사각형
        (4, 3),     # 1.33  일반 사진
        (3, 4),     # 0.75  세로 사진
        (16, 9),    # 1.78  와이드
        (9, 16),    # 0.56  세로 와이드
        (3, 2),     # 1.5   일반 문서
        (2, 3),     # 0.67  세로 문서
        (1, 2),     # 0.5   긴 세로 (영수증)
        (2, 1),     # 2.0   긴 가로 (배너)
        (5, 7),     # 0.71  A4 세로
        (7, 5),     # 1.4   A4 가로
        (3, 5),     # 0.6   세로 문서 (좁은)
        (5, 3),     # 1.67  가로 문서 (넓은)
    ]

    def __init__(
        self,
        target_pixels: int = 448 * 448,
        patch_size: int = 14,
        ratios: list[tuple[int, int]] | None = None,
        min_size: int = 56,
    ):
        """
        Args:
            target_pixels: 버킷당 목표 총 픽셀 수
            patch_size: ViT 패치 크기 (정렬 단위)
            ratios: 버킷 종횡비 리스트
            min_size: 최소 변 길이
        """
        self.target_pixels = target_pixels
        self.patch_size = patch_size
        self.min_size = min_size

        ratios = ratios or self.DEFAULT_RATIOS
        self.buckets = self._compute_bucket_sizes(ratios)

        logger.info(
            f"버킷 {len(self.buckets)}개 생성 "
            f"(target_pixels={target_pixels}, patch_size={patch_size})"
        )
        for ratio, (w, h) in sorted(
            self.buckets.items(), key=lambda x: x[0]
        ):
            logger.debug(f"  ratio={ratio:.3f} → {w}×{h}")

    def _compute_bucket_sizes(
        self,
        ratios: list[tuple[int, int]],
    ) -> dict[float, tuple[int, int]]:
        """
        종횡비별 버킷 크기 계산

        목표 총 픽셀 수를 유지하면서 패치 크기의 배수로 정렬한다.
        """
        buckets = {}
        for rw, rh in ratios:
            ratio = rw / rh

            # H = sqrt(T / r), W = r * H
            h = math.sqrt(self.target_pixels / ratio)
            w = ratio * h

            # 패치 크기 배수로 정렬
            h = max(self.patch_size * round(h / self.patch_size), self.min_size)
            w = max(self.patch_size * round(w / self.patch_size), self.min_size)

            actual_ratio = w / h
            buckets[actual_ratio] = (int(w), int(h))

        return buckets

    def assign_bucket(
        self,
        width: int,
        height: int,
    ) -> tuple[float, tuple[int, int]]:
        """
        이미지 크기 → 가장 적합한 버킷 할당

        Args:
            width: 이미지 너비
            height: 이미지 높이

        Returns:
            (종횡비, (버킷_너비, 버킷_높이))
        """
        image_ratio = width / height

        best_ratio = min(
            self.buckets.keys(),
            key=lambda r: abs(r - image_ratio),
        )
        return best_ratio, self.buckets[best_ratio]

    def compute_padding_efficiency(
        self,
        image_sizes: list[tuple[int, int]],
    ) -> dict[str, float]:
        """
        패딩 효율 분석

        Args:
            image_sizes: [(width, height), ...] 이미지 크기 리스트

        Returns:
            효율 통계 딕셔너리
        """
        total_actual_pixels = 0
        total_bucket_pixels = 0
        total_fixed_pixels = 0

        max_w = max(w for w, h in image_sizes)
        max_h = max(h for w, h in image_sizes)

        for w, h in image_sizes:
            total_actual_pixels += w * h

            _, (bw, bh) = self.assign_bucket(w, h)
            total_bucket_pixels += bw * bh

            total_fixed_pixels += max_w * max_h

        n = len(image_sizes)
        bucket_efficiency = total_actual_pixels / total_bucket_pixels
        fixed_efficiency = total_actual_pixels / total_fixed_pixels

        return {
            "num_images": n,
            "bucket_efficiency": bucket_efficiency,
            "fixed_efficiency": fixed_efficiency,
            "bucket_waste": 1 - bucket_efficiency,
            "fixed_waste": 1 - fixed_efficiency,
            "improvement": (1 - fixed_efficiency) - (1 - bucket_efficiency),
        }


# ──────────────────────────────────────────────
# 2. Aspect Ratio Bucket Sampler
# ──────────────────────────────────────────────

class AspectRatioBucketSampler(Sampler):
    """
    PyTorch Sampler: 종횡비 기준으로 배치를 구성한다.

    같은 버킷의 이미지끼리 배치를 만들어 패딩을 최소화한다.
    에폭마다 버킷 내부를 셔플하고, 버킷 순서도 셔플한다.
    """

    def __init__(
        self,
        dataset: Dataset,
        bucket_manager: AspectRatioBucket,
        batch_size: int = 4,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            dataset: 이미지 크기 정보를 가진 데이터셋
            bucket_manager: AspectRatioBucket 인스턴스
            batch_size: 배치 크기
            drop_last: 마지막 불완전 배치 버림 여부
            shuffle: 셔플 여부
            seed: 랜덤 시드
        """
        self.dataset = dataset
        self.bucket_manager = bucket_manager
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # 이미지 크기별 버킷 할당
        self.bucket_indices = self._assign_all_to_buckets()

        total_samples = sum(len(v) for v in self.bucket_indices.values())
        logger.info(
            f"BucketSampler: {total_samples}개 샘플 → "
            f"{len(self.bucket_indices)}개 버킷"
        )

    def _assign_all_to_buckets(self) -> dict[float, list[int]]:
        """모든 샘플을 버킷에 할당"""
        bucket_indices: dict[float, list[int]] = defaultdict(list)

        for idx in range(len(self.dataset)):
            # 데이터셋에서 이미지 크기 정보 가져오기
            size = self._get_image_size(idx)
            if size is None:
                continue

            w, h = size
            ratio, _ = self.bucket_manager.assign_bucket(w, h)
            bucket_indices[ratio].append(idx)

        return dict(bucket_indices)

    def _get_image_size(self, idx: int) -> tuple[int, int] | None:
        """데이터셋에서 이미지 크기 추출 (서브클래스에서 오버라이드 가능)"""
        if hasattr(self.dataset, "get_image_size"):
            return self.dataset.get_image_size(idx)

        # fallback: 메타데이터에서 추출
        if hasattr(self.dataset, "samples"):
            sample = self.dataset.samples[idx]
            meta = sample.get("metadata", {})
            if "image_resolution" in meta:
                return tuple(meta["image_resolution"])
            if "width" in sample and "height" in sample:
                return (sample["width"], sample["height"])

        return None

    def __iter__(self):
        """배치 단위 인덱스 생성"""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        all_batches = []

        for ratio, indices in self.bucket_indices.items():
            # 버킷 내부 셔플
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in perm]

            # 배치 단위로 분할
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        # 버킷 간 순서 셔플
        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in perm]

        # Flatten: 배치 내 인덱스를 순서대로 yield
        for batch in all_batches:
            yield from batch

    def __len__(self) -> int:
        total = sum(len(v) for v in self.bucket_indices.values())
        if self.drop_last:
            return (total // self.batch_size) * self.batch_size
        return total

    def set_epoch(self, epoch: int) -> None:
        """에폭 설정 (DDP에서 매 에폭 호출)"""
        self.epoch = epoch


# ──────────────────────────────────────────────
# 3. Dynamic Padding Collator
# ──────────────────────────────────────────────

class DynamicPaddingCollator:
    """
    동적 패딩 Collate 함수

    배치 내 이미지 크기가 다를 때,
    가장 큰 이미지 크기에 맞춰 패딩한다.
    Attention Mask도 함께 생성한다.
    """

    def __init__(
        self,
        pad_value: float = 0.0,
        patch_size: int = 14,
        pad_to_multiple: bool = True,
    ):
        """
        Args:
            pad_value: 패딩 값 (보통 0)
            patch_size: 패치 크기 배수로 패딩
            pad_to_multiple: 패치 크기 배수로 정렬할지
        """
        self.pad_value = pad_value
        self.patch_size = patch_size
        self.pad_to_multiple = pad_to_multiple

    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        """
        배치 collate

        Args:
            batch: [{"id": str, "image": Tensor(C,H,W), "conversations": list}]

        Returns:
            {
                "ids": list[str],
                "images": Tensor(B,C,H_max,W_max),
                "attention_mask": Tensor(B,H_max,W_max),  # 1=유효, 0=패딩
                "original_sizes": list[tuple[int,int]],
                "conversations": list[list],
            }
        """
        ids = [item["id"] for item in batch]
        conversations = [item["conversations"] for item in batch]

        # 이미지 처리
        images = [item["image"] for item in batch if item.get("image") is not None]
        if not images:
            return {
                "ids": ids,
                "images": None,
                "attention_mask": None,
                "original_sizes": [],
                "conversations": conversations,
            }

        # 최대 크기 계산
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)

        # 패치 크기 배수로 올림
        if self.pad_to_multiple:
            max_h = self._ceil_to_multiple(max_h, self.patch_size)
            max_w = self._ceil_to_multiple(max_w, self.patch_size)

        # 패딩 + 마스크 생성
        channels = images[0].shape[0]
        padded_images = torch.full(
            (len(images), channels, max_h, max_w),
            fill_value=self.pad_value,
        )
        attention_mask = torch.zeros(len(images), max_h, max_w)
        original_sizes = []

        for i, img in enumerate(images):
            c, h, w = img.shape
            padded_images[i, :, :h, :w] = img
            attention_mask[i, :h, :w] = 1.0
            original_sizes.append((w, h))

        return {
            "ids": ids,
            "images": padded_images,
            "attention_mask": attention_mask,
            "original_sizes": original_sizes,
            "conversations": conversations,
        }

    @staticmethod
    def _ceil_to_multiple(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple


# ──────────────────────────────────────────────
# 4. Multi-resolution 이미지 전처리기
# ──────────────────────────────────────────────

class MultiResolutionProcessor:
    """
    가변 해상도 이미지 전처리기

    원본 종횡비를 유지하면서 목표 해상도 범위 내로 리사이즈한다.
    """

    def __init__(
        self,
        min_pixels: int = 224 * 224,
        max_pixels: int = 1024 * 1024,
        patch_size: int = 14,
        interpolation: str = "bicubic",
    ):
        """
        Args:
            min_pixels: 최소 총 픽셀 수
            max_pixels: 최대 총 픽셀 수
            patch_size: 패치 크기 (정렬 단위)
            interpolation: 보간법 ("bilinear" 또는 "bicubic")
        """
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size

        interp_map = {
            "bilinear": "bilinear",
            "bicubic": "bicubic",
        }
        self.interpolation = interp_map.get(interpolation, "bicubic")

    def resize_image(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        종횡비 유지 리사이즈

        Args:
            image: (C, H, W) 텐서

        Returns:
            (리사이즈된 이미지, 메타정보)
        """
        c, h, w = image.shape
        original_pixels = h * w

        # 스케일 팩터 결정
        if original_pixels > self.max_pixels:
            scale = math.sqrt(self.max_pixels / original_pixels)
        elif original_pixels < self.min_pixels:
            scale = math.sqrt(self.min_pixels / original_pixels)
        else:
            scale = 1.0

        new_h = int(h * scale)
        new_w = int(w * scale)

        # 패치 크기 배수로 정렬
        new_h = max(
            self.patch_size,
            self.patch_size * round(new_h / self.patch_size),
        )
        new_w = max(
            self.patch_size,
            self.patch_size * round(new_w / self.patch_size),
        )

        # 리사이즈
        resized = F.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode=self.interpolation,
            align_corners=False if self.interpolation == "bicubic" else None,
        ).squeeze(0)

        meta = {
            "original_size": (w, h),
            "resized_size": (new_w, new_h),
            "scale": scale,
            "num_patches": (new_h // self.patch_size) * (new_w // self.patch_size),
        }

        return resized, meta

    def process_batch_variable(
        self,
        images: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[dict]]:
        """
        배치 내 각 이미지를 개별 해상도로 처리

        Returns:
            (리사이즈된 이미지 리스트, 메타정보 리스트)
        """
        results = []
        metas = []
        for img in images:
            resized, meta = self.resize_image(img)
            results.append(resized)
            metas.append(meta)
        return results, metas


# ──────────────────────────────────────────────
# 5. 타일 기반 처리 (LLaVA-NeXT / InternVL2 스타일)
# ──────────────────────────────────────────────

class TileProcessor:
    """
    고해상도 이미지를 타일로 분할하여 처리

    큰 이미지를 여러 개의 작은 타일로 나누고,
    전체 이미지의 축소본(thumbnail)과 함께 처리한다.
    """

    def __init__(
        self,
        tile_size: int = 448,
        max_tiles: int = 12,
        min_tiles: int = 1,
        use_thumbnail: bool = True,
    ):
        """
        Args:
            tile_size: 타일 한 변 크기
            max_tiles: 최대 타일 수
            min_tiles: 최소 타일 수
            use_thumbnail: 전체 축소본 추가 여부
        """
        self.tile_size = tile_size
        self.max_tiles = max_tiles
        self.min_tiles = min_tiles
        self.use_thumbnail = use_thumbnail

    def compute_tile_grid(
        self,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        """
        이미지 크기 → 최적 타일 그리드 결정

        목표: 원본 비율을 유지하면서 max_tiles 이내로

        Args:
            width, height: 원본 이미지 크기

        Returns:
            (cols, rows): 타일 그리드
        """
        ratio = width / height
        best_grid = (1, 1)
        best_waste = float("inf")

        for total in range(self.min_tiles, self.max_tiles + 1):
            for cols in range(1, total + 1):
                rows = total // cols
                if rows == 0 or cols * rows > self.max_tiles:
                    continue

                grid_ratio = cols / rows
                # 종횡비 차이 + 타일 낭비 최소화
                ratio_diff = abs(grid_ratio - ratio) / max(grid_ratio, ratio)
                used = cols * rows
                waste = ratio_diff + 0.1 * (1 - used / self.max_tiles)

                if waste < best_waste:
                    best_waste = waste
                    best_grid = (cols, rows)

        return best_grid

    def split_into_tiles(
        self,
        image: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None, dict]:
        """
        이미지 → 타일 분할

        Args:
            image: (C, H, W) 텐서

        Returns:
            (타일 리스트, 썸네일, 메타정보)
        """
        c, h, w = image.shape
        cols, rows = self.compute_tile_grid(w, h)

        # 그리드에 맞게 리사이즈
        grid_w = cols * self.tile_size
        grid_h = rows * self.tile_size

        resized = F.interpolate(
            image.unsqueeze(0),
            size=(grid_h, grid_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        # 타일 분할
        tiles = []
        for row in range(rows):
            for col in range(cols):
                y1 = row * self.tile_size
                y2 = y1 + self.tile_size
                x1 = col * self.tile_size
                x2 = x1 + self.tile_size
                tile = resized[:, y1:y2, x1:x2]
                tiles.append(tile)

        # 썸네일 (전체 이미지 축소)
        thumbnail = None
        if self.use_thumbnail:
            thumbnail = F.interpolate(
                image.unsqueeze(0),
                size=(self.tile_size, self.tile_size),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)

        meta = {
            "original_size": (w, h),
            "grid": (cols, rows),
            "num_tiles": len(tiles),
            "has_thumbnail": self.use_thumbnail,
            "total_tokens": (
                len(tiles) + (1 if self.use_thumbnail else 0)
            ) * (self.tile_size // 14) ** 2,
        }

        return tiles, thumbnail, meta

    def merge_tile_features(
        self,
        tile_features: list[torch.Tensor],
        grid: tuple[int, int],
    ) -> torch.Tensor:
        """
        타일별 feature를 원본 공간 배치로 합침

        Args:
            tile_features: 각 타일의 feature [(N_patch, D), ...]
            grid: (cols, rows)

        Returns:
            (total_patches, D)
        """
        cols, rows = grid
        row_features = []

        for r in range(rows):
            col_features = []
            for c in range(cols):
                idx = r * cols + c
                col_features.append(tile_features[idx])
            row_features.append(torch.cat(col_features, dim=0))

        return torch.cat(row_features, dim=0)


# ──────────────────────────────────────────────
# 6. 종합 Multi-resolution DataLoader
# ──────────────────────────────────────────────

class MultiResolutionDataLoader:
    """
    Multi-resolution 전체 파이프라인을 통합한 DataLoader 생성기
    """

    @staticmethod
    def create(
        dataset: Dataset,
        batch_size: int = 4,
        target_pixels: int = 448 * 448,
        patch_size: int = 14,
        num_workers: int = 4,
        strategy: str = "bucket",
    ) -> DataLoader:
        """
        Multi-resolution DataLoader 생성

        Args:
            dataset: 데이터셋
            batch_size: 배치 크기
            target_pixels: 목표 총 픽셀 수
            patch_size: 패치 크기
            num_workers: 워커 수
            strategy: "bucket" (Aspect Ratio Bucketing) 또는 "dynamic" (Dynamic Padding)
        """
        if strategy == "bucket":
            bucket_manager = AspectRatioBucket(
                target_pixels=target_pixels,
                patch_size=patch_size,
            )
            sampler = AspectRatioBucketSampler(
                dataset=dataset,
                bucket_manager=bucket_manager,
                batch_size=batch_size,
                shuffle=True,
            )
            collator = DynamicPaddingCollator(
                patch_size=patch_size,
                pad_to_multiple=True,
            )

            return DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collator,
                persistent_workers=num_workers > 0,
                prefetch_factor=2 if num_workers > 0 else None,
            )

        elif strategy == "dynamic":
            collator = DynamicPaddingCollator(
                patch_size=patch_size,
                pad_to_multiple=True,
            )

            return DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collator,
                persistent_workers=num_workers > 0,
                prefetch_factor=2 if num_workers > 0 else None,
            )

        else:
            raise ValueError(f"지원하지 않는 전략: {strategy}")
```

---

## 9.3.3 패딩 효율 분석 실험

```python
def analyze_padding_efficiency():
    """의료 문서 데이터셋의 패딩 효율 분석"""

    # 의료 문서 해상도 분포 시뮬레이션
    medical_doc_sizes = [
        # (width, height)
        (2480, 3508),  # A4 300dpi (진단서)
        (2480, 3508),
        (2480, 3508),
        (1700, 2200),  # 보험 청구서
        (1700, 2200),
        (400, 1200),   # 영수증
        (400, 1200),
        (400, 1200),
        (3000, 4000),  # 약봉지 사진
        (1920, 1080),  # 스크린 캡처
        (1920, 1080),
        (800, 600),    # 저해상도 스캔
        (640, 480),    # 촬영 사진
        (1024, 768),   # 태블릿 캡처
        (3024, 4032),  # 스마트폰 세로
        (4032, 3024),  # 스마트폰 가로
    ]

    # 1. 기본 (버킷 없음) vs 버킷 비교
    bucket = AspectRatioBucket(target_pixels=448 * 448)
    stats = bucket.compute_padding_efficiency(medical_doc_sizes)

    print("=" * 60)
    print("패딩 효율 분석 (의료 문서 데이터셋)")
    print("=" * 60)
    print(f"이미지 수: {stats['num_images']}")
    print(f"\n고정 크기 패딩:")
    print(f"  효율: {stats['fixed_efficiency']:.1%}")
    print(f"  낭비: {stats['fixed_waste']:.1%}")
    print(f"\nAspect Ratio Bucketing:")
    print(f"  효율: {stats['bucket_efficiency']:.1%}")
    print(f"  낭비: {stats['bucket_waste']:.1%}")
    print(f"\n개선량: {stats['improvement']:.1%}p")

    # 2. 버킷 할당 결과
    print("\n[버킷 할당 결과]")
    for w, h in medical_doc_sizes[:6]:
        ratio, (bw, bh) = bucket.assign_bucket(w, h)
        print(
            f"  {w}×{h} (r={w/h:.2f}) → "
            f"버킷 {bw}×{bh} (r={ratio:.2f})"
        )

    # 3. 타일 분할 분석
    print("\n[타일 분할 분석]")
    tile_proc = TileProcessor(tile_size=448, max_tiles=12)
    for w, h in [(2480, 3508), (400, 1200), (3000, 4000)]:
        cols, rows = tile_proc.compute_tile_grid(w, h)
        num_tiles = cols * rows
        total_tokens = (num_tiles + 1) * (448 // 14) ** 2  # +1 thumbnail
        print(
            f"  {w}×{h} → {cols}×{rows} grid "
            f"({num_tiles} tiles, {total_tokens} tokens)"
        )


if __name__ == "__main__":
    analyze_padding_efficiency()
```

---

## 9.3.4 실전 팁

### 1. 버킷 수 선택 가이드

버킷이 너무 적으면 패딩 낭비가 크고, 너무 많으면 버킷당 샘플 수가 적어져서 배치 구성이 비효율적이다.

```
버킷 수 vs 패딩 효율:
  K=4  → 약 30% 패딩 낭비
  K=8  → 약 15% 패딩 낭비
  K=13 → 약 8% 패딩 낭비
  K=20 → 약 5% 패딩 낭비 (but 배치 불균형 위험)
```

의료 문서처럼 종횡비 분산이 큰 데이터셋은 $K=12\sim16$이 적절하다.

### 2. DDP에서의 주의사항

분산 학습(DDP)에서 BucketSampler를 쓸 때 `set_epoch()`을 반드시 호출해야 한다. 안 하면 매 에폭 같은 셔플 순서가 반복된다.

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # 필수!
    for batch in dataloader:
        ...
```

### 3. 메모리 관리

고해상도 이미지를 처리하면 GPU 메모리가 폭발할 수 있다. 배치 내 최대 총 토큰 수를 제한하는 것이 안전하다.

```python
# 배치 크기를 토큰 수 기준으로 동적 조절
MAX_TOKENS_PER_BATCH = 4096

def dynamic_batch_size(image_sizes, base_batch=8, patch_size=14):
    """토큰 수 기준 동적 배치 크기 결정"""
    tokens_per_image = [
        (w // patch_size) * (h // patch_size) for w, h in image_sizes
    ]
    max_tokens = max(tokens_per_image)
    return max(1, MAX_TOKENS_PER_BATCH // max_tokens)
```

### 4. Bicubic vs Bilinear 성능 차이

의료 문서 OCR에서 보간법 선택은 텍스트 판독 정확도에 직결된다:

| 보간법 | 텍스트 선명도 | 연산 비용 | 권장 용도 |
|--------|------------|----------|----------|
| Nearest | 계단 현상 (최악) | 1x | 절대 비권장 |
| Bilinear | 보통 | 1x | 실시간 추론 |
| Bicubic | 선명 | ~4x | 학습 데이터, OCR |
| Lanczos | 가장 선명 | ~16x | 전처리 (오프라인) |

---

## 용어 체크리스트

학습 후 아래 질문에 답할 수 있는지 스스로 점검해라.

| # | 체크 항목 | 핵심 키워드 |
|---|----------|------------|
| 1 | 고정 해상도 리사이즈의 문제점을 설명할 수 있는가? | 종횡비 왜곡, 정보 손실, 작은 글씨 |
| 2 | Aspect Ratio Bucketing의 수학적 원리를 설명할 수 있는가? | 종횡비 매칭, 목표 픽셀 제약, 패치 정렬 |
| 3 | 패딩 효율 공식을 유도할 수 있는가? | $\eta = 1 - \text{actual}/\text{padded}$ |
| 4 | Bilinear Interpolation의 수식을 쓸 수 있는가? | 4개 인접 픽셀, $a$, $b$ 가중치 |
| 5 | Bicubic이 Bilinear보다 OCR에서 나은 이유는? | 선명도, 텍스트 경계 보존 |
| 6 | 해상도 2배 → 연산량 16배인 이유를 유도할 수 있는가? | $O(N^2)$, $N \propto WH/P^2$ |
| 7 | 타일 기반 처리(LLaVA-NeXT)의 장단점은? | 고해상도 처리 가능, 타일 경계 정보 손실 |
| 8 | AspectRatioBucketSampler가 DDP에서 `set_epoch()` 필요한 이유는? | 셔플 시드 변경, 에폭간 다양성 |
| 9 | Dynamic Padding에서 Attention Mask의 역할은? | 패딩 영역 attention 제외 |
| 10 | 배치 크기를 토큰 수 기준으로 동적 조절하는 이유는? | GPU 메모리 제한, 가변 해상도 |
