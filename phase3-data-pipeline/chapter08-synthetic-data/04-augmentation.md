# 8.4 데이터 증강 기법

학습 데이터가 충분하더라도, 데이터 증강(augmentation)은 모델의 일반화 성능을 끌어올리는 핵심 전략이다. 특히 OCR에서는 텍스트 영역의 가독성을 보존하면서 다양한 변형을 가해야 하므로, 일반적인 이미지 분류 증강과는 다른 접근이 필요하다. 이 장에서는 MixUp, CutMix, CutOut 등의 고급 증강 기법과 OCR 특화 증강 전략을 다룬다.

---

> **핵심 용어 박스**
>
> | 용어 | 정의 |
> |------|------|
> | **Augmentation** | 기존 학습 데이터에 변형을 가해 가상의 새로운 샘플을 생성하는 정규화 기법 |
> | **CutOut** | 이미지의 랜덤 직사각형 영역을 0으로 마스킹하여 과적합을 방지하는 기법 |
> | **MixUp** | 두 이미지와 레이블을 선형 보간하여 새로운 학습 샘플을 생성하는 기법 |
> | **CutMix** | 한 이미지의 일부 영역을 다른 이미지의 해당 영역으로 대체하는 기법 |
> | **Mosaic** | 4개 이미지를 하나의 이미지로 합성하여 다양한 스케일과 컨텍스트를 학습하는 기법 |

---

## 8.4.1 데이터 증강의 원리

### 왜 증강이 필요한가?

학습 데이터 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$이 유한하면, 모델은 학습 데이터에 과적합(overfitting)할 위험이 있다. 증강은 데이터 분포 $p(x, y)$를 확장하여 이를 완화한다.

**기대 리스크**:

$$
R(f) = \mathbb{E}_{(x,y) \sim p(x,y)} [\mathcal{L}(f(x), y)]
$$

학습 데이터 $\mathcal{D}$로는 경험 리스크(empirical risk)만 최소화할 수 있다:

$$
\hat{R}(f) = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(f(x_i), y_i)
$$

증강은 변환 $T$를 적용하여 데이터셋을 확장한다:

$$
\mathcal{D}_{\text{aug}} = \{(T(x_i), y_i) \mid x_i \in \mathcal{D}, T \in \mathcal{T}\}
$$

이때 $\mathcal{T}$는 레이블을 보존하는 변환 집합이다. OCR에서는 텍스트 내용과 위치 정보가 보존되어야 한다.

### 증강 기법 분류

| 범주 | 기법 | 적용 대상 | OCR 적합성 |
|------|------|----------|-----------|
| 기하학적 | 회전, 스케일, 플립 | 위치/형태 | 주의 필요 (텍스트 방향) |
| 색상 | 밝기, 대비, 채도 | 픽셀 값 | 안전 |
| 노이즈 | 가우시안, S&P | 픽셀 값 | 안전 |
| 삭제 | CutOut, Random Erasing | 영역 마스킹 | 주의 필요 (텍스트 보존) |
| 혼합 | MixUp, CutMix | 이미지 + 레이블 | 제한적 |
| 모자이크 | Mosaic, Copy-Paste | 레이아웃 | 적합 |

---

## 수학적 원리

### MixUp

두 학습 샘플 $(x_i, y_i)$와 $(x_j, y_j)$를 선형 보간하여 새 샘플을 생성한다:

$$
\tilde{x} = \lambda x_i + (1 - \lambda) x_j
$$

$$
\tilde{y} = \lambda y_i + (1 - \lambda) y_j
$$

혼합 비율 $\lambda$는 Beta 분포에서 샘플링한다:

$$
\lambda \sim \text{Beta}(\alpha, \alpha), \quad \alpha > 0
$$

Beta 분포의 확률밀도함수:

$$
f(\lambda; \alpha, \alpha) = \frac{\lambda^{\alpha-1}(1-\lambda)^{\alpha-1}}{B(\alpha, \alpha)}
$$

여기서 $B(\alpha, \alpha)$는 베타 함수다.

**$\alpha$에 따른 분포 특성**:
- $\alpha \to 0$: $\lambda \to 0$ 또는 $\lambda \to 1$ (거의 혼합 안 함)
- $\alpha = 1$: $\lambda \sim \mathcal{U}(0, 1)$ (균일 분포)
- $\alpha > 1$: $\lambda \to 0.5$ (강하게 혼합)

OCR에서는 $\alpha \in [0.1, 0.4]$가 적절하다. 너무 강한 혼합은 텍스트 가독성을 파괴한다.

### CutMix

이미지 $x_i$의 직사각형 영역을 $x_j$의 해당 영역으로 대체한다:

$$
\tilde{x} = M \odot x_i + (1 - M) \odot x_j
$$

여기서 $M \in \{0, 1\}^{W \times H}$는 이진 마스크, $\odot$는 원소별 곱이다.

마스크 영역의 비율:

$$
\lambda = 1 - \frac{r_w \cdot r_h}{W \cdot H}
$$

절단 영역의 너비 $r_w$와 높이 $r_h$:

$$
r_w = W\sqrt{1 - \lambda}, \quad r_h = H\sqrt{1 - \lambda}
$$

$\lambda \sim \text{Beta}(\alpha, \alpha)$에서 샘플링한다.

레이블도 면적 비율에 따라 혼합한다:

$$
\tilde{y} = \lambda y_i + (1 - \lambda) y_j
$$

### CutOut (Random Erasing)

이미지의 랜덤 직사각형 영역을 마스킹(0 또는 랜덤 값)한다:

$$
\tilde{x}(p, q) = \begin{cases} 0 & \text{if } (p, q) \in R \\ x(p, q) & \text{otherwise} \end{cases}
$$

여기서 $R$은 랜덤하게 선택된 직사각형 영역이다.

**영역 크기 결정**: 면적 비율 $s$를 균일 분포에서 샘플링:

$$
s \sim \mathcal{U}(s_{\min}, s_{\max}), \quad s_{\min} = 0.02, \; s_{\max} = 0.33
$$

종횡비 $r$도 랜덤 샘플링:

$$
r \sim \mathcal{U}(r_{\min}, r_{\max}), \quad r_{\min} = 0.3, \; r_{\max} = 3.3
$$

$$
h = \sqrt{s \cdot H \cdot W \cdot r}, \quad w = \sqrt{\frac{s \cdot H \cdot W}{r}}
$$

### Mosaic

4개 이미지를 2x2 격자로 합성한다. 분할점 $(c_x, c_y)$를 랜덤하게 선택:

$$
c_x \sim \mathcal{U}(0.25W, 0.75W), \quad c_y \sim \mathcal{U}(0.25H, 0.75H)
$$

각 이미지는 해당 사분면에 크기를 맞춰 배치된다. 레이블(바운딩 박스)도 좌표를 변환하여 합성한다.

---

## 8.4.2 OCR 특화 증강 전략

### 텍스트 영역 보존 증강

일반적인 CutOut/CutMix는 텍스트 영역을 무차별적으로 가릴 수 있다. OCR에서는 텍스트 영역의 바운딩 박스 정보를 활용하여 가독성을 보존해야 한다.

```python
import cv2
import numpy as np
import random
import logging
from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BBox:
    """바운딩 박스."""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str = ""

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def iou(self, other: "BBox") -> float:
        """IoU (Intersection over Union)를 계산한다."""
        xi1 = max(self.x1, other.x1)
        yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2)
        yi2 = min(self.y2, other.y2)

        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = self.area + other.area - inter

        return inter / max(union, 1e-6)

    def overlap_ratio(self, other: "BBox") -> float:
        """self 기준 겹침 비율을 계산한다."""
        xi1 = max(self.x1, other.x1)
        yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2)
        yi2 = min(self.y2, other.y2)

        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        return inter / max(self.area, 1e-6)


@dataclass
class AugmentationConfig:
    """증강 파이프라인 설정."""
    # CutOut
    cutout_prob: float = 0.3
    cutout_num_holes: int = 3
    cutout_max_ratio: float = 0.15
    cutout_fill_value: int = 128

    # MixUp
    mixup_prob: float = 0.2
    mixup_alpha: float = 0.2

    # CutMix
    cutmix_prob: float = 0.2
    cutmix_alpha: float = 1.0

    # 기하학적 변환
    rotate_prob: float = 0.3
    rotate_max_angle: float = 5.0
    scale_prob: float = 0.3
    scale_range: tuple = (0.9, 1.1)

    # 색상 변환
    brightness_prob: float = 0.4
    brightness_range: tuple = (-30, 30)
    contrast_prob: float = 0.4
    contrast_range: tuple = (0.7, 1.3)

    # OCR 특화
    text_safe_cutout: bool = True     # 텍스트 영역 보존 CutOut
    preserve_text_ratio: float = 0.8  # 텍스트 보존 비율 임계값


class OCRAugmentation:
    """OCR 학습에 특화된 데이터 증강 엔진.

    텍스트 영역의 가독성을 보존하면서 다양한 증강을 적용한다.
    albumentations 스타일의 파이프라인 구조를 채택한다.
    """

    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or AugmentationConfig()
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _is_safe_region(
        self,
        region: BBox,
        text_bboxes: list,
        max_overlap: float = 0.2,
    ) -> bool:
        """영역이 텍스트와 과도하게 겹치지 않는지 확인한다."""
        for bbox in text_bboxes:
            if region.overlap_ratio(bbox) > max_overlap:
                return False
        return True

    # ─────────────────────────────────────────────
    # CutOut (텍스트 보존)
    # ─────────────────────────────────────────────

    def apply_cutout(
        self,
        image: np.ndarray,
        text_bboxes: Optional[list] = None,
        num_holes: Optional[int] = None,
        max_ratio: Optional[float] = None,
    ) -> np.ndarray:
        """CutOut을 적용한다. 텍스트 영역은 보존한다.

        x̃(p,q) = 0 if (p,q) ∈ R, else x(p,q)
        """
        h, w = image.shape[:2]
        num_holes = num_holes or self.config.cutout_num_holes
        max_ratio = max_ratio or self.config.cutout_max_ratio

        result = image.copy()

        for _ in range(num_holes):
            # 랜덤 영역 크기
            hole_h = int(h * np.random.uniform(0.02, max_ratio))
            hole_w = int(w * np.random.uniform(0.02, max_ratio))

            # 최대 시도 횟수
            for attempt in range(50):
                # 랜덤 위치
                top = np.random.randint(0, h - hole_h)
                left = np.random.randint(0, w - hole_w)

                hole_bbox = BBox(
                    x1=left, y1=top,
                    x2=left + hole_w, y2=top + hole_h,
                )

                # 텍스트 보존 모드
                if self.config.text_safe_cutout and text_bboxes:
                    if self._is_safe_region(hole_bbox, text_bboxes):
                        result[top:top+hole_h, left:left+hole_w] = self.config.cutout_fill_value
                        break
                else:
                    result[top:top+hole_h, left:left+hole_w] = self.config.cutout_fill_value
                    break

        return result

    # ─────────────────────────────────────────────
    # MixUp
    # ─────────────────────────────────────────────

    def apply_mixup(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        label1: Optional[dict] = None,
        label2: Optional[dict] = None,
        alpha: Optional[float] = None,
    ) -> tuple:
        """MixUp을 적용한다.

        x̃ = λ * x_i + (1-λ) * x_j,  λ ~ Beta(α, α)
        """
        alpha = alpha or self.config.mixup_alpha

        # Beta 분포에서 λ 샘플링
        lam = np.random.beta(alpha, alpha)

        # 크기 맞추기
        h = min(image1.shape[0], image2.shape[0])
        w = min(image1.shape[1], image2.shape[1])

        img1_resized = cv2.resize(image1, (w, h))
        img2_resized = cv2.resize(image2, (w, h))

        # 선형 보간
        mixed = (lam * img1_resized.astype(np.float32) +
                 (1 - lam) * img2_resized.astype(np.float32))
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)

        # 레이블 혼합 (분류 태스크용)
        mixed_label = None
        if label1 is not None and label2 is not None:
            mixed_label = {"lambda": lam, "label1": label1, "label2": label2}

        logger.debug(f"MixUp 적용: λ={lam:.3f}, α={alpha}")
        return mixed, mixed_label

    # ─────────────────────────────────────────────
    # CutMix
    # ─────────────────────────────────────────────

    def apply_cutmix(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        bboxes1: Optional[list] = None,
        bboxes2: Optional[list] = None,
        alpha: Optional[float] = None,
    ) -> tuple:
        """CutMix를 적용한다.

        x̃ = M ⊙ x_i + (1-M) ⊙ x_j
        λ = 1 - (r_w * r_h) / (W * H)
        """
        alpha = alpha or self.config.cutmix_alpha

        h = min(image1.shape[0], image2.shape[0])
        w = min(image1.shape[1], image2.shape[1])

        img1 = cv2.resize(image1, (w, h))
        img2 = cv2.resize(image2, (w, h))

        # λ 샘플링
        lam = np.random.beta(alpha, alpha)

        # 절단 영역 계산
        cut_ratio = np.sqrt(1 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)

        # 중심점 랜덤 선택
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)

        # 절단 영역 경계
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(w, cx + cut_w // 2)
        y2 = min(h, cy + cut_h // 2)

        # 실제 λ 재계산
        actual_lam = 1 - (x2 - x1) * (y2 - y1) / (w * h)

        # 이미지 합성
        result = img1.copy()
        result[y1:y2, x1:x2] = img2[y1:y2, x1:x2]

        # 바운딩 박스 조정
        merged_bboxes = []
        if bboxes1:
            for bbox in bboxes1:
                # img1에서 절단 영역 밖의 bbox만 유지
                cut_region = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
                if bbox.overlap_ratio(cut_region) < self.config.preserve_text_ratio:
                    merged_bboxes.append(bbox)

        if bboxes2:
            for bbox in bboxes2:
                # img2에서 절단 영역 안의 bbox만 유지
                if (bbox.x1 >= x1 and bbox.y1 >= y1 and
                    bbox.x2 <= x2 and bbox.y2 <= y2):
                    merged_bboxes.append(bbox)

        logger.debug(f"CutMix 적용: λ={actual_lam:.3f}, 절단영역=({x1},{y1})-({x2},{y2})")
        return result, merged_bboxes, actual_lam

    # ─────────────────────────────────────────────
    # Mosaic
    # ─────────────────────────────────────────────

    def apply_mosaic(
        self,
        images: list,
        bboxes_list: Optional[list] = None,
        output_size: tuple = (1024, 1024),
    ) -> tuple:
        """4개 이미지를 Mosaic으로 합성한다.

        분할점 (c_x, c_y)를 랜덤 선택하여 4개 이미지를 배치한다.
        """
        if len(images) != 4:
            raise ValueError(f"Mosaic은 정확히 4개 이미지가 필요하다. (입력: {len(images)}개)")

        out_h, out_w = output_size

        # 분할점 랜덤 선택
        cx = np.random.randint(int(out_w * 0.25), int(out_w * 0.75))
        cy = np.random.randint(int(out_h * 0.25), int(out_h * 0.75))

        result = np.full((out_h, out_w, 3), 128, dtype=np.uint8)
        all_bboxes = []

        # 각 사분면에 이미지 배치
        regions = [
            (0, 0, cx, cy),              # 좌상
            (cx, 0, out_w, cy),          # 우상
            (0, cy, cx, out_h),          # 좌하
            (cx, cy, out_w, out_h),      # 우하
        ]

        for idx, (rx1, ry1, rx2, ry2) in enumerate(regions):
            rw = rx2 - rx1
            rh = ry2 - ry1

            if rw <= 0 or rh <= 0:
                continue

            img = cv2.resize(images[idx], (rw, rh))
            result[ry1:ry2, rx1:rx2] = img

            # 바운딩 박스 좌표 변환
            if bboxes_list and idx < len(bboxes_list):
                orig_h, orig_w = images[idx].shape[:2]
                scale_x = rw / orig_w
                scale_y = rh / orig_h

                for bbox in bboxes_list[idx]:
                    new_bbox = BBox(
                        x1=int(bbox.x1 * scale_x + rx1),
                        y1=int(bbox.y1 * scale_y + ry1),
                        x2=int(bbox.x2 * scale_x + rx1),
                        y2=int(bbox.y2 * scale_y + ry1),
                        label=bbox.label,
                    )
                    # 출력 이미지 범위 클리핑
                    new_bbox.x1 = max(0, min(new_bbox.x1, out_w))
                    new_bbox.y1 = max(0, min(new_bbox.y1, out_h))
                    new_bbox.x2 = max(0, min(new_bbox.x2, out_w))
                    new_bbox.y2 = max(0, min(new_bbox.y2, out_h))

                    if new_bbox.area > 0:
                        all_bboxes.append(new_bbox)

        logger.debug(f"Mosaic 적용: 분할점=({cx},{cy}), bbox {len(all_bboxes)}개")
        return result, all_bboxes

    # ─────────────────────────────────────────────
    # 기하학적 변환 (OCR 안전)
    # ─────────────────────────────────────────────

    def apply_safe_rotation(
        self,
        image: np.ndarray,
        bboxes: Optional[list] = None,
        max_angle: Optional[float] = None,
    ) -> tuple:
        """OCR에 안전한 미세 회전을 적용한다.

        큰 각도 회전은 텍스트 방향을 바꿀 수 있으므로
        ±5° 이내로 제한한다.
        """
        max_angle = max_angle or self.config.rotate_max_angle
        angle = np.random.uniform(-max_angle, max_angle)

        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            borderValue=(255, 255, 255),
        )

        # 바운딩 박스 변환
        if bboxes:
            rotated_bboxes = []
            for bbox in bboxes:
                corners = np.array([
                    [bbox.x1, bbox.y1, 1],
                    [bbox.x2, bbox.y1, 1],
                    [bbox.x2, bbox.y2, 1],
                    [bbox.x1, bbox.y2, 1],
                ], dtype=np.float32)

                transformed = (M @ corners.T).T
                new_x1 = max(0, int(transformed[:, 0].min()))
                new_y1 = max(0, int(transformed[:, 1].min()))
                new_x2 = min(w, int(transformed[:, 0].max()))
                new_y2 = min(h, int(transformed[:, 1].max()))

                rotated_bboxes.append(BBox(
                    x1=new_x1, y1=new_y1, x2=new_x2, y2=new_y2,
                    label=bbox.label,
                ))
            return rotated, rotated_bboxes

        return rotated, bboxes

    def apply_safe_scale(
        self,
        image: np.ndarray,
        bboxes: Optional[list] = None,
    ) -> tuple:
        """OCR에 안전한 스케일 변환을 적용한다."""
        scale = np.random.uniform(*self.config.scale_range)
        h, w = image.shape[:2]

        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(image, (new_w, new_h))

        # 원래 크기로 패딩 또는 크롭
        if scale > 1.0:
            # 크롭
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            result = scaled[start_y:start_y+h, start_x:start_x+w]
        else:
            # 패딩
            result = np.full((h, w, 3), 255, dtype=np.uint8)
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            result[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = scaled

        # bbox 변환
        if bboxes:
            adjusted_bboxes = []
            for bbox in bboxes:
                if scale > 1.0:
                    nb = BBox(
                        x1=int(bbox.x1 * scale - (new_w - w) // 2),
                        y1=int(bbox.y1 * scale - (new_h - h) // 2),
                        x2=int(bbox.x2 * scale - (new_w - w) // 2),
                        y2=int(bbox.y2 * scale - (new_h - h) // 2),
                        label=bbox.label,
                    )
                else:
                    nb = BBox(
                        x1=int(bbox.x1 * scale + (w - new_w) // 2),
                        y1=int(bbox.y1 * scale + (h - new_h) // 2),
                        x2=int(bbox.x2 * scale + (w - new_w) // 2),
                        y2=int(bbox.y2 * scale + (h - new_h) // 2),
                        label=bbox.label,
                    )
                # 클리핑
                nb.x1 = max(0, min(nb.x1, w))
                nb.y1 = max(0, min(nb.y1, h))
                nb.x2 = max(0, min(nb.x2, w))
                nb.y2 = max(0, min(nb.y2, h))

                if nb.area > 0:
                    adjusted_bboxes.append(nb)

            return result, adjusted_bboxes

        return result, bboxes

    # ─────────────────────────────────────────────
    # 색상 변환
    # ─────────────────────────────────────────────

    def apply_color_jitter(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """색상 변형을 적용한다. (밝기, 대비, 채도)"""
        result = image.copy()

        # 밝기
        if np.random.random() < self.config.brightness_prob:
            beta = np.random.randint(*self.config.brightness_range)
            result = cv2.convertScaleAbs(result, alpha=1.0, beta=beta)

        # 대비
        if np.random.random() < self.config.contrast_prob:
            alpha = np.random.uniform(*self.config.contrast_range)
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=0)

        # 채도 (HSV 변환)
        if np.random.random() < 0.3 and len(result.shape) == 3:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= np.random.uniform(0.7, 1.3)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return result
```

---

## 8.4.3 OCR 레이아웃 변형

문서 OCR에서는 텍스트의 레이아웃 자체를 변형하는 증강도 효과적이다.

```python
class LayoutAugmentation:
    """문서 레이아웃 수준의 증강.

    텍스트 블록의 위치, 간격, 배열을 변형하여
    다양한 문서 레이아웃에 대한 강건성을 확보한다.
    """

    def apply_text_region_shuffle(
        self,
        image: np.ndarray,
        text_regions: list,
        shuffle_prob: float = 0.3,
    ) -> tuple:
        """텍스트 영역의 순서를 셔플한다.

        표 형식 문서에서 행의 순서를 바꿔 모델이
        위치가 아닌 내용에 기반하여 인식하도록 유도한다.
        """
        if np.random.random() > shuffle_prob or len(text_regions) < 2:
            return image, text_regions

        result = image.copy()

        # 영역별 이미지 추출
        region_images = []
        for region in text_regions:
            crop = image[region.y1:region.y2, region.x1:region.x2].copy()
            region_images.append(crop)

        # 셔플
        indices = list(range(len(text_regions)))
        random.shuffle(indices)

        # 재배치
        shuffled_regions = []
        for new_idx, orig_idx in enumerate(indices):
            target = text_regions[new_idx]
            source_img = region_images[orig_idx]

            # 크기 맞추기
            h = target.y2 - target.y1
            w = target.x2 - target.x1
            resized = cv2.resize(source_img, (w, h))

            result[target.y1:target.y2, target.x1:target.x2] = resized

            shuffled_regions.append(BBox(
                x1=target.x1, y1=target.y1,
                x2=target.x2, y2=target.y2,
                label=text_regions[orig_idx].label,
            ))

        return result, shuffled_regions

    def apply_column_shift(
        self,
        image: np.ndarray,
        shift_range: int = 20,
    ) -> np.ndarray:
        """열(column) 방향으로 미세 시프트를 적용한다.

        다단 레이아웃 문서에서 열 간 정렬이 미세하게
        어긋나는 현상을 시뮬레이션한다.
        """
        h, w = image.shape[:2]
        result = np.full_like(image, 255)

        # 이미지를 세로 스트립으로 분할
        num_strips = np.random.randint(3, 8)
        strip_width = w // num_strips

        for i in range(num_strips):
            x_start = i * strip_width
            x_end = min((i + 1) * strip_width, w)

            shift_y = np.random.randint(-shift_range, shift_range)

            src_y_start = max(0, -shift_y)
            src_y_end = min(h, h - shift_y)
            dst_y_start = max(0, shift_y)
            dst_y_end = min(h, h + shift_y)

            strip_h = min(src_y_end - src_y_start, dst_y_end - dst_y_start)
            if strip_h > 0:
                result[dst_y_start:dst_y_start+strip_h, x_start:x_end] = \
                    image[src_y_start:src_y_start+strip_h, x_start:x_end]

        return result

    def apply_elastic_distortion(
        self,
        image: np.ndarray,
        alpha: float = 30.0,
        sigma: float = 5.0,
    ) -> np.ndarray:
        """탄성 변형(Elastic Distortion)을 적용한다.

        수기 작성 문서의 불규칙한 줄 간격을 시뮬레이션한다.

        dx(x,y) = α * G_σ * U(-1,1)
        dy(x,y) = α * G_σ * U(-1,1)
        """
        h, w = image.shape[:2]

        # 랜덤 변위 필드 생성
        dx = np.random.uniform(-1, 1, (h, w)).astype(np.float32)
        dy = np.random.uniform(-1, 1, (h, w)).astype(np.float32)

        # 가우시안 스무딩
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1

        dx = cv2.GaussianBlur(dx, (ksize, ksize), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (ksize, ksize), sigma) * alpha

        # 변위 맵 생성
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x_coords + dx).astype(np.float32)
        map_y = (y_coords + dy).astype(np.float32)

        result = cv2.remap(
            image, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        return result
```

---

## 8.4.4 Albumentations 기반 파이프라인

```python
try:
    import albumentations as A
    from albumentations.core.transforms_interface import ImageOnlyTransform
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    logger.warning("albumentations 미설치. pip install albumentations 실행 필요.")


class OCRSafeRandomCrop(ImageOnlyTransform):
    """텍스트 영역을 보존하는 랜덤 크롭.

    albumentations 커스텀 Transform으로 구현.
    텍스트 바운딩 박스가 잘리지 않도록 크롭 영역을 제한한다.
    """

    def __init__(self, crop_ratio: float = 0.9, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.crop_ratio = crop_ratio

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        crop_h = int(h * self.crop_ratio)
        crop_w = int(w * self.crop_ratio)

        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)

        cropped = img[top:top+crop_h, left:left+crop_w]
        return cv2.resize(cropped, (w, h))

    def get_transform_init_args_names(self):
        return ("crop_ratio",)


def build_ocr_augmentation_pipeline(
    difficulty: str = "medium",
) -> "A.Compose":
    """OCR 학습용 albumentations 파이프라인을 구성한다.

    Args:
        difficulty: "light", "medium", "heavy"

    Returns:
        albumentations.Compose 파이프라인
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations가 필요하다: pip install albumentations")

    common_transforms = [
        # 색상 변환 (OCR에 안전)
        A.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.1, hue=0.05,
            p=0.5,
        ),
        # 선명화/블러
        A.OneOf([
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),
    ]

    if difficulty == "light":
        transforms = common_transforms + [
            A.GaussNoise(var_limit=(5, 20), p=0.2),
            A.Affine(
                scale=(0.95, 1.05),
                rotate=(-2, 2),
                shear=(-2, 2),
                p=0.3,
            ),
        ]

    elif difficulty == "medium":
        transforms = common_transforms + [
            A.GaussNoise(var_limit=(10, 40), p=0.4),
            A.Affine(
                scale=(0.9, 1.1),
                rotate=(-5, 5),
                shear=(-5, 5),
                p=0.4,
            ),
            A.CoarseDropout(
                max_holes=5,
                max_height=30,
                max_width=30,
                fill_value=128,
                p=0.3,
            ),
            A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),
        ]

    elif difficulty == "heavy":
        transforms = common_transforms + [
            A.GaussNoise(var_limit=(20, 60), p=0.5),
            A.Affine(
                scale=(0.85, 1.15),
                rotate=(-8, 8),
                shear=(-8, 8),
                p=0.5,
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=50,
                max_width=50,
                fill_value=0,
                p=0.4,
            ),
            A.ImageCompression(quality_lower=30, quality_upper=80, p=0.5),
            A.MotionBlur(blur_limit=(3, 9), p=0.3),
            A.Perspective(scale=(0.02, 0.05), p=0.3),
            OCRSafeRandomCrop(crop_ratio=0.9, p=0.2),
        ]
    else:
        raise ValueError(f"지원하지 않는 난이도: {difficulty}")

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.3,  # 30% 이상 보이는 bbox만 유지
        ),
    )


# 사용 예시
def augment_ocr_dataset(
    image_dir: str,
    annotation_dir: str,
    output_dir: str,
    difficulty: str = "medium",
    num_augmented_per_image: int = 5,
):
    """OCR 데이터셋에 증강을 적용한다.

    Args:
        image_dir: 원본 이미지 디렉토리
        annotation_dir: 어노테이션 JSON 디렉토리
        output_dir: 증강된 데이터 출력 디렉토리
        difficulty: 증강 난이도
        num_augmented_per_image: 이미지당 증강 횟수
    """
    import json
    import time

    image_path = Path(image_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "images").mkdir(exist_ok=True)
    (output_path / "annotations").mkdir(exist_ok=True)

    pipeline = build_ocr_augmentation_pipeline(difficulty)
    ocr_aug = OCRAugmentation()

    image_files = list(image_path.glob("*.png")) + list(image_path.glob("*.jpg"))
    total_generated = 0
    start_time = time.time()

    for img_file in image_files:
        image = cv2.imread(str(img_file))
        if image is None:
            continue

        # 어노테이션 로드
        anno_file = Path(annotation_dir) / f"{img_file.stem}.json"
        bboxes = []
        labels = []

        if anno_file.exists():
            with open(anno_file, "r", encoding="utf-8") as f:
                anno = json.load(f)

            for entry in anno.get("annotations", []):
                bbox = entry.get("bbox", {})
                if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
                    bboxes.append([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
                    labels.append(entry.get("text", ""))

        for aug_idx in range(num_augmented_per_image):
            try:
                # albumentations 파이프라인 적용
                if bboxes:
                    transformed = pipeline(
                        image=image,
                        bboxes=bboxes,
                        labels=labels,
                    )
                else:
                    transformed = pipeline(image=image, bboxes=[], labels=[])

                aug_image = transformed["image"]

                # 추가 OCR 특화 증강 (확률적)
                if np.random.random() < 0.2:
                    text_bboxes = [
                        BBox(x1=int(b[0]), y1=int(b[1]), x2=int(b[2]), y2=int(b[3]))
                        for b in transformed.get("bboxes", [])
                    ]
                    aug_image = ocr_aug.apply_cutout(
                        aug_image, text_bboxes=text_bboxes,
                    )

                # 저장
                out_name = f"{img_file.stem}_aug{aug_idx:02d}.png"
                cv2.imwrite(str(output_path / "images" / out_name), aug_image)

                # 변환된 어노테이션 저장
                aug_anno = {
                    "image": out_name,
                    "source": img_file.name,
                    "augmentation": difficulty,
                    "annotations": [
                        {
                            "bbox": {
                                "x1": int(b[0]), "y1": int(b[1]),
                                "x2": int(b[2]), "y2": int(b[3]),
                            },
                            "text": l,
                        }
                        for b, l in zip(
                            transformed.get("bboxes", []),
                            transformed.get("labels", []),
                        )
                    ],
                }

                anno_out = output_path / "annotations" / f"{img_file.stem}_aug{aug_idx:02d}.json"
                with open(anno_out, "w", encoding="utf-8") as f:
                    json.dump(aug_anno, f, ensure_ascii=False, indent=2)

                total_generated += 1

            except Exception as e:
                logger.error(f"증강 실패: {img_file.name} aug{aug_idx}: {e}")

    elapsed = time.time() - start_time
    logger.info(
        f"증강 완료: {total_generated}개 생성, "
        f"소요: {elapsed:.1f}초 ({total_generated / max(elapsed, 0.01):.1f} img/s)"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    augment_ocr_dataset(
        image_dir="./output/synthetic_medical_docs",
        annotation_dir="./output/synthetic_medical_docs",
        output_dir="./output/augmented_docs",
        difficulty="medium",
        num_augmented_per_image=5,
    )
```

---

## 8.4.5 증강 효과 분석

```python
class AugmentationAnalyzer:
    """증강 전후 데이터 분포 변화를 분석한다."""

    @staticmethod
    def compare_pixel_distributions(
        original: np.ndarray,
        augmented: np.ndarray,
    ) -> dict:
        """원본과 증강 이미지의 픽셀 분포를 비교한다."""
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
        aug_gray = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY) if len(augmented.shape) == 3 else augmented

        return {
            "original_mean": float(orig_gray.mean()),
            "original_std": float(orig_gray.std()),
            "augmented_mean": float(aug_gray.mean()),
            "augmented_std": float(aug_gray.std()),
            "pixel_diff_mean": float(np.abs(
                orig_gray.astype(float) - aug_gray.astype(float)
            ).mean()),
        }

    @staticmethod
    def measure_text_readability(
        image: np.ndarray,
        text_bboxes: list,
    ) -> dict:
        """텍스트 영역의 가독성 지표를 측정한다.

        텍스트/배경 간 대비, 선명도 등을 정량화한다.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        contrasts = []
        sharpness_scores = []

        for bbox in text_bboxes:
            roi = gray[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
            if roi.size == 0:
                continue

            # 대비: 표준편차
            contrast = float(roi.std())
            contrasts.append(contrast)

            # 선명도: 라플라시안의 분산
            laplacian = cv2.Laplacian(roi, cv2.CV_64F)
            sharpness = float(laplacian.var())
            sharpness_scores.append(sharpness)

        return {
            "mean_contrast": float(np.mean(contrasts)) if contrasts else 0,
            "mean_sharpness": float(np.mean(sharpness_scores)) if sharpness_scores else 0,
            "min_contrast": float(np.min(contrasts)) if contrasts else 0,
            "num_readable_regions": sum(1 for c in contrasts if c > 30),
            "total_regions": len(text_bboxes),
        }
```

---

## 용어 체크리스트

학습 후 아래 항목을 설명할 수 있는지 점검하라.

- [ ] **Augmentation**: 데이터 증강이 경험 리스크와 기대 리스크의 갭을 줄이는 원리를 설명할 수 있는가?
- [ ] **MixUp**: $\tilde{x} = \lambda x_i + (1-\lambda) x_j$ 수식에서 $\lambda \sim \text{Beta}(\alpha, \alpha)$의 $\alpha$ 값에 따른 분포 변화를 설명할 수 있는가?
- [ ] **CutMix**: 절단 영역의 면적 비율 $\lambda = 1 - r_w \cdot r_h / (W \cdot H)$를 유도하고, 레이블 혼합 방식을 설명할 수 있는가?
- [ ] **CutOut**: 랜덤 영역 마스킹이 과적합을 방지하는 원리와, OCR에서의 주의점(텍스트 보존)을 아는가?
- [ ] **Mosaic**: 4개 이미지 합성에서 분할점 선택과 바운딩 박스 좌표 변환 과정을 이해하는가?
- [ ] **Beta 분포**: $\text{Beta}(\alpha, \alpha)$의 $\alpha < 1$, $\alpha = 1$, $\alpha > 1$ 각각의 분포 형태를 설명할 수 있는가?
- [ ] **OCR 안전 증강**: 텍스트 영역 보존 CutOut, 미세 회전 제한 등 OCR 특화 제약 조건을 이해하는가?
- [ ] **Elastic Distortion**: 변위 필드 $dx, dy$를 가우시안 스무딩하여 탄성 변형을 생성하는 원리를 아는가?
- [ ] **albumentations**: `A.Compose`에서 `bbox_params`를 설정하여 바운딩 박스와 이미지를 동시에 변환하는 방법을 아는가?
- [ ] **가독성 측정**: 대비(표준편차)와 선명도(라플라시안 분산)로 텍스트 가독성을 정량화하는 원리를 설명할 수 있는가?
