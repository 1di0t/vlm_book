# 8.2 노이즈 & 변형 추가

합성 데이터가 아무리 정교해도, 노이즈가 없으면 실제 문서와 괴리가 생긴다. 실제 의료 문서는 스캐너, 팩스, 복사기를 거치면서 가우시안 노이즈, 모션 블러, 원근 왜곡 등 다양한 열화(degradation)를 겪는다. 이 장에서는 이런 현실적 노이즈를 프로그래밍적으로 시뮬레이션하는 기법을 다룬다.

---

> **핵심 용어 박스**
>
> | 용어 | 정의 |
> |------|------|
> | **Gaussian Noise** | 평균 0, 분산 $\sigma^2$인 정규분포를 따르는 가산 노이즈. 센서 열잡음 모사 |
> | **Salt-and-Pepper** | 픽셀이 무작위로 최대값(255) 또는 최소값(0)으로 치환되는 임펄스 노이즈 |
> | **Motion Blur** | 촬영 시 카메라/문서 이동으로 발생하는 방향성 흐림. 방향 커널과의 컨볼루션 |
> | **Perspective Transform** | 3D 투영에 의한 원근 왜곡. 호모그래피 행렬 $H \in \mathbb{R}^{3 \times 3}$로 표현 |

---

## 8.2.1 노이즈 모델의 필요성

### 실제 문서 열화 유형

```
원본 문서 → 인쇄 → 접힘/구겨짐 → 스캔/팩스 → 디지털 이미지
                ↓          ↓            ↓
           프린터 노이즈  물리적 손상    센서 노이즈
           잉크 번짐      주름/접힌 자국  해상도 저하
           토너 얼룩      찢어짐         모아레 패턴
```

| 열화 유형 | 발생 원인 | 수학적 모델 |
|----------|----------|------------|
| 가우시안 노이즈 | 센서 열잡음, 전기 잡음 | $I' = I + \mathcal{N}(0, \sigma^2)$ |
| Salt-and-Pepper | 전송 오류, 센서 불량 | 확률 $p$로 0 또는 255 치환 |
| 모션 블러 | 스캔 시 문서 이동 | $I' = I * K_{\text{motion}}$ |
| 가우시안 블러 | 초점 미맞춤, 저해상도 | $I' = I * G_\sigma$ |
| 원근 왜곡 | 비평면 스캔, 카메라 촬영 | $\mathbf{p}' = H\mathbf{p}$ |
| JPEG 압축 | 파일 저장/전송 | 양자화 블록 아티팩트 |
| 밝기/대비 변형 | 스캐너 설정 차이 | $I' = \alpha I + \beta$ |

---

## 수학적 원리

### 가우시안 노이즈 (Gaussian Noise)

가장 기본적인 노이즈 모델이다. 각 픽셀에 독립적인 정규분포 노이즈를 가산한다:

$$
I'(x, y) = I(x, y) + N(0, \sigma^2)
$$

여기서:
- $I(x, y)$: 원본 픽셀 값
- $N(0, \sigma^2)$: 평균 0, 분산 $\sigma^2$인 가우시안 랜덤 변수
- $\sigma$: 노이즈 강도 (표준편차)

$\sigma$가 클수록 노이즈가 강하다. 의료 문서 스캔의 경우 $\sigma \in [5, 25]$ 범위가 현실적이다.

**SNR (Signal-to-Noise Ratio)**:

$$
\text{SNR} = 10 \log_{10} \frac{\sum_{x,y} I(x,y)^2}{\sum_{x,y} [I'(x,y) - I(x,y)]^2} \text{ dB}
$$

### 모션 블러 (Motion Blur)

방향성 커널 $K$와의 컨볼루션으로 모델링한다:

$$
I'(x, y) = (I * K)(x, y) = \sum_{i=-k}^{k} \sum_{j=-k}^{k} I(x-i, y-j) \cdot K(i, j)
$$

각도 $\theta$, 길이 $L$인 모션 블러 커널:

$$
K_{\text{motion}}(i, j) = \begin{cases} \frac{1}{L} & \text{if } (i, j) \text{ lies on line at angle } \theta \\ 0 & \text{otherwise} \end{cases}
$$

구체적으로, 커널 크기 $L \times L$에서 중심을 지나는 각도 $\theta$의 직선 위 픽셀만 $\frac{1}{L}$ 값을 갖는다:

$$
K_{\text{motion}}[i][j] = \frac{1}{L} \cdot \mathbb{1}\left[ \left| j \cos\theta - i \sin\theta \right| < 0.5 \right]
$$

### 원근 변환 (Perspective Transform)

호모그래피 행렬 $H \in \mathbb{R}^{3 \times 3}$로 원근 변환을 표현한다:

$$
\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

실제 좌표는 동차 좌표를 정규화하여 구한다:

$$
x_{\text{out}} = \frac{x'}{w'} = \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + h_{33}}, \quad y_{\text{out}} = \frac{y'}{w'} = \frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + h_{33}}
$$

4개 대응점 쌍 $(x_i, y_i) \leftrightarrow (x_i', y_i')$로부터 $H$의 8개 자유 파라미터를 결정한다 ($h_{33} = 1$ 정규화).

### Salt-and-Pepper 노이즈

확률 $p$로 각 픽셀이 손상된다:

$$
I'(x, y) = \begin{cases} 0 & \text{with probability } p/2 \\ 255 & \text{with probability } p/2 \\ I(x, y) & \text{with probability } 1 - p \end{cases}
$$

의료 문서에서는 $p \in [0.001, 0.01]$ 범위가 현실적이다. 팩스 전송 시뮬레이션에서는 $p$를 높일 수 있다.

---

## 8.2.2 NoiseAugmentor 클래스

```python
import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class NoiseType(Enum):
    """지원하는 노이즈 유형."""
    GAUSSIAN = "gaussian"
    SALT_AND_PEPPER = "salt_and_pepper"
    SPECKLE = "speckle"
    POISSON = "poisson"


class BlurType(Enum):
    """지원하는 블러 유형."""
    GAUSSIAN = "gaussian"
    MOTION = "motion"
    MEDIAN = "median"
    DEFOCUS = "defocus"


@dataclass
class NoiseConfig:
    """노이즈 파라미터 설정."""
    # 가우시안 노이즈
    gaussian_sigma_range: tuple = (3.0, 25.0)

    # Salt-and-Pepper
    sp_prob_range: tuple = (0.001, 0.01)

    # 모션 블러
    motion_kernel_range: tuple = (3, 15)
    motion_angle_range: tuple = (0, 360)

    # 가우시안 블러
    gaussian_blur_range: tuple = (1, 5)

    # 원근 변환
    perspective_strength: float = 0.02

    # 밝기/대비
    brightness_range: tuple = (-30, 30)
    contrast_range: tuple = (0.7, 1.3)

    # JPEG 압축
    jpeg_quality_range: tuple = (30, 95)

    # 적용 확률
    noise_prob: float = 0.5
    blur_prob: float = 0.3
    perspective_prob: float = 0.2
    brightness_prob: float = 0.4
    jpeg_prob: float = 0.3


class NoiseAugmentor:
    """OCR 학습용 노이즈 및 변형을 적용하는 증강 엔진.

    실제 문서 스캔/팩스/촬영 과정에서 발생하는 다양한
    열화를 시뮬레이션한다. OpenCV 기반으로 구현되어
    대량 처리에 최적화되어 있다.
    """

    def __init__(self, config: Optional[NoiseConfig] = None, seed: Optional[int] = None):
        self.config = config or NoiseConfig()
        if seed is not None:
            np.random.seed(seed)

        self._augment_pipeline: list[tuple[str, Callable, float]] = []
        self._build_pipeline()

    def _build_pipeline(self):
        """증강 파이프라인을 구성한다."""
        self._augment_pipeline = [
            ("gaussian_noise", self.add_gaussian_noise, self.config.noise_prob),
            ("salt_pepper", self.add_salt_and_pepper, self.config.noise_prob * 0.5),
            ("motion_blur", self.add_motion_blur, self.config.blur_prob),
            ("gaussian_blur", self.add_gaussian_blur, self.config.blur_prob * 0.5),
            ("perspective", self.apply_perspective_transform, self.config.perspective_prob),
            ("brightness_contrast", self.adjust_brightness_contrast, self.config.brightness_prob),
            ("jpeg_compress", self.simulate_jpeg_compression, self.config.jpeg_prob),
        ]

    def add_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """가우시안 노이즈를 추가한다.

        I'(x,y) = I(x,y) + N(0, σ²)
        """
        sigma = np.random.uniform(*self.config.gaussian_sigma_range)

        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        logger.debug(f"가우시안 노이즈 적용: σ={sigma:.1f}")
        return noisy

    def add_salt_and_pepper(self, image: np.ndarray) -> np.ndarray:
        """Salt-and-Pepper 노이즈를 추가한다.

        확률 p로 각 픽셀을 0 또는 255로 치환한다.
        """
        prob = np.random.uniform(*self.config.sp_prob_range)
        noisy = image.copy()

        # Salt (흰 점)
        salt_mask = np.random.random(image.shape[:2]) < prob / 2
        noisy[salt_mask] = 255

        # Pepper (검은 점)
        pepper_mask = np.random.random(image.shape[:2]) < prob / 2
        noisy[pepper_mask] = 0

        logger.debug(f"Salt-and-Pepper 적용: p={prob:.4f}")
        return noisy

    def add_speckle_noise(self, image: np.ndarray) -> np.ndarray:
        """스펙클 노이즈를 추가한다.

        I' = I + I * N(0, σ²)  (곱셈 노이즈)
        """
        sigma = np.random.uniform(0.05, 0.2)
        noise = np.random.normal(0, sigma, image.shape)
        noisy = image.astype(np.float32) + image.astype(np.float32) * noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _create_motion_kernel(self, size: int, angle: float) -> np.ndarray:
        """방향성 모션 블러 커널을 생성한다.

        각도 θ, 길이 L인 직선 커널:
        K[i][j] = 1/L  if (i,j) on line at angle θ
        """
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2

        # 각도를 라디안으로 변환
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        for i in range(size):
            offset = i - center
            x = int(center + offset * cos_a + 0.5)
            y = int(center + offset * sin_a + 0.5)

            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0

        # 정규화: 커널 합 = 1
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum

        return kernel

    def add_motion_blur(self, image: np.ndarray) -> np.ndarray:
        """모션 블러를 적용한다.

        I' = I * K_motion (컨볼루션)
        """
        kernel_size = np.random.randint(*self.config.motion_kernel_range)
        if kernel_size % 2 == 0:
            kernel_size += 1  # 홀수 보장

        angle = np.random.uniform(*self.config.motion_angle_range)
        kernel = self._create_motion_kernel(kernel_size, angle)

        blurred = cv2.filter2D(image, -1, kernel)

        logger.debug(f"모션 블러 적용: size={kernel_size}, angle={angle:.1f}°")
        return blurred

    def add_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """가우시안 블러를 적용한다.

        G(x,y) = (1/2πσ²) exp(-(x²+y²)/2σ²)
        I' = I * G
        """
        ksize = np.random.choice(range(
            self.config.gaussian_blur_range[0],
            self.config.gaussian_blur_range[1] + 1,
            2,
        ))
        if ksize % 2 == 0:
            ksize += 1

        sigma = np.random.uniform(0.5, 2.0)
        blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)

        logger.debug(f"가우시안 블러 적용: ksize={ksize}, σ={sigma:.2f}")
        return blurred

    def apply_perspective_transform(self, image: np.ndarray) -> np.ndarray:
        """원근 변환을 적용한다.

        4개 코너점에 랜덤 변형을 가해 호모그래피 H를 생성한다.
        """
        h, w = image.shape[:2]
        strength = self.config.perspective_strength

        # 원본 4개 코너
        src_points = np.float32([
            [0, 0], [w, 0], [w, h], [0, h]
        ])

        # 랜덤 변형된 코너
        max_shift_x = int(w * strength)
        max_shift_y = int(h * strength)

        dst_points = np.float32([
            [np.random.randint(0, max_shift_x),
             np.random.randint(0, max_shift_y)],
            [w - np.random.randint(0, max_shift_x),
             np.random.randint(0, max_shift_y)],
            [w - np.random.randint(0, max_shift_x),
             h - np.random.randint(0, max_shift_y)],
            [np.random.randint(0, max_shift_x),
             h - np.random.randint(0, max_shift_y)],
        ])

        # 호모그래피 행렬 계산: p' = H * p
        H = cv2.getPerspectiveTransform(src_points, dst_points)

        transformed = cv2.warpPerspective(
            image, H, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        logger.debug(f"원근 변환 적용: strength={strength}")
        return transformed

    def adjust_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """밝기와 대비를 조절한다.

        I' = α * I + β
        α: 대비 계수, β: 밝기 오프셋
        """
        alpha = np.random.uniform(*self.config.contrast_range)
        beta = np.random.randint(*self.config.brightness_range)

        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        logger.debug(f"밝기/대비 조절: α={alpha:.2f}, β={beta}")
        return adjusted

    def simulate_jpeg_compression(self, image: np.ndarray) -> np.ndarray:
        """JPEG 압축 아티팩트를 시뮬레이션한다."""
        quality = np.random.randint(*self.config.jpeg_quality_range)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        logger.debug(f"JPEG 압축 적용: quality={quality}")
        return decoded

    def augment(
        self,
        image: np.ndarray,
        apply_all: bool = False,
    ) -> np.ndarray:
        """이미지에 랜덤 노이즈/변형을 적용한다.

        Args:
            image: 입력 이미지 (H, W, C), uint8
            apply_all: True면 모든 증강을 적용, False면 확률적 적용

        Returns:
            증강된 이미지
        """
        result = image.copy()

        for name, func, prob in self._augment_pipeline:
            if apply_all or np.random.random() < prob:
                try:
                    result = func(result)
                except Exception as e:
                    logger.error(f"증강 실패 ({name}): {e}")
                    continue

        return result
```

---

## 8.2.3 스캔 시뮬레이션

실제 스캐너/팩스 출력의 특성을 모사하는 전용 시뮬레이터다.

```python
class ScanSimulator:
    """스캐너/팩스/복사기 출력을 시뮬레이션한다.

    해상도 저하, 흑백 변환, 모아레 패턴, 스캔 라인 등
    특정 장비의 출력 특성을 재현한다.
    """

    def __init__(self, target_dpi: int = 150):
        self.target_dpi = target_dpi

    def simulate_low_resolution(
        self,
        image: np.ndarray,
        source_dpi: int = 300,
    ) -> np.ndarray:
        """해상도 저하를 시뮬레이션한다.

        축소 후 원래 크기로 확대하여 정보 손실을 재현한다.
        """
        scale = self.target_dpi / source_dpi
        h, w = image.shape[:2]

        # 축소
        small = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

        # 다시 확대 (정보 손실 발생)
        restored = cv2.resize(
            small,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        return restored

    def convert_to_binary(
        self,
        image: np.ndarray,
        method: str = "otsu",
    ) -> np.ndarray:
        """팩스 전송을 위한 이진화(binarization)를 수행한다.

        Otsu's method: 클래스 간 분산 σ²_B(t)를 최대화하는 임계값 t* 탐색
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if method == "otsu":
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif method == "adaptive":
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2,
            )
        else:
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # 3채널로 복원 (파이프라인 호환)
        if len(image.shape) == 3:
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        return binary

    def add_moire_pattern(
        self,
        image: np.ndarray,
        frequency: float = 0.05,
        amplitude: float = 15.0,
    ) -> np.ndarray:
        """모아레 패턴을 추가한다.

        두 주기 패턴의 간섭으로 발생하는 줄무늬:
        M(x,y) = A * sin(2π * f * (x*cos(θ) + y*sin(θ)))
        """
        h, w = image.shape[:2]
        theta = np.random.uniform(0, np.pi)

        y_coords, x_coords = np.mgrid[0:h, 0:w]
        pattern = amplitude * np.sin(
            2 * np.pi * frequency * (
                x_coords * np.cos(theta) + y_coords * np.sin(theta)
            )
        )

        if len(image.shape) == 3:
            pattern = np.stack([pattern] * 3, axis=-1)

        result = image.astype(np.float32) + pattern.astype(np.float32)
        return np.clip(result, 0, 255).astype(np.uint8)

    def add_scan_lines(
        self,
        image: np.ndarray,
        line_spacing: int = 4,
        line_intensity: float = 0.05,
    ) -> np.ndarray:
        """스캔 라인 아티팩트를 추가한다.

        CCD 스캐너의 라인 센서 특성을 모사한다.
        """
        result = image.astype(np.float32)
        h = image.shape[0]

        for y in range(0, h, line_spacing):
            darkening = 1.0 - line_intensity * np.random.random()
            result[y] = result[y] * darkening

        return np.clip(result, 0, 255).astype(np.uint8)

    def add_page_curl(
        self,
        image: np.ndarray,
        curl_amount: float = 0.02,
    ) -> np.ndarray:
        """페이지 말림(curl) 효과를 시뮬레이션한다.

        책이나 두꺼운 문서를 스캔할 때 가장자리가 말리는 현상이다.
        사인 함수로 비선형 왜곡을 적용한다.
        """
        h, w = image.shape[:2]
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                # 가장자리에서 중심으로의 정규화 거리
                dx = (x - w / 2) / (w / 2)
                curl = curl_amount * w * np.sin(np.pi * dx) * (abs(dx) ** 2)
                map_x[y, x] = x + curl
                map_y[y, x] = y

        result = cv2.remap(
            image, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        return result

    def simulate_scanner(
        self,
        image: np.ndarray,
        scanner_type: str = "flatbed",
    ) -> np.ndarray:
        """특정 스캐너 유형의 출력을 시뮬레이션한다.

        Args:
            image: 입력 이미지
            scanner_type: "flatbed", "adf", "fax" 중 하나
        """
        result = image.copy()

        if scanner_type == "flatbed":
            # 플랫베드: 비교적 깨끗, 약간의 블러와 스캔 라인
            result = cv2.GaussianBlur(result, (3, 3), 0.5)
            result = self.add_scan_lines(result, line_intensity=0.02)

        elif scanner_type == "adf":
            # ADF (자동 문서 급지기): 스큐, 약간의 모션 블러
            angle = np.random.uniform(-1.0, 1.0)
            M = cv2.getRotationMatrix2D(
                (image.shape[1] / 2, image.shape[0] / 2), angle, 1.0
            )
            result = cv2.warpAffine(
                result, M, (image.shape[1], image.shape[0]),
                borderValue=(255, 255, 255),
            )
            # 미세한 모션 블러
            kernel = np.zeros((3, 3))
            kernel[1, :] = 1.0 / 3
            result = cv2.filter2D(result, -1, kernel)

        elif scanner_type == "fax":
            # 팩스: 저해상도, 이진화, 노이즈
            result = self.simulate_low_resolution(result, source_dpi=300)
            result = self.convert_to_binary(result, method="otsu")

            # Salt-and-Pepper 노이즈 (전송 오류 모사)
            noise_augmentor = NoiseAugmentor(NoiseConfig(sp_prob_range=(0.005, 0.02)))
            result = noise_augmentor.add_salt_and_pepper(result)

            # 모아레 패턴
            if np.random.random() < 0.3:
                result = self.add_moire_pattern(result)

        return result
```

---

## 8.2.4 복합 열화 파이프라인

실제 문서는 여러 유형의 열화가 순차적으로 적용된다. 이를 모사하는 파이프라인이다.

```python
import time
from pathlib import Path
from typing import Union


class DegradationPipeline:
    """여러 열화를 순차적으로 적용하는 파이프라인.

    실제 문서 생애주기를 모사한다:
    인쇄 → 물리적 손상 → 스캔/촬영 → 디지털 저장
    """

    def __init__(
        self,
        noise_config: Optional[NoiseConfig] = None,
        scanner_type: str = "flatbed",
        seed: Optional[int] = None,
    ):
        self.noise_augmentor = NoiseAugmentor(config=noise_config, seed=seed)
        self.scan_simulator = ScanSimulator()
        self.scanner_type = scanner_type

        # 난이도별 프리셋
        self._presets = {
            "clean": {
                "noise_sigma": (1, 5),
                "blur_prob": 0.1,
                "perspective_prob": 0.05,
            },
            "mild": {
                "noise_sigma": (5, 15),
                "blur_prob": 0.3,
                "perspective_prob": 0.15,
            },
            "moderate": {
                "noise_sigma": (10, 25),
                "blur_prob": 0.5,
                "perspective_prob": 0.25,
            },
            "severe": {
                "noise_sigma": (20, 40),
                "blur_prob": 0.7,
                "perspective_prob": 0.4,
            },
        }

    def apply_preset(
        self,
        image: np.ndarray,
        difficulty: str = "moderate",
    ) -> np.ndarray:
        """난이도 프리셋에 따라 열화를 적용한다.

        Args:
            image: 입력 이미지
            difficulty: "clean", "mild", "moderate", "severe"
        """
        preset = self._presets.get(difficulty, self._presets["moderate"])

        result = image.copy()

        # 1. 밝기/대비 변형
        if np.random.random() < 0.4:
            result = self.noise_augmentor.adjust_brightness_contrast(result)

        # 2. 가우시안 노이즈
        sigma_range = preset["noise_sigma"]
        self.noise_augmentor.config.gaussian_sigma_range = sigma_range
        if np.random.random() < 0.5:
            result = self.noise_augmentor.add_gaussian_noise(result)

        # 3. 블러 (모션 또는 가우시안)
        if np.random.random() < preset["blur_prob"]:
            if np.random.random() < 0.5:
                result = self.noise_augmentor.add_motion_blur(result)
            else:
                result = self.noise_augmentor.add_gaussian_blur(result)

        # 4. 원근 변환
        if np.random.random() < preset["perspective_prob"]:
            result = self.noise_augmentor.apply_perspective_transform(result)

        # 5. 스캐너 시뮬레이션
        if np.random.random() < 0.3:
            result = self.scan_simulator.simulate_scanner(
                result, self.scanner_type
            )

        # 6. JPEG 압축
        if np.random.random() < 0.3:
            result = self.noise_augmentor.simulate_jpeg_compression(result)

        return result

    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        difficulty: str = "moderate",
        num_variants: int = 3,
    ) -> list:
        """디렉토리 내 모든 이미지에 열화를 적용한다.

        Args:
            input_dir: 원본 이미지 디렉토리
            output_dir: 출력 디렉토리
            difficulty: 열화 난이도
            num_variants: 이미지당 생성할 변형 수

        Returns:
            생성된 파일 경로 목록
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        generated = []
        start_time = time.time()

        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None:
                logger.warning(f"이미지 로드 실패: {img_file}")
                continue

            for variant_idx in range(num_variants):
                try:
                    degraded = self.apply_preset(image, difficulty)
                    out_name = f"{img_file.stem}_deg_{difficulty}_{variant_idx}{img_file.suffix}"
                    out_path = output_path / out_name
                    cv2.imwrite(str(out_path), degraded)
                    generated.append(str(out_path))

                except Exception as e:
                    logger.error(f"열화 적용 실패: {img_file.name} v{variant_idx}: {e}")

        elapsed = time.time() - start_time
        logger.info(
            f"배치 처리 완료: {len(generated)}개 생성, "
            f"소요 시간: {elapsed:.1f}초 "
            f"({len(generated) / max(elapsed, 0.01):.1f} img/s)"
        )

        return generated
```

---

## 8.2.5 노이즈 강도 분석

생성된 노이즈의 통계적 특성을 분석하는 유틸리티다.

```python
class NoiseAnalyzer:
    """노이즈 특성을 분석하고 시각화한다."""

    @staticmethod
    def compute_psnr(original: np.ndarray, noisy: np.ndarray) -> float:
        """PSNR (Peak Signal-to-Noise Ratio)을 계산한다.

        PSNR = 10 * log10(MAX² / MSE)
        """
        mse = np.mean((original.astype(np.float64) - noisy.astype(np.float64)) ** 2)
        if mse == 0:
            return float("inf")

        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return psnr

    @staticmethod
    def compute_ssim(
        original: np.ndarray,
        noisy: np.ndarray,
        window_size: int = 11,
    ) -> float:
        """SSIM (Structural Similarity Index)을 계산한다.

        SSIM(x, y) = (2μ_xμ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
        """
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        if len(original.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)

        original = original.astype(np.float64)
        noisy = noisy.astype(np.float64)

        kernel = cv2.getGaussianKernel(window_size, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(original, -1, window)
        mu2 = cv2.filter2D(noisy, -1, window)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.filter2D(original ** 2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(noisy ** 2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(original * noisy, -1, window) - mu1_mu2

        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = numerator / denominator
        return float(ssim_map.mean())

    def analyze_degradation(
        self,
        original: np.ndarray,
        degraded: np.ndarray,
    ) -> dict:
        """원본 대비 열화 이미지의 품질 지표를 계산한다."""
        psnr = self.compute_psnr(original, degraded)
        ssim = self.compute_ssim(original, degraded)

        # 노이즈 분포 분석
        diff = original.astype(np.float64) - degraded.astype(np.float64)
        noise_mean = np.mean(diff)
        noise_std = np.std(diff)

        return {
            "psnr_db": round(psnr, 2),
            "ssim": round(ssim, 4),
            "noise_mean": round(noise_mean, 2),
            "noise_std": round(noise_std, 2),
        }
```

---

## 8.2.6 실행 예시

```python
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1. 기본 노이즈 증강
    image = cv2.imread("input_document.png")

    augmentor = NoiseAugmentor(NoiseConfig(
        gaussian_sigma_range=(5, 20),
        motion_kernel_range=(3, 11),
        perspective_strength=0.015,
    ))

    # 단일 이미지 증강
    augmented = augmentor.augment(image)
    cv2.imwrite("augmented_output.png", augmented)

    # 2. 스캐너 시뮬레이션
    scanner = ScanSimulator(target_dpi=150)

    flatbed_result = scanner.simulate_scanner(image, "flatbed")
    fax_result = scanner.simulate_scanner(image, "fax")
    adf_result = scanner.simulate_scanner(image, "adf")

    # 3. 배치 처리
    pipeline = DegradationPipeline(scanner_type="flatbed")
    pipeline.process_batch(
        input_dir="./output/synthetic_medical_docs",
        output_dir="./output/degraded_docs",
        difficulty="moderate",
        num_variants=3,
    )

    # 4. 품질 분석
    analyzer = NoiseAnalyzer()
    metrics = analyzer.analyze_degradation(image, augmented)
    print(f"PSNR: {metrics['psnr_db']} dB")
    print(f"SSIM: {metrics['ssim']}")
    print(f"노이즈 평균: {metrics['noise_mean']}, 표준편차: {metrics['noise_std']}")
```

---

## 용어 체크리스트

학습 후 아래 항목을 설명할 수 있는지 점검하라.

- [ ] **Gaussian Noise**: $I'(x,y) = I(x,y) + N(0, \sigma^2)$ 수식의 각 항을 설명하고, $\sigma$가 이미지에 미치는 영향을 아는가?
- [ ] **Salt-and-Pepper**: 임펄스 노이즈의 확률 모델을 수식으로 쓸 수 있는가?
- [ ] **Motion Blur**: 방향 커널 $K$의 구성 원리와 컨볼루션으로 블러를 적용하는 과정을 설명할 수 있는가?
- [ ] **Perspective Transform**: 호모그래피 행렬 $H$의 8개 자유도와, 4개 대응점으로부터 $H$를 구하는 원리를 아는가?
- [ ] **SNR/PSNR**: 신호 대 잡음비의 정의와 dB 단위 계산법을 알고 있는가?
- [ ] **SSIM**: 구조적 유사도의 수식과 PSNR 대비 장점을 설명할 수 있는가?
- [ ] **모아레 패턴**: 두 주기 패턴의 간섭으로 발생하는 원리와 수학적 모델을 아는가?
- [ ] **Otsu's method**: 클래스 간 분산 $\sigma^2_B(t)$ 최대화로 이진화 임계값을 구하는 과정을 이해하는가?
- [ ] **열화 파이프라인**: 여러 노이즈/변형을 순차 적용할 때의 순서 의존성과 실제 문서 열화 과정의 대응 관계를 설명할 수 있는가?
