---
---

# 6.4 보험금 청구 의료 문서 OCR

> 의료 문서 OCR은 일반 문서 OCR과 다르다.
> **금액 1원 오류가 청구 거절**로 이어지고, **진단코드 한 글자 오류가 보험 심사 반려**로 이어진다.
> 이 챕터에서는 보험금 청구에 사용되는 의료 문서의 유형, OCR 난이도 요소, 전처리 파이프라인,
> 그리고 필드 추출 후처리를 다룬다.

---

## 핵심 용어

| 용어 | 정의 | 왜 중요한가 |
|------|------|-------------|
| **비식별화 (De-identification)** | 환자 개인정보(이름, 주민번호 등)를 제거하거나 마스킹하는 과정 | 개인정보보호법·의료법 준수 필수. 위반 시 법적 제재 |
| **PHI (Protected Health Information)** | 보호대상 건강정보. 환자를 식별할 수 있는 18가지 유형의 정보 | HIPAA(미국)/개인정보보호법(한국)에서 엄격히 보호 |
| **Pre-printed Form** | 양식이 미리 인쇄된 문서. 빈칸에 수기/타이핑으로 내용을 기재 | OCR이 양식 텍스트와 기재 내용을 구분해야 한다 |
| **KCD 코드** | 한국표준질병사인분류. ICD-10의 한국판 (예: S72.0 = 대퇴골 경부 골절) | 보험 심사의 핵심. 코드 오류 시 청구 반려 |
| **의료행위코드** | 건강보험 수가 체계의 행위 코드 (예: N0611 = 관혈적 정복술) | 수술/시술 내역 확인, 보험금 산정의 근거 |
| **EDI (Electronic Data Interchange)** | 의료기관↔보험사 간 전자 데이터 교환 표준 | 자동화된 청구 처리의 표준 포맷 |

---

## 보험 청구 문서 유형

### 1. 진단서

| 항목 | 설명 | OCR 주의점 |
|------|------|-----------|
| 상병명 | 질병/부상 명칭 (한글) | 의학 용어 인식 정확도 |
| KCD 진단코드 | 알파벳 + 숫자 혼합 (예: S72.0) | 대소문자 구분, 마침표 위치 |
| 발병일 | 질병 발생 추정일 | 날짜 형식 파싱 |
| 진단일 | 진단 확정일 | 날짜 형식 파싱 |
| 의사 성명/면허번호 | 진단 의사 정보 | 도장/서명과 겹치는 경우 |

**KCD 코드 형식 규칙**:
```
형식: [알파벳 1자리][숫자 2자리].[숫자 0~2자리]

예시:
  S72.0  → S: 외상, 72: 대퇴골 골절, 0: 경부
  K25.0  → K: 소화기계, 25: 위궤양, 0: 급성 출혈 동반
  M54.5  → M: 근골격계, 54: 배아픔, 5: 허리통증
  I10    → I: 순환기계, 10: 본태성 고혈압 (소수점 이하 없음)
  C34.1  → C: 악성 신생물, 34: 기관지/폐, 1: 상엽
```

### 2. 수술기록

| 항목 | 설명 | OCR 주의점 |
|------|------|-----------|
| 수술명 | 한글/영문 혼재 가능 | 의학 영문 약어 인식 |
| 수술코드 | 건강보험 수가코드 | 알파벳+숫자 혼합 |
| 수술일시 | 시작/종료 시간 포함 | 시간 형식 파싱 |
| 마취 방법 | 전신/부분 마취 | 고정 어휘 매칭 |
| 집도의/보조의 | 의사 정보 | 서명과 분리 |

### 3. 입퇴원확인서

| 항목 | 설명 | OCR 주의점 |
|------|------|-----------|
| 입원일 | YYYY-MM-DD 또는 YYYY.MM.DD | 날짜 형식 다양 |
| 퇴원일 | 날짜 | 재원일수 계산 검증 가능 |
| 진료과 | 내과, 외과, 정형외과 등 | 고정 어휘 매칭 |
| 상병명 | 진단서와 동일 | 교차 검증 가능 |
| 입원사유 | 자유 텍스트 | 정형화되지 않은 텍스트 |

### 4. 진료비 세부내역서

가장 OCR 난이도가 높은 문서다.

| 항목 | 설명 | OCR 주의점 |
|------|------|-----------|
| 항목코드 | 약품코드, 행위코드 | 코드 정확도 필수 |
| 항목명 | 시술명, 약품명 | 긴 텍스트, 줄바꿈 |
| 단가 | 원 단위 금액 | 1원 단위 정확성 |
| 횟수/일수 | 정수 | 단순하지만 0과 O 혼동 |
| 금액 | 단가 × 횟수 | 합계 검증 |
| 급여/비급여 | 건강보험 적용 여부 | 보험금 산정에 직결 |
| 본인부담금 | 환자 실부담 | 최종 보험금 계산 근거 |
| **합계** | 모든 항목 합산 | **checksum으로 검증 가능** |

### 5. 보험 청구 양식

보험사별로 양식이 다르다. 공통 필수 항목:

| 항목 | 설명 |
|------|------|
| 피보험자 정보 | 이름, 주민번호, 연락처 |
| 사고/질병 경위 | 자유 텍스트 |
| 청구 금액 | 진료비 세부내역 기반 |
| 계좌 정보 | 은행명, 계좌번호 |
| 첨부 서류 체크 | 체크박스 |

---

## OCR 난이도 요소

### 1. Pre-printed Form + 수기/출력 텍스트 혼재

이건 의료 문서 OCR의 최대 난제 중 하나다.

```
┌─────────────────────────────────┐
│  [인쇄] 성  명:  [수기] 홍길동    │  ← 인쇄체와 수기가 같은 줄
│  [인쇄] 생년월일: [출력] 1990.01.01│  ← 인쇄체와 프린터 출력
│  [인쇄] 진단코드: [수기] S72.0    │  ← 코드를 손으로 적은 경우
│                                 │
│  [인쇄] 의사 서명: [도장+서명]     │  ← 도장이 텍스트를 가림
└─────────────────────────────────┘
```

구분 전략:
- **색상 차이**: 인쇄체(검정/회색) vs 수기(파란색/검정)
- **폰트 차이**: 인쇄체(균일) vs 수기(불규칙)
- **위치 기반**: 고정 양식의 레이블 위치를 템플릿으로 학습

### 2. 금액 인식 정확도

진료비 세부내역서에서 금액 인식 오류는 곧 보험금 산정 오류다.

문제 상황:
- `1,250,000` → `1,250,0OO` (0과 O 혼동)
- `15,000` → `1S,000` (5와 S 혼동)
- `3,500` → `3.500` (쉼표와 마침표 혼동)
- 인쇄 품질 저하로 숫자가 뭉개진 경우

### 3. 진단코드/의료행위코드 정확도

알파벳과 숫자가 혼합된 코드에서 자주 발생하는 오류:

| 정답 | 오인식 | 원인 |
|------|--------|------|
| S72.0 | S72.O | 0 → O |
| K25.1 | K2S.1 | 5 → S |
| M54.5 | M545 | 마침표 누락 |
| I10 | 110 | I → 1 |
| C34.1 | C34.l | 1 → l |
| N0611 | NO611 | 0 → O |

### 4. 도장/서명 영역과 텍스트 분리

의료 문서에는 의사 도장, 병원 직인, 환자 서명이 텍스트 위에 겹친다.

```python
import torch
import torch.nn as nn


class StampTextSeparator(nn.Module):
    """
    도장/서명 영역과 텍스트 영역을 분리하는 U-Net 기반 모듈.

    입력: 문서 이미지
    출력: 텍스트 마스크 (도장/서명이 제거된 깨끗한 텍스트 영역)
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)

        # 출력: 3채널 (텍스트, 도장, 배경)
        self.output = nn.Conv2d(64, 3, 1)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) 문서 이미지

        Returns:
            masks: (B, 3, H, W) 클래스별 확률맵 [텍스트, 도장, 배경]
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.output(d1)
```

### 5. 팩스/스캔 품질 저하

현실적으로 보험 청구 문서의 상당수는 팩스로 전송되거나 저해상도로 스캔된다.

일반적인 품질 문제:
- 해상도: 150 DPI 이하 (권장 300 DPI)
- 기울어짐: 1~5도 회전
- 잡음: 팩스 전송 시 발생하는 가로줄 잡음
- 농도: 너무 밝거나 어두운 스캔
- 접힘 자국: 우편 발송 시 생긴 접힌 선

---

## 수학적 원리

### 1. 이미지 전처리: Binarization (Otsu's Method)

문서 이미지에서 텍스트와 배경을 분리하는 가장 기본적인 기법이다.

**Otsu's Method**: 이미지의 히스토그램에서 클래스 간 분산을 최대화하는 임계값을 자동으로 찾는다.

픽셀 강도 $i \in [0, 255]$에 대해 히스토그램 $p(i) = n_i / N$ ($n_i$: 강도 $i$의 픽셀 수, $N$: 전체 픽셀 수).

임계값 $t$로 이분류할 때:

클래스 0 (배경): 강도 $[0, t]$

$$
\omega_0(t) = \sum_{i=0}^{t} p(i), \quad \mu_0(t) = \frac{\sum_{i=0}^{t} i \cdot p(i)}{\omega_0(t)}
$$

클래스 1 (전경=텍스트): 강도 $[t+1, 255]$

$$
\omega_1(t) = \sum_{i=t+1}^{255} p(i), \quad \mu_1(t) = \frac{\sum_{i=t+1}^{255} i \cdot p(i)}{\omega_1(t)}
$$

클래스 간 분산:

$$
\sigma_B^2(t) = \omega_0(t) \omega_1(t) \left[\mu_0(t) - \mu_1(t)\right]^2
$$

최적 임계값:

$$
t^* = \arg\max_t \sigma_B^2(t)
$$

```python
import numpy as np
from PIL import Image
import torch


def otsu_threshold(image: np.ndarray) -> int:
    """
    Otsu's method로 최적 이진화 임계값을 찾는다.

    Args:
        image: (H, W) 그레이스케일 이미지 (0-255)

    Returns:
        최적 임계값 (0-255)
    """
    # 히스토그램 계산
    hist = np.zeros(256, dtype=np.float64)
    for val in image.ravel():
        hist[val] += 1
    hist /= hist.sum()  # 정규화 → 확률

    best_threshold = 0
    best_variance = 0.0

    for t in range(256):
        # 클래스 0 (배경): [0, t]
        w0 = hist[:t+1].sum()
        # 클래스 1 (전경): [t+1, 255]
        w1 = hist[t+1:].sum()

        if w0 == 0 or w1 == 0:
            continue

        # 클래스별 평균
        mu0 = np.sum(np.arange(t+1) * hist[:t+1]) / w0
        mu1 = np.sum(np.arange(t+1, 256) * hist[t+1:]) / w1

        # 클래스 간 분산
        variance = w0 * w1 * (mu0 - mu1) ** 2

        if variance > best_variance:
            best_variance = variance
            best_threshold = t

    return best_threshold


def binarize_document(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    문서 이미지를 이진화한다.

    Args:
        image: (H, W, 3) 또는 (H, W) 이미지
        method: "otsu" 또는 "adaptive"

    Returns:
        binary: (H, W) 이진 이미지 (0 또는 255)
    """
    # 그레이스케일 변환
    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray = image.copy()

    if method == "otsu":
        threshold = otsu_threshold(gray)
        binary = (gray > threshold).astype(np.uint8) * 255

    elif method == "adaptive":
        # 적응적 이진화: 지역별로 다른 임계값 사용
        # 의료 문서처럼 조명이 균일하지 않은 경우에 효과적
        block_size = 51
        constant = 10
        binary = adaptive_threshold(gray, block_size, constant)

    return binary


def adaptive_threshold(
    gray: np.ndarray,
    block_size: int = 51,
    constant: int = 10,
) -> np.ndarray:
    """
    적응적 임계값 이진화.

    각 픽셀의 임계값을 주변 block_size × block_size 영역의
    평균에서 constant를 뺀 값으로 설정한다.

    Args:
        gray: (H, W) 그레이스케일 이미지
        block_size: 지역 블록 크기 (홀수)
        constant: 평균에서 빼는 상수

    Returns:
        binary: (H, W) 이진 이미지
    """
    H, W = gray.shape
    pad = block_size // 2

    # 패딩
    padded = np.pad(gray.astype(np.float64), pad, mode="reflect")

    # 적분 이미지로 빠른 지역 평균 계산
    integral = padded.cumsum(axis=0).cumsum(axis=1)

    # 지역 평균 계산
    local_mean = np.zeros_like(gray, dtype=np.float64)
    for y in range(H):
        for x in range(W):
            y1, x1 = y, x
            y2, x2 = y + block_size, x + block_size
            area = block_size * block_size
            local_sum = (
                integral[y2, x2]
                - integral[y1, x2]
                - integral[y2, x1]
                + integral[y1, x1]
            )
            local_mean[y, x] = local_sum / area

    # 이진화
    binary = (gray > (local_mean - constant)).astype(np.uint8) * 255

    return binary
```

### 2. 노이즈 제거

#### Gaussian Filter

가우시안 블러로 고주파 잡음을 제거한다.

2D 가우시안 커널:

$$
G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)
$$

필터링 결과:

$$
I_{\text{filtered}}(x, y) = (I * G)(x, y) = \sum_{i}\sum_{j} I(x-i, y-j) \cdot G(i, j)
$$

#### Morphological Operations

형태학적 연산으로 문서 이미지의 구조적 잡음을 처리한다.

**침식 (Erosion)**: 작은 잡음 제거, 텍스트가 살짝 얇아짐

$$
(I \ominus K)(x, y) = \min_{(i,j) \in K} I(x+i, y+j)
$$

**팽창 (Dilation)**: 끊어진 획 연결, 텍스트가 살짝 두꺼워짐

$$
(I \oplus K)(x, y) = \max_{(i,j) \in K} I(x+i, y+j)
$$

**열림 (Opening)** = 침식 → 팽창: 잡음 제거에 사용

$$
I \circ K = (I \ominus K) \oplus K
$$

**닫힘 (Closing)** = 팽창 → 침식: 끊어진 획 연결에 사용

$$
I \bullet K = (I \oplus K) \ominus K
$$

```python
import numpy as np
from typing import Tuple


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    2D 가우시안 커널을 생성한다.

    Args:
        size: 커널 크기 (홀수)
        sigma: 표준편차

    Returns:
        kernel: (size, size) 정규화된 가우시안 커널
    """
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def apply_gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    가우시안 필터를 적용하여 잡음을 제거한다.

    Args:
        image: (H, W) 그레이스케일 이미지
        kernel_size: 커널 크기
        sigma: 가우시안 표준편차

    Returns:
        filtered: (H, W) 필터링된 이미지
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    pad = kernel_size // 2

    padded = np.pad(image.astype(np.float64), pad, mode="reflect")
    H, W = image.shape
    filtered = np.zeros_like(image, dtype=np.float64)

    for y in range(H):
        for x in range(W):
            region = padded[y:y+kernel_size, x:x+kernel_size]
            filtered[y, x] = np.sum(region * kernel)

    return filtered.astype(np.uint8)


def morphological_open(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    형태학적 열림(Opening) 연산. 작은 잡음 제거에 사용.

    Args:
        binary: (H, W) 이진 이미지 (0 또는 255)
        kernel_size: 구조 요소 크기

    Returns:
        opened: (H, W) 열림 연산 결과
    """
    # 침식
    eroded = morphological_erode(binary, kernel_size)
    # 팽창
    opened = morphological_dilate(eroded, kernel_size)
    return opened


def morphological_close(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    형태학적 닫힘(Closing) 연산. 끊어진 획 연결에 사용.

    Args:
        binary: (H, W) 이진 이미지
        kernel_size: 구조 요소 크기

    Returns:
        closed: (H, W) 닫힘 연산 결과
    """
    # 팽창
    dilated = morphological_dilate(binary, kernel_size)
    # 침식
    closed = morphological_erode(dilated, kernel_size)
    return closed


def morphological_erode(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """침식 연산."""
    pad = kernel_size // 2
    padded = np.pad(binary, pad, mode="constant", constant_values=255)
    H, W = binary.shape
    result = np.zeros_like(binary)

    for y in range(H):
        for x in range(W):
            region = padded[y:y+kernel_size, x:x+kernel_size]
            result[y, x] = region.min()

    return result


def morphological_dilate(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """팽창 연산."""
    pad = kernel_size // 2
    padded = np.pad(binary, pad, mode="constant", constant_values=0)
    H, W = binary.shape
    result = np.zeros_like(binary)

    for y in range(H):
        for x in range(W):
            region = padded[y:y+kernel_size, x:x+kernel_size]
            result[y, x] = region.max()

    return result
```

### 3. 숫자 인식 검증: Checksum & 금액 합계 일관성

의료 문서에서 금액의 정확성을 검증하는 방법이다.

#### 합계 일관성 검증

진료비 세부내역서에서 각 항목의 금액과 합계가 일치하는지 검증:

$$
\text{Verified} = \begin{cases}
\text{True} & \text{if } \sum_{i=1}^{n} \text{amount}_i = \text{total} \\
\text{False} & \text{otherwise}
\end{cases}
$$

오차 허용 범위를 두는 경우:

$$
|\sum_{i=1}^{n} \text{amount}_i - \text{total}| \leq \epsilon
$$

보험 청구에서는 $\epsilon = 0$이어야 한다. 1원이라도 맞지 않으면 검증 실패.

#### 단가 × 횟수 검증

$$
\text{amount}_i = \text{unit\_price}_i \times \text{quantity}_i \times \text{days}_i
$$

이 관계를 이용하면 OCR 오류를 역으로 추적할 수 있다.

```python
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class MedicalBillingItem:
    """진료비 세부내역 항목."""
    code: str               # 항목코드 (약품코드/행위코드)
    name: str               # 항목명
    unit_price: int          # 단가 (원)
    quantity: int            # 횟수
    days: int               # 일수
    amount: int              # 금액
    insurance_covered: bool  # 급여 여부
    patient_copay: int       # 본인부담금


@dataclass
class BillingVerificationResult:
    """청구서 검증 결과."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    corrected_items: List[Tuple[int, str, int, int]]  # (항목인덱스, 필드, 기존값, 수정값)


class MedicalBillingVerifier:
    """
    진료비 세부내역서의 금액 정합성을 검증한다.

    OCR 결과의 숫자 오류를 탐지하고, 가능한 경우 자동 수정을 제안한다.
    """

    def verify(
        self,
        items: List[MedicalBillingItem],
        claimed_total: int,
        claimed_copay_total: int,
    ) -> BillingVerificationResult:
        """
        청구서 전체를 검증한다.

        Args:
            items: 세부내역 항목 리스트
            claimed_total: 청구서에 기재된 총 금액
            claimed_copay_total: 청구서에 기재된 총 본인부담금

        Returns:
            검증 결과
        """
        errors = []
        warnings = []
        corrections = []

        # 1. 항목별 단가×횟수×일수 = 금액 검증
        for i, item in enumerate(items):
            expected = item.unit_price * item.quantity * item.days
            if expected != item.amount:
                errors.append(
                    f"항목 {i} ({item.name}): "
                    f"단가({item.unit_price}) × 횟수({item.quantity}) × 일수({item.days}) "
                    f"= {expected} ≠ 금액({item.amount})"
                )
                # 자동 수정 제안: 금액을 계산값으로
                corrections.append((i, "amount", item.amount, expected))

        # 2. 합계 검증
        calculated_total = sum(item.amount for item in items)
        if calculated_total != claimed_total:
            errors.append(
                f"합계 불일치: 항목 합산({calculated_total}) ≠ 기재 합계({claimed_total})"
            )

        # 3. 본인부담금 합계 검증
        calculated_copay = sum(item.patient_copay for item in items)
        if calculated_copay != claimed_copay_total:
            errors.append(
                f"본인부담금 합계 불일치: "
                f"항목 합산({calculated_copay}) ≠ 기재 합계({claimed_copay_total})"
            )

        # 4. 금액 범위 검증 (비정상적으로 높거나 낮은 금액)
        for i, item in enumerate(items):
            if item.unit_price < 0 or item.amount < 0:
                errors.append(f"항목 {i} ({item.name}): 음수 금액 탐지")
            if item.unit_price > 50_000_000:  # 단가 5천만원 초과
                warnings.append(f"항목 {i} ({item.name}): 비정상적으로 높은 단가({item.unit_price})")

        # 5. 본인부담금이 금액을 초과하는지 검증
        for i, item in enumerate(items):
            if item.patient_copay > item.amount:
                errors.append(
                    f"항목 {i} ({item.name}): "
                    f"본인부담금({item.patient_copay}) > 금액({item.amount})"
                )

        is_valid = len(errors) == 0

        return BillingVerificationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            corrected_items=corrections,
        )


# ---- 사용 예시 ----
items = [
    MedicalBillingItem("642801ATB", "세프트리악손주1g", 3500, 2, 5, 35000, True, 10500),
    MedicalBillingItem("N0611", "관혈적 정복술", 850000, 1, 1, 850000, True, 255000),
    MedicalBillingItem("LA311", "일반 마취", 350000, 1, 1, 350000, True, 105000),
    MedicalBillingItem("AJ100", "입원료(1인실)", 150000, 1, 5, 750000, False, 750000),
]

verifier = MedicalBillingVerifier()
result = verifier.verify(
    items=items,
    claimed_total=1_985_000,
    claimed_copay_total=1_120_500,
)

print(f"검증 통과: {result.is_valid}")
for err in result.errors:
    print(f"  [오류] {err}")
for warn in result.warnings:
    print(f"  [경고] {warn}")
```

---

## 의료 문서 전처리 파이프라인

전체 파이프라인을 하나로 통합한다.

```python
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Dict, Any
import math


@dataclass
class PreprocessConfig:
    """전처리 설정."""
    target_dpi: int = 300
    binarize: bool = True
    binarize_method: str = "adaptive"
    denoise: bool = True
    denoise_sigma: float = 1.0
    deskew: bool = True
    deskew_max_angle: float = 5.0
    remove_stamps: bool = False
    enhance_contrast: bool = True


class MedicalDocumentPreprocessor:
    """
    의료 문서 전처리 파이프라인.

    스캔/팩스 품질의 의료 문서를 OCR에 최적화된 상태로 전처리한다.
    """

    def __init__(self, config: PreprocessConfig = None):
        self.config = config or PreprocessConfig()

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        전처리 파이프라인 실행.

        Args:
            image: (H, W, 3) 또는 (H, W) 원본 이미지

        Returns:
            processed: 전처리된 이미지
        """
        # 그레이스케일 변환
        if image.ndim == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.copy()

        # 1. 대비 향상
        if self.config.enhance_contrast:
            gray = self._enhance_contrast(gray)

        # 2. 기울기 보정
        if self.config.deskew:
            gray = self._deskew(gray)

        # 3. 잡음 제거
        if self.config.denoise:
            gray = apply_gaussian_filter(gray, kernel_size=3, sigma=self.config.denoise_sigma)

        # 4. 이진화
        if self.config.binarize:
            gray = binarize_document(gray, method=self.config.binarize_method)

        # 5. 형태학적 후처리 (작은 잡음 제거 + 끊어진 획 연결)
        gray = morphological_open(gray, kernel_size=2)   # 잡음 제거
        gray = morphological_close(gray, kernel_size=2)  # 획 연결

        return gray

    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """
        히스토그램 평활화로 대비를 향상한다.

        CLAHE(Contrast Limited Adaptive Histogram Equalization) 방식.
        """
        # 간단한 히스토그램 평활화
        hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        equalized = cdf_normalized[gray].astype(np.uint8)
        return equalized

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """
        문서 기울기를 보정한다.

        Hough Transform 기반으로 텍스트 줄의 각도를 추정하여 회전 보정한다.
        여기서는 프로젝션 프로파일 방식을 사용한다.
        """
        # 프로젝션 프로파일 방식으로 각도 추정
        best_angle = 0
        best_score = 0

        # -5도 ~ +5도 범위에서 0.5도 단위 탐색
        for angle_10x in range(-50, 51, 5):
            angle = angle_10x / 10.0
            rotated = self._rotate(gray, angle)
            # 수평 프로젝션의 분산이 최대인 각도가 최적
            projection = rotated.mean(axis=1)
            score = projection.var()
            if score > best_score:
                best_score = score
                best_angle = angle

        if abs(best_angle) > 0.1:
            return self._rotate(gray, best_angle)
        return gray

    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        이미지를 회전한다 (간소화 버전).

        Args:
            image: (H, W) 이미지
            angle: 회전 각도 (도)

        Returns:
            rotated: 회전된 이미지
        """
        H, W = image.shape
        center_y, center_x = H / 2, W / 2
        rad = math.radians(-angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        rotated = np.full_like(image, 255)  # 배경은 흰색

        for y in range(H):
            for x in range(W):
                # 역변환으로 원본 좌표 계산
                dx = x - center_x
                dy = y - center_y
                src_x = int(cos_a * dx - sin_a * dy + center_x)
                src_y = int(sin_a * dx + cos_a * dy + center_y)

                if 0 <= src_x < W and 0 <= src_y < H:
                    rotated[y, x] = image[src_y, src_x]

        return rotated
```

---

## 필드 추출 후처리

OCR 결과에서 의료 문서의 각 필드를 구조화하는 후처리 모듈이다.

```python
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime


@dataclass
class DiagnosisInfo:
    """진단 정보."""
    disease_name: str               # 상병명
    kcd_code: str                   # KCD 진단코드
    onset_date: Optional[str]       # 발병일
    diagnosis_date: Optional[str]   # 진단일
    doctor_name: Optional[str]      # 의사 성명
    doctor_license: Optional[str]   # 면허번호


@dataclass
class SurgeryInfo:
    """수술 정보."""
    surgery_name: str               # 수술명
    surgery_code: str               # 수술코드
    surgery_date: Optional[str]     # 수술일
    anesthesia_type: Optional[str]  # 마취 방법
    surgeon_name: Optional[str]     # 집도의


@dataclass
class HospitalizationInfo:
    """입퇴원 정보."""
    admission_date: str             # 입원일
    discharge_date: str             # 퇴원일
    department: str                 # 진료과
    disease_name: str               # 상병명
    total_days: int                 # 재원일수


class MedicalFieldExtractor:
    """
    OCR 결과에서 의료 문서의 필드를 추출하고 검증하는 모듈.
    """

    # KCD 코드 패턴: 알파벳 1자리 + 숫자 2~3자리 + 선택적(.숫자)
    KCD_PATTERN = re.compile(r'[A-Z]\d{2,3}(?:\.\d{1,2})?')

    # 의료행위코드 패턴: 알파벳+숫자 혼합 (2~6자리)
    PROCEDURE_CODE_PATTERN = re.compile(r'[A-Z]{1,2}\d{3,5}')

    # 날짜 패턴들
    DATE_PATTERNS = [
        re.compile(r'(\d{4})[.\-/년]\s*(\d{1,2})[.\-/월]\s*(\d{1,2})[일]?'),
        re.compile(r'(\d{4})(\d{2})(\d{2})'),  # YYYYMMDD
    ]

    # 금액 패턴: 쉼표 포함 숫자 + 선택적 "원"
    AMOUNT_PATTERN = re.compile(r'[\d,]+(?:\s*원)?')

    # KCD 코드 오류 수정 맵
    KCD_CORRECTIONS = {
        'O': '0',   # 대문자 O → 숫자 0 (코드 내부에서)
        'l': '1',   # 소문자 l → 숫자 1
        'I': '1',   # 대문자 I → 숫자 1 (첫 글자 제외)
        'S': '5',   # S → 5 (숫자 위치에서)
        'B': '8',   # B → 8 (숫자 위치에서)
    }

    def extract_kcd_code(self, text: str) -> List[str]:
        """
        텍스트에서 KCD 진단코드를 추출한다.

        OCR 오류를 보정하여 유효한 코드를 반환한다.

        Args:
            text: OCR 결과 텍스트

        Returns:
            유효한 KCD 코드 리스트
        """
        candidates = []

        # 1. 정규식으로 후보 추출
        raw_matches = self.KCD_PATTERN.findall(text)
        candidates.extend(raw_matches)

        # 2. OCR 오류 보정 후 재탐색
        corrected_text = self._correct_code_ocr_errors(text)
        corrected_matches = self.KCD_PATTERN.findall(corrected_text)
        candidates.extend(corrected_matches)

        # 3. 중복 제거 및 유효성 검증
        valid_codes = []
        seen = set()
        for code in candidates:
            code = code.upper().strip()
            if code not in seen and self._is_valid_kcd(code):
                valid_codes.append(code)
                seen.add(code)

        return valid_codes

    def _correct_code_ocr_errors(self, text: str) -> str:
        """
        코드 영역의 OCR 오류를 보정한다.

        예: S72.O → S72.0, K2S.1 → K25.1
        """
        result = list(text)

        # KCD 코드 패턴 주변에서만 보정
        # 간단한 휴리스틱: 대문자 뒤에 숫자가 오는 패턴에서 보정
        for i in range(1, len(result)):
            # 첫 글자(알파벳)는 건드리지 않음
            # 나머지 위치에서 숫자여야 할 곳의 문자를 보정
            if i > 0 and result[i-1].isdigit() or (i > 0 and result[i-1] == '.'):
                if result[i] in self.KCD_CORRECTIONS:
                    result[i] = self.KCD_CORRECTIONS[result[i]]

        return ''.join(result)

    def _is_valid_kcd(self, code: str) -> bool:
        """
        KCD 코드의 유효성을 검증한다.

        유효한 첫 글자: A~Z (실제로는 특정 범위만 사용)
        유효한 KCD 챕터:
          A00-B99: 감염
          C00-D48: 신생물
          E00-E90: 내분비
          F00-F99: 정신
          G00-G99: 신경계
          H00-H59: 눈
          H60-H95: 귀
          I00-I99: 순환기
          J00-J99: 호흡기
          K00-K93: 소화기
          L00-L99: 피부
          M00-M99: 근골격
          N00-N99: 비뇨생식
          O00-O99: 임신
          P00-P96: 주산기
          Q00-Q99: 선천기형
          R00-R99: 증상/징후
          S00-T98: 외상
          V01-Y98: 외인
          Z00-Z99: 건강상태
        """
        if not code or len(code) < 3:
            return False

        first_char = code[0].upper()
        valid_chapters = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        if first_char not in valid_chapters:
            return False

        # 첫 글자 이후는 숫자와 . 만 허용
        rest = code[1:]
        if not re.match(r'^\d{2,3}(\.\d{1,2})?$', rest):
            return False

        return True

    def extract_amount(self, text: str) -> Optional[int]:
        """
        텍스트에서 금액을 추출한다.

        쉼표 제거, "원" 제거 후 정수로 변환한다.

        Args:
            text: OCR 결과 텍스트

        Returns:
            금액 (원 단위) 또는 None
        """
        # 쉼표 포함 숫자 추출
        match = self.AMOUNT_PATTERN.search(text)
        if not match:
            return None

        amount_str = match.group()
        # "원" 제거
        amount_str = amount_str.replace('원', '').strip()
        # 쉼표 제거
        amount_str = amount_str.replace(',', '')

        # OCR 오류 보정: O→0, l→1 등
        amount_str = amount_str.replace('O', '0')
        amount_str = amount_str.replace('o', '0')
        amount_str = amount_str.replace('l', '1')
        amount_str = amount_str.replace('I', '1')

        try:
            return int(amount_str)
        except ValueError:
            return None

    def extract_date(self, text: str) -> Optional[str]:
        """
        텍스트에서 날짜를 추출한다.

        다양한 날짜 형식을 YYYY-MM-DD로 통일한다.

        Args:
            text: OCR 결과 텍스트

        Returns:
            "YYYY-MM-DD" 형식 날짜 또는 None
        """
        for pattern in self.DATE_PATTERNS:
            match = pattern.search(text)
            if match:
                year, month, day = match.group(1), match.group(2), match.group(3)
                try:
                    # 유효성 검증
                    dt = datetime(int(year), int(month), int(day))
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue

        return None

    def extract_diagnosis(self, ocr_text: str) -> DiagnosisInfo:
        """
        OCR 결과에서 진단 정보를 추출한다.

        Args:
            ocr_text: 진단서 OCR 결과 전문

        Returns:
            진단 정보 구조체
        """
        kcd_codes = self.extract_kcd_code(ocr_text)
        dates = []
        for line in ocr_text.split('\n'):
            date = self.extract_date(line)
            if date:
                dates.append(date)

        # 상병명 추출: "진단명" 또는 "상병명" 레이블 뒤의 텍스트
        disease_match = re.search(r'(?:진단명|상병명|병명)\s*[:：]?\s*(.+?)(?:\n|$)', ocr_text)
        disease_name = disease_match.group(1).strip() if disease_match else ""

        # 의사 성명 추출
        doctor_match = re.search(r'(?:의사|담당의|주치의)\s*[:：]?\s*(\S+)', ocr_text)
        doctor_name = doctor_match.group(1).strip() if doctor_match else None

        # 면허번호 추출
        license_match = re.search(r'(?:면허|면허번호)\s*[:：]?\s*(\d+)', ocr_text)
        license_no = license_match.group(1).strip() if license_match else None

        return DiagnosisInfo(
            disease_name=disease_name,
            kcd_code=kcd_codes[0] if kcd_codes else "",
            onset_date=dates[0] if len(dates) > 0 else None,
            diagnosis_date=dates[1] if len(dates) > 1 else dates[0] if dates else None,
            doctor_name=doctor_name,
            doctor_license=license_no,
        )


# ---- 사용 예시 ----
extractor = MedicalFieldExtractor()

# KCD 코드 추출 테스트
test_texts = [
    "진단코드: S72.0 (대퇴골 경부 골절)",
    "상병코드 K25.1",
    "진단: M54.5 요통",
    "코드: S72.O",        # OCR 오류: 0 → O
    "I1O 본태성 고혈압",  # OCR 오류: 0 → O
]

print("=== KCD 코드 추출 ===")
for text in test_texts:
    codes = extractor.extract_kcd_code(text)
    print(f"  입력: {text}")
    print(f"  추출: {codes}")

# 금액 추출 테스트
amount_texts = [
    "본인부담금: 1,250,000원",
    "금액 350000",
    "합계: 2,5OO,OOO원",  # OCR 오류: 0 → O
]

print("\n=== 금액 추출 ===")
for text in amount_texts:
    amount = extractor.extract_amount(text)
    print(f"  입력: {text}")
    print(f"  추출: {amount}")

# 날짜 추출 테스트
date_texts = [
    "입원일: 2024년 3월 15일",
    "퇴원일: 2024.03.25",
    "수술일: 2024-03-16",
]

print("\n=== 날짜 추출 ===")
for text in date_texts:
    date = extractor.extract_date(text)
    print(f"  입력: {text}")
    print(f"  추출: {date}")
```

---

## 비식별화 (De-identification) 처리

의료 문서에는 환자 개인정보가 포함되어 있다. OCR 후 반드시 비식별화를 수행해야 한다.

### PHI (Protected Health Information) 유형

| PHI 유형 | 예시 | 마스킹 방법 |
|----------|------|------------|
| 이름 | 홍길동 | [이름] |
| 주민등록번호 | 900101-1234567 | [주민번호] |
| 연락처 | 010-1234-5678 | [전화번호] |
| 주소 | 서울시 강남구 역삼동 | [주소] |
| 생년월일 | 1990.01.01 | [생년월일] |
| 의료기록번호 | MRN-12345678 | [의료기록번호] |

```python
import re
from typing import Dict, List, Tuple


class MedicalDeidentifier:
    """
    의료 문서 비식별화 모듈.

    OCR 결과에서 PHI(Protected Health Information)를 탐지하고 마스킹한다.

    주의: 이 모듈은 기본적인 패턴 매칭 기반이다. 프로덕션에서는
    NER(Named Entity Recognition) 모델과 결합하여 사용해야 한다.
    """

    # 주민등록번호 패턴: YYMMDD-N######
    RRN_PATTERN = re.compile(r'\d{6}\s*[-]\s*[1-4]\d{6}')

    # 전화번호 패턴
    PHONE_PATTERNS = [
        re.compile(r'01[016789][-.\s]?\d{3,4}[-.\s]?\d{4}'),   # 휴대폰
        re.compile(r'0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}'),      # 일반 전화
    ]

    # 이메일 패턴
    EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

    # 의료기록번호 패턴
    MRN_PATTERNS = [
        re.compile(r'(?:MRN|환자번호|등록번호)\s*[:：]?\s*(\w+)'),
    ]

    def __init__(self, custom_names: List[str] = None):
        """
        Args:
            custom_names: 추가 이름 목록 (환자명 사전)
        """
        self.custom_names = custom_names or []

    def deidentify(self, text: str) -> Tuple[str, Dict[str, List[str]]]:
        """
        텍스트에서 PHI를 탐지하고 마스킹한다.

        Args:
            text: OCR 결과 텍스트

        Returns:
            (마스킹된 텍스트, 탐지된 PHI 딕셔너리)
        """
        detected_phi: Dict[str, List[str]] = {
            "주민번호": [],
            "전화번호": [],
            "이메일": [],
            "의료기록번호": [],
        }

        masked_text = text

        # 1. 주민등록번호 마스킹
        for match in self.RRN_PATTERN.finditer(text):
            detected_phi["주민번호"].append(match.group())
            masked_text = masked_text.replace(match.group(), "[주민번호]")

        # 2. 전화번호 마스킹
        for pattern in self.PHONE_PATTERNS:
            for match in pattern.finditer(masked_text):
                detected_phi["전화번호"].append(match.group())
                masked_text = masked_text.replace(match.group(), "[전화번호]")

        # 3. 이메일 마스킹
        for match in self.EMAIL_PATTERN.finditer(masked_text):
            detected_phi["이메일"].append(match.group())
            masked_text = masked_text.replace(match.group(), "[이메일]")

        # 4. 의료기록번호 마스킹
        for pattern in self.MRN_PATTERNS:
            for match in pattern.finditer(masked_text):
                if match.group(1):
                    detected_phi["의료기록번호"].append(match.group(1))
                    masked_text = masked_text.replace(
                        match.group(), match.group().replace(match.group(1), "[의료기록번호]")
                    )

        return masked_text, detected_phi


# ---- 사용 예시 ----
deidentifier = MedicalDeidentifier()

sample_ocr = """
진 단 서

성명: 홍길동
주민등록번호: 900101-1234567
연락처: 010-1234-5678
의료기록번호: MRN:A12345678

진단명: 대퇴골 경부 골절
진단코드: S72.0
발병일: 2024년 3월 10일
진단일: 2024년 3월 10일

상기 환자는 위 진단을 받았음을 확인합니다.

담당의: 김의사 (면허번호: 12345)
"""

masked, phi_found = deidentifier.deidentify(sample_ocr)
print("=== 비식별화 결과 ===")
print(masked)
print("\n=== 탐지된 PHI ===")
for phi_type, values in phi_found.items():
    if values:
        print(f"  {phi_type}: {values}")
```

---

## 전체 의료 문서 OCR 파이프라인

모든 모듈을 통합한 엔드투엔드 파이프라인이다.

```python
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class MedicalOCRResult:
    """의료 문서 OCR 최종 결과."""
    document_type: str                              # 문서 유형
    raw_text: str                                    # 원본 OCR 텍스트
    deidentified_text: str                           # 비식별화 텍스트
    extracted_fields: Dict[str, Any] = field(default_factory=dict)
    verification: Optional[Dict] = None              # 검증 결과
    confidence: float = 0.0
    phi_detected: Dict[str, List[str]] = field(default_factory=dict)


class MedicalDocumentOCRPipeline:
    """
    보험금 청구용 의료 문서 OCR 파이프라인.

    전처리 → OCR → 필드 추출 → 검증 → 비식별화의 전체 과정을 통합한다.

    사용법:
        pipeline = MedicalDocumentOCRPipeline()
        result = pipeline.process(image, document_type="diagnosis")
    """

    def __init__(
        self,
        preprocess_config: PreprocessConfig = None,
        ocr_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        enable_deidentification: bool = True,
    ):
        self.preprocessor = MedicalDocumentPreprocessor(
            preprocess_config or PreprocessConfig()
        )
        self.field_extractor = MedicalFieldExtractor()
        self.billing_verifier = MedicalBillingVerifier()
        self.deidentifier = MedicalDeidentifier()
        self.enable_deidentification = enable_deidentification

        # OCR 모델은 지연 로드
        self._ocr_model = None
        self._ocr_model_name = ocr_model_name

    # 문서 유형별 OCR 프롬프트
    OCR_PROMPTS = {
        "diagnosis": (
            "이 진단서를 분석해라. 다음 필드를 JSON으로 추출해라:\n"
            "- 성명\n- 주민등록번호\n- 진단명(상병명)\n- 진단코드(KCD)\n"
            "- 발병일\n- 진단일\n- 의사성명\n- 면허번호\n"
            "정확한 코드와 날짜를 추출하는 것이 중요하다."
        ),
        "surgery": (
            "이 수술기록을 분석해라. 다음 필드를 JSON으로 추출해라:\n"
            "- 수술명\n- 수술코드\n- 수술일시\n- 마취방법\n"
            "- 집도의\n- 보조의"
        ),
        "hospitalization": (
            "이 입퇴원확인서를 분석해라. 다음 필드를 JSON으로 추출해라:\n"
            "- 입원일\n- 퇴원일\n- 진료과\n- 상병명\n- 재원일수"
        ),
        "billing_detail": (
            "이 진료비 세부내역서의 모든 항목을 JSON 배열로 추출해라.\n"
            "각 항목: {코드, 항목명, 단가, 횟수, 일수, 금액, 급여여부, 본인부담금}\n"
            "마지막에 합계 금액과 본인부담금 합계도 포함해라.\n"
            "금액은 반드시 숫자로, 쉼표 없이 정수로 변환해라."
        ),
    }

    def process(
        self,
        image: np.ndarray,
        document_type: str = "diagnosis",
    ) -> MedicalOCRResult:
        """
        의료 문서 OCR 전체 파이프라인 실행.

        Args:
            image: (H, W, 3) 또는 (H, W) 원본 이미지
            document_type: "diagnosis" | "surgery" | "hospitalization" | "billing_detail"

        Returns:
            OCR 결과
        """
        try:
            # Step 1: 전처리
            logger.info(f"[Step 1] 전처리 시작 (문서 유형: {document_type})")
            preprocessed = self.preprocessor.process(image)

            # Step 2: OCR 실행
            logger.info("[Step 2] OCR 실행")
            prompt = self.OCR_PROMPTS.get(document_type, self.OCR_PROMPTS["diagnosis"])
            raw_text = self._run_ocr(preprocessed, prompt)

            # Step 3: 필드 추출
            logger.info("[Step 3] 필드 추출")
            extracted_fields = self._extract_fields(raw_text, document_type)

            # Step 4: 검증
            logger.info("[Step 4] 검증")
            verification = None
            if document_type == "billing_detail":
                verification = self._verify_billing(extracted_fields)

            # Step 5: 비식별화
            logger.info("[Step 5] 비식별화")
            deidentified_text = raw_text
            phi_detected = {}
            if self.enable_deidentification:
                deidentified_text, phi_detected = self.deidentifier.deidentify(raw_text)

            return MedicalOCRResult(
                document_type=document_type,
                raw_text=raw_text,
                deidentified_text=deidentified_text,
                extracted_fields=extracted_fields,
                verification=verification,
                confidence=0.95,  # 실제로는 OCR 모델의 confidence 사용
                phi_detected=phi_detected,
            )

        except Exception as e:
            logger.error(f"OCR 파이프라인 오류: {str(e)}", exc_info=True)
            return MedicalOCRResult(
                document_type=document_type,
                raw_text="",
                deidentified_text="",
                confidence=0.0,
            )

    def _run_ocr(self, image: np.ndarray, prompt: str) -> str:
        """
        OCR 모델을 실행한다.

        실제 구현에서는 Qwen2.5-VL 등 VLM 모델 사용.
        여기서는 인터페이스만 정의.
        """
        # 실제 구현:
        # pil_image = Image.fromarray(image)
        # return run_qwen25vl_ocr(pil_image, prompt=prompt)

        # Placeholder
        logger.warning("OCR 모델 미로드. Placeholder 결과 반환.")
        return "[OCR 결과 placeholder]"

    def _extract_fields(self, text: str, document_type: str) -> Dict[str, Any]:
        """문서 유형에 맞는 필드를 추출한다."""
        if document_type == "diagnosis":
            info = self.field_extractor.extract_diagnosis(text)
            return {
                "disease_name": info.disease_name,
                "kcd_code": info.kcd_code,
                "onset_date": info.onset_date,
                "diagnosis_date": info.diagnosis_date,
                "doctor_name": info.doctor_name,
                "doctor_license": info.doctor_license,
            }
        elif document_type == "billing_detail":
            # JSON 파싱 시도
            try:
                # OCR 결과에서 JSON 블록 추출
                json_match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', text)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

            return {"raw_text": text}

        return {"raw_text": text}

    def _verify_billing(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """진료비 내역을 검증한다."""
        if "items" not in fields:
            return {"status": "skip", "reason": "항목 데이터 없음"}

        items = []
        for item_data in fields["items"]:
            try:
                items.append(MedicalBillingItem(
                    code=item_data.get("code", ""),
                    name=item_data.get("name", ""),
                    unit_price=int(item_data.get("unit_price", 0)),
                    quantity=int(item_data.get("quantity", 1)),
                    days=int(item_data.get("days", 1)),
                    amount=int(item_data.get("amount", 0)),
                    insurance_covered=item_data.get("insurance_covered", True),
                    patient_copay=int(item_data.get("patient_copay", 0)),
                ))
            except (ValueError, TypeError) as e:
                logger.warning(f"항목 파싱 오류: {e}")

        if not items:
            return {"status": "error", "reason": "유효한 항목 없음"}

        result = self.billing_verifier.verify(
            items=items,
            claimed_total=int(fields.get("total", 0)),
            claimed_copay_total=int(fields.get("copay_total", 0)),
        )

        return {
            "status": "pass" if result.is_valid else "fail",
            "errors": result.errors,
            "warnings": result.warnings,
            "corrections": [
                {"item_index": c[0], "field": c[1], "old": c[2], "new": c[3]}
                for c in result.corrected_items
            ],
        }


# ---- 사용 예시 ----
def demo_pipeline():
    """파이프라인 데모 (실제 이미지 없이)."""
    pipeline = MedicalDocumentOCRPipeline(
        enable_deidentification=True,
    )

    # 가상 이미지 (실제로는 스캔 이미지)
    dummy_image = np.random.randint(0, 255, (3508, 2480, 3), dtype=np.uint8)

    result = pipeline.process(dummy_image, document_type="diagnosis")

    print(f"문서 유형: {result.document_type}")
    print(f"신뢰도: {result.confidence}")
    print(f"비식별화 텍스트: {result.deidentified_text[:100]}...")
    print(f"추출 필드: {json.dumps(result.extracted_fields, ensure_ascii=False, indent=2)}")


# demo_pipeline()
```

---

## 프로덕션 배포 고려사항

| 고려사항 | 설명 | 권장 방안 |
|----------|------|----------|
| **개인정보** | 의료 정보는 민감 데이터 | 비식별화 필수, 암호화 저장, 접근 로그 |
| **정확도 SLA** | 금액 오류 = 보험 사고 | 인간 검토(Human-in-the-loop) 병행 |
| **처리 속도** | 보험 청구 대량 처리 | GPU 서버 + 배치 처리 + 큐 시스템 |
| **감사 추적** | 규제 준수 | 모든 처리 단계 로깅, 원본 이미지 보관 |
| **모델 업데이트** | 새로운 양식 대응 | A/B 테스트 + 점진적 롤아웃 |
| **장애 대응** | OCR 실패 시 | 수동 입력 폴백 + 알림 |

---

## 용어 체크리스트

아래 용어들을 설명할 수 있으면 이 챕터를 이해한 거다.

- [ ] **비식별화**: PHI의 18가지 유형 중 의료 문서에서 가장 흔한 것을 나열할 수 있는가?
- [ ] **PHI**: 개인정보보호법과 HIPAA에서 PHI 처리 규정을 설명할 수 있는가?
- [ ] **Pre-printed Form**: 인쇄 양식과 수기 텍스트를 구분하는 전략을 설명할 수 있는가?
- [ ] **KCD 코드**: 형식 규칙을 알고, OCR 오인식 패턴(0↔O, 1↔l)을 보정하는 방법을 설명할 수 있는가?
- [ ] **의료행위코드**: 코드 형식과 보험금 산정에서의 역할을 설명할 수 있는가?
- [ ] **Otsu's Method**: 클래스 간 분산 최대화 공식을 유도할 수 있는가?
- [ ] **Morphological Operations**: 열림/닫힘 연산의 효과와 문서 전처리에서의 용도를 설명할 수 있는가?
- [ ] **금액 검증**: 단가×횟수=금액, 항목합=합계 검증 방법을 설명할 수 있는가?
- [ ] **적응적 이진화**: Otsu vs Adaptive Threshold의 차이와 각각 적합한 상황을 설명할 수 있는가?
- [ ] **Deskew**: 문서 기울기 보정의 원리(프로젝션 프로파일)를 설명할 수 있는가?
- [ ] **EDI**: 의료기관↔보험사 간 전자 데이터 교환의 흐름을 설명할 수 있는가?
