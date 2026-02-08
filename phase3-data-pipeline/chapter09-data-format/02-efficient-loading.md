# 9.2 효율적 데이터 로딩

GPU는 비싸다. GPU가 데이터를 기다리며 놀고 있으면 돈을 태우는 거다. VLM 학습에서 이미지 데이터 로딩은 가장 흔한 병목 중 하나이며, 특히 고해상도 의료 문서는 파일 하나가 수 MB에 달한다. 이 절에서는 I/O 병목을 체계적으로 분석하고, Memory Mapping, Prefetching, WebDataset 등 실전 기법을 코드와 함께 다룬다.

---

## 핵심 용어

| 용어 | 정의 | 관련 기술 |
|------|------|-----------|
| **Memory Mapping (mmap)** | 파일을 가상 메모리에 매핑하여 디스크 I/O 없이 접근하는 기법 | `numpy.memmap`, Arrow |
| **Lazy Loading** | 데이터를 실제 사용 시점에 로딩하여 초기 메모리 사용량을 줄이는 전략 | HuggingFace Datasets |
| **Prefetching** | 현재 배치 처리 중 다음 배치를 미리 로딩하여 I/O 대기를 제거 | `prefetch_factor` |
| **WebDataset** | tar 아카이브 기반 순차 읽기 최적화 데이터셋 포맷 | wds 라이브러리 |
| **Apache Arrow** | 컬럼 지향 메모리 포맷으로 zero-copy 읽기를 지원 | HuggingFace Datasets |
| **Pin Memory** | 호스트 메모리를 page-locked로 할당하여 GPU 전송 속도를 높이는 기법 | `pin_memory=True` |
| **num_workers** | DataLoader의 병렬 데이터 로딩 프로세스 수 | PyTorch DataLoader |
| **Collate Function** | 개별 샘플을 배치로 합치는 함수 | 패딩, 동적 배치 |

---

## 9.2.1 I/O 병목 분석

### 데이터 로딩의 전체 흐름

```
디스크 → 호스트 메모리 → 전처리(CPU) → GPU 메모리 → 학습(GPU)
  (I/O)     (Decode)      (Transform)    (Transfer)    (Compute)
```

각 단계에서 병목이 발생할 수 있고, 전체 처리량은 가장 느린 단계에 의해 결정된다.

### 병목 위치 진단

```python
import time
import torch
from torch.utils.data import DataLoader

def diagnose_bottleneck(dataloader: DataLoader, num_batches: int = 50):
    """
    DataLoader 병목 진단

    I/O 시간 > Compute 시간 → I/O 병목
    I/O 시간 < Compute 시간 → Compute 병목 (이상적)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Phase 1: 데이터 로딩 시간 측정
    load_times = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        t0 = time.perf_counter()
        _ = batch  # 로딩만 수행
        load_times.append(time.perf_counter() - t0)

    avg_load = sum(load_times) / len(load_times)

    # Phase 2: GPU 전송 시간 측정
    transfer_times = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        t0 = time.perf_counter()
        if isinstance(batch, dict):
            _ = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        torch.cuda.synchronize()
        transfer_times.append(time.perf_counter() - t0)

    avg_transfer = sum(transfer_times) / len(transfer_times)

    print(f"평균 로딩 시간:  {avg_load*1000:.1f} ms/batch")
    print(f"평균 전송 시간:  {avg_transfer*1000:.1f} ms/batch")

    if avg_load > avg_transfer * 2:
        print("→ I/O 병목. num_workers 증가 또는 데이터 포맷 최적화 필요.")
    else:
        print("→ Compute 병목 또는 균형 상태. 현재 설정 적절.")

    return {"load_ms": avg_load * 1000, "transfer_ms": avg_transfer * 1000}
```

---

## 수학적 원리

### I/O vs Compute 처리량 분석

학습 파이프라인의 실효 처리량(effective throughput)은 I/O와 Compute 중 느린 쪽에 의해 결정된다:

$$
\text{Throughput}_{\text{eff}} = \min\left(\text{Throughput}_{\text{I/O}},\ \text{Throughput}_{\text{compute}}\right)
$$

여기서 각 처리량은:

$$
\text{Throughput}_{\text{I/O}} = \frac{B \cdot S_{\text{avg}}}{T_{\text{read}} + T_{\text{decode}} + T_{\text{transform}}}
$$

- $B$: 배치 크기
- $S_{\text{avg}}$: 평균 샘플 크기 (bytes)
- $T_{\text{read}}$: 디스크 읽기 시간
- $T_{\text{decode}}$: 이미지 디코딩 시간
- $T_{\text{transform}}$: 전처리 시간

$$
\text{Throughput}_{\text{compute}} = \frac{B}{T_{\text{forward}} + T_{\text{backward}}}
$$

GPU 활용률(GPU Utilization)은:

$$
U_{\text{GPU}} = \frac{T_{\text{compute}}}{T_{\text{compute}} + T_{\text{idle}}}
$$

$T_{\text{idle}}$이 0이면 GPU 활용률 100%. 목표는 $U_{\text{GPU}} \to 1$이다.

### Prefetch 깊이 최적화

Prefetch는 현재 배치를 GPU에서 처리하는 동안 다음 배치를 미리 CPU에서 로딩하는 기법이다. Prefetch 깊이 $d$는 미리 준비할 배치 수를 결정한다.

Prefetch가 없을 때의 총 학습 시간 ($N$개 배치):

$$
T_{\text{no\_prefetch}} = N \cdot (T_{\text{load}} + T_{\text{compute}})
$$

Prefetch가 있을 때:

$$
T_{\text{prefetch}} = T_{\text{load}} + N \cdot \max(T_{\text{load}},\ T_{\text{compute}})
$$

속도 향상 비율:

$$
\text{Speedup} = \frac{T_{\text{load}} + T_{\text{compute}}}{\max(T_{\text{load}},\ T_{\text{compute}})}
$$

$T_{\text{load}} = T_{\text{compute}}$일 때 이론적 최대 2배 속도 향상이다.

#### Prefetch 깊이에 따른 메모리 트레이드오프

Prefetch 깊이 $d$일 때 추가 메모리 사용량:

$$
M_{\text{prefetch}} = d \cdot B \cdot \left(S_{\text{img}} + S_{\text{text}}\right)
$$

여기서:
- $B$: 배치 크기
- $S_{\text{img}}$: 이미지 텐서 크기 (예: $3 \times 448 \times 448 \times 4\text{bytes} \approx 2.3\text{MB}$)
- $S_{\text{text}}$: 텍스트 텐서 크기

Prefetch 깊이를 늘려도 $T_{\text{load}} \leq T_{\text{compute}}$이면 추가 이득이 없다. 일반적으로 $d = 2$면 충분하다.

### 병렬 워커 확장성

$W$개의 DataLoader 워커가 있을 때, 이상적 I/O 처리량:

$$
\text{Throughput}_{\text{I/O}}(W) = \min\left(W \cdot \text{Throughput}_{\text{single}},\ \text{BW}_{\text{disk}}\right)
$$

디스크 대역폭 $\text{BW}_{\text{disk}}$이 상한이므로, 워커를 무한히 늘려도 디스크 속도 이상은 못 낸다. SSD에서는 보통 $W = 4 \sim 8$이 최적점이다.

---

## 9.2.2 PyTorch DataLoader 최적화

### 기본 VLM 데이터셋 구현

```python
"""
Chapter 9.2 - 효율적 데이터 로딩
PyTorch DataLoader 최적화 + WebDataset + Memory-mapped 데이터셋
"""

import io
import json
import logging
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. 기본 VLM 데이터셋
# ──────────────────────────────────────────────

class VLMDataset(Dataset):
    """
    VLM 학습용 기본 데이터셋

    JSONL 형식의 대화 데이터를 로딩하고
    이미지 전처리를 수행한다.
    """

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        image_size: int = 448,
        max_length: int = 2048,
        lazy_image: bool = True,
    ):
        """
        Args:
            data_path: JSONL 데이터 파일 경로
            image_dir: 이미지 디렉토리 경로
            image_size: 타겟 이미지 크기
            max_length: 최대 시퀀스 길이
            lazy_image: True면 이미지를 사용 시점에 로딩 (Lazy Loading)
        """
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.max_length = max_length
        self.lazy_image = lazy_image

        # 메타데이터만 먼저 로딩 (Lazy Loading 전략)
        self.samples = self._load_metadata(data_path)

        # 이미지 전처리 파이프라인
        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Eager Loading 시 이미지 캐시
        if not lazy_image:
            self._preload_images()

        logger.info(
            f"데이터셋 초기화: {len(self.samples)}개 샘플, "
            f"lazy_image={lazy_image}"
        )

    def _load_metadata(self, data_path: str) -> list[dict]:
        """JSONL 메타데이터 로딩 (이미지 제외)"""
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num} 파싱 실패: {e}")
        return samples

    def _preload_images(self) -> None:
        """전체 이미지 미리 로딩 (메모리 여유 있을 때)"""
        self._image_cache: dict[str, torch.Tensor] = {}
        for sample in self.samples:
            image_path = self._get_image_path(sample)
            if image_path and str(image_path) not in self._image_cache:
                try:
                    img = Image.open(image_path).convert("RGB")
                    self._image_cache[str(image_path)] = self.image_transform(img)
                except (OSError, IOError) as e:
                    logger.warning(f"이미지 로딩 실패: {image_path}: {e}")
        logger.info(f"이미지 캐시: {len(self._image_cache)}개 로딩 완료")

    def _get_image_path(self, sample: dict) -> Path | None:
        """샘플에서 이미지 경로 추출"""
        if "image" in sample:
            return self.image_dir / sample["image"]
        if "conversations" in sample:
            for msg in sample["conversations"]:
                if isinstance(msg.get("content"), list):
                    for part in msg["content"]:
                        if part.get("type") == "image":
                            return self.image_dir / part["image"]
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image_path = self._get_image_path(sample)

        # 이미지 로딩
        image_tensor = None
        if image_path:
            if self.lazy_image:
                try:
                    img = Image.open(image_path).convert("RGB")
                    image_tensor = self.image_transform(img)
                except (OSError, IOError) as e:
                    logger.warning(f"이미지 로딩 실패 [{idx}]: {e}")
                    image_tensor = torch.zeros(3, self.image_size, self.image_size)
            else:
                image_tensor = self._image_cache.get(
                    str(image_path),
                    torch.zeros(3, self.image_size, self.image_size),
                )

        return {
            "id": sample.get("id", str(idx)),
            "image": image_tensor,
            "conversations": sample.get("conversations", []),
        }


# ──────────────────────────────────────────────
# 2. 최적화된 DataLoader 팩토리
# ──────────────────────────────────────────────

class OptimizedDataLoaderFactory:
    """
    VLM 학습용 최적화된 DataLoader 생성 팩토리

    num_workers, pin_memory, prefetch_factor 등을
    하드웨어에 맞게 자동 조정한다.
    """

    @staticmethod
    def create(
        dataset: Dataset,
        batch_size: int = 4,
        num_workers: int | None = None,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        shuffle: bool = True,
        drop_last: bool = True,
        persistent_workers: bool = True,
        collate_fn=None,
        sampler: Sampler | None = None,
    ) -> DataLoader:
        """
        최적화된 DataLoader 생성

        Args:
            dataset: 데이터셋
            batch_size: 배치 크기
            num_workers: 워커 수 (None이면 자동 감지)
            pin_memory: Page-locked 메모리 사용 여부
            prefetch_factor: 워커당 미리 로딩할 배치 수
            shuffle: 셔플 여부 (sampler 사용 시 무시)
            drop_last: 마지막 불완전 배치 버릴지
            persistent_workers: 에폭 간 워커 프로세스 유지
            collate_fn: 커스텀 collate 함수
            sampler: 커스텀 샘플러
        """
        if num_workers is None:
            num_workers = OptimizedDataLoaderFactory._auto_detect_workers()

        # sampler 사용 시 shuffle 비활성화
        if sampler is not None:
            shuffle = False

        # persistent_workers는 num_workers > 0일 때만
        if num_workers == 0:
            persistent_workers = False
            prefetch_factor = None  # 0 워커에서는 prefetch 불가

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn or VLMCollator(),
            sampler=sampler,
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )

        logger.info(
            f"DataLoader 생성: batch_size={batch_size}, "
            f"num_workers={num_workers}, pin_memory={pin_memory}, "
            f"prefetch_factor={prefetch_factor}"
        )
        return loader

    @staticmethod
    def _auto_detect_workers() -> int:
        """CPU 코어 수 기반 워커 수 자동 감지"""
        import os
        cpu_count = os.cpu_count() or 4
        # GPU당 적절한 워커 수: 보통 CPU 코어의 절반
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        workers = min(cpu_count // max(gpu_count, 1), 8)
        return max(workers, 2)


class VLMCollator:
    """
    VLM 배치 collate 함수

    이미지 텐서와 대화 데이터를 배치로 합친다.
    이미지 크기가 동일하다고 가정 (Dynamic Resolution은 ch9.3에서).
    """

    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        ids = [item["id"] for item in batch]
        conversations = [item["conversations"] for item in batch]

        # 이미지 텐서 스택
        images = []
        for item in batch:
            if item["image"] is not None:
                images.append(item["image"])

        image_tensor = torch.stack(images) if images else None

        return {
            "ids": ids,
            "images": image_tensor,
            "conversations": conversations,
        }


# ──────────────────────────────────────────────
# 3. Memory-Mapped 데이터셋
# ──────────────────────────────────────────────

class MemoryMappedImageDataset(Dataset):
    """
    Memory-mapped 이미지 데이터셋

    이미지를 하나의 바이너리 파일에 연속 저장하고
    mmap으로 접근하여 I/O 오버헤드를 최소화한다.

    파일 구조:
    [Header: 8 bytes (num_samples)]
    [Index: num_samples * 16 bytes (offset, size per sample)]
    [Data: raw image bytes ...]
    """

    HEADER_SIZE = 8     # uint64: num_samples
    INDEX_ENTRY_SIZE = 16  # uint64 offset + uint64 size

    def __init__(
        self,
        data_file: str,
        metadata_file: str,
        image_size: int = 448,
    ):
        """
        Args:
            data_file: mmap 바이너리 데이터 파일 경로
            metadata_file: 메타데이터 JSON 파일 경로
            image_size: 타겟 이미지 크기
        """
        self.data_path = Path(data_file)
        self.image_size = image_size

        # 메타데이터 로딩
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # mmap 파일 열기
        self.mmap_file = np.memmap(
            str(self.data_path),
            dtype=np.uint8,
            mode="r",
        )

        # 인덱스 읽기
        self.num_samples = struct.unpack(
            "<Q", self.mmap_file[:self.HEADER_SIZE].tobytes()
        )[0]

        self.index = self._read_index()

        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        logger.info(
            f"MemoryMapped 데이터셋: {self.num_samples}개 샘플, "
            f"파일 크기: {self.data_path.stat().st_size / 1e9:.2f} GB"
        )

    def _read_index(self) -> list[tuple[int, int]]:
        """인덱스 테이블 읽기: [(offset, size), ...]"""
        index = []
        idx_start = self.HEADER_SIZE
        for i in range(self.num_samples):
            entry_offset = idx_start + i * self.INDEX_ENTRY_SIZE
            entry_bytes = self.mmap_file[
                entry_offset:entry_offset + self.INDEX_ENTRY_SIZE
            ].tobytes()
            offset, size = struct.unpack("<QQ", entry_bytes)
            index.append((offset, size))
        return index

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        offset, size = self.index[idx]
        # mmap에서 직접 읽기 (실제 디스크 I/O는 OS가 최적화)
        raw_bytes = self.mmap_file[offset:offset + size].tobytes()

        # bytes → PIL Image → Tensor
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        image_tensor = self.image_transform(img)

        meta = self.metadata[idx] if idx < len(self.metadata) else {}

        return {
            "id": meta.get("id", str(idx)),
            "image": image_tensor,
            "conversations": meta.get("conversations", []),
        }

    @staticmethod
    def build_mmap_file(
        image_paths: list[str],
        output_path: str,
    ) -> None:
        """
        이미지 파일 리스트 → mmap 바이너리 파일 생성

        Args:
            image_paths: 이미지 파일 경로 리스트
            output_path: 출력 바이너리 파일 경로
        """
        num_samples = len(image_paths)
        header_and_index_size = (
            MemoryMappedImageDataset.HEADER_SIZE
            + num_samples * MemoryMappedImageDataset.INDEX_ENTRY_SIZE
        )

        with open(output_path, "wb") as f:
            # 헤더: 샘플 수
            f.write(struct.pack("<Q", num_samples))

            # 인덱스 공간 예약 (나중에 채움)
            f.write(b"\x00" * num_samples * MemoryMappedImageDataset.INDEX_ENTRY_SIZE)

            # 데이터 기록 + 인덱스 수집
            index_entries = []
            for img_path in image_paths:
                offset = f.tell()
                with open(img_path, "rb") as img_f:
                    data = img_f.read()
                f.write(data)
                index_entries.append((offset, len(data)))

            # 인덱스 채우기
            f.seek(MemoryMappedImageDataset.HEADER_SIZE)
            for offset, size in index_entries:
                f.write(struct.pack("<QQ", offset, size))

        total_size = Path(output_path).stat().st_size
        logger.info(
            f"mmap 파일 생성: {num_samples}개 샘플, "
            f"{total_size / 1e9:.2f} GB → {output_path}"
        )
```

### WebDataset 기반 스트리밍 로딩

```python
# ──────────────────────────────────────────────
# 4. WebDataset 기반 스트리밍 데이터셋
# ──────────────────────────────────────────────

try:
    import webdataset as wds
    HAS_WEBDATASET = True
except ImportError:
    HAS_WEBDATASET = False
    logger.warning("webdataset 미설치. pip install webdataset 필요.")


class WebDatasetBuilder:
    """
    WebDataset(tar 기반) 데이터셋 빌더

    장점:
    - 순차 읽기 최적화 → HDD/NFS에서도 빠름
    - 셔플이 tar 내부에서 수행 → 메모리 효율적
    - 스트리밍 가능 → 거대 데이터셋도 메모리 부족 없음
    """

    @staticmethod
    def create_tar(
        samples: list[dict],
        image_dir: str,
        output_path: str,
        shard_size: int = 1000,
    ) -> list[str]:
        """
        샘플 리스트 → WebDataset tar 샤드 파일 생성

        Args:
            samples: [{"id": "...", "image": "path", "conversations": [...]}]
            image_dir: 이미지 디렉토리
            output_path: 출력 경로 패턴 (예: "data/train-%05d.tar")
            shard_size: 샤드당 샘플 수

        Returns:
            생성된 tar 파일 경로 리스트
        """
        if not HAS_WEBDATASET:
            raise ImportError("webdataset 패키지가 필요하다.")

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        shard_paths = []
        sink = wds.ShardWriter(output_path, maxcount=shard_size)

        for sample in samples:
            img_path = Path(image_dir) / sample["image"]
            if not img_path.exists():
                logger.warning(f"이미지 없음: {img_path}")
                continue

            with open(img_path, "rb") as f:
                image_data = f.read()

            # tar 내부 파일명은 __key__ 기반
            sink.write({
                "__key__": sample["id"],
                "jpg": image_data,
                "json": json.dumps(
                    sample["conversations"], ensure_ascii=False
                ).encode("utf-8"),
            })

        sink.close()
        logger.info(f"WebDataset 생성 완료: {output_path}")
        return shard_paths

    @staticmethod
    def create_dataloader(
        tar_pattern: str,
        image_size: int = 448,
        batch_size: int = 4,
        num_workers: int = 4,
        shuffle_buffer: int = 1000,
    ) -> Any:
        """
        WebDataset tar 패턴 → DataLoader 생성

        Args:
            tar_pattern: tar 파일 glob 패턴 (예: "data/train-{00000..00099}.tar")
            image_size: 이미지 크기
            batch_size: 배치 크기
            num_workers: 워커 수
            shuffle_buffer: 셔플 버퍼 크기
        """
        if not HAS_WEBDATASET:
            raise ImportError("webdataset 패키지가 필요하다.")

        image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        def preprocess(sample: dict) -> dict:
            """WebDataset 샘플 전처리"""
            # jpg → PIL → Tensor
            img = Image.open(io.BytesIO(sample["jpg"])).convert("RGB")
            image_tensor = image_transform(img)

            # json → conversations
            conversations = json.loads(sample["json"].decode("utf-8"))

            return {
                "id": sample["__key__"],
                "image": image_tensor,
                "conversations": conversations,
            }

        dataset = (
            wds.WebDataset(tar_pattern)
            .shuffle(shuffle_buffer)
            .decode("pill")       # 자동 이미지 디코딩
            .map(preprocess)
            .batched(batch_size)
        )

        loader = wds.WebLoader(
            dataset,
            batch_size=None,  # WebDataset이 자체 배칭 수행
            num_workers=num_workers,
            pin_memory=True,
        )

        return loader
```

---

## 9.2.3 Arrow 기반 데이터셋 (HuggingFace Datasets)

```python
# ──────────────────────────────────────────────
# 5. HuggingFace Datasets (Arrow) 활용
# ──────────────────────────────────────────────

class ArrowDatasetManager:
    """
    Apache Arrow 기반 데이터셋 관리자 (HuggingFace Datasets)

    장점:
    - Zero-copy 읽기: 메모리 매핑으로 디스크 데이터를 직접 참조
    - 컬럼 지향: 필요한 컬럼만 읽기 가능
    - 자동 캐싱: 전처리 결과 캐싱으로 재실행 시 빠름
    """

    @staticmethod
    def from_jsonl(
        data_path: str,
        image_dir: str,
        cache_dir: str = ".cache/arrow_dataset",
    ):
        """JSONL → Arrow 데이터셋 변환"""
        try:
            from datasets import Dataset as HFDataset, Features, Value, Sequence
        except ImportError:
            raise ImportError("datasets 패키지가 필요하다. pip install datasets")

        # JSONL 로딩
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        # Arrow Dataset 생성
        dataset = HFDataset.from_list(samples)

        # 디스크에 Arrow 형식으로 저장 (캐싱)
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(cache_path))

        logger.info(
            f"Arrow 데이터셋 생성: {len(dataset)}개 샘플 → {cache_dir}"
        )
        return dataset

    @staticmethod
    def create_torch_dataset(
        arrow_dataset,
        image_dir: str,
        image_size: int = 448,
    ) -> Dataset:
        """Arrow 데이터셋 → PyTorch Dataset 래퍼"""

        image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        class ArrowTorchDataset(Dataset):
            def __init__(self, hf_dataset, img_dir, transform):
                self.dataset = hf_dataset
                self.img_dir = Path(img_dir)
                self.transform = transform

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                item = self.dataset[idx]
                image_tensor = None

                if "image" in item and item["image"]:
                    img_path = self.img_dir / item["image"]
                    try:
                        img = Image.open(img_path).convert("RGB")
                        image_tensor = self.transform(img)
                    except (OSError, IOError):
                        image_tensor = torch.zeros(3, image_size, image_size)

                return {
                    "id": item.get("id", str(idx)),
                    "image": image_tensor,
                    "conversations": item.get("conversations", []),
                }

        return ArrowTorchDataset(arrow_dataset, image_dir, image_transform)
```

---

## 9.2.4 DataLoader 성능 벤치마크

```python
# ──────────────────────────────────────────────
# 6. 벤치마크 유틸리티
# ──────────────────────────────────────────────

class DataLoaderBenchmark:
    """DataLoader 설정별 성능 벤치마크"""

    def __init__(self, dataset: Dataset, num_batches: int = 100):
        self.dataset = dataset
        self.num_batches = num_batches

    def benchmark_workers(
        self,
        batch_size: int = 4,
        worker_range: list[int] | None = None,
    ) -> dict[int, float]:
        """
        num_workers별 처리량 측정

        Returns:
            {num_workers: samples_per_second}
        """
        if worker_range is None:
            worker_range = [0, 1, 2, 4, 8]

        results = {}
        for nw in worker_range:
            loader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                num_workers=nw,
                pin_memory=True,
                prefetch_factor=2 if nw > 0 else None,
                persistent_workers=nw > 0,
            )

            t0 = time.perf_counter()
            count = 0
            for i, batch in enumerate(loader):
                if i >= self.num_batches:
                    break
                count += batch_size

            elapsed = time.perf_counter() - t0
            throughput = count / elapsed
            results[nw] = throughput

            logger.info(
                f"num_workers={nw}: {throughput:.1f} samples/sec "
                f"({elapsed:.2f}s for {count} samples)"
            )

        return results

    def benchmark_prefetch(
        self,
        batch_size: int = 4,
        num_workers: int = 4,
        prefetch_range: list[int] | None = None,
    ) -> dict[int, float]:
        """prefetch_factor별 처리량 측정"""
        if prefetch_range is None:
            prefetch_range = [1, 2, 4, 8]

        results = {}
        for pf in prefetch_range:
            loader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=pf,
                persistent_workers=True,
            )

            t0 = time.perf_counter()
            count = 0
            for i, batch in enumerate(loader):
                if i >= self.num_batches:
                    break
                count += batch_size

            elapsed = time.perf_counter() - t0
            throughput = count / elapsed
            results[pf] = throughput

            logger.info(
                f"prefetch_factor={pf}: {throughput:.1f} samples/sec"
            )

        return results

    def print_report(
        self,
        worker_results: dict[int, float],
        prefetch_results: dict[int, float],
    ) -> None:
        """벤치마크 결과 리포트 출력"""
        print("\n" + "=" * 60)
        print("DataLoader 벤치마크 결과")
        print("=" * 60)

        print("\n[num_workers 별 처리량]")
        print(f"{'Workers':>8} | {'Throughput':>15} | {'Speedup':>8}")
        print("-" * 40)
        base = worker_results.get(0, 1.0)
        for nw, tp in sorted(worker_results.items()):
            print(f"{nw:>8} | {tp:>12.1f} s/s | {tp/base:>7.2f}x")

        print("\n[prefetch_factor 별 처리량]")
        print(f"{'Prefetch':>8} | {'Throughput':>15}")
        print("-" * 30)
        for pf, tp in sorted(prefetch_results.items()):
            print(f"{pf:>8} | {tp:>12.1f} s/s")
```

---

## 9.2.5 실전 설정 가이드

### 하드웨어별 권장 설정

| 환경 | num_workers | prefetch_factor | pin_memory | 비고 |
|------|------------|----------------|------------|------|
| 단일 GPU + SSD | 4-8 | 2 | True | 가장 일반적 |
| 단일 GPU + HDD | 8-16 | 4 | True | I/O 보상 필요 |
| 멀티 GPU (DDP) | 4 per GPU | 2 | True | 총 워커 = 4 × GPU수 |
| 클라우드 NFS | 2-4 | 2 | True | 네트워크 지연 고려 |
| WebDataset (S3) | 4 | 2 | True | 순차읽기 최적화됨 |

### 데이터 포맷별 I/O 성능 비교

| 포맷 | Random Read | Sequential Read | 메모리 사용 | 적합 케이스 |
|------|------------|----------------|------------|------------|
| 개별 이미지 파일 | 느림 (seek) | 보통 | 낮음 | 소규모, 디버깅 |
| Memory-mapped | 빠름 | 빠름 | OS 관리 | 중규모, 반복 접근 |
| WebDataset (tar) | 불가 | 매우 빠름 | 낮음 | 대규모, 스트리밍 |
| Arrow | 빠름 | 빠름 | 중간 | HuggingFace 생태계 |
| LMDB | 매우 빠름 | 빠름 | 높음 | 랜덤 접근 빈번 |

### 주의사항

```python
# 안 좋은 패턴 1: 워커에서 전역 상태 공유
# → 각 워커는 독립 프로세스. 전역 변수 변경은 메인에 반영 안 됨.

# 안 좋은 패턴 2: __getitem__에서 무거운 초기화
class BadDataset(Dataset):
    def __getitem__(self, idx):
        model = load_heavy_model()  # 매 호출마다 로딩 → 극악 성능
        return model.process(self.data[idx])

# 좋은 패턴: worker_init_fn으로 워커별 한 번만 초기화
def worker_init_fn(worker_id):
    """각 워커에서 한 번만 실행"""
    np.random.seed(worker_id)
    # 필요한 리소스를 여기서 한 번만 로딩

loader = DataLoader(
    dataset,
    num_workers=4,
    worker_init_fn=worker_init_fn,
)
```

```python
# 안 좋은 패턴 3: persistent_workers 없이 매 에폭 워커 재생성
DataLoader(dataset, num_workers=8, persistent_workers=False)
# → 에폭마다 8개 프로세스 생성/종료. 수십 초 낭비.

# 좋은 패턴: persistent_workers=True
DataLoader(dataset, num_workers=8, persistent_workers=True)
# → 워커가 에폭 간 유지됨. 초기화 1회.
```

---

## 9.2.6 종합 사용 예시

```python
def full_pipeline_example():
    """전체 데이터 로딩 파이프라인 예시"""

    # ── 1. 기본 DataLoader ──
    dataset = VLMDataset(
        data_path="data/train.jsonl",
        image_dir="data/images",
        image_size=448,
        lazy_image=True,  # 메모리 절약
    )

    loader = OptimizedDataLoaderFactory.create(
        dataset=dataset,
        batch_size=4,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    # ── 2. 학습 루프 ──
    for epoch in range(3):
        for batch in loader:
            images = batch["images"]    # (B, 3, 448, 448)
            convs = batch["conversations"]
            # ... 모델 forward/backward ...
            pass

    # ── 3. Memory-mapped 데이터셋 (대규모) ──
    # 먼저 mmap 파일 생성 (1회)
    # MemoryMappedImageDataset.build_mmap_file(
    #     image_paths=["img1.jpg", "img2.jpg", ...],
    #     output_path="data/images.mmap",
    # )

    # mmap_dataset = MemoryMappedImageDataset(
    #     data_file="data/images.mmap",
    #     metadata_file="data/metadata.json",
    # )

    # ── 4. WebDataset (초대규모, S3 스트리밍) ──
    # wds_loader = WebDatasetBuilder.create_dataloader(
    #     tar_pattern="s3://bucket/train-{00000..00099}.tar",
    #     batch_size=8,
    #     shuffle_buffer=5000,
    # )

    # ── 5. 벤치마크 ──
    bench = DataLoaderBenchmark(dataset, num_batches=50)
    worker_results = bench.benchmark_workers(batch_size=4)
    prefetch_results = bench.benchmark_prefetch(batch_size=4, num_workers=4)
    bench.print_report(worker_results, prefetch_results)


if __name__ == "__main__":
    import time
    full_pipeline_example()
```

---

## 용어 체크리스트

학습 후 아래 질문에 답할 수 있는지 스스로 점검해라.

| # | 체크 항목 | 핵심 키워드 |
|---|----------|------------|
| 1 | I/O 병목과 Compute 병목을 어떻게 구분하는가? | 로딩 시간 vs 연산 시간, GPU 활용률 |
| 2 | Prefetch가 왜 최대 2배 속도 향상인가? | 파이프라이닝, $\max(T_{\text{load}}, T_{\text{compute}})$ |
| 3 | num_workers를 무한히 늘리면 안 되는 이유는? | 디스크 대역폭 상한, 프로세스 오버헤드 |
| 4 | pin_memory가 GPU 전송을 빠르게 하는 원리는? | Page-locked 메모리, DMA 직접 전송 |
| 5 | Memory Mapping의 장점과 한계는? | OS 페이지 캐시, 랜덤 접근 효율, 파일 크기 제한 |
| 6 | WebDataset이 HDD/NFS에서 왜 빠른가? | 순차 읽기, tar 아카이브, seek 최소화 |
| 7 | Arrow 포맷의 zero-copy 읽기란? | 메모리 매핑, 직렬화 없이 직접 참조 |
| 8 | Lazy Loading과 Eager Loading의 트레이드오프는? | 초기 시간 vs 접근 시간, 메모리 사용량 |
| 9 | persistent_workers가 왜 중요한가? | 워커 재생성 비용, 에폭 간 상태 유지 |
| 10 | 데이터 규모별 최적 포맷(개별파일/mmap/WebDataset/Arrow)을 선택할 수 있는가? | 규모, 접근 패턴, 인프라 |
