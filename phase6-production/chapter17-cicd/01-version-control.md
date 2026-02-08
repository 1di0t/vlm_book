---
---

# 17.1 모델 버전 관리

ML 모델은 소프트웨어 코드와 달리 데이터, 하이퍼파라미터, 학습 환경까지 함께 버전 관리해야 한다. 모델 아티팩트만 저장하면 재현이 불가능하다. 여기서는 프로덕션 수준의 모델 버전 관리 체계를 다룬다.

> **핵심 용어**
>
> | 용어 | 정의 |
> |------|------|
> | **Model Registry** | 학습된 모델 아티팩트, 메타데이터, 버전 정보를 **중앙 집중형**으로 관리하는 저장소. 모델의 라이프사이클(Staging → Production → Archived) 관리를 담당한다. |
> | **DVC** (Data Version Control) | Git과 연동되는 **데이터 및 모델 버전 관리** 도구. 대용량 바이너리 파일을 Git 외부 스토리지에 저장하고, 메타데이터(.dvc 파일)만 Git에 커밋한다. |
> | **MLflow** | 실험 추적, 모델 등록, 배포를 통합 관리하는 **오픈소스 ML 플랫폼**. Tracking, Models, Registry, Projects 4개 컴포넌트로 구성된다. |
> | **Artifact Storage** | 모델 가중치, 토크나이저, 전처리 파이프라인 등 학습 산출물을 저장하는 **영구 스토리지**. S3, GCS, Azure Blob 등을 백엔드로 사용한다. |
> | **Experiment Tracking** | 각 학습 실행(run)의 하이퍼파라미터, 메트릭, 코드 커밋, 데이터 버전을 **자동 기록**하여 실험 간 비교와 재현을 가능하게 하는 시스템. |
> | **Lineage Tracking** | 모델이 어떤 데이터, 코드, 파이프라인을 거쳐 생성되었는지 **추적 가능한 계보**를 기록하는 것. 감사(Audit)와 디버깅에 필수다. |
> | **Model Card** | 모델의 용도, 성능, 한계, 편향성 등을 **표준 양식**으로 문서화한 메타데이터. Google이 제안한 개념이다. |

---

## 모델 메타데이터 관리

모델 버전을 관리할 때 가중치 파일만 저장하면 안 된다. 재현성과 추적성을 위해 다음 메타데이터를 반드시 함께 기록해야 한다.

### 필수 메타데이터 항목

| 카테고리 | 항목 | 설명 |
|----------|------|------|
| **학습 Config** | learning_rate, batch_size, epochs, optimizer | 동일한 결과를 재현하기 위한 하이퍼파라미터 |
| **메트릭** | train_loss, val_loss, accuracy, CER, WER | 모델 성능의 정량적 기록 |
| **데이터 버전** | dataset_hash, data_version, split_ratio | 어떤 데이터로 학습했는지 추적 |
| **코드 커밋** | git_commit_hash, branch, tag | 어떤 코드 버전으로 학습했는지 추적 |
| **환경 정보** | python_version, cuda_version, torch_version | 라이브러리 버전 불일치로 인한 재현 실패 방지 |
| **모델 구조** | model_architecture, num_params, config_json | 모델 구조 변경 이력 추적 |
| **학습 인프라** | gpu_type, num_gpus, training_time | 리소스 비용 추적 및 최적화 |

### 메타데이터 스키마 정의

```python
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import hashlib
import json
import subprocess


@dataclass
class ModelMetadata:
    """모델 버전에 필수로 기록해야 하는 메타데이터 스키마."""

    # 모델 식별
    model_name: str
    model_version: str
    model_architecture: str

    # 학습 설정
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str
    scheduler: Optional[str] = None
    weight_decay: float = 0.0

    # 성능 메트릭
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    val_cer: Optional[float] = None
    val_wer: Optional[float] = None

    # 데이터 버전
    dataset_name: str = ""
    dataset_version: str = ""
    dataset_hash: str = ""
    train_samples: int = 0
    val_samples: int = 0

    # 코드 버전
    git_commit: str = ""
    git_branch: str = ""
    git_tag: str = ""

    # 환경 정보
    python_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    gpu_type: str = ""
    num_gpus: int = 1

    # 시간 정보
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    training_duration_seconds: float = 0.0

    # 라이프사이클
    stage: str = "development"  # development → staging → production → archived

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "ModelMetadata":
        return cls(**json.loads(json_str))

    def compute_fingerprint(self) -> str:
        """메타데이터 기반 고유 핑거프린트 생성."""
        key_fields = f"{self.model_architecture}_{self.dataset_hash}_{self.git_commit}"
        return hashlib.sha256(key_fields.encode()).hexdigest()[:12]


def get_git_info() -> dict:
    """현재 Git 정보를 자동 수집한다."""
    def run_git(cmd: str) -> str:
        try:
            result = subprocess.run(
                ["git"] + cmd.split(),
                capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip()
        except Exception:
            return ""

    return {
        "git_commit": run_git("rev-parse HEAD"),
        "git_branch": run_git("rev-parse --abbrev-ref HEAD"),
        "git_tag": run_git("describe --tags --always"),
    }
```

---

## 버전 명명 규칙: Semantic Versioning for ML

소프트웨어의 SemVer(`MAJOR.MINOR.PATCH`)를 ML 모델에 맞게 확장한다.

### ML Semantic Versioning 규칙

$$
\text{Version} = \texttt{MAJOR.MINOR.PATCH[-STAGE][+BUILD]}
$$

| 구성 요소 | 변경 시점 | 예시 |
|-----------|----------|------|
| **MAJOR** | 모델 아키텍처 변경, 입출력 스키마 변경 (하위 호환 깨짐) | `2.0.0` — ViT → Swin Transformer로 교체 |
| **MINOR** | 학습 데이터 추가, 하이퍼파라미터 변경 (하위 호환 유지) | `1.3.0` — 학습 데이터 50만 → 80만 확대 |
| **PATCH** | 버그 수정, 미세 성능 개선 | `1.3.1` — 전처리 정규화 버그 수정 |
| **STAGE** | 라이프사이클 단계 | `1.3.0-rc1` — Release Candidate 1 |
| **BUILD** | 빌드/실험 식별자 | `1.3.0+exp42` — 42번 실험 |

### 버전 비교 로직

```python
import re
from functools import total_ordering


@total_ordering
class MLModelVersion:
    """ML 모델 전용 Semantic Version 파서 및 비교기."""

    _PATTERN = re.compile(
        r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
        r"(?:-(?P<stage>[a-zA-Z0-9.]+))?"
        r"(?:\+(?P<build>[a-zA-Z0-9.]+))?$"
    )

    STAGE_ORDER = {"dev": 0, "alpha": 1, "beta": 2, "rc": 3}

    def __init__(self, version_str: str):
        match = self._PATTERN.match(version_str)
        if not match:
            raise ValueError(f"잘못된 버전 형식: {version_str}")

        self.major = int(match.group("major"))
        self.minor = int(match.group("minor"))
        self.patch = int(match.group("patch"))
        self.stage = match.group("stage")
        self.build = match.group("build")

    def __repr__(self) -> str:
        v = f"{self.major}.{self.minor}.{self.patch}"
        if self.stage:
            v += f"-{self.stage}"
        if self.build:
            v += f"+{self.build}"
        return v

    def __eq__(self, other) -> bool:
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __lt__(self, other) -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def bump_major(self) -> "MLModelVersion":
        return MLModelVersion(f"{self.major + 1}.0.0")

    def bump_minor(self) -> "MLModelVersion":
        return MLModelVersion(f"{self.major}.{self.minor + 1}.0")

    def bump_patch(self) -> "MLModelVersion":
        return MLModelVersion(f"{self.major}.{self.minor}.{self.patch + 1}")

    def is_compatible(self, other: "MLModelVersion") -> bool:
        """MAJOR가 같으면 하위 호환성 유지."""
        return self.major == other.major


# 사용 예시
v1 = MLModelVersion("1.2.3-rc1+exp42")
v2 = MLModelVersion("1.3.0")
print(f"{v1} < {v2}: {v1 < v2}")       # True
print(f"호환: {v1.is_compatible(v2)}")   # True (같은 MAJOR)
```

---

## 도구 비교: MLflow vs DVC vs W&B vs HuggingFace Hub

| 기능 | **MLflow** | **DVC** | **W&B Artifacts** | **HuggingFace Hub** |
|------|-----------|---------|-------------------|---------------------|
| **실험 추적** | O (Tracking Server) | △ (외부 도구 필요) | O (대시보드 우수) | X |
| **모델 레지스트리** | O (내장) | △ (GTO 플러그인) | O (내장) | O (Model Card) |
| **데이터 버전 관리** | △ (제한적) | **O** (핵심 기능) | O | △ (datasets 라이브러리) |
| **파이프라인 정의** | O (Projects) | **O** (dvc.yaml) | X | X |
| **스토리지 백엔드** | S3, GCS, HDFS, Local | S3, GCS, Azure, SSH | W&B Cloud | HF Cloud, S3 |
| **셀프 호스팅** | **O** | **O** | 유료 | △ (Enterprise) |
| **Git 통합** | △ | **O** (Git 네이티브) | △ | **O** (git-lfs) |
| **학습 곡선** | 중간 | 낮음 | 낮음 | 매우 낮음 |
| **비용** | 무료 (OSS) | 무료 (OSS) | 무료 티어 + 유료 | 무료 티어 + 유료 |
| **OCR 모델 적합도** | 높음 | 높음 | 중간 | 높음 |

### 선택 기준

- **실험 관리 중심** → MLflow 또는 W&B
- **데이터 파이프라인 중심** → DVC
- **커뮤니티 공유 중심** → HuggingFace Hub
- **엔터프라이즈 올인원** → MLflow + DVC 조합

---

## 코드: MLflow 모델 등록/로딩

### MLflow Tracking Server 설정

```python
import os
import mlflow
from mlflow.tracking import MlflowClient
import torch
import logging

logger = logging.getLogger(__name__)

# ── MLflow 설정 ──────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class MLflowModelManager:
    """MLflow 기반 모델 버전 관리 매니저.

    주요 기능:
    - 실험 생성/조회
    - 모델 학습 실행 기록 (파라미터, 메트릭, 아티팩트)
    - 모델 레지스트리 등록/로딩
    - 모델 라이프사이클 전환 (Staging → Production)
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        self._ensure_experiment()

    def _ensure_experiment(self):
        """실험이 없으면 생성한다."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                self.experiment_name,
                artifact_location=f"s3://mlflow-artifacts/{self.experiment_name}"
            )
            logger.info(f"실험 생성: {self.experiment_name}")
        mlflow.set_experiment(self.experiment_name)

    def log_training_run(
        self,
        model: torch.nn.Module,
        params: dict,
        metrics: dict,
        metadata: "ModelMetadata",
        model_name: str,
        artifacts: dict[str, str] | None = None,
    ) -> str:
        """학습 실행을 MLflow에 기록하고 모델을 등록한다.

        Args:
            model: PyTorch 모델
            params: 하이퍼파라미터 dict
            metrics: 평가 메트릭 dict
            metadata: ModelMetadata 인스턴스
            model_name: 레지스트리에 등록할 모델 이름
            artifacts: 추가 아티팩트 {이름: 파일경로}

        Returns:
            run_id: MLflow Run ID
        """
        with mlflow.start_run(run_name=f"{model_name}-{metadata.model_version}") as run:
            # 하이퍼파라미터 기록
            mlflow.log_params(params)

            # 메트릭 기록
            mlflow.log_metrics(metrics)

            # 커스텀 태그 기록
            mlflow.set_tags({
                "model_version": metadata.model_version,
                "git_commit": metadata.git_commit,
                "dataset_version": metadata.dataset_version,
                "model_architecture": metadata.model_architecture,
                "stage": metadata.stage,
            })

            # 메타데이터 JSON 저장
            metadata_path = "/tmp/model_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write(metadata.to_json())
            mlflow.log_artifact(metadata_path, artifact_path="metadata")

            # 추가 아티팩트 저장
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, artifact_path=name)

            # PyTorch 모델 저장 + 레지스트리 등록
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=model_name,
            )

            logger.info(f"학습 기록 완료: run_id={run.info.run_id}")
            return run.info.run_id

    def promote_model(self, model_name: str, version: int, stage: str):
        """모델 버전의 라이프사이클 스테이지를 전환한다.

        Args:
            model_name: 등록된 모델 이름
            version: 모델 버전 번호
            stage: "Staging" | "Production" | "Archived"
        """
        valid_stages = {"Staging", "Production", "Archived", "None"}
        if stage not in valid_stages:
            raise ValueError(f"잘못된 stage: {stage}. 허용: {valid_stages}")

        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=(stage == "Production"),
        )
        logger.info(f"모델 전환: {model_name} v{version} → {stage}")

    def load_production_model(self, model_name: str) -> torch.nn.Module:
        """Production 스테이지의 최신 모델을 로드한다."""
        model_uri = f"models:/{model_name}/Production"
        try:
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Production 모델 로드 성공: {model_name}")
            return model
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise

    def load_model_by_version(self, model_name: str, version: int) -> torch.nn.Module:
        """특정 버전의 모델을 로드한다."""
        model_uri = f"models:/{model_name}/{version}"
        return mlflow.pytorch.load_model(model_uri)

    def compare_runs(self, run_ids: list[str]) -> list[dict]:
        """여러 실행의 메트릭을 비교한다."""
        results = []
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            results.append({
                "run_id": run_id,
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags,
            })
        return results

    def get_latest_versions(self, model_name: str) -> list[dict]:
        """모델의 각 스테이지별 최신 버전 조회."""
        versions = self.client.get_latest_versions(model_name)
        return [
            {
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "status": v.status,
            }
            for v in versions
        ]
```

### 사용 예시: 학습 후 등록

```python
# ── 실제 학습 워크플로우 예시 ──────────────────────────────────
import torch
import torch.nn as nn

# 1. 모델 정의 (예시: 간단한 OCR 모델)
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(32, 100),
)

# 2. 메타데이터 준비
metadata = ModelMetadata(
    model_name="ocr-receipt-v1",
    model_version="1.2.0",
    model_architecture="CNN-OCR",
    learning_rate=3e-4,
    batch_size=32,
    epochs=50,
    optimizer="AdamW",
    dataset_name="receipt-kr-v3",
    dataset_version="3.1.0",
    dataset_hash="a1b2c3d4e5f6",
    train_samples=150000,
    val_samples=15000,
    val_accuracy=0.957,
    val_cer=0.032,
    **get_git_info(),
)

# 3. MLflow에 기록
manager = MLflowModelManager(experiment_name="ocr-receipt")
run_id = manager.log_training_run(
    model=model,
    params={"lr": 3e-4, "batch_size": 32, "epochs": 50, "optimizer": "AdamW"},
    metrics={"val_accuracy": 0.957, "val_cer": 0.032, "val_wer": 0.078},
    metadata=metadata,
    model_name="ocr-receipt",
)

# 4. 검증 후 Production 승격
manager.promote_model("ocr-receipt", version=3, stage="Production")

# 5. Production 모델 로드
prod_model = manager.load_production_model("ocr-receipt")
```

---

## 코드: DVC 파이프라인 설정

### DVC 초기화 및 데이터 추적

```bash
# DVC 초기화 (Git repo 내에서 실행)
dvc init

# 리모트 스토리지 설정 (S3)
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc remote modify myremote endpointurl $AWS_ENDPOINT_URL

# 대용량 데이터 추적
dvc add data/training/receipt-images/
dvc add models/checkpoints/ocr-receipt-v1.2.0.pt

# Git에 .dvc 파일만 커밋
git add data/training/receipt-images.dvc models/checkpoints/ocr-receipt-v1.2.0.pt.dvc .gitignore
git commit -m "feat: add receipt training data v3.1.0 and model v1.2.0"

# DVC push (실제 파일을 리모트 스토리지에 업로드)
dvc push
```

### DVC 파이프라인 (dvc.yaml)

```yaml
# dvc.yaml — ML 파이프라인 정의
stages:
  preprocess:
    cmd: python src/preprocess.py --input data/raw/ --output data/processed/
    deps:
      - src/preprocess.py
      - data/raw/
    outs:
      - data/processed/
    params:
      - preprocess.image_size
      - preprocess.normalize

  train:
    cmd: python src/train.py --config configs/train.yaml
    deps:
      - src/train.py
      - data/processed/
      - configs/train.yaml
    params:
      - train.learning_rate
      - train.batch_size
      - train.epochs
    outs:
      - models/checkpoints/:
          persist: true
    metrics:
      - metrics/train_metrics.json:
          cache: false
    plots:
      - plots/loss_curve.csv:
          x: epoch
          y: loss

  evaluate:
    cmd: python src/evaluate.py --model models/checkpoints/best.pt --data data/test/
    deps:
      - src/evaluate.py
      - models/checkpoints/best.pt
      - data/test/
    metrics:
      - metrics/eval_metrics.json:
          cache: false
    plots:
      - plots/confusion_matrix.csv:
          x: predicted
          y: actual
```

### DVC 파라미터 파일 (params.yaml)

```yaml
# params.yaml — 하이퍼파라미터 중앙 관리
preprocess:
  image_size: [224, 224]
  normalize: true
  augmentation:
    random_rotation: 5
    random_crop: 0.9

train:
  learning_rate: 3.0e-4
  batch_size: 32
  epochs: 50
  optimizer: AdamW
  scheduler: CosineAnnealingLR
  weight_decay: 0.01

evaluate:
  threshold: 0.5
  beam_width: 5
```

### DVC 파이프라인 실행 및 실험 추적

```bash
# 파이프라인 전체 실행 (의존성 변경된 스테이지만 재실행)
dvc repro

# 실험 실행 (파라미터 오버라이드)
dvc exp run --set-param train.learning_rate=1e-4
dvc exp run --set-param train.batch_size=64

# 실험 비교
dvc exp show --sort-by metrics/eval_metrics.json:accuracy

# 최적 실험을 브랜치로 승격
dvc exp apply exp-abc123
git add . && git commit -m "feat: apply best experiment (lr=1e-4, acc=0.962)"

# 특정 버전의 데이터/모델 체크아웃
git checkout v1.2.0
dvc checkout
```

### DVC + Python API 연동

```python
import json
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DVCPipelineManager:
    """DVC 파이프라인을 Python에서 관리하는 래퍼 클래스."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        if not (self.repo_path / ".dvc").exists():
            raise FileNotFoundError(f"DVC repo가 아님: {repo_path}")

    def _run_cmd(self, cmd: list[str]) -> str:
        """DVC CLI 명령을 실행한다."""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                logger.error(f"DVC 명령 실패: {result.stderr}")
                raise RuntimeError(result.stderr)
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.error("DVC 명령 타임아웃 (600초)")
            raise

    def repro(self, stage: str | None = None) -> str:
        """파이프라인을 재실행한다."""
        cmd = ["dvc", "repro"]
        if stage:
            cmd.append(stage)
        return self._run_cmd(cmd)

    def run_experiment(self, param_overrides: dict) -> str:
        """파라미터를 오버라이드하여 실험을 실행한다."""
        cmd = ["dvc", "exp", "run"]
        for key, value in param_overrides.items():
            cmd.extend(["--set-param", f"{key}={value}"])
        return self._run_cmd(cmd)

    def get_metrics(self) -> dict:
        """현재 메트릭을 조회한다."""
        output = self._run_cmd(["dvc", "metrics", "show", "--json"])
        return json.loads(output)

    def push(self) -> str:
        """아티팩트를 리모트 스토리지에 업로드한다."""
        return self._run_cmd(["dvc", "push"])

    def pull(self) -> str:
        """리모트 스토리지에서 아티팩트를 다운로드한다."""
        return self._run_cmd(["dvc", "pull"])

    def list_experiments(self) -> str:
        """실험 목록을 조회한다."""
        return self._run_cmd(["dvc", "exp", "show", "--json"])


# ── 사용 예시 ──────────────────────────────────────────────────
# dvc_mgr = DVCPipelineManager("/path/to/repo")
# dvc_mgr.run_experiment({"train.learning_rate": 1e-4, "train.batch_size": 64})
# metrics = dvc_mgr.get_metrics()
# print(f"Accuracy: {metrics['metrics/eval_metrics.json']['accuracy']}")
```

---

## 전체 아키텍처: 모델 버전 관리 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                    Model Version Control Pipeline                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   [코드]──Git──→ GitHub/GitLab                                   │
│     │                                                            │
│   [데이터]──DVC──→ S3/GCS (Artifact Storage)                     │
│     │                                                            │
│   [학습]──MLflow──→ Tracking Server                              │
│     │                 ├── Parameters                             │
│     │                 ├── Metrics                                │
│     │                 └── Artifacts                              │
│     │                                                            │
│   [모델]──Registry──→ Model Registry                             │
│     │                 ├── v1.0.0 (Archived)                     │
│     │                 ├── v1.1.0 (Staging)                      │
│     │                 └── v1.2.0 (Production) ←── 현재 서빙     │
│     │                                                            │
│   [배포]──CI/CD──→ Kubernetes / Docker                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 용어 체크리스트

학습을 마친 뒤 아래 항목을 스스로 점검해봐라.

- [ ] **Model Registry**가 왜 필요한지, 파일 시스템에 직접 저장하는 것과 차이를 설명할 수 있는가?
- [ ] **DVC**가 Git과 어떻게 연동되는지, `.dvc` 파일의 역할을 이해했는가?
- [ ] **MLflow**의 4개 컴포넌트(Tracking, Models, Registry, Projects)를 구분할 수 있는가?
- [ ] **Artifact Storage**로 S3/GCS를 사용하는 이유와 설정 방법을 알고 있는가?
- [ ] **Experiment Tracking**에서 기록해야 할 필수 메타데이터 7가지를 나열할 수 있는가?
- [ ] **Semantic Versioning for ML**에서 MAJOR/MINOR/PATCH 변경 기준을 설명할 수 있는가?
- [ ] **Lineage Tracking**의 목적과 감사(Audit) 시나리오를 이해했는가?
- [ ] MLflow와 DVC를 **조합**해서 사용하는 아키텍처를 설계할 수 있는가?
- [ ] `dvc repro`와 `dvc exp run`의 차이를 설명할 수 있는가?
- [ ] 모델 라이프사이클(Development → Staging → Production → Archived) 전환 기준을 정의할 수 있는가?
