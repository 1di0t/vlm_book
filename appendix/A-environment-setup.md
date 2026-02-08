---
---

# Appendix A: 개발 환경 설정

OCR/VLM 모델을 학습하고 추론하기 위한 전체 개발 환경 설정 가이드다.
Docker 기반 재현 가능한 환경부터, 로컬 conda 환경, GPU 드라이버 호환성까지 전부 다룬다.

---

## A.1 사전 요구사항

시작하기 전에 아래 항목이 호스트 머신에 설치되어 있어야 한다.

| 항목 | 최소 버전 | 권장 버전 | 확인 명령어 |
|------|-----------|-----------|-------------|
| NVIDIA Driver | 525.60+ | 550.54+ | `nvidia-smi` |
| CUDA Toolkit | 11.8 | 12.4 | `nvcc --version` |
| Docker Engine | 24.0+ | 27.0+ | `docker --version` |
| NVIDIA Container Toolkit | 1.14+ | 1.16+ | `nvidia-ctk --version` |
| Python | 3.10+ | 3.11 | `python --version` |
| Git | 2.30+ | 2.40+ | `git --version` |

---

## A.2 NVIDIA 드라이버 + CUDA 설치

### A.2.1 GPU 드라이버 + CUDA 버전 호환성 표

CUDA Toolkit과 드라이버 간 호환성을 반드시 확인해야 한다. 잘못 맞추면 런타임에서 `CUDA error: no kernel image is available` 같은 에러가 터진다.

| CUDA Toolkit | 최소 드라이버 (Linux) | 최소 드라이버 (Windows) | 지원 GPU Compute Capability |
|-------------|----------------------|------------------------|---------------------------|
| CUDA 12.4 | 550.54.14 | 551.61 | 5.0+ (Maxwell~Hopper) |
| CUDA 12.2 | 535.86.10 | 536.25 | 5.0+ |
| CUDA 12.1 | 530.30.02 | 531.14 | 5.0+ |
| CUDA 11.8 | 520.61.05 | 520.06 | 3.5+ (Kepler~Hopper) |
| CUDA 11.7 | 515.43.04 | 516.01 | 3.5+ |

> **주의**: PyTorch 2.x는 기본적으로 CUDA 11.8 또는 12.1/12.4를 지원한다.
> bitsandbytes는 CUDA 11.8+ 필수, Flash Attention 2는 Ampere(SM80) 이상 GPU에서만 동작한다.

### A.2.2 Ubuntu에서 드라이버 설치

```bash
# 기존 드라이버 완전 제거
sudo apt-get purge -y 'nvidia-*' 'cuda-*' 'libnvidia-*'
sudo apt-get autoremove -y
sudo rm -rf /usr/local/cuda*

# NVIDIA 공식 저장소 등록
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 드라이버 + CUDA Toolkit 설치
sudo apt-get update
sudo apt-get install -y nvidia-driver-550 cuda-toolkit-12-4

# 환경 변수 설정
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 설치 확인
nvidia-smi
nvcc --version
```

---

## A.3 Docker + NVIDIA Container Toolkit 설정

Docker 기반 환경이 재현성 측면에서 가장 안정적이다. 팀원 간 "내 컴퓨터에서는 되는데" 문제를 원천 차단한다.

### A.3.1 Docker Engine 설치

```bash
# Docker 공식 설치 스크립트 (Ubuntu/Debian)
curl -fsSL https://get.docker.com | sudo sh

# 현재 사용자를 docker 그룹에 추가 (sudo 없이 docker 사용)
sudo usermod -aG docker $USER
newgrp docker

# 설치 확인
docker run hello-world
```

### A.3.2 NVIDIA Container Toolkit 설치

```bash
# NVIDIA Container Toolkit 설치
sudo apt-get install -y nvidia-container-toolkit

# Docker 데몬에 NVIDIA 런타임 등록
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# GPU 접근 확인
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

위 명령이 GPU 정보를 정상 출력하면 Docker에서 GPU 사용 준비가 된 거다.

### A.3.3 학습용 Dockerfile

```dockerfile
# === 학습용 Dockerfile ===
# 파일명: Dockerfile.train
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

LABEL maintainer="ocrbook-team"
LABEL description="OCR/VLM Training Environment"

# 비대화형 설정 + 타임존
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git git-lfs wget curl vim \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Python 심볼릭 링크
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch 설치 (CUDA 12.4)
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 핵심 학습 라이브러리
RUN pip install --no-cache-dir \
    transformers==4.46.0 \
    datasets==3.0.0 \
    accelerate==1.0.0 \
    peft==0.13.0 \
    bitsandbytes==0.44.0 \
    trl==0.11.0 \
    sentencepiece==0.2.0 \
    protobuf==5.28.0

# Flash Attention 2 (Ampere 이상 GPU 전용)
RUN pip install --no-cache-dir flash-attn==2.6.3 --no-build-isolation

# DeepSpeed
RUN pip install --no-cache-dir deepspeed==0.15.0

# 모니터링 + 유틸
RUN pip install --no-cache-dir \
    wandb==0.18.0 \
    tensorboard==2.18.0 \
    matplotlib==3.9.0 \
    seaborn==0.13.0 \
    pillow==10.4.0 \
    opencv-python-headless==4.10.0.84 \
    einops==0.8.0 \
    scipy==1.14.0

# 작업 디렉토리
WORKDIR /workspace

# Git LFS 초기화
RUN git lfs install

# 기본 환경 변수
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV HF_HOME=/workspace/.cache/huggingface
ENV WANDB_DIR=/workspace/.cache/wandb
ENV TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions
ENV CUDA_HOME=/usr/local/cuda

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

CMD ["/bin/bash"]
```

### A.3.4 추론용 Dockerfile

```dockerfile
# === 추론용 Dockerfile ===
# 파일명: Dockerfile.serve
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

LABEL maintainer="ocrbook-team"
LABEL description="OCR/VLM Inference Server"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git wget curl \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch (CUDA 12.4)
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124

# vLLM + 추론 라이브러리
RUN pip install --no-cache-dir \
    vllm==0.6.2 \
    transformers==4.46.0 \
    accelerate==1.0.0 \
    bitsandbytes==0.44.0 \
    sentencepiece==0.2.0 \
    protobuf==5.28.0 \
    pillow==10.4.0 \
    einops==0.8.0

# API 서빙
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.31.0 \
    python-multipart==0.0.12 \
    pydantic==2.9.0

WORKDIR /workspace

ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV HF_HOME=/workspace/.cache/huggingface
ENV CUDA_HOME=/usr/local/cuda

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/workspace/model", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
```

### A.3.5 docker-compose.yml (GPU 지원)

```yaml
# docker-compose.yml
version: "3.9"

services:
  # === 학습 서비스 ===
  train:
    build:
      context: .
      dockerfile: Dockerfile.train
    container_name: ocrbook-train
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all          # 모든 GPU 사용. 특정 GPU만 쓰려면 device_ids: ["0","1"]
              capabilities: [gpu]
    volumes:
      - ./workspace:/workspace    # 코드 + 데이터 마운트
      - ./cache:/workspace/.cache # 캐시 영속화 (모델 재다운로드 방지)
      - /dev/shm:/dev/shm         # 공유 메모리 (DataLoader num_workers > 0 필수)
    shm_size: "16gb"              # 공유 메모리 크기
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - WANDB_API_KEY=${WANDB_API_KEY}
      - HF_TOKEN=${HF_TOKEN}
      - NCCL_P2P_DISABLE=0
      - NCCL_IB_DISABLE=1
    ulimits:
      memlock:
        soft: -1
        hard: -1
    stdin_open: true
    tty: true

  # === 추론 서비스 ===
  serve:
    build:
      context: .
      dockerfile: Dockerfile.serve
    container_name: ocrbook-serve
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models:/workspace/model:ro  # 모델 가중치 (읽기 전용)
      - ./cache:/workspace/.cache
    ports:
      - "8000:8000"
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - HF_TOKEN=${HF_TOKEN}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5

  # === 모니터링 ===
  tensorboard:
    image: tensorflow/tensorflow:latest
    container_name: ocrbook-tensorboard
    ports:
      - "6006:6006"
    volumes:
      - ./workspace/logs:/logs:ro
    command: tensorboard --logdir=/logs --host=0.0.0.0 --port=6006
    restart: unless-stopped
```

사용 방법:

```bash
# .env 파일 생성
cat > .env << 'EOF'
WANDB_API_KEY=your_wandb_key_here
HF_TOKEN=your_huggingface_token_here
EOF

# 학습 컨테이너 빌드 + 실행
docker compose up -d train

# 학습 컨테이너 접속
docker compose exec train bash

# 추론 서버 실행
docker compose up -d serve

# 텐서보드 모니터링
docker compose up -d tensorboard
```

---

## A.4 conda 가상환경 설정

Docker 없이 로컬에서 직접 환경을 구축하는 경우다.

### A.4.1 Miniconda 설치

```bash
# Miniconda 설치
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc
```

### A.4.2 가상환경 생성 + PyTorch 설치

```bash
# 가상환경 생성
conda create -n ocrbook python=3.11 -y
conda activate ocrbook

# PyTorch + CUDA 12.4
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# GPU 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

### A.4.3 핵심 라이브러리 설치

```bash
# === HuggingFace 생태계 ===
pip install transformers==4.46.0     # 모델 로딩 + 토크나이저
pip install datasets==3.0.0          # 데이터셋 처리
pip install accelerate==1.0.0        # 분산 학습 + mixed precision
pip install peft==0.13.0             # LoRA, QLoRA 등 PEFT 기법
pip install trl==0.11.0              # RLHF, SFT Trainer

# === 양자화 ===
pip install bitsandbytes==0.44.0     # 4-bit/8-bit 양자화
pip install auto-gptq==0.7.1         # GPTQ 양자화
pip install autoawq==0.2.6           # AWQ 양자화

# === 추론 엔진 ===
pip install vllm==0.6.2              # 고성능 추론 서버

# === 분산 학습 ===
pip install deepspeed==0.15.0        # ZeRO 최적화

# === Flash Attention ===
pip install flash-attn==2.6.3 --no-build-isolation  # Ampere+ GPU 필수

# === 모니터링 ===
pip install wandb==0.18.0            # 실험 추적
pip install tensorboard==2.18.0      # 학습 시각화

# === 이미지 처리 (OCR/VLM 필수) ===
pip install pillow==10.4.0
pip install opencv-python-headless==4.10.0.84

# === 기타 유틸 ===
pip install einops==0.8.0            # 텐서 연산 표기
pip install sentencepiece==0.2.0     # 토크나이저 백엔드
pip install protobuf==5.28.0
pip install scipy==1.14.0
pip install matplotlib==3.9.0
pip install seaborn==0.13.0
```

또는 `requirements.txt`로 한 번에 설치:

```bash
pip install -r requirements.txt
```

```text
# requirements.txt
torch==2.4.0
torchvision==0.19.0
torchaudio==2.4.0
transformers==4.46.0
datasets==3.0.0
accelerate==1.0.0
peft==0.13.0
bitsandbytes==0.44.0
trl==0.11.0
auto-gptq==0.7.1
autoawq==0.2.6
vllm==0.6.2
deepspeed==0.15.0
flash-attn==2.6.3
wandb==0.18.0
tensorboard==2.18.0
pillow==10.4.0
opencv-python-headless==4.10.0.84
einops==0.8.0
sentencepiece==0.2.0
protobuf==5.28.0
scipy==1.14.0
matplotlib==3.9.0
seaborn==0.13.0
```

---

## A.5 환경 검증 스크립트

환경 설정 후 반드시 아래 스크립트로 전체 검증을 돌려야 한다. 하나라도 실패하면 학습/추론 중에 터진다.

```python
#!/usr/bin/env python3
"""
환경 검증 스크립트 (verify_env.py)

사용법:
    python verify_env.py
    python verify_env.py --verbose    # 상세 출력
"""

import sys
import platform
import subprocess
import importlib
from pathlib import Path


def check_python_version():
    """Python 버전 확인 (3.10+ 필수)"""
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    return ok, f"Python {v.major}.{v.minor}.{v.micro}"


def check_gpu():
    """NVIDIA GPU 인식 확인"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpus = result.stdout.strip().split("\n")
            return True, gpus
        return False, "nvidia-smi 실행 실패"
    except FileNotFoundError:
        return False, "nvidia-smi를 찾을 수 없음 (드라이버 미설치)"
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi 타임아웃"


def check_cuda_torch():
    """PyTorch CUDA 지원 확인"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            info = {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cudnn_version": str(torch.backends.cudnn.version()),
                "gpu_count": torch.cuda.device_count(),
                "gpu_names": [torch.cuda.get_device_name(i)
                              for i in range(torch.cuda.device_count())],
                "bf16_support": torch.cuda.is_bf16_supported(),
            }
            return True, info
        return False, {"torch_version": torch.__version__, "error": "CUDA 사용 불가"}
    except ImportError:
        return False, "PyTorch가 설치되지 않음"


def check_flash_attention():
    """Flash Attention 2 확인"""
    try:
        import flash_attn
        return True, f"flash-attn {flash_attn.__version__}"
    except ImportError:
        return False, "flash-attn 미설치 (Ampere+ GPU 필요)"


def check_libraries():
    """핵심 라이브러리 버전 확인"""
    libs = [
        "transformers", "datasets", "accelerate", "peft",
        "bitsandbytes", "trl", "deepspeed", "vllm",
        "wandb", "tensorboard", "PIL", "cv2", "einops",
        "sentencepiece", "scipy", "matplotlib",
    ]
    results = {}
    for lib in libs:
        try:
            mod = importlib.import_module(lib)
            version = getattr(mod, "__version__", "unknown")
            results[lib] = {"installed": True, "version": version}
        except ImportError:
            results[lib] = {"installed": False, "version": None}
    return results


def check_deepspeed_ops():
    """DeepSpeed CPU/CUDA ops 빌드 확인"""
    try:
        import deepspeed
        report = deepspeed.ops.op_builder.ALL_OPS
        return True, f"DeepSpeed {deepspeed.__version__} - {len(report)} ops 감지"
    except Exception as e:
        return False, str(e)


def check_bitsandbytes():
    """bitsandbytes CUDA 바인딩 확인"""
    try:
        import bitsandbytes as bnb
        # 간단한 양자화 테스트
        import torch
        linear = bnb.nn.Linear8bitLt(64, 64, has_fp16_weights=False)
        return True, f"bitsandbytes {bnb.__version__}"
    except Exception as e:
        return False, f"bitsandbytes 오류: {e}"


def run_gpu_stress_test():
    """간단한 GPU 연산 테스트"""
    try:
        import torch
        device = torch.device("cuda:0")

        # FP16 행렬곱
        a = torch.randn(1024, 1024, dtype=torch.float16, device=device)
        b = torch.randn(1024, 1024, dtype=torch.float16, device=device)
        c = torch.mm(a, b)

        # BF16 행렬곱 (Ampere+)
        if torch.cuda.is_bf16_supported():
            a_bf16 = a.to(torch.bfloat16)
            b_bf16 = b.to(torch.bfloat16)
            c_bf16 = torch.mm(a_bf16, b_bf16)

        torch.cuda.synchronize()
        return True, "FP16/BF16 행렬곱 정상"
    except Exception as e:
        return False, str(e)


def main():
    verbose = "--verbose" in sys.argv
    print("=" * 60)
    print("  OCR/VLM 개발 환경 검증")
    print("=" * 60)

    checks = []

    # 1. Python
    ok, info = check_python_version()
    checks.append(("Python Version", ok, info))

    # 2. OS
    checks.append(("OS", True, f"{platform.system()} {platform.release()}"))

    # 3. GPU
    ok, info = check_gpu()
    checks.append(("NVIDIA GPU", ok, info))

    # 4. PyTorch + CUDA
    ok, info = check_cuda_torch()
    checks.append(("PyTorch + CUDA", ok, info))

    # 5. Flash Attention
    ok, info = check_flash_attention()
    checks.append(("Flash Attention", ok, info))

    # 6. bitsandbytes
    ok, info = check_bitsandbytes()
    checks.append(("bitsandbytes", ok, info))

    # 7. GPU 연산 테스트
    ok, info = run_gpu_stress_test()
    checks.append(("GPU Stress Test", ok, info))

    # 결과 출력
    print()
    all_pass = True
    for name, ok, info in checks:
        status = "PASS" if ok else "FAIL"
        icon = "[+]" if ok else "[-]"
        print(f"  {icon} {name}: {status}")
        if verbose or not ok:
            if isinstance(info, dict):
                for k, v in info.items():
                    print(f"      {k}: {v}")
            elif isinstance(info, list):
                for item in info:
                    print(f"      {item}")
            else:
                print(f"      {info}")
        if not ok:
            all_pass = False

    # 라이브러리 체크
    print(f"\n{'─' * 60}")
    print("  라이브러리 설치 현황:")
    print(f"{'─' * 60}")
    lib_results = check_libraries()
    for lib, result in lib_results.items():
        if result["installed"]:
            print(f"  [+] {lib:20s} v{result['version']}")
        else:
            print(f"  [-] {lib:20s} 미설치")
            all_pass = False

    # 최종 결과
    print(f"\n{'=' * 60}")
    if all_pass:
        print("  전체 검증 통과. 학습/추론 환경 준비 완료.")
    else:
        print("  일부 항목 실패. 위 [-] 항목을 확인하고 재설치해라.")
    print(f"{'=' * 60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## A.6 Weights & Biases (wandb) 설정

```bash
# wandb 로그인
wandb login

# 또는 환경변수로 설정 (CI/CD에서 사용)
export WANDB_API_KEY="your-api-key"
export WANDB_PROJECT="ocrbook"
export WANDB_ENTITY="your-team"
```

학습 코드에서 사용:

```python
import wandb

wandb.init(
    project="ocrbook",
    name="qwen2-vl-7b-lora-ocr",
    config={
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "method": "LoRA",
        "lora_rank": 64,
        "learning_rate": 2e-4,
        "batch_size": 4,
        "epochs": 3,
    },
    tags=["ocr", "lora", "qwen2-vl"],
)
```

---

## A.7 HuggingFace Hub 설정

Gated 모델(Llama 3, Gemma 등)을 다운로드하려면 HuggingFace 토큰이 필요하다.

```bash
# CLI 로그인
huggingface-cli login

# 또는 환경 변수
export HF_TOKEN="hf_your_token_here"

# 모델 다운로드 캐시 디렉토리 (디스크 용량 주의)
export HF_HOME="/data/huggingface"
export TRANSFORMERS_CACHE="/data/huggingface/hub"
```

---

## A.8 DeepSpeed 설정

DeepSpeed ZeRO를 사용하려면 별도 설정 파일이 필요하다.

```json
// ds_config_zero2.json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none"
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

---

## A.9 일반적인 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `CUDA out of memory` | VRAM 부족 | batch size 줄이기, gradient checkpointing 활성화, QLoRA 사용 |
| `No kernel image available` | CUDA/드라이버 버전 불일치 | 호환성 표 참고하여 재설치 |
| `bitsandbytes` 로딩 실패 | CUDA 라이브러리 경로 문제 | `export LD_LIBRARY_PATH=/usr/local/cuda/lib64` |
| `flash-attn` 빌드 실패 | SM 버전 미지원 (Turing 이하) | `--no-build-isolation` 플래그, 또는 포기하고 SDPA 사용 |
| `NCCL error` (멀티 GPU) | GPU 간 통신 실패 | `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1` 설정 |
| `DataLoader worker killed` | 공유 메모리 부족 | `--shm-size 16g` (Docker), `num_workers` 줄이기 |
| `tokenizers parallelism` 경고 | Fork 안전성 | `TOKENIZERS_PARALLELISM=false` |
| `ImportError: libcudnn` | cuDNN 미설치 | `apt install libcudnn8 libcudnn8-dev` |

---

## A.10 환경 변수 요약 (`.env` 템플릿)

```bash
# .env
# === API Keys ===
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# === CUDA ===
CUDA_HOME=/usr/local/cuda-12.4
LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# === HuggingFace ===
HF_HOME=/data/huggingface
TRANSFORMERS_CACHE=/data/huggingface/hub
HF_DATASETS_CACHE=/data/huggingface/datasets

# === wandb ===
WANDB_PROJECT=ocrbook
WANDB_ENTITY=your-team

# === Performance ===
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8

# === 멀티 GPU (NCCL) ===
NCCL_P2P_DISABLE=0
NCCL_IB_DISABLE=1

# === vLLM ===
VLLM_WORKER_MULTIPROC_METHOD=spawn
```

> **보안 주의**: `.env` 파일은 반드시 `.gitignore`에 추가해라. 절대 Git에 커밋하면 안 된다.
