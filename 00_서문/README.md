# VLM Fine-tuning & Deployment 완벽 가이드

## 서문

본 교재는 Vision-Language Model(VLM)의 Fine-tuning과 프로덕션 배포에 대한 체계적인 학습 로드맵을 제공한다. 딥러닝의 수학적 기초부터 Transformer 아키텍처, Vision Transformer, 멀티모달 모델의 구조와 학습, 그리고 실제 프로덕션 환경에서의 배포까지 전 과정을 다룬다.

---

## 학습 대상

- **머신러닝 엔지니어**: VLM 파인튜닝 및 배포 실무 역량 강화
- **연구원**: VLM 아키텍처 심층 이해 및 연구 기반 구축
- **MLOps 엔지니어**: 대규모 모델의 효율적 서빙 및 운영

---

## 사전 요구사항

- Python 프로그래밍 중급 이상
- PyTorch 기본 사용 경험
- 선형대수, 미적분 기초 지식
- Linux 명령어 및 Docker 기본 이해

---

## 교재 구성

| Phase | 주제 | 학습 기간 |
|-------|------|----------|
| 1 | 딥러닝 & Transformer 기초 | 3-4주 |
| 2 | Vision & VLM 아키텍처 | 4-5주 |
| 3 | 데이터 파이프라인 | 3-4주 |
| 4 | Fine-tuning 실전 | 4-5주 |
| 5 | 평가 & 최적화 | 2-3주 |
| 6 | 프로덕션 배포 | 2-4주 |

**총 학습 기간**: 약 18-25주 (4-6개월)

---

## 핵심 참고 논문

### Transformer & Language Models
- Vaswani et al. (2017). **"Attention Is All You Need"** - [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Devlin et al. (2018). **"BERT: Pre-training of Deep Bidirectional Transformers"** - [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Brown et al. (2020). **"Language Models are Few-Shot Learners"** (GPT-3) - [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

### Vision Transformer
- Dosovitskiy et al. (2020). **"An Image is Worth 16x16 Words"** (ViT) - [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Liu et al. (2021). **"Swin Transformer"** - [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
- Radford et al. (2021). **"Learning Transferable Visual Models"** (CLIP) - [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

### Vision-Language Models
- Alayrac et al. (2022). **"Flamingo: a Visual Language Model"** - [arXiv:2204.14198](https://arxiv.org/abs/2204.14198)
- Liu et al. (2023). **"Visual Instruction Tuning"** (LLaVA) - [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)
- Wang et al. (2024). **"Qwen2-VL"** - [arXiv:2409.12191](https://arxiv.org/abs/2409.12191)

### Fine-tuning & Optimization
- Hu et al. (2021). **"LoRA: Low-Rank Adaptation"** - [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Dettmers et al. (2023). **"QLoRA: Efficient Finetuning"** - [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Dao et al. (2022). **"FlashAttention"** - [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

### Inference & Deployment
- Kwon et al. (2023). **"vLLM: PagedAttention"** - [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Zheng et al. (2024). **"SGLang: Efficient Execution"** - [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)

---

## 강의 및 튜토리얼 자료

### 대학 강의
- **Stanford CS224N**: Natural Language Processing with Deep Learning
  - 공식 사이트: https://web.stanford.edu/class/cs224n/
  - Transformer 강의 슬라이드 및 노트 제공

- **Stanford CS231n**: Deep Learning for Computer Vision
  - Backpropagation 설명: https://cs231n.github.io/optimization-2/

### 실습 튜토리얼
- **Andrej Karpathy의 "Neural Networks: Zero to Hero"**
  - "Let's build GPT: from scratch, in code, spelled out"
  - 공식 사이트: https://karpathy.ai/zero-to-hero.html
  - GitHub: https://github.com/karpathy/nanoGPT

- **Jay Alammar의 "The Illustrated Transformer"**
  - Transformer 아키텍처 시각화 설명

---

## 개발 환경

### 권장 하드웨어
- **GPU**: NVIDIA RTX 4090 / A100 / H100
- **VRAM**: 최소 24GB (풀 파인튜닝 시 48GB+ 권장)
- **RAM**: 64GB 이상
- **Storage**: NVMe SSD 1TB 이상

### 소프트웨어 스택
```
Python 3.10+
PyTorch 2.0+
Transformers 4.35+
PEFT (LoRA/QLoRA)
DeepSpeed / FSDP
vLLM / SGLang
Docker / Kubernetes
```

---

## 실습 코드 저장소

각 챕터별 실습 코드는 해당 폴더의 `code/` 디렉토리에 제공된다.

```
vlm_book/
├── 01_딥러닝_Transformer_기초/
│   └── code/
│       ├── attention_from_scratch.py
│       └── mini_transformer.py
├── 02_Vision_VLM_아키텍처/
│   └── code/
│       ├── vit_implementation.py
│       └── vlm_architecture.py
...
```

---

## 업데이트 내역

- **2024.02**: 초판 작성
- Qwen2.5-VL, InternVL2.5 등 최신 모델 반영
- vLLM, SGLang 최신 버전 반영

---

## 피드백

오류 수정, 내용 추가 제안은 이슈를 통해 접수 가능하다.
