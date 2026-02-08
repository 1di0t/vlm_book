---
layout: default
title: VLM Book
render_with_liquid: true
---

# VLM Fine-tuning & Deployment 완벽 가이드

> 보험금 청구 의료 문서 OCR을 중심으로 한 Vision-Language Model 학습 교재
> 딥러닝 기초부터 프로덕션 배포까지 — 수학 공식 100+개, PyTorch 구현 50+개

---

## Phase 1: 딥러닝 & Transformer 기초

| Chapter | 주제 | 핵심 내용 |
|---------|------|-----------|
| [1.1 선형대수](phase1-foundations/chapter01-math/01-linear-algebra) | 수학적 기초 | 행렬 연산, SVD, Norm |
| [1.2 미적분](phase1-foundations/chapter01-math/02-calculus) | | Chain Rule, Jacobian, Softmax 미분 |
| [1.3 확률과 통계](phase1-foundations/chapter01-math/03-probability) | | Cross-Entropy, KL Divergence |
| [2.1 Self-Attention](phase1-foundations/chapter02-transformer/01-self-attention) | Transformer | Scaled Dot-Product, Multi-Head |
| [2.2 Positional Encoding](phase1-foundations/chapter02-transformer/02-positional-encoding) | | Sinusoidal, RoPE |
| [2.3 Transformer Block](phase1-foundations/chapter02-transformer/03-transformer-block) | | LayerNorm, FFN, SwiGLU |
| [2.4 Encoder-Decoder](phase1-foundations/chapter02-transformer/04-encoder-decoder) | | 전체 구조 |
| [3.1 Tokenization](phase1-foundations/chapter03-llm-basics/01-tokenization) | LLM 기초 | BPE, SentencePiece |
| [3.2 Autoregressive](phase1-foundations/chapter03-llm-basics/02-autoregressive) | | 자기회귀 생성 |
| [3.3 KV Cache](phase1-foundations/chapter03-llm-basics/03-kv-cache) | | 추론 최적화 |
| [3.4 Inference](phase1-foundations/chapter03-llm-basics/04-inference) | | Sampling 전략 |

## Phase 2: Vision & VLM 아키텍처

| Chapter | 주제 | 핵심 내용 |
|---------|------|-----------|
| [4.1 Patch Embedding](phase2-vlm-architecture/chapter04-vit/01-patch-embedding) | ViT | 이미지 → 토큰 변환 |
| [4.2 CLS & Position](phase2-vlm-architecture/chapter04-vit/02-cls-position) | | 2D PE, 보간 |
| [4.3 ViT Variants](phase2-vlm-architecture/chapter04-vit/03-vit-variants) | | Swin, DeiT |
| [4.4 ViT vs CNN](phase2-vlm-architecture/chapter04-vit/04-vit-vs-cnn) | | Inductive Bias 분석 |
| [5.1 Cross-Attention](phase2-vlm-architecture/chapter05-vlm-patterns/01-cross-attention) | VLM 패턴 | Flamingo, BLIP-2 |
| [5.2 Projection](phase2-vlm-architecture/chapter05-vlm-patterns/02-projection) | | LLaVA 방식 |
| [5.3 Early Fusion](phase2-vlm-architecture/chapter05-vlm-patterns/03-early-fusion) | | Fuyu 방식 |
| [5.4 패턴 비교](phase2-vlm-architecture/chapter05-vlm-patterns/04-pattern-comparison) | | OCR 최적 선택 |
| [6.1 Qwen2.5-VL](phase2-vlm-architecture/chapter06-ocr-vlm/01-qwen2.5-vl) | OCR VLM | M-RoPE, Dynamic Resolution |
| [6.2 OLMoCR](phase2-vlm-architecture/chapter06-ocr-vlm/02-olmocr) | | 문서 OCR 특화 |
| [6.3 문서 이해 VLM](phase2-vlm-architecture/chapter06-ocr-vlm/03-document-vlm) | | DocTR, Donut, Pix2Struct |
| [6.4 의료 문서 OCR](phase2-vlm-architecture/chapter06-ocr-vlm/04-medical-ocr) | | 보험 청구 핵심 |

## Phase 3: 데이터 파이프라인

| Chapter | 주제 | 핵심 내용 |
|---------|------|-----------|
| [7.1 문서 유형 분류](phase3-data-pipeline/chapter07-data-collection/01-document-types) | 데이터 수집 | 진단서, 수술기록, 세부내역서 |
| [7.2 어노테이션](phase3-data-pipeline/chapter07-data-collection/02-annotation) | | IoU, NMS, Cohen's Kappa |
| [7.3 품질 관리](phase3-data-pipeline/chapter07-data-collection/03-quality-control) | | Label Noise 탐지 |
| [7.4 개인정보 처리](phase3-data-pipeline/chapter07-data-collection/04-privacy) | | k-anonymity, DP |
| [8.1 템플릿 생성](phase3-data-pipeline/chapter08-synthetic-data/01-template-generation) | 합성 데이터 | 의료 문서 렌더링 |
| [8.2 노이즈 & 변형](phase3-data-pipeline/chapter08-synthetic-data/02-noise-augmentation) | | 스캔/팩스 시뮬레이션 |
| [8.3 Pre-printed 시뮬레이션](phase3-data-pipeline/chapter08-synthetic-data/03-preprinted-simulation) | | 양식 배경 합성 |
| [8.4 데이터 증강](phase3-data-pipeline/chapter08-synthetic-data/04-augmentation) | | MixUp, CutMix |
| [9.1 Conversation Format](phase3-data-pipeline/chapter09-data-format/01-conversation-format) | 데이터 포맷 | VLM 학습용 JSON |
| [9.2 효율적 로딩](phase3-data-pipeline/chapter09-data-format/02-efficient-loading) | | WebDataset, mmap |
| [9.3 Multi-resolution](phase3-data-pipeline/chapter09-data-format/03-multi-resolution) | | Aspect Ratio Bucketing |

## Phase 4: Fine-tuning 실전

| Chapter | 주제 | 핵심 내용 |
|---------|------|-----------|
| [10.1 Full Fine-tuning](phase4-finetuning/chapter10-techniques/01-full-finetuning) | 기법 이론 | Catastrophic Forgetting |
| [10.2 LoRA](phase4-finetuning/chapter10-techniques/02-lora) | | Low-Rank Adaptation 핵심 |
| [10.3 QLoRA](phase4-finetuning/chapter10-techniques/03-qlora) | | NF4, Double Quantization |
| [10.4 Freeze 전략](phase4-finetuning/chapter10-techniques/04-freeze-strategy) | | Progressive Unfreezing |
| [11.1 분산 학습](phase4-finetuning/chapter11-infrastructure/01-distributed-training) | 인프라 | DDP, FSDP, ZeRO |
| [11.2 Mixed Precision](phase4-finetuning/chapter11-infrastructure/02-mixed-precision) | | FP16, BF16, Loss Scaling |
| [11.3 Config 예시](phase4-finetuning/chapter11-infrastructure/03-config-examples) | | Qwen2.5-VL + LoRA 설정 |
| [11.4 모니터링](phase4-finetuning/chapter11-infrastructure/04-monitoring) | | WandB, 이상 징후 감지 |
| [12.1 Sanity Check](phase4-finetuning/chapter12-training/01-sanity-check) | 학습 실행 | Overfit Test, LR Finder |
| [12.2 Small Scale](phase4-finetuning/chapter12-training/02-small-scale) | | 하이퍼파라미터 탐색 |
| [12.3 Full Scale](phase4-finetuning/chapter12-training/03-full-scale) | | 체크포인트, Early Stop |
| [12.4 트러블슈팅](phase4-finetuning/chapter12-training/04-troubleshooting) | | Loss NaN, OOM 해결 |

## Phase 5: 평가 & 최적화

| Chapter | 주제 | 핵심 내용 |
|---------|------|-----------|
| [13.1 OCR 메트릭](phase5-evaluation/chapter13-evaluation/01-ocr-metrics) | 평가 체계 | CER, WER, BLEU |
| [13.2 구조화 출력 평가](phase5-evaluation/chapter13-evaluation/02-structured-output) | | JSON Accuracy |
| [13.3 의료 문서 평가](phase5-evaluation/chapter13-evaluation/03-medical-specific) | | Critical Field 가중 평가 |
| [13.4 Human Evaluation](phase5-evaluation/chapter13-evaluation/04-human-evaluation) | | Cohen's Kappa, IAA |
| [14.1 Quantization](phase5-evaluation/chapter14-optimization/01-quantization) | 최적화 | AWQ, GPTQ |
| [14.2 KV Cache 최적화](phase5-evaluation/chapter14-optimization/02-kv-cache-optimization) | | PagedAttention |
| [14.3 추론 엔진](phase5-evaluation/chapter14-optimization/03-inference-engines) | | vLLM, SGLang 비교 |
| [14.4 벤치마크](phase5-evaluation/chapter14-optimization/04-benchmark) | | Roofline Model |

## Phase 6: 프로덕션 배포

| Chapter | 주제 | 핵심 내용 |
|---------|------|-----------|
| [15.1 서빙 아키텍처](phase6-production/chapter15-serving/01-architecture) | 서빙 | API Gateway, LB |
| [15.2 vLLM + FastAPI](phase6-production/chapter15-serving/02-vllm-fastapi) | | Dockerfile 포함 |
| [15.3 Batching 전략](phase6-production/chapter15-serving/03-batching) | | Continuous Batching |
| [16.1 Health Check](phase6-production/chapter16-stability/01-health-check) | 안정성 | K8s Probe |
| [16.2 24시간 안정성](phase6-production/chapter16-stability/02-24h-stability) | | Memory Leak 대응 |
| [16.3 로깅 & 모니터링](phase6-production/chapter16-stability/03-logging-monitoring) | | Prometheus, Grafana |
| [17.1 모델 버전 관리](phase6-production/chapter17-cicd/01-version-control) | CI/CD | MLflow, DVC |
| [17.2 배포 자동화](phase6-production/chapter17-cicd/02-deployment-automation) | | Canary, Blue-Green |
| [17.3 Rollback 전략](phase6-production/chapter17-cicd/03-rollback-strategy) | | 자동 롤백 |

## 부록

| 부록 | 내용 |
|------|------|
| [A. 환경 설정](appendix/A-environment-setup) | Docker, CUDA, PyTorch 설치 |
| [B. 하드웨어 가이드](appendix/B-hardware-guide) | GPU 비교, VRAM 계산 |
| [C. 참고 자료](appendix/C-references) | 핵심 논문 50+편 |
| [D. 용어 사전](appendix/D-glossary) | 126개 용어 정리 |
