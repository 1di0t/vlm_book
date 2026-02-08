---
---

# 목차

## Phase 1: 딥러닝 & Transformer 기초

### Chapter 1: 수학적 기초
- [1.1 선형대수](phase1-foundations/chapter01-math/01-linear-algebra.md)
- [1.2 미적분](phase1-foundations/chapter01-math/02-calculus.md)
- [1.3 확률과 통계](phase1-foundations/chapter01-math/03-probability.md)

### Chapter 2: Transformer 아키텍처
- [2.1 Self-Attention 메커니즘](phase1-foundations/chapter02-transformer/01-self-attention.md)
- [2.2 Positional Encoding](phase1-foundations/chapter02-transformer/02-positional-encoding.md)
- [2.3 Transformer Block 구성요소](phase1-foundations/chapter02-transformer/03-transformer-block.md)
- [2.4 Encoder vs Decoder](phase1-foundations/chapter02-transformer/04-encoder-decoder.md)

### Chapter 3: LLM 기초
- [3.1 Tokenization](phase1-foundations/chapter03-llm-basics/01-tokenization.md)
- [3.2 Autoregressive Generation](phase1-foundations/chapter03-llm-basics/02-autoregressive.md)
- [3.3 KV Cache](phase1-foundations/chapter03-llm-basics/03-kv-cache.md)
- [3.4 Inference 특성](phase1-foundations/chapter03-llm-basics/04-inference.md)

---

## Phase 2: Vision & VLM 아키텍처

### Chapter 4: Vision Transformer (ViT)
- [4.1 Patch Embedding](phase2-vlm-architecture/chapter04-vit/01-patch-embedding.md)
- [4.2 CLS Token & Position Embedding](phase2-vlm-architecture/chapter04-vit/02-cls-position.md)
- [4.3 ViT Variants](phase2-vlm-architecture/chapter04-vit/03-vit-variants.md)
- [4.4 ViT vs CNN](phase2-vlm-architecture/chapter04-vit/04-vit-vs-cnn.md)

### Chapter 5: VLM 아키텍처 패턴
- [5.1 Cross-Attention 방식](phase2-vlm-architecture/chapter05-vlm-patterns/01-cross-attention.md)
- [5.2 Projection/Connector 방식](phase2-vlm-architecture/chapter05-vlm-patterns/02-projection.md)
- [5.3 Early Fusion 방식](phase2-vlm-architecture/chapter05-vlm-patterns/03-early-fusion.md)
- [5.4 패턴 비교 분석](phase2-vlm-architecture/chapter05-vlm-patterns/04-pattern-comparison.md)

### Chapter 6: OCR 특화 VLM
- [6.1 Qwen2.5-VL 심층 분석](phase2-vlm-architecture/chapter06-ocr-vlm/01-qwen2.5-vl.md)
- [6.2 OLMoCR 분석](phase2-vlm-architecture/chapter06-ocr-vlm/02-olmocr.md)
- [6.3 문서 이해 VLM](phase2-vlm-architecture/chapter06-ocr-vlm/03-document-vlm.md)
- [6.4 보험금 청구 의료 문서 OCR](phase2-vlm-architecture/chapter06-ocr-vlm/04-medical-ocr.md)

---

## Phase 3: 데이터 파이프라인

### Chapter 7: 데이터 수집 & 정제
- [7.1 보험 청구 의료 문서 유형 분류](phase3-data-pipeline/chapter07-data-collection/01-document-types.md)
- [7.2 어노테이션 전략](phase3-data-pipeline/chapter07-data-collection/02-annotation.md)
- [7.3 데이터 품질 관리](phase3-data-pipeline/chapter07-data-collection/03-quality-control.md)
- [7.4 개인정보 처리](phase3-data-pipeline/chapter07-data-collection/04-privacy.md)

### Chapter 8: 합성 데이터 생성
- [8.1 템플릿 기반 생성](phase3-data-pipeline/chapter08-synthetic-data/01-template-generation.md)
- [8.2 노이즈 & 변형 추가](phase3-data-pipeline/chapter08-synthetic-data/02-noise-augmentation.md)
- [8.3 Pre-printed 라인 간섭 시뮬레이션](phase3-data-pipeline/chapter08-synthetic-data/03-preprinted-simulation.md)
- [8.4 데이터 증강 기법](phase3-data-pipeline/chapter08-synthetic-data/04-augmentation.md)

### Chapter 9: 데이터 포맷 & 로더
- [9.1 Conversation Format 변환](phase3-data-pipeline/chapter09-data-format/01-conversation-format.md)
- [9.2 효율적 데이터 로딩](phase3-data-pipeline/chapter09-data-format/02-efficient-loading.md)
- [9.3 Multi-resolution 처리](phase3-data-pipeline/chapter09-data-format/03-multi-resolution.md)

---

## Phase 4: Fine-tuning 실전

### Chapter 10: Fine-tuning 기법 이론
- [10.1 Full Fine-tuning](phase4-finetuning/chapter10-techniques/01-full-finetuning.md)
- [10.2 LoRA (Low-Rank Adaptation)](phase4-finetuning/chapter10-techniques/02-lora.md)
- [10.3 QLoRA](phase4-finetuning/chapter10-techniques/03-qlora.md)
- [10.4 Freeze 전략](phase4-finetuning/chapter10-techniques/04-freeze-strategy.md)

### Chapter 11: 학습 인프라 설정
- [11.1 분산 학습](phase4-finetuning/chapter11-infrastructure/01-distributed-training.md)
- [11.2 Mixed Precision Training](phase4-finetuning/chapter11-infrastructure/02-mixed-precision.md)
- [11.3 Config 예시](phase4-finetuning/chapter11-infrastructure/03-config-examples.md)
- [11.4 모니터링 설정](phase4-finetuning/chapter11-infrastructure/04-monitoring.md)

### Chapter 12: 학습 실행 & 디버깅
- [12.1 Sanity Check](phase4-finetuning/chapter12-training/01-sanity-check.md)
- [12.2 Small Scale 학습](phase4-finetuning/chapter12-training/02-small-scale.md)
- [12.3 Full Scale 학습](phase4-finetuning/chapter12-training/03-full-scale.md)
- [12.4 트러블슈팅](phase4-finetuning/chapter12-training/04-troubleshooting.md)

---

## Phase 5: 평가 & 최적화

### Chapter 13: 평가 체계 구축
- [13.1 OCR 메트릭](phase5-evaluation/chapter13-evaluation/01-ocr-metrics.md)
- [13.2 구조화 출력 평가](phase5-evaluation/chapter13-evaluation/02-structured-output.md)
- [13.3 보험 청구 문서 특화 평가](phase5-evaluation/chapter13-evaluation/03-medical-specific.md)
- [13.4 Human Evaluation](phase5-evaluation/chapter13-evaluation/04-human-evaluation.md)

### Chapter 14: 모델 최적화
- [14.1 Quantization](phase5-evaluation/chapter14-optimization/01-quantization.md)
- [14.2 KV Cache 최적화](phase5-evaluation/chapter14-optimization/02-kv-cache-optimization.md)
- [14.3 추론 엔진](phase5-evaluation/chapter14-optimization/03-inference-engines.md)
- [14.4 벤치마크](phase5-evaluation/chapter14-optimization/04-benchmark.md)

---

## Phase 6: 프로덕션 배포

### Chapter 15: 서빙 아키텍처
- [15.1 기본 구조](phase6-production/chapter15-serving/01-architecture.md)
- [15.2 vLLM + FastAPI 설정](phase6-production/chapter15-serving/02-vllm-fastapi.md)
- [15.3 Batching 전략](phase6-production/chapter15-serving/03-batching.md)

### Chapter 16: 안정성 & 모니터링
- [16.1 Health Check](phase6-production/chapter16-stability/01-health-check.md)
- [16.2 24시간 안정성](phase6-production/chapter16-stability/02-24h-stability.md)
- [16.3 로깅 & 모니터링](phase6-production/chapter16-stability/03-logging-monitoring.md)

### Chapter 17: CI/CD & MLOps
- [17.1 모델 버전 관리](phase6-production/chapter17-cicd/01-version-control.md)
- [17.2 배포 자동화](phase6-production/chapter17-cicd/02-deployment-automation.md)
- [17.3 Rollback 전략](phase6-production/chapter17-cicd/03-rollback-strategy.md)

---

## 부록

- [A. 개발 환경 설정](appendix/A-environment-setup.md)
- [B. 하드웨어 가이드](appendix/B-hardware-guide.md)
- [C. 참고 자료](appendix/C-references.md)
- [D. 용어 사전](appendix/D-glossary.md)
