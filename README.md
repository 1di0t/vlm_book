# VLM Fine-tuning & Deployment 완벽 가이드

Vision-Language Model(VLM)의 Fine-tuning부터 프로덕션 배포까지, 의료 문서 OCR을 중심으로 한 실무 교과서.

## 대상 독자

- 딥러닝 기초 지식이 있는 ML 엔지니어
- VLM/LLM을 실무에 적용하려는 개발자
- OCR 시스템을 구축하려는 팀

## 사전 요구 지식

- Python 중급 이상
- PyTorch 기본 사용법
- 선형대수, 미적분 기초 개념

## 구성

| Phase | 주제 | 핵심 내용 |
|-------|------|-----------|
| 1 | 딥러닝 & Transformer 기초 | 수학적 기초, Transformer, LLM |
| 2 | Vision & VLM 아키텍처 | ViT, VLM 패턴, OCR 특화 모델 |
| 3 | 데이터 파이프라인 | 수집, 합성 데이터, 데이터 로더 |
| 4 | Fine-tuning 실전 | LoRA, QLoRA, 학습 인프라 |
| 5 | 평가 & 최적화 | 메트릭, Quantization, 추론 최적화 |
| 6 | 프로덕션 배포 | vLLM, 모니터링, MLOps |

## 실습 환경

```bash
# 권장 환경
Python >= 3.10
PyTorch >= 2.0
CUDA >= 12.0
GPU: RTX 4090 / A100 / RTX 6000 Pro
```

## 사용법

### 마크다운 문서
각 챕터의 `.md` 파일을 순서대로 읽는다.

### Jupyter Notebook
```bash
cd phase1-foundations/chapter01-math/notebooks
jupyter notebook 01-numpy-neural-network.ipynb
```

## 목차

전체 목차는 [SUMMARY.md](SUMMARY.md) 참조.

## 라이선스

이 책의 내용은 학습 목적으로 자유롭게 사용할 수 있다.
