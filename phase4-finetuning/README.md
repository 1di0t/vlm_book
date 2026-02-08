---
---

# Phase 4: Fine-tuning 실전

사전 학습된 VLM을 우리 도메인(의료 문서)에 맞게 튜닝한다. LoRA, QLoRA 등 효율적인 방법부터 분산 학습까지 다룬다.

## 학습 목표

- Full Fine-tuning vs LoRA vs QLoRA 선택 기준 이해
- 어떤 레이어를 freeze할지 결정하는 방법
- DeepSpeed, FSDP를 활용한 분산 학습 설정
- 학습 중 발생하는 문제 진단 및 해결

## 구성

| Chapter | 주제 | 핵심 내용 |
|---------|------|-----------|
| 10 | Fine-tuning 기법 이론 | Full, LoRA, QLoRA, Freeze 전략 |
| 11 | 학습 인프라 설정 | 분산 학습, Mixed Precision, 모니터링 |
| 12 | 학습 실행 & 디버깅 | Sanity Check, 스케일링, 트러블슈팅 |

## 실습

| 노트북 | 설명 |
|--------|------|
| `01-lora-basics.ipynb` | LoRA 기본 개념 실습 |
| `01-lora-training-e2e.ipynb` | End-to-End LoRA 튜닝 |
| `02-ablation-study.ipynb` | Hyperparameter Ablation |

## 예상 소요 시간

4-5주

## 사전 요구 지식

- Phase 1, 2, 3 완료
- GPU 서버 접근 권한
- 기본적인 Linux 명령어
