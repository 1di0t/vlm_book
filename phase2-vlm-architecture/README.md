# Phase 2: Vision & VLM 아키텍처

이미지를 이해하는 AI의 핵심인 Vision Transformer(ViT)와 이를 언어 모델과 결합하는 VLM 아키텍처를 다룬다.

## 학습 목표

- ViT가 이미지를 처리하는 방식 이해
- VLM의 세 가지 주요 아키텍처 패턴 비교
- OCR 특화 VLM의 특징과 선택 기준 파악
- 의료 문서 OCR의 특수한 요구사항 이해

## 구성

| Chapter | 주제 | 핵심 내용 |
|---------|------|-----------|
| 4 | Vision Transformer | Patch Embedding, CLS Token, ViT Variants |
| 5 | VLM 아키텍처 패턴 | Cross-Attention, Projection, Early Fusion |
| 6 | OCR 특화 VLM | Qwen2.5-VL, OLMoCR, 의료 문서 처리 |

## 실습

| 노트북 | 설명 |
|--------|------|
| `01-vit-classification.ipynb` | ViT로 이미지 분류 |
| `02-attention-visualization.ipynb` | Attention Map 시각화 |
| `01-llava-code-analysis.ipynb` | LLaVA 구조 분석 |
| `01-qwen-vl-test.ipynb` | Qwen-VL 의료 문서 테스트 |

## 예상 소요 시간

4-5주

## 사전 요구 지식

- Phase 1 완료
- PyTorch 기본 사용법
- CNN 기초 개념 (권장)
