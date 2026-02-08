---
---

# Phase 1: 딥러닝 & Transformer 기초

VLM을 제대로 이해하고 튜닝하려면 기초가 탄탄해야 한다. 이 Phase에서는 Transformer의 핵심 개념부터 LLM의 동작 원리까지 다룬다.

## 학습 목표

- Transformer의 Self-Attention 메커니즘을 수식 레벨에서 이해
- Positional Encoding의 종류와 동작 원리 파악
- LLM의 tokenization과 autoregressive generation 이해
- KV Cache의 필요성과 메모리 사용량 계산

## 구성

| Chapter | 주제 | 핵심 내용 |
|---------|------|-----------|
| 1 | 수학적 기초 | 선형대수, 미적분, 확률/통계 |
| 2 | Transformer | Attention, Positional Encoding, Block 구조 |
| 3 | LLM 기초 | Tokenization, Generation, KV Cache |

## 실습

| 노트북 | 설명 |
|--------|------|
| `01-numpy-neural-network.ipynb` | NumPy로 forward/backward pass 구현 |
| `01-attention-implementation.ipynb` | Multi-head Attention 직접 구현 |
| `02-mini-transformer.ipynb` | 작은 Transformer로 문자 생성 |
| `01-bpe-tokenizer.ipynb` | BPE 알고리즘 구현 |
| `02-text-generation.ipynb` | HuggingFace로 텍스트 생성 |

## 예상 소요 시간

3-4주 (하루 2-3시간 기준)

## 사전 요구 지식

- Python 기본 문법
- NumPy 배열 연산
- 고등학교 수준의 수학 (행렬, 미분 개념)
