# Appendix D: 용어 사전 (Glossary)

> 본 가이드에서 사용하는 핵심 용어 110+개를 카테고리별로 정리한다.
> 각 용어는 한글명, 영문명, 정의, 수식(해당 시), 본문 참조를 포함한다.

---

## 1. 수학 기초 (Mathematics)

| # | 용어 (한글) | 영문 | 정의 | 수식 | 참조 |
|---|------------|------|------|------|------|
| 1 | 내적 | Dot Product | 두 벡터의 원소별 곱의 합 | $a \cdot b = \sum_i a_i b_i$ | Phase 1 Ch.1 |
| 2 | 행렬 곱셈 | Matrix Multiplication | $(m \times k)$와 $(k \times n)$ 행렬의 곱으로 $(m \times n)$ 행렬 생성 | $C_{ij} = \sum_k A_{ik} B_{kj}$ | Phase 1 Ch.1 |
| 3 | 고유값 분해 | Eigendecomposition | 행렬을 고유값과 고유벡터로 분해 | $Av = \lambda v$ | Phase 1 Ch.1 |
| 4 | 특이값 분해 | SVD (Singular Value Decomposition) | 임의 행렬을 $U\Sigma V^T$로 분해 | $W = U\Sigma V^T$ | Phase 1 Ch.1, Phase 4 Ch.10 |
| 5 | 노름 | Norm | 벡터 또는 행렬의 크기를 측정하는 함수 | $\|x\|_p = (\sum_i |x_i|^p)^{1/p}$ | Phase 1 Ch.1 |
| 6 | 그래디언트 | Gradient | 다변수 함수의 각 변수에 대한 편미분 벡터 | $\nabla f = [\partial f/\partial x_1, \ldots]^T$ | Phase 1 Ch.1 |
| 7 | 야코비안 | Jacobian | 벡터 함수의 1차 편미분 행렬 | $J_{ij} = \partial f_i / \partial x_j$ | Phase 1 Ch.1 |
| 8 | 헤시안 | Hessian | 스칼라 함수의 2차 편미분 행렬 | $H_{ij} = \partial^2 f / \partial x_i \partial x_j$ | Phase 1 Ch.1 |
| 9 | 체인 룰 | Chain Rule | 합성 함수의 미분 법칙 | $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$ | Phase 1 Ch.1 |
| 10 | 소프트맥스 | Softmax | 로짓을 확률 분포로 변환하는 함수 | $\text{softmax}(x_i) = e^{x_i} / \sum_j e^{x_j}$ | Phase 1 Ch.1 |
| 11 | 크로스 엔트로피 | Cross-Entropy | 두 확률 분포 간 차이 측정 | $H(p,q) = -\sum_i p_i \log q_i$ | Phase 1 Ch.1 |
| 12 | KL 발산 | KL Divergence | 두 확률 분포 간 비대칭 거리 | $D_{KL}(p\|q) = \sum_i p_i \log(p_i/q_i)$ | Phase 1 Ch.1 |

---

## 2. Transformer 아키텍처 (Transformer Architecture)

| # | 용어 (한글) | 영문 | 정의 | 수식 | 참조 |
|---|------------|------|------|------|------|
| 13 | 셀프 어텐션 | Self-Attention | 시퀀스 내 모든 위치 간 관계를 계산하는 메커니즘 | $\text{Attn}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$ | Phase 1 Ch.2 |
| 14 | 멀티헤드 어텐션 | Multi-Head Attention | 여러 어텐션 헤드를 병렬로 수행 후 결합 | $\text{MHA} = \text{Concat}(h_1,\ldots,h_H)W^O$ | Phase 1 Ch.2 |
| 15 | 쿼리/키/밸류 | Query/Key/Value | 어텐션 연산의 세 가지 투영 벡터 | $Q=XW^Q, K=XW^K, V=XW^V$ | Phase 1 Ch.2 |
| 16 | 위치 인코딩 | Positional Encoding | 토큰의 순서 정보를 벡터에 주입하는 방법 | $PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$ | Phase 1 Ch.2 |
| 17 | RoPE | Rotary Position Embedding | 회전 행렬 기반 상대적 위치 인코딩 | $f(x,m) = R_{\Theta,m} x$ | Phase 1 Ch.2 |
| 18 | 레이어 정규화 | Layer Normalization | 레이어 출력을 평균 0, 분산 1로 정규화 | $\text{LN}(x) = \gamma \odot \frac{x-\mu}{\sigma+\epsilon} + \beta$ | Phase 1 Ch.2 |
| 19 | RMSNorm | Root Mean Square Normalization | 평균 제거 없이 RMS로만 정규화 | $\text{RMS}(x) = x / \sqrt{\frac{1}{n}\sum x_i^2}$ | Phase 1 Ch.2 |
| 20 | 피드포워드 네트워크 | Feed-Forward Network (FFN) | 어텐션 후 위치별 비선형 변환 | $\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$ | Phase 1 Ch.2 |
| 21 | SwiGLU | SwiGLU Activation | Swish + Gated Linear Unit 결합 활성화 함수 | $\text{SwiGLU}(x) = (\text{Swish}(xW_1)) \odot (xW_2)$ | Phase 1 Ch.2 |
| 22 | 잔차 연결 | Residual Connection | 입력을 출력에 더하여 그래디언트 소실 방지 | $y = x + F(x)$ | Phase 1 Ch.2 |
| 23 | 인코더-디코더 | Encoder-Decoder | 인코더가 입력 처리, 디코더가 출력 생성하는 구조 | — | Phase 1 Ch.2 |

---

## 3. LLM 기초 (Large Language Model Basics)

| # | 용어 (한글) | 영문 | 정의 | 수식 | 참조 |
|---|------------|------|------|------|------|
| 24 | 토크나이제이션 | Tokenization | 텍스트를 모델이 처리할 수 있는 토큰 단위로 분할 | — | Phase 1 Ch.3 |
| 25 | BPE | Byte Pair Encoding | 빈도 기반으로 서브워드를 병합하는 토크나이저 | — | Phase 1 Ch.3 |
| 26 | 자기회귀 생성 | Autoregressive Generation | 이전 토큰들을 조건으로 다음 토큰을 순차 예측 | $P(x_t | x_{<t})$ | Phase 1 Ch.3 |
| 27 | KV 캐시 | KV Cache | 추론 시 이전 Key/Value를 저장하여 재계산 방지 | 메모리: $2 \times L \times d \times n_{seq}$ | Phase 1 Ch.3 |
| 28 | 온도 | Temperature | 소프트맥스 분포의 날카로움을 조절하는 하이퍼파라미터 | $p_i = e^{z_i/T} / \sum_j e^{z_j/T}$ | Phase 1 Ch.3 |
| 29 | Top-k 샘플링 | Top-k Sampling | 확률 상위 k개 토큰에서만 샘플링 | — | Phase 1 Ch.3 |
| 30 | Top-p 샘플링 | Nucleus Sampling | 누적 확률이 p 이하인 토큰 집합에서 샘플링 | $\sum_{x \in V_p} P(x) \geq p$ | Phase 1 Ch.3 |

---

## 4. Vision Transformer (ViT)

| # | 용어 (한글) | 영문 | 정의 | 수식 | 참조 |
|---|------------|------|------|------|------|
| 31 | 패치 임베딩 | Patch Embedding | 이미지를 고정 크기 패치로 분할 후 선형 투영 | $z = xE + e_{pos}$, $N = HW/P^2$ | Phase 2 Ch.4 |
| 32 | CLS 토큰 | CLS Token | 분류를 위한 학습 가능한 특수 토큰 | $x_{class} \in \mathbb{R}^D$ | Phase 2 Ch.4 |
| 33 | 2D 위치 임베딩 | 2D Position Embedding | 행/열 분리하여 2차원 위치 정보 인코딩 | $(i,j) \rightarrow [PE_{row}(i); PE_{col}(j)]$ | Phase 2 Ch.4 |
| 34 | 윈도우 어텐션 | Window Attention | 고정 크기 윈도우 내에서만 어텐션 계산 (Swin) | $O(M^2 N)$ vs $O(N^2)$ | Phase 2 Ch.4 |
| 35 | 시프트 윈도우 | Shifted Window | 윈도우를 이동하여 윈도우 간 정보 교환 | — | Phase 2 Ch.4 |
| 36 | DeiT | Data-efficient Image Transformer | 지식 증류로 데이터 효율성을 높인 ViT | — | Phase 2 Ch.4 |
| 37 | Swin Transformer | Shifted Window Transformer | 계층적 윈도우 기반 Vision Transformer | — | Phase 2 Ch.4 |
| 38 | 귀납적 편향 | Inductive Bias | 모델에 내재된 구조적 가정 (locality, translation equivariance 등) | — | Phase 2 Ch.4 |
| 39 | 전역 수용장 | Global Receptive Field | 모든 위치의 정보를 한 번에 참조할 수 있는 범위 | — | Phase 2 Ch.4 |

---

## 5. VLM 아키텍처 (Vision-Language Model Architecture)

| # | 용어 (한글) | 영문 | 정의 | 수식 | 참조 |
|---|------------|------|------|------|------|
| 40 | 크로스 어텐션 | Cross-Attention | 서로 다른 모달리티 간 어텐션 수행 | $\text{CA}(Q_{text}, K_{img}, V_{img})$ | Phase 2 Ch.5 |
| 41 | Q-Former | Querying Transformer (BLIP-2) | 학습 가능한 쿼리로 visual feature를 고정 수 토큰으로 압축 | $Q \in \mathbb{R}^{M \times D}$ | Phase 2 Ch.5 |
| 42 | 게이트 크로스 어텐션 | Gated Cross-Attention | 게이트로 visual 정보 주입량을 조절 (Flamingo) | $h = h + \tanh(\alpha) \cdot \text{CA}(h,v)$ | Phase 2 Ch.5 |
| 43 | 선형 프로젝션 | Linear Projection | 선형 변환으로 차원 매핑 | $z_{LLM} = Wz_{vision} + b$ | Phase 2 Ch.5 |
| 44 | MLP 커넥터 | MLP Connector | 비선형 활성화 포함 다층 투영 | $z = W_2 \cdot \text{GELU}(W_1 z + b_1) + b_2$ | Phase 2 Ch.5 |
| 45 | 비주얼 토큰 | Visual Token | 이미지 패치를 LLM 입력 토큰으로 변환한 벡터 | — | Phase 2 Ch.5 |
| 46 | 피처 정렬 | Feature Alignment | Vision과 Language 임베딩 공간을 정렬 | — | Phase 2 Ch.5 |
| 47 | 얼리 퓨전 | Early Fusion | 이미지/텍스트 토큰을 결합 후 단일 모델로 처리 | $Z = [v_1,\ldots,v_m, t_1,\ldots,t_n]$ | Phase 2 Ch.5 |
| 48 | 토큰 결합 | Token Concatenation | 비주얼 토큰과 텍스트 토큰을 시퀀스로 이어붙이기 | — | Phase 2 Ch.5 |
| 49 | 정보 병목 | Information Bottleneck | 모달리티 간 정보 전달의 제한점 | — | Phase 2 Ch.5 |

---

## 6. OCR 특화 VLM (OCR-Specialized VLM)

| # | 용어 (한글) | 영문 | 정의 | 수식 | 참조 |
|---|------------|------|------|------|------|
| 50 | 동적 해상도 | Dynamic Resolution | 입력 이미지 비율을 유지하며 가변 패치 수 생성 | — | Phase 2 Ch.6 |
| 51 | M-RoPE | Multimodal RoPE | 시간/높이/너비 3축 회전 위치 인코딩 | $f(q, m_t, m_h, m_w)$ | Phase 2 Ch.6 |
| 52 | 토큰 병합 | Token Merging | 인접 패치를 합쳐 토큰 수 축소 | 2×2 → 토큰 수 1/4 | Phase 2 Ch.6 |
| 53 | 문서 이해 | Document Understanding | 문서의 텍스트, 레이아웃, 표, 그림을 종합 이해 | — | Phase 2 Ch.6 |
| 54 | 표 인식 | Table Recognition | 표의 행/열 구조를 인식하고 셀 내용 추출 | — | Phase 2 Ch.6 |
| 55 | 키-값 추출 | Key-Value Extraction | 문서에서 필드명과 값의 쌍을 추출 | — | Phase 2 Ch.6 |
| 56 | KCD 코드 | Korean Classification of Diseases | 한국표준질병사인분류 코드 (예: S72.0) | — | Phase 2 Ch.6 |
| 57 | 의료행위코드 | Medical Procedure Code | 건강보험 의료행위 분류 코드 (EDI) | — | Phase 2 Ch.6 |
| 58 | 비식별화 | De-identification | 개인식별정보(PHI)를 제거/대체하여 익명화 | — | Phase 2 Ch.6 |
| 59 | Pre-printed 양식 | Pre-printed Form | 미리 인쇄된 양식 위에 수기/출력 텍스트가 혼재하는 문서 | — | Phase 2 Ch.6 |

---

## 7. 데이터 파이프라인 (Data Pipeline)

| # | 용어 (한글) | 영문 | 정의 | 수식 | 참조 |
|---|------------|------|------|------|------|
| 60 | 바운딩 박스 | Bounding Box | 객체를 감싸는 직사각형 좌표 $(x, y, w, h)$ | — | Phase 3 Ch.7 |
| 61 | IoU | Intersection over Union | 두 영역의 겹침 비율 | $\text{IoU} = |A \cap B| / |A \cup B|$ | Phase 3 Ch.7 |
| 62 | NMS | Non-Maximum Suppression | IoU 기반으로 중복 검출 박스 제거 | — | Phase 3 Ch.7 |
| 63 | 코헨 카파 | Cohen's Kappa | 평가자 간 일치도 측정 (우연 일치 보정) | $\kappa = (p_o - p_e)/(1 - p_e)$ | Phase 3 Ch.7 |
| 64 | k-익명성 | k-Anonymity | 각 레코드가 최소 k-1개와 구별 불가한 상태 | — | Phase 3 Ch.7 |
| 65 | 차분 프라이버시 | Differential Privacy | 데이터 한 건의 포함 여부가 출력에 미치는 영향을 제한 | $\Pr[M(D) \in S] \leq e^\epsilon \Pr[M(D') \in S]$ | Phase 3 Ch.7 |
| 66 | 아핀 변환 | Affine Transform | 회전, 크기 조절, 이동, 기울기를 포함하는 선형 변환 | $p' = Ap + t$ | Phase 3 Ch.8 |
| 67 | 호모그래피 | Homography | 평면 간 투영 변환을 나타내는 3×3 행렬 | $p' = Hp$ | Phase 3 Ch.8 |
| 68 | MixUp | MixUp Augmentation | 두 이미지/레이블을 선형 보간하여 새 샘플 생성 | $\tilde{x} = \lambda x_i + (1-\lambda)x_j$ | Phase 3 Ch.8 |
| 69 | CutMix | CutMix Augmentation | 한 이미지의 영역을 다른 이미지 영역으로 대체 | $\lambda = 1 - r_w r_h / WH$ | Phase 3 Ch.8 |
| 70 | 허프 변환 | Hough Transform | 이미지에서 직선/원 등의 기하학적 형태 검출 | $(x,y) \rightarrow (\rho, \theta)$ | Phase 3 Ch.8 |
| 71 | 오츠 이진화 | Otsu's Binarization | 클래스 간 분산을 최대화하는 임계값 자동 결정 | $\sigma_B^2(t) = \frac{[\mu_T\omega(t)-\mu(t)]^2}{\omega(t)[1-\omega(t)]}$ | Phase 3 Ch.8 |
| 72 | 대화 형식 | Conversation Format | VLM 학습용 System/User/Assistant 역할 기반 JSON 형식 | — | Phase 3 Ch.9 |
| 73 | 종횡비 버케팅 | Aspect Ratio Bucketing | 유사 비율 이미지를 그룹화하여 패딩 최소화 | — | Phase 3 Ch.9 |
| 74 | 동적 배칭 | Dynamic Batching | 요청 길이에 따라 배치 크기를 동적 조절 | — | Phase 3 Ch.9 |

---

## 8. Fine-tuning 기법 (Fine-tuning Techniques)

| # | 용어 (한글) | 영문 | 정의 | 수식 | 참조 |
|---|------------|------|------|------|------|
| 75 | 풀 파인튜닝 | Full Fine-tuning | 모든 파라미터를 업데이트하는 학습 방식 | $\theta_{t+1} = \theta_t - \eta \nabla L$ | Phase 4 Ch.10 |
| 76 | 파국적 망각 | Catastrophic Forgetting | 새 태스크 학습 시 이전 지식이 손실되는 현상 | — | Phase 4 Ch.10 |
| 77 | LoRA | Low-Rank Adaptation | 저랭크 행렬 분해로 적은 파라미터만 학습 | $h = W_0 x + BAx$, $r \ll d$ | Phase 4 Ch.10 |
| 78 | 랭크 | Rank (LoRA) | LoRA 분해의 차원 수, 표현력과 효율성의 트레이드오프 | $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$ | Phase 4 Ch.10 |
| 79 | 알파 | Alpha (LoRA) | LoRA 스케일링 팩터, 학습률 스케일링 역할 | $\Delta W = (\alpha/r) BA$ | Phase 4 Ch.10 |
| 80 | QLoRA | Quantized LoRA | 4-bit 양자화 모델 위에 LoRA 적용 | — | Phase 4 Ch.10 |
| 81 | NF4 | NormalFloat4 | 정규분포 가정 하 최적 4-bit 양자화 | $q_i = \Phi^{-1}((i+0.5)/2^b)$ | Phase 4 Ch.10 |
| 82 | 이중 양자화 | Double Quantization | 양자화 상수를 FP8로 재양자화하여 메모리 절약 | ~4.1 bit/param | Phase 4 Ch.10 |
| 83 | 점진적 해동 | Progressive Unfreezing | epoch마다 하위 레이어부터 순차적으로 학습 가능하게 전환 | — | Phase 4 Ch.10 |
| 84 | 차별적 학습률 | Discriminative Learning Rate | 레이어별 다른 학습률 적용 | $\eta_l = \eta_{base} \times \xi^{L-l}$ | Phase 4 Ch.10 |
| 85 | 가중치 감쇠 | Weight Decay | L2 정규화로 과적합 방지 | $L_{total} = L + (\lambda/2)\|\theta\|^2$ | Phase 4 Ch.10 |

---

## 9. 분산 학습 & 인프라 (Distributed Training & Infrastructure)

| # | 용어 (한글) | 영문 | 정의 | 수식 | 참조 |
|---|------------|------|------|------|------|
| 86 | DDP | Distributed Data Parallel | 각 GPU에 모델 복제 후 gradient 동기화 | $\nabla L = \frac{1}{N}\sum \nabla L_i$ | Phase 4 Ch.11 |
| 87 | FSDP | Fully Sharded Data Parallel | 모델 파라미터를 GPU 간 샤딩하여 메모리 절약 | — | Phase 4 Ch.11 |
| 88 | ZeRO | Zero Redundancy Optimizer | optimizer/gradient/parameter 단계별 분할 (DeepSpeed) | Stage 1/2/3 | Phase 4 Ch.11 |
| 89 | 파이프라인 병렬 | Pipeline Parallelism | 모델을 레이어 단위로 여러 GPU에 분할 배치 | bubble ratio = $(P-1)/(P-1+M)$ | Phase 4 Ch.11 |
| 90 | 혼합 정밀도 학습 | Mixed Precision Training | FP16/BF16과 FP32를 혼합하여 학습 속도 향상 | — | Phase 4 Ch.11 |
| 91 | 손실 스케일링 | Loss Scaling | FP16 underflow 방지를 위해 loss에 상수를 곱함 | $\text{scaled\_loss} = \text{loss} \times s$ | Phase 4 Ch.11 |
| 92 | 코사인 어닐링 | Cosine Annealing | 코사인 함수로 학습률을 점진적 감소 | $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max}-\eta_{min})(1+\cos(\frac{t}{T}\pi))$ | Phase 4 Ch.12 |
| 93 | 워밍업 | Warmup | 학습 초기에 학습률을 0에서 점진적 증가 | — | Phase 4 Ch.12 |

---

## 10. 평가 메트릭 (Evaluation Metrics)

| # | 용어 (한글) | 영문 | 정의 | 수식 | 참조 |
|---|------------|------|------|------|------|
| 94 | 편집 거리 | Edit Distance (Levenshtein) | 한 문자열을 다른 문자열로 변환하는 최소 편집 횟수 | DP 기반 $d(i,j)$ | Phase 5 Ch.13 |
| 95 | CER | Character Error Rate | 문자 단위 오류율 | $(S+D+I)/N$ | Phase 5 Ch.13 |
| 96 | WER | Word Error Rate | 단어 단위 오류율 | $(S+D+I)/N$ | Phase 5 Ch.13 |
| 97 | 정규화 편집 거리 | Normalized Edit Distance | 최대 길이로 정규화한 편집 거리 | $\text{NED} = d(s_1,s_2)/\max(|s_1|,|s_2|)$ | Phase 5 Ch.13 |
| 98 | BLEU | Bilingual Evaluation Understudy | n-gram 기반 텍스트 생성 품질 평가 | $\text{BLEU} = BP \cdot \exp(\sum w_n \log p_n)$ | Phase 5 Ch.13 |
| 99 | F1 스코어 | F1 Score | Precision과 Recall의 조화 평균 | $F1 = 2PR/(P+R)$ | Phase 5 Ch.13 |
| 100 | 트리 편집 거리 | Tree Edit Distance | JSON/XML 트리 구조 간 최소 편집 비용 | — | Phase 5 Ch.13 |
| 101 | 필드별 정확도 | Field-level Accuracy | 구조화 출력의 각 필드 단위 정확도 | $\text{Acc} = \text{correct}/\text{total}$ | Phase 5 Ch.13 |
| 102 | 청구 오류율 | Claim Error Rate | 보험 청구 시 OCR 오류로 인한 거절/오류 비율 | — | Phase 5 Ch.13 |

---

## 11. 모델 최적화 (Model Optimization)

| # | 용어 (한글) | 영문 | 정의 | 수식 | 참조 |
|---|------------|------|------|------|------|
| 103 | 양자화 | Quantization | 모델 가중치/활성화를 저정밀도로 변환 | $q = \text{round}(x/s) + z$ | Phase 5 Ch.14 |
| 104 | PTQ | Post-Training Quantization | 학습 후 양자화 적용 | — | Phase 5 Ch.14 |
| 105 | QAT | Quantization-Aware Training | 양자화를 고려한 학습 | — | Phase 5 Ch.14 |
| 106 | AWQ | Activation-Aware Weight Quantization | 활성화 분포 기반 중요 채널 가중치 보존 양자화 | — | Phase 5 Ch.14 |
| 107 | GPTQ | GPTQ | Hessian 기반 layer-wise 양자화 | $\delta_q = (w_q - \hat{w}_q)/[H^{-1}]_{qq}$ | Phase 5 Ch.14 |
| 108 | PagedAttention | PagedAttention | 가상 메모리 개념으로 KV Cache 비연속 관리 | — | Phase 5 Ch.14 |
| 109 | 프리픽스 캐싱 | Prefix Caching | 공통 prefix의 KV Cache를 요청 간 공유 | — | Phase 5 Ch.14 |
| 110 | 루프라인 모델 | Roofline Model | compute vs memory bound 병목 분석 | — | Phase 5 Ch.14 |

---

## 12. 프로덕션 배포 (Production Deployment)

| # | 용어 (한글) | 영문 | 정의 | 수식 | 참조 |
|---|------------|------|------|------|------|
| 111 | 연속 배칭 | Continuous Batching | 개별 요청 완료 시 즉시 다음 요청을 투입하는 배칭 | — | Phase 6 Ch.15 |
| 112 | 리틀의 법칙 | Little's Law | 대기열 시스템의 기본 법칙 | $L = \lambda W$ | Phase 6 Ch.15 |
| 113 | 활성 프로브 | Liveness Probe | 프로세스 생존 여부 확인 (Kubernetes) | — | Phase 6 Ch.16 |
| 114 | 준비 프로브 | Readiness Probe | 트래픽 수신 준비 여부 확인 | — | Phase 6 Ch.16 |
| 115 | 그레이스풀 저하 | Graceful Degradation | 장애 시 핵심 기능은 유지하며 부분 기능만 제한 | — | Phase 6 Ch.16 |
| 116 | 백분위 지연 | Percentile Latency | p50/p95/p99 등 응답 시간의 분포 지표 | — | Phase 6 Ch.16 |
| 117 | 블루-그린 배포 | Blue-Green Deployment | 두 환경을 교차하여 무중단 배포 | — | Phase 6 Ch.17 |
| 118 | 카나리 배포 | Canary Deployment | 소수 트래픽에 먼저 배포 후 점진 확대 | — | Phase 6 Ch.17 |
| 119 | A/B 테스트 | A/B Test | 두 버전 간 통계적 유의성 비교 | $z = (\bar{x}_A - \bar{x}_B)/\sqrt{s_A^2/n_A + s_B^2/n_B}$ | Phase 6 Ch.17 |
| 120 | 롤백 | Rollback | 배포 실패 시 이전 버전으로 복원 | — | Phase 6 Ch.17 |
| 121 | 모델 레지스트리 | Model Registry | 모델 아티팩트, 메타데이터, 버전을 체계적으로 관리 | — | Phase 6 Ch.17 |
| 122 | SLO/SLA | Service Level Objective/Agreement | 서비스 품질 목표/약속 (latency, uptime 등) | — | Phase 6 Ch.16 |

---

## 13. 하드웨어 & 메모리 (Hardware & Memory)

| # | 용어 (한글) | 영문 | 정의 | 수식 | 참조 |
|---|------------|------|------|------|------|
| 123 | VRAM | Video RAM | GPU의 전용 메모리 | — | Appendix B |
| 124 | 메모리 계산 공식 | Memory Calculation | 모델 학습 시 필요 메모리 산정 | $M = P(b_p + b_g + b_o) + \text{KV} + \text{Act}$ | Appendix B |
| 125 | FP32/FP16/BF16 | Floating Point Precision | 부동소수점 정밀도 (32/16비트) | FP16: 1+5+10 bit | Phase 4 Ch.11 |
| 126 | 텐서 코어 | Tensor Core | 행렬 곱셈 가속 전용 하드웨어 (NVIDIA GPU) | — | Appendix B |

---

## 용어 색인 (가나다순)

| 가나다 | 용어 | # |
|--------|------|---|
| ㄱ | 가중치 감쇠 | 85 |
| ㄱ | 게이트 크로스 어텐션 | 42 |
| ㄱ | 고유값 분해 | 3 |
| ㄱ | 그래디언트 | 6 |
| ㄱ | 그레이스풀 저하 | 115 |
| ㄱ | 귀납적 편향 | 38 |
| ㄴ | 내적 | 1 |
| ㄴ | 노름 | 5 |
| ㄷ | 동적 배칭 | 74 |
| ㄷ | 동적 해상도 | 50 |
| ㄷ | 대화 형식 | 72 |
| ㄹ | 랭크 (LoRA) | 78 |
| ㄹ | 레이어 정규화 | 18 |
| ㄹ | 리틀의 법칙 | 112 |
| ㄹ | 롤백 | 120 |
| ㄹ | 루프라인 모델 | 110 |
| ㅁ | 멀티헤드 어텐션 | 14 |
| ㅁ | 모델 레지스트리 | 121 |
| ㅁ | 문서 이해 | 53 |
| ㅂ | 바운딩 박스 | 60 |
| ㅂ | 백분위 지연 | 116 |
| ㅂ | 비식별화 | 58 |
| ㅂ | 비주얼 토큰 | 45 |
| ㅂ | 블루-그린 배포 | 117 |
| ㅅ | 소프트맥스 | 10 |
| ㅅ | 손실 스케일링 | 91 |
| ㅅ | 선형 프로젝션 | 43 |
| ㅅ | 셀프 어텐션 | 13 |
| ㅅ | 시프트 윈도우 | 35 |
| ㅇ | 아핀 변환 | 66 |
| ㅇ | 알파 (LoRA) | 79 |
| ㅇ | 양자화 | 103 |
| ㅇ | 얼리 퓨전 | 47 |
| ㅇ | 연속 배칭 | 111 |
| ㅇ | 오츠 이진화 | 71 |
| ㅇ | 온도 | 28 |
| ㅇ | 위치 인코딩 | 16 |
| ㅇ | 윈도우 어텐션 | 34 |
| ㅇ | 의료행위코드 | 57 |
| ㅇ | 이중 양자화 | 82 |
| ㅈ | 자기회귀 생성 | 26 |
| ㅈ | 잔차 연결 | 22 |
| ㅈ | 전역 수용장 | 39 |
| ㅈ | 점진적 해동 | 83 |
| ㅈ | 정규화 편집 거리 | 97 |
| ㅈ | 종횡비 버케팅 | 73 |
| ㅊ | 차별적 학습률 | 84 |
| ㅊ | 차분 프라이버시 | 65 |
| ㅊ | 청구 오류율 | 102 |
| ㅊ | 체인 룰 | 9 |
| ㅋ | 카나리 배포 | 118 |
| ㅋ | 코사인 어닐링 | 92 |
| ㅋ | 코헨 카파 | 63 |
| ㅋ | 크로스 어텐션 | 40 |
| ㅋ | 크로스 엔트로피 | 11 |
| ㅌ | 토크나이제이션 | 24 |
| ㅌ | 토큰 결합 | 48 |
| ㅌ | 토큰 병합 | 52 |
| ㅌ | 트리 편집 거리 | 100 |
| ㅌ | 텐서 코어 | 126 |
| ㅍ | 파국적 망각 | 76 |
| ㅍ | 파이프라인 병렬 | 89 |
| ㅍ | 패치 임베딩 | 31 |
| ㅍ | 편집 거리 | 94 |
| ㅍ | 피드포워드 네트워크 | 20 |
| ㅍ | 피처 정렬 | 46 |
| ㅍ | 프리픽스 캐싱 | 109 |
| ㅍ | 풀 파인튜닝 | 75 |
| ㅍ | 필드별 정확도 | 101 |
| ㅎ | 허프 변환 | 70 |
| ㅎ | 헤시안 | 8 |
| ㅎ | 호모그래피 | 67 |
| ㅎ | 혼합 정밀도 학습 | 90 |
| ㅎ | 활성 프로브 | 113 |

---

> **참고**: 이 용어 사전은 본문 작성이 진행됨에 따라 지속적으로 업데이트된다.
> 각 용어의 상세 설명은 본문 참조 링크를 따라가면 확인할 수 있다.
