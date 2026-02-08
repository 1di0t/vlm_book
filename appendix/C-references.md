---
---

# Appendix C: 참고 자료

이 부록에서는 OCR/VLM 파인튜닝과 관련된 핵심 논문, 오픈소스 저장소, 온라인 강의, 블로그 등을 총정리한다.
논문은 카테고리별로 분류하고, 각각 [제목, 저자, 연도, 한 줄 요약, 핵심 기여] 형식으로 정리했다.

---

## C.1 핵심 논문

### C.1.1 Transformer 기초

| # | 제목 | 저자 | 연도 | 한 줄 요약 | 핵심 기여 |
|---|------|------|------|-----------|----------|
| 1 | **Attention Is All You Need** | Vaswani et al. | 2017 | Self-attention 기반 Transformer 아키텍처 제안 | RNN/CNN 없이 attention만으로 seq2seq 구현, 병렬화 극대화 |
| 2 | **BERT: Pre-training of Deep Bidirectional Transformers** | Devlin et al. | 2019 | 양방향 Transformer 사전학습으로 NLU 혁신 | Masked Language Model + Next Sentence Prediction |
| 3 | **Language Models are Unsupervised Multitask Learners (GPT-2)** | Radford et al. | 2019 | 대규모 언어 모델의 zero-shot 능력 입증 | WebText 데이터셋, autoregressive LM의 가능성 |
| 4 | **Language Models are Few-Shot Learners (GPT-3)** | Brown et al. | 2020 | 175B 파라미터 LLM으로 few-shot learning 달성 | In-context learning, scaling law 실증 |
| 5 | **Training Compute-Optimal Large Language Models (Chinchilla)** | Hoffmann et al. | 2022 | 최적 학습을 위한 모델-데이터 비율 규명 | Chinchilla scaling law: 토큰 수 = 20 × 파라미터 수 |
| 6 | **LLaMA: Open and Efficient Foundation Language Models** | Touvron et al. | 2023 | 오픈소스 LLM의 시작점 | 공개 데이터만으로 GPT-3급 성능 달성 |
| 7 | **Llama 2: Open Foundation and Fine-Tuned Chat Models** | Touvron et al. | 2023 | RLHF 적용 오픈소스 Chat LLM | Safety alignment + 상업적 사용 허용 라이선스 |

### C.1.2 Vision Transformer (ViT) 계열

| # | 제목 | 저자 | 연도 | 한 줄 요약 | 핵심 기여 |
|---|------|------|------|-----------|----------|
| 8 | **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)** | Dosovitskiy et al. | 2020 | 이미지를 패치로 분할하여 Transformer에 입력 | 비전에서 Transformer 직접 적용 가능성 입증 |
| 9 | **Training Data-Efficient Image Transformers (DeiT)** | Touvron et al. | 2021 | 데이터 효율적 ViT 학습 기법 | Knowledge distillation + augmentation 전략 |
| 10 | **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows** | Liu et al. | 2021 | 윈도우 기반 계층적 ViT | 선형 복잡도 attention, 다양한 비전 태스크에 범용 적용 |
| 11 | **Scaling Vision Transformers (ViT-22B)** | Dehghani et al. | 2023 | 22B 파라미터 ViT | ViT scaling의 한계 실험, 대규모 비전 모델 가능성 |
| 12 | **InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions** | Wang et al. | 2023 | Deformable convolution 기반 대규모 비전 모델 | CNN 기반 대안으로 ViT에 필적하는 성능 |

### C.1.3 Vision-Language Models (VLM)

| # | 제목 | 저자 | 연도 | 한 줄 요약 | 핵심 기여 |
|---|------|------|------|-----------|----------|
| 13 | **Learning Transferable Visual Models From Natural Language Supervision (CLIP)** | Radford et al. | 2021 | 텍스트-이미지 대조학습으로 범용 비전 표현 학습 | Contrastive learning, zero-shot 이미지 분류 |
| 14 | **Flamingo: a Visual Language Model for Few-Shot Learning** | Alayrac et al. | 2022 | 인터리브 이미지-텍스트 입력 처리 VLM | Perceiver Resampler, few-shot VQA/캡셔닝 |
| 15 | **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models** | Li et al. | 2023 | Frozen ViT + LLM을 Q-Former로 연결 | 효율적 비전-언어 연결, 2단계 사전학습 |
| 16 | **Visual Instruction Tuning (LLaVA)** | Liu et al. | 2023 | GPT-4 생성 데이터로 visual instruction tuning | 간단한 선형 프로젝터로 ViT와 LLM 연결, 학습 효율성 |
| 17 | **Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)** | Liu et al. | 2023 | LLaVA 개선: MLP 프로젝터 + 더 많은 데이터 | 최소한의 변경으로 SOTA 달성, 재현 가능한 학습 레시피 |
| 18 | **LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge** | Liu et al. | 2024 | 고해상도 입력 + 다양한 LLM 백본 지원 | Dynamic high-resolution, OCR 성능 대폭 향상 |
| 19 | **Qwen-VL: A Versatile Vision-Language Model** | Bai et al. | 2023 | 다목적 VLM (OCR, VQA, Grounding 등) | Bounding box 입력/출력, 다국어(중/영) 지원 |
| 20 | **Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution** | Wang et al. | 2024 | 임의 해상도 + 동적 토큰 할당 VLM | Naive Dynamic Resolution, M-RoPE, 비디오 이해 |
| 21 | **InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks** | Chen et al. | 2024 | 대규모 비전 인코더 + LLM 정렬 | InternViT-6B, progressive alignment 전략 |
| 22 | **InternVL 1.5: How Far Are We to GPT-4V?** | Chen et al. | 2024 | InternVL의 개선 버전, GPT-4V와의 격차 분석 | Dynamic high-resolution, 강력한 OCR/문서 이해 |
| 23 | **CogVLM: Visual Expert for Pretrained Language Models** | Wang et al. | 2023 | Visual expert 모듈로 비전-언어 깊은 융합 | 기존 LLM 가중치 보존하면서 비전 능력 추가 |
| 24 | **PaLI: A Jointly-Scaled Multilingual Language-Image Model** | Chen et al. | 2022 | 다국어 비전-언어 사전학습 모델 | ViT-e (4B) + mT5 결합, 109개 언어 지원 |

### C.1.4 Parameter-Efficient Fine-Tuning (PEFT)

| # | 제목 | 저자 | 연도 | 한 줄 요약 | 핵심 기여 |
|---|------|------|------|-----------|----------|
| 25 | **LoRA: Low-Rank Adaptation of Large Language Models** | Hu et al. | 2021 | 저랭크 행렬 분해로 효율적 파인튜닝 | 파라미터 0.1~1%만 학습하면서 Full FT에 근접한 성능 |
| 26 | **QLoRA: Efficient Finetuning of Quantized LLMs** | Dettmers et al. | 2023 | 4-bit 양자화 + LoRA로 메모리 극한 절약 | NF4 양자화, Double Quantization, Paged Optimizer |
| 27 | **LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning** | Hu et al. | 2023 | 다양한 어댑터 기법 통합 비교 | Series/Parallel adapter, Prefix Tuning 등 체계적 비교 |
| 28 | **DoRA: Weight-Decomposed Low-Rank Adaptation** | Liu et al. | 2024 | 가중치를 방향과 크기로 분해한 LoRA 변형 | LoRA 대비 일관된 성능 향상, 특히 low-rank에서 효과적 |
| 29 | **LoRA+: Efficient Low Rank Adaptation of Large Models** | Hayou et al. | 2024 | LoRA의 A/B 행렬에 다른 학습률 적용 | 수렴 속도 향상 + 성능 개선, 구현 변경 최소화 |

### C.1.5 OCR / Document Understanding

| # | 제목 | 저자 | 연도 | 한 줄 요약 | 핵심 기여 |
|---|------|------|------|-----------|----------|
| 30 | **Nougat: Neural Optical Understanding for Academic Documents** | Blecher et al. | 2023 | 학술 PDF를 마크다운으로 변환하는 VLM | End-to-end OCR, 수식/표 인식, Swin + mBART |
| 31 | **Donut: Document Understanding Transformer** | Kim et al. | 2022 | OCR 없는 문서 이해 Transformer | OCR-free end-to-end 아키텍처, Swin + BART |
| 32 | **Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding** | Lee et al. | 2023 | 스크린샷 파싱으로 사전학습한 VLM | Screenshot parsing pretraining, 가변 해상도 입력 |
| 33 | **DocTR: Document Text Recognition** | Mindee | 2021 | 경량 문서 텍스트 인식 프레임워크 | 2-stage (detection + recognition), 프로덕션 최적화 |
| 34 | **TrOCR: Transformer-based Optical Character Recognition** | Li et al. | 2023 | 사전학습 Transformer 기반 OCR | BEiT encoder + GPT-2 decoder, end-to-end |
| 35 | **GOT: General OCR Theory** | Wei et al. | 2024 | 범용 OCR 이론 + 통합 모델 | Scene text, document, formula 등 통합 처리 |
| 36 | **TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document** | Liu et al. | 2024 | OCR-free 대형 멀티모달 문서 이해 모델 | Shifted window attention, token resampling |

### C.1.6 Efficient Inference / Serving

| # | 제목 | 저자 | 연도 | 한 줄 요약 | 핵심 기여 |
|---|------|------|------|-----------|----------|
| 37 | **Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)** | Kwon et al. | 2023 | 페이지 기반 KV Cache 관리로 추론 효율 극대화 | PagedAttention: OS 가상 메모리 개념을 KV cache에 적용 |
| 38 | **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** | Dao et al. | 2022 | IO-aware attention 구현으로 속도/메모리 동시 개선 | Tiling + kernel fusion, HBM 접근 최소화 |
| 39 | **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** | Dao | 2023 | FlashAttention 개선: 더 나은 병렬화 | 비대칭 work partitioning, sequence parallel |
| 40 | **SGLang: Efficient Execution of Structured Language Model Programs** | Zheng et al. | 2024 | 구조화된 LLM 프로그램 실행 엔진 | RadixAttention, compressed FSM, 배치 최적화 |
| 41 | **FlashDecoding++: Faster Large Language Model Inference on GPUs** | Hong et al. | 2023 | 디코딩 단계 최적화 | Split-K attention, softmax 최적화 |

### C.1.7 Quantization

| # | 제목 | 저자 | 연도 | 한 줄 요약 | 핵심 기여 |
|---|------|------|------|-----------|----------|
| 42 | **GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers** | Frantar et al. | 2022 | Hessian 기반 정확한 사후 양자화 | OBQ 기반 레이어별 양자화, 3-4bit에서 성능 유지 |
| 43 | **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** | Lin et al. | 2023 | Activation 분포 기반 가중치 양자화 | 중요 채널 보호, hardware-efficient 커널 |
| 44 | **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models** | Xiao et al. | 2023 | Activation smoothing으로 양자화 난이도 완화 | Activation outlier를 weight로 이전, W8A8 양자화 |
| 45 | **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** | Dettmers et al. | 2022 | Outlier 분리 기반 8-bit 양자화 | Mixed-precision decomposition, bitsandbytes |
| 46 | **AQLM: Additive Quantization for Language Models** | Egiazarian et al. | 2024 | Additive quantization으로 극한 압축 | 2-bit 양자화에서도 준수한 성능 유지 |

### C.1.8 학습 최적화 / Scaling

| # | 제목 | 저자 | 연도 | 한 줄 요약 | 핵심 기여 |
|---|------|------|------|-----------|----------|
| 47 | **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (DeepSpeed)** | Rajbhandari et al. | 2020 | 분산 학습 메모리 최적화 | Optimizer/Gradient/Parameter partitioning 3단계 |
| 48 | **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** | Shoeybi et al. | 2019 | 모델 병렬화 기반 대규모 LM 학습 | Tensor parallelism, efficient pipeline parallelism |
| 49 | **Scaling Data-Constrained Language Models** | Muennighoff et al. | 2023 | 데이터 제한 상황에서의 스케일링 전략 | 데이터 반복 학습의 효과와 한계 분석 |
| 50 | **GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection** | Zhao et al. | 2024 | 그래디언트 저랭크 투영으로 메모리 절약 | Full-rank 학습과 유사한 성능, 8-bit optimizer 호환 |

---

## C.2 오픈소스 저장소

### C.2.1 모델

| 프로젝트 | GitHub / HuggingFace | 설명 |
|---------|---------------------|------|
| **Qwen2-VL** | [QwenLM/Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) | 알리바바의 VLM. 임의 해상도, 비디오 지원 |
| **LLaVA** | [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) | 대표적 오픈소스 VLM. 학습 레시피 공개 |
| **LLaVA-NeXT** | [LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) | LLaVA 후속 버전. 고해상도, OCR 강화 |
| **InternVL** | [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL) | 대규모 비전 인코더 기반 VLM |
| **CogVLM** | [THUDM/CogVLM](https://github.com/THUDM/CogVLM) | Visual expert 기반 VLM |
| **Nougat** | [facebookresearch/nougat](https://github.com/facebookresearch/nougat) | 학술 PDF OCR |
| **Donut** | [clovaai/donut](https://github.com/clovaai/donut) | OCR-free 문서 이해 |
| **GOT-OCR** | [Ucas-HaoranWei/GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0) | 범용 OCR 모델 |
| **Llama 3** | [meta-llama/llama3](https://github.com/meta-llama/llama3) | Meta의 오픈소스 LLM |
| **Mistral** | [mistralai/mistral-src](https://github.com/mistralai/mistral-src) | 효율적 오픈소스 LLM |
| **Phi-3-Vision** | [microsoft/Phi-3-vision](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) | MS의 소형 VLM |
| **Florence-2** | [microsoft/Florence-2](https://huggingface.co/microsoft/Florence-2-large) | MS의 범용 비전 모델 |

### C.2.2 프레임워크 / 라이브러리

| 프로젝트 | GitHub | 설명 |
|---------|--------|------|
| **HuggingFace Transformers** | [huggingface/transformers](https://github.com/huggingface/transformers) | 사실상 표준 모델 허브. 모델 로딩/학습/추론 |
| **PEFT** | [huggingface/peft](https://github.com/huggingface/peft) | LoRA, QLoRA 등 PEFT 기법 구현 |
| **TRL** | [huggingface/trl](https://github.com/huggingface/trl) | SFTTrainer, DPO, RLHF 학습 |
| **Accelerate** | [huggingface/accelerate](https://github.com/huggingface/accelerate) | 분산 학습 추상화 레이어 |
| **vLLM** | [vllm-project/vllm](https://github.com/vllm-project/vllm) | PagedAttention 기반 고성능 추론 |
| **SGLang** | [sgl-project/sglang](https://github.com/sgl-project/sglang) | 구조화된 LLM 프로그램 실행 |
| **DeepSpeed** | [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) | ZeRO 최적화, 분산 학습 |
| **Flash Attention** | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | IO-aware attention 커널 |
| **bitsandbytes** | [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | 양자화 라이브러리 (4-bit, 8-bit) |
| **AutoGPTQ** | [AutoGPTQ/AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) | GPTQ 양자화 구현 |
| **AutoAWQ** | [casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ) | AWQ 양자화 구현 |
| **LLaMA-Factory** | [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 100+ LLM/VLM 통합 파인튜닝 프레임워크 |
| **Axolotl** | [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl) | 다양한 파인튜닝 설정 지원 |
| **Unsloth** | [unslothai/unsloth](https://github.com/unslothai/unsloth) | 2배 빠른 LoRA/QLoRA 학습 |
| **LitGPT** | [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt) | Lightning 기반 GPT 학습/추론 |

### C.2.3 데이터셋 / 벤치마크

| 데이터셋 | 링크 | 설명 |
|---------|------|------|
| **DocVQA** | [HF: datasets/docvqa](https://huggingface.co/datasets/lmms-lab/DocVQA) | 문서 VQA 벤치마크 |
| **ChartQA** | [HF: datasets/chartqa](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) | 차트 이해 VQA |
| **OCRBench** | [GitHub](https://github.com/Yuliang-Liu/MultimodalOCR) | VLM OCR 성능 종합 벤치마크 |
| **TextVQA** | [textvqa.org](https://textvqa.org/) | 이미지 내 텍스트 읽기 VQA |
| **SROIE** | [ICDAR 2019](https://rrc.cvc.uab.es/?ch=13) | 영수증 OCR 데이터셋 |
| **IAM Handwriting** | [fki.tic.heia-fr.ch](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) | 필기체 인식 데이터셋 |
| **MMMU** | [HF: datasets/mmmu](https://huggingface.co/datasets/MMMU/MMMU) | 멀티모달 대학 수준 벤치마크 |
| **MathVista** | [HF: datasets/mathvista](https://huggingface.co/datasets/AI4Math/MathVista) | 수학 + 비전 통합 벤치마크 |
| **ShareGPT4V** | [HF: datasets/sharegpt4v](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V) | GPT-4V 생성 고품질 캡션 데이터 |

---

## C.3 온라인 강의

### C.3.1 대학 강의

| 강의 | 대학 | 강사 | 주제 | 링크 |
|------|------|------|------|------|
| **CS224N: NLP with Deep Learning** | Stanford | Chris Manning | NLP 전반, Transformer, LLM | [cs224n.stanford.edu](https://web.stanford.edu/class/cs224n/) |
| **CS231N: CNNs for Visual Recognition** | Stanford | Fei-Fei Li, Andrej Karpathy | 컴퓨터 비전, CNN, ViT | [cs231n.stanford.edu](http://cs231n.stanford.edu/) |
| **CS25: Transformers United** | Stanford | 여러 연사 | Transformer 심층 세미나 | [web.stanford.edu/class/cs25](https://web.stanford.edu/class/cs25/) |
| **CS324: Large Language Models** | Stanford | Percy Liang et al. | LLM 전반 (학습, 평가, 사회적 영향) | [stanford-cs324.github.io](https://stanford-cs324.github.io/winter2022/) |
| **CS336: Language Modeling from Scratch** | Stanford | Percy Liang, Tatsunori Hashimoto | LM을 처음부터 구현 | [stanford-cs336.github.io](https://stanford-cs336.github.io/spring2024/) |
| **11-785: Introduction to Deep Learning** | CMU | Bhiksha Raj | 딥러닝 이론 + 실습 | [deeplearning.cs.cmu.edu](https://deeplearning.cs.cmu.edu/) |
| **MIT 6.S191: Intro to Deep Learning** | MIT | Alexander Amini | 딥러닝 입문, 최신 트렌드 | [introtodeeplearning.com](http://introtodeeplearning.com/) |

### C.3.2 온라인 코스

| 강의 | 플랫폼 | 특징 |
|------|--------|------|
| **Practical Deep Learning for Coders** | fast.ai | 코드 중심, top-down 방식 교육 |
| **Deep Learning Specialization** | Coursera (Andrew Ng) | 딥러닝 기초부터 체계적 |
| **HuggingFace NLP Course** | HuggingFace | Transformers 라이브러리 실습 중심 |
| **HuggingFace Diffusion Course** | HuggingFace | 생성 모델 실습 |
| **Full Stack LLM Bootcamp** | FSDL | LLM 애플리케이션 개발 전과정 |
| **LLM University** | Cohere | LLM 개념부터 배포까지 |

---

## C.4 블로그 / 튜토리얼

### C.4.1 영문 블로그

| 블로그 | 저자/조직 | 대표 콘텐츠 | 링크 |
|--------|---------|------------|------|
| **The Illustrated Transformer** | Jay Alammar | Transformer 시각화 설명 (최고의 입문 자료) | [jalammar.github.io](https://jalammar.github.io/illustrated-transformer/) |
| **The Illustrated GPT-2** | Jay Alammar | GPT-2 아키텍처 시각화 | [jalammar.github.io](https://jalammar.github.io/illustrated-gpt2/) |
| **Lil'Log** | Lilian Weng (OpenAI) | LLM, Attention, Agent 등 심층 분석 | [lilianweng.github.io](https://lilianweng.github.io/) |
| **HuggingFace Blog** | HuggingFace | 최신 모델/기법 튜토리얼 | [huggingface.co/blog](https://huggingface.co/blog) |
| **Sebastian Raschka** | Sebastian Raschka | LLM 학습/파인튜닝 심층 분석 | [magazine.sebastianraschka.com](https://magazine.sebastianraschka.com/) |
| **The Gradient** | 커뮤니티 | AI 연구 심층 리뷰 | [thegradient.pub](https://thegradient.pub/) |
| **Distill.pub** | 커뮤니티 | 인터랙티브 ML 시각화 (업데이트 중단) | [distill.pub](https://distill.pub/) |
| **Andrej Karpathy (YouTube)** | Andrej Karpathy | "Let's build GPT from scratch" 등 | [youtube.com/@andrejkarpathy](https://www.youtube.com/@andrejkarpathy) |
| **Chip Huyen** | Chip Huyen | MLOps, LLM 시스템 설계 | [huyenchip.com](https://huyenchip.com/) |
| **Cameron Wolfe** | Cameron Wolfe | 논문 리뷰 + 실용적 해설 | [cameronrwolfe.substack.com](https://cameronrwolfe.substack.com/) |

### C.4.2 핵심 블로그 포스트 (필독)

| 제목 | 저자 | 주제 | 링크 |
|------|------|------|------|
| **The Annotated Transformer** | Harvard NLP | Transformer 코드 라인별 해설 | [nlp.seas.harvard.edu](https://nlp.seas.harvard.edu/annotated-transformer/) |
| **nanoGPT** | Andrej Karpathy | GPT를 처음부터 구현하는 미니멀 코드 | [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) |
| **A Visual Guide to Quantization** | Maarten Grootendorst | 양자화 시각적 가이드 | [newsletter.maartengrootendorst.com](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) |
| **RLHF: Reinforcement Learning from Human Feedback** | HuggingFace | RLHF 파이프라인 해설 | [huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf) |
| **Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA** | HuggingFace | QLoRA 실습 가이드 | [huggingface.co/blog/4bit-transformers-bitsandbytes](https://huggingface.co/blog/4bit-transformers-bitsandbytes) |
| **A Gentle Introduction to 8-bit Matrix Multiplication** | HuggingFace | 8-bit 양자화 원리 | [huggingface.co/blog/hf-bitsandbytes-integration](https://huggingface.co/blog/hf-bitsandbytes-integration) |
| **How to Fine-Tune LLMs in 2024 with Hugging Face** | Philipp Schmid | 최신 파인튜닝 가이드 | [philschmid.de](https://www.philschmid.de/) |
| **Mixture of Experts Explained** | HuggingFace | MoE 아키텍처 해설 | [huggingface.co/blog/moe](https://huggingface.co/blog/moe) |

---

## C.5 한국어 자료

### C.5.1 기업 기술 블로그

| 블로그 | 조직 | 대표 콘텐츠 | 링크 |
|--------|------|------------|------|
| **NAVER D2** | 네이버 | 검색, NLP, 추천 시스템 기술 | [d2.naver.com](https://d2.naver.com/home) |
| **NAVER CLOVA AI** | 네이버 클로바 | OCR, 비전, 음성 AI 기술 | [clova.ai/research](https://clova-ai.github.io/) |
| **Kakao AI Report** | 카카오 | AI 기술 트렌드 분석 | [tech.kakao.com](https://tech.kakao.com/) |
| **NCSOFT AI** | 엔씨소프트 | NLP, 대화 AI, LLM | [ncsoft.github.io](https://ncsoft.github.io/) |
| **Upstage Tech Blog** | 업스테이지 | Document AI, OCR, LLM | [upstage.ai/blog](https://www.upstage.ai/blog) |
| **DeepNatural AI Blog** | 딥내추럴 | 데이터 라벨링, AI 서비스 | [deepnatural.ai](https://www.deepnatural.ai/) |
| **Riiid Tech Blog** | 뤼이드 | 교육 AI, 지식 추적 | [riiid.com](https://riiid.com/) |
| **Lablup Tech Blog** | 래블업 | MLOps, Backend.AI | [blog.lablup.com](https://blog.lablup.com/) |

### C.5.2 한국어 오픈소스 / 커뮤니티

| 프로젝트/커뮤니티 | 설명 | 링크 |
|-----------------|------|------|
| **KoAlpaca** | 한국어 Alpaca 데이터셋 + 모델 | [github.com/Beomi/KoAlpaca](https://github.com/Beomi/KoAlpaca) |
| **KULLM (구름)** | 고려대 한국어 LLM | [github.com/nlpai-lab/KULLM](https://github.com/nlpai-lab/KULLM) |
| **Polyglot-Ko** | 한국어 특화 다국어 LLM | [github.com/EleutherAI/polyglot](https://github.com/EleutherAI/polyglot) |
| **Korean-LLM-Leaderboard** | 한국어 LLM 벤치마크 리더보드 | [github.com/Beomi](https://github.com/Beomi) |
| **모두의 MLOps** | MLOps 한국어 교재 | [mlops-for-all.github.io](https://mlops-for-all.github.io/) |
| **TensorFlow Korea** | TF 한국어 커뮤니티 | [tensorflow.blog](https://tensorflow.blog/) |
| **PyTorch Korea** | PyTorch 한국어 튜토리얼 | [tutorials.pytorch.kr](https://tutorials.pytorch.kr/) |
| **AI Korea (Facebook)** | 한국 AI 커뮤니티 | Facebook Group |

### C.5.3 한국어 강의 / 교재

| 자료 | 저자/기관 | 특징 | 링크 |
|------|---------|------|------|
| **딥러닝을 이용한 자연어 처리 입문** | Won Joon Yoo | NLP 전반 한국어 교재 (무료 온라인) | [wikidocs.net/book/2155](https://wikidocs.net/book/2155) |
| **PyTorch로 시작하는 딥러닝 입문** | Won Joon Yoo | PyTorch 기초 한국어 교재 | [wikidocs.net/book/2788](https://wikidocs.net/book/2788) |
| **핸즈온 머신러닝** | Aurélien Géron (한국어판) | ML/DL 종합 교재 | 한빛미디어 |
| **트랜스포머를 활용한 자연어 처리** | Lewis Tunstall et al. (한국어판) | HuggingFace 공식 교재 | 한빛미디어 |
| **LangChain으로 구현하는 LLM** | 서지영 | LLM 애플리케이션 실습 | 위키독스 |

---

## C.6 유용한 도구 / 서비스

| 도구 | 용도 | 링크 |
|------|------|------|
| **Weights & Biases** | 실험 추적, 하이퍼파라미터 관리 | [wandb.ai](https://wandb.ai/) |
| **HuggingFace Hub** | 모델/데이터셋 공유 플랫폼 | [huggingface.co](https://huggingface.co/) |
| **Gradio** | 빠른 ML 데모 UI 제작 | [gradio.app](https://gradio.app/) |
| **Streamlit** | 데이터 앱 프레임워크 | [streamlit.io](https://streamlit.io/) |
| **Label Studio** | 데이터 라벨링 도구 | [labelstud.io](https://labelstud.io/) |
| **DVC (Data Version Control)** | 데이터/모델 버전 관리 | [dvc.org](https://dvc.org/) |
| **MLflow** | ML 실험 관리 + 모델 레지스트리 | [mlflow.org](https://mlflow.org/) |
| **Triton Inference Server** | NVIDIA 고성능 추론 서버 | [github.com/triton-inference-server](https://github.com/triton-inference-server/server) |
| **BentoML** | ML 모델 서빙 프레임워크 | [bentoml.com](https://www.bentoml.com/) |
| **LitServe** | Lightning 기반 모델 서빙 | [github.com/Lightning-AI/LitServe](https://github.com/Lightning-AI/LitServe) |

---

## C.7 논문 검색 / 추적 도구

| 도구 | 설명 | 링크 |
|------|------|------|
| **arXiv** | 프리프린트 서버 (AI 논문 대부분 여기 올라옴) | [arxiv.org](https://arxiv.org/) |
| **Semantic Scholar** | AI 기반 논문 검색 + 인용 추적 | [semanticscholar.org](https://www.semanticscholar.org/) |
| **Papers with Code** | 논문 + 코드 + 벤치마크 통합 | [paperswithcode.com](https://paperswithcode.com/) |
| **Connected Papers** | 논문 관계 시각화 | [connectedpapers.com](https://www.connectedpapers.com/) |
| **Google Scholar** | 논문 검색 + 인용 추적 | [scholar.google.com](https://scholar.google.com/) |
| **Daily Papers (HuggingFace)** | 매일 주요 AI 논문 큐레이션 | [huggingface.co/papers](https://huggingface.co/papers) |
| **AK's Daily Paper** | AK (@_akhaliq)의 논문 큐레이션 | Twitter/X |
| **AI Newsletter (The Batch)** | Andrew Ng의 주간 AI 뉴스레터 | [deeplearning.ai/the-batch](https://www.deeplearning.ai/the-batch/) |

---

## C.8 읽기 순서 추천

OCR/VLM 파인튜닝을 처음 접하는 경우, 아래 순서로 읽는 걸 추천한다.

### 기초 단계 (1~2주)
1. **The Illustrated Transformer** (Jay Alammar) - Transformer 직관적 이해
2. **Attention Is All You Need** (Vaswani et al. 2017) - 원논문
3. **ViT** (Dosovitskiy et al. 2020) - 비전 Transformer 기초
4. **CLIP** (Radford et al. 2021) - 비전-언어 연결의 시작

### 핵심 단계 (2~3주)
5. **LLaVA** (Liu et al. 2023) - VLM 기본 구조 이해
6. **LoRA** (Hu et al. 2021) - PEFT의 핵심
7. **QLoRA** (Dettmers et al. 2023) - 실용적 파인튜닝
8. **Qwen2-VL** (Wang et al. 2024) - 최신 VLM 아키텍처

### 심화 단계 (3~4주)
9. **FlashAttention** (Dao et al. 2022) - 효율적 어텐션
10. **vLLM/PagedAttention** (Kwon et al. 2023) - 추론 최적화
11. **Nougat** (Blecher et al. 2023) - 문서 OCR 특화
12. **GPTQ** / **AWQ** - 양자화 기법

### 실전 단계 (지속)
- HuggingFace Blog의 최신 튜토리얼 팔로우
- Papers with Code에서 벤치마크 동향 추적
- LLaMA-Factory / Axolotl 등으로 실제 파인튜닝 실습
