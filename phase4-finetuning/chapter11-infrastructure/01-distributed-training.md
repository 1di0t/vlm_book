---
---

# 11.1 분산 학습

## 핵심 용어 박스

| 용어 | 정의 |
|------|------|
| **DDP (DistributedDataParallel)** | 각 GPU에 모델 전체 복사본을 두고, 미니배치를 분할하여 병렬 처리 후 gradient를 **All-Reduce**로 동기화하는 데이터 병렬 기법. PyTorch 표준. |
| **FSDP (FullyShardedDataParallel)** | 모델 파라미터, gradient, optimizer state를 GPU 간에 **샤딩(분할)**하여 메모리를 절약하는 PyTorch 네이티브 기법. ZeRO Stage 3와 동일 개념. |
| **DeepSpeed ZeRO** | Microsoft의 메모리 최적화 기술. Stage 1(optimizer), Stage 2(+gradient), Stage 3(+parameter) 단계별로 메모리를 분할하여 단일 GPU 부담을 줄인다. |
| **Pipeline Parallelism** | 모델의 레이어를 여러 GPU에 **순차적으로 배치**하고, 마이크로배치를 파이프라인으로 흘려보내는 모델 병렬 기법. bubble overhead가 존재한다. |
| **Tensor Parallelism** | 단일 레이어의 행렬 연산을 여러 GPU에 **분할**하여 병렬 수행하는 기법. Megatron-LM 스타일. 고속 interconnect(NVLink)가 필수다. |
| **All-Reduce** | 분산 환경에서 모든 프로세스의 텐서를 **집계(reduce)**한 뒤 결과를 **모든 프로세스에 배포(broadcast)**하는 집합 통신 연산. |
| **NCCL** | NVIDIA Collective Communications Library. GPU 간 All-Reduce, Broadcast 등 집합 통신을 최적화한 라이브러리. |

---

## 개요

대규모 모델(7B+)을 학습하려면 단일 GPU로는 메모리와 연산이 부족하다. 분산 학습은 여러 GPU에 연산과 메모리를 분배하여 이 한계를 극복한다.

분산 학습 전략은 크게 세 가지 축으로 나뉜다:

| 전략 | 분할 대상 | 대표 기법 | 통신 패턴 |
|------|-----------|-----------|-----------|
| **Data Parallelism** | 데이터(미니배치) | DDP, FSDP, ZeRO | All-Reduce (gradient) |
| **Pipeline Parallelism** | 모델 레이어 | GPipe, PipeDream | Point-to-Point (activation) |
| **Tensor Parallelism** | 레이어 내부 행렬 | Megatron-LM | All-Reduce (partial result) |

실전에서는 이 세 가지를 조합한 **3D Parallelism**을 사용한다.

---

## 수학적 원리

### 1. Data Parallelism: Gradient 동기화

$N$ 개의 GPU가 있고, 각 GPU $i$ 가 미니배치 $\mathcal{B}_i$에 대해 local gradient $\nabla \mathcal{L}_i$를 계산한다. 동기화된 전체 gradient는:

$$\nabla \mathcal{L} = \frac{1}{N_{\text{gpu}}} \sum_{i=1}^{N_{\text{gpu}}} \nabla \mathcal{L}_i$$

이 연산은 All-Reduce로 수행된다. 각 GPU가 동일한 gradient를 갖게 되므로, 동일한 파라미터 업데이트가 보장된다.

**Effective batch size**는 다음과 같다:

$$B_{\text{eff}} = B_{\text{per\_gpu}} \times N_{\text{gpu}} \times N_{\text{accum}}$$

여기서 $B_{\text{per\_gpu}}$는 GPU당 미니배치 크기, $N_{\text{accum}}$은 gradient accumulation 스텝 수다.

### 2. All-Reduce 통신량

Ring All-Reduce 알고리즘에서 $N$개 GPU, 모델 크기 $M$ bytes일 때 총 통신량은:

$$\text{Communication} = 2 \cdot \frac{N-1}{N} \times M \text{ bytes}$$

두 단계로 나뉜다:
- **Reduce-Scatter**: 각 GPU가 $\frac{M}{N}$ 크기의 청크를 $N-1$번 전송 → $\frac{N-1}{N} \times M$
- **All-Gather**: 집계된 결과를 다시 $N-1$번 전송 → $\frac{N-1}{N} \times M$

$N$이 커져도 각 GPU의 통신량은 거의 $2M$으로 수렴한다. 즉 GPU 수에 대해 **거의 선형 확장**이 가능하다.

### 3. ZeRO 메모리 분할

AdamW optimizer로 FP16 모델을 학습할 때, 단일 GPU가 보유해야 하는 메모리를 분석한다. 파라미터 수 $\Psi$일 때:

| 항목 | 바이트/파라미터 | 설명 |
|------|----------------|------|
| FP16 파라미터 | 2 | 모델 weight |
| FP16 gradient | 2 | backward 결과 |
| FP32 파라미터 사본 | 4 | optimizer용 master weight |
| FP32 1차 모멘트 ($m$) | 4 | Adam의 running mean |
| FP32 2차 모멘트 ($v$) | 4 | Adam의 running variance |
| **합계** | **16** | $16\Psi$ bytes |

7B 모델이면: $16 \times 7 \times 10^9 = 112$ GB (단일 GPU).

**ZeRO Stage 1: Optimizer State 분할**

Optimizer state (FP32 파라미터 + $m$ + $v$ = $12\Psi$ bytes)를 $N$개 GPU에 분할:

$$\text{Memory}_{\text{stage1}} = 2\Psi + 2\Psi + \frac{12\Psi}{N} = 4\Psi + \frac{12\Psi}{N}$$

$N=4$이면: $4\Psi + 3\Psi = 7\Psi$ bytes → 기존 $16\Psi$ 대비 **약 2.3배 절약**.
$N \to \infty$이면: $4\Psi$ bytes → **4배 절약**.

**ZeRO Stage 2: + Gradient 분할**

Gradient ($2\Psi$ bytes)도 분할:

$$\text{Memory}_{\text{stage2}} = 2\Psi + \frac{2\Psi}{N} + \frac{12\Psi}{N} = 2\Psi + \frac{14\Psi}{N}$$

$N \to \infty$이면: $2\Psi$ bytes → **8배 절약**.

**ZeRO Stage 3: + Parameter 분할**

파라미터 자체도 분할:

$$\text{Memory}_{\text{stage3}} = \frac{2\Psi}{N} + \frac{2\Psi}{N} + \frac{12\Psi}{N} = \frac{16\Psi}{N}$$

$N$개 GPU에 **완벽히 분산** → **$N$배 절약**. 단, forward/backward 시 필요한 파라미터를 그때그때 All-Gather해야 하므로 통신 오버헤드가 증가한다.

| Stage | GPU당 메모리 ($N \to \infty$) | 절약 배수 | 추가 통신 |
|-------|-------------------------------|-----------|-----------|
| 없음 | $16\Psi$ | 1x | 0 |
| Stage 1 | $4\Psi$ | 4x | 없음 (기존 All-Reduce 동일) |
| Stage 2 | $2\Psi$ | 8x | Reduce-Scatter만 |
| Stage 3 | $\frac{16\Psi}{N}$ | $N$x | All-Gather (forward + backward) |

### 4. Pipeline Parallelism: Bubble Ratio

$P$개의 파이프라인 스테이지(GPU), $M$개의 마이크로배치가 있을 때, GPipe 스타일의 bubble ratio는:

$$\text{Bubble Ratio} = \frac{P - 1}{P - 1 + M}$$

이 값은 전체 파이프라인 시간 중 GPU가 **유휴(idle)** 상태인 비율이다.

예를 들어 $P=4$, $M=12$이면:

$$\text{Bubble} = \frac{3}{3 + 12} = 20\%$$

Bubble을 줄이려면:
- $M$을 키운다 (마이크로배치 수 증가)
- Interleaved schedule 사용 (1F1B 등)
- $P$를 줄인다 (파이프라인 스테이지 축소)

이상적으로 $M \gg P$이면 bubble은 무시할 수 있다.

### 5. Tensor Parallelism: 행렬 분할

Transformer의 MLP 레이어 $Y = \text{GeLU}(XA)B$에서 행렬 $A$를 열 방향으로 $N$개 GPU에 분할한다:

$$A = [A_1, A_2, \ldots, A_N]$$

각 GPU $i$에서:

$$Y_i = \text{GeLU}(X A_i) B_i$$

최종 결과는 All-Reduce로 합산:

$$Y = \sum_{i=1}^{N} Y_i$$

Attention 레이어에서는 head를 GPU별로 분할하는 것이 자연스럽다. $H$개의 head, $N$개 GPU면 GPU당 $H/N$개 head를 담당한다.

통신량: 각 레이어마다 All-Reduce가 필요하므로 레이어 수에 비례하여 통신이 증가한다. NVLink(600 GB/s)급 고속 interconnect가 필수인 이유다.

---

## 코드: DDP 설정

### 기본 DDP 학습

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int):
    """분산 환경 초기화."""
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "29500")

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """분산 환경 정리."""
    dist.destroy_process_group()


def train_ddp(rank: int, world_size: int, config: dict):
    """DDP 학습 메인 함수."""
    setup_distributed(rank, world_size)

    try:
        # 모델 로드 (각 GPU에 전체 모델 복사)
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            torch_dtype=torch.bfloat16,
        ).to(rank)

        # DDP 래핑
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False,  # 사용하지 않는 파라미터가 없으면 False
            gradient_as_bucket_view=True,  # 메모리 최적화
        )

        # DistributedSampler: 각 GPU가 다른 데이터 부분집합을 받도록 보장
        train_sampler = DistributedSampler(
            config["train_dataset"],
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

        train_dataloader = DataLoader(
            config["train_dataset"],
            batch_size=config["batch_size_per_gpu"],
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        # 학습 루프
        for epoch in range(config["num_epochs"]):
            train_sampler.set_epoch(epoch)  # 매 에폭마다 셔플 시드 변경
            model.train()

            for step, batch in enumerate(train_dataloader):
                input_ids = batch["input_ids"].to(rank)
                attention_mask = batch["attention_mask"].to(rank)
                labels = batch["labels"].to(rank)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                loss.backward()  # DDP가 자동으로 gradient All-Reduce 수행

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                if rank == 0 and step % 50 == 0:
                    logger.info(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")

            # 체크포인트 저장 (rank 0만)
            if rank == 0:
                model.module.save_pretrained(f"checkpoint-epoch{epoch}")

    finally:
        cleanup_distributed()


def launch_ddp_training(config: dict):
    """DDP 학습 런처."""
    world_size = torch.cuda.device_count()
    logger.info(f"GPU {world_size}개로 DDP 학습 시작")
    mp.spawn(train_ddp, args=(world_size, config), nprocs=world_size, join=True)
```

**torchrun 실행:**

```bash
# 단일 노드, 4 GPU
torchrun --nproc_per_node=4 train.py

# 멀티 노드 (2노드, 각 4 GPU)
# 노드 0:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=10.0.0.1 --master_port=29500 train.py
# 노드 1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=10.0.0.1 --master_port=29500 train.py
```

---

## 코드: FSDP 설정

```python
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from functools import partial
import logging

logger = logging.getLogger(__name__)


def create_fsdp_model(
    model_name: str,
    rank: int,
    sharding_strategy: str = "full_shard",
    cpu_offload: bool = False,
) -> FSDP:
    """FSDP 래핑된 모델 생성."""

    # 모델을 meta device에 로드 (메모리 절약)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )

    # Mixed Precision 설정
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,    # gradient All-Reduce 시 dtype
        buffer_dtype=torch.bfloat16,
    )

    # Sharding 전략 선택
    strategy_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,       # ZeRO Stage 3
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP, # ZeRO Stage 2
        "no_shard": ShardingStrategy.NO_SHARD,           # DDP
    }
    strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

    # Auto Wrap Policy: Transformer 레이어 단위로 래핑
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    # FSDP 래핑
    fsdp_model = FSDP(
        model,
        sharding_strategy=strategy,
        mixed_precision=bf16_policy,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=rank,
        limit_all_gathers=True,  # 메모리 피크 제한
        use_orig_params=True,    # optimizer에서 원본 파라미터명 사용 가능
    )

    # 학습 가능 파라미터 확인
    if rank == 0:
        total_params = sum(p.numel() for p in fsdp_model.parameters())
        logger.info(f"FSDP 모델 파라미터: {total_params:,}")
        logger.info(f"Sharding 전략: {sharding_strategy}")

    return fsdp_model


def save_fsdp_checkpoint(model: FSDP, optimizer, epoch: int, save_dir: str):
    """FSDP 체크포인트 저장.

    FSDP는 파라미터가 분산되어 있으므로 특별한 저장 로직이 필요하다.
    """
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        StateDictType,
    )

    # Full state dict: rank 0에 모든 파라미터를 모아서 저장
    full_state_config = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_config):
        state_dict = model.state_dict()

        if torch.distributed.get_rank() == 0:
            save_path = f"{save_dir}/checkpoint-epoch{epoch}.pt"
            torch.save({"model": state_dict, "epoch": epoch}, save_path)
            logger.info(f"FSDP 체크포인트 저장: {save_path}")
```

---

## 코드: DeepSpeed 설정

### DeepSpeed ZeRO Stage 2 Config (JSON)

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

### DeepSpeed ZeRO Stage 3 Config (JSON)

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

### DeepSpeed ZeRO Stage 3 + CPU Offload Config

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": 8,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

### DeepSpeed 학습 스크립트

```python
import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def train_with_deepspeed(args, train_dataset):
    """DeepSpeed 학습 루프."""

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )

    # DeepSpeed 초기화 (optimizer, scheduler, dataloader를 자동 래핑)
    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        training_data=train_dataset,
    )

    for epoch in range(args.num_epochs):
        model_engine.train()

        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(model_engine.device)
            attention_mask = batch["attention_mask"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)

            outputs = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # DeepSpeed가 backward + gradient sync + optimizer step을 관리
            model_engine.backward(loss)
            model_engine.step()

            if model_engine.local_rank == 0 and step % 50 == 0:
                logger.info(
                    f"Epoch {epoch} Step {step} "
                    f"Loss: {loss.item():.4f} "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.2e}"
                )

        # 체크포인트 저장
        model_engine.save_checkpoint(
            save_dir=f"checkpoints/epoch-{epoch}",
            tag=f"epoch_{epoch}",
        )

    return model_engine
```

**DeepSpeed 실행:**

```bash
# ZeRO Stage 2, 4 GPU
deepspeed --num_gpus=4 train.py \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --deepspeed ds_config_zero2.json

# ZeRO Stage 3 + CPU Offload, 2 GPU
deepspeed --num_gpus=2 train.py \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --deepspeed ds_config_zero3_offload.json

# 멀티 노드
deepspeed --hostfile hostfile.txt --num_gpus=4 train.py \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --deepspeed ds_config_zero3.json
```

---

## VLM 분산 학습 시 주의사항

VLM(Vision-Language Model)은 Vision Encoder + Connector + LLM으로 구성되므로, 분산 학습에서 추가적인 고려가 필요하다.

### 1. 모듈별 Sharding 전략

| 구성요소 | 크기 (Qwen2.5-VL-7B) | 권장 Sharding | 이유 |
|----------|----------------------|---------------|------|
| Vision Encoder (ViT) | ~0.6B | FSDP wrapping 또는 freeze | 상대적으로 작음 |
| Connector (MLP) | ~0.01B | DDP (no shard) | 매우 작으므로 sharding 오버헤드 > 이득 |
| LLM Backbone | ~6.4B | FSDP full shard | 대부분의 메모리를 차지 |

### 2. FSDP Wrap Policy for VLM

```python
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    _or_policy,
    lambda_auto_wrap_policy,
)
from functools import partial


def create_vlm_wrap_policy(model_type: str = "qwen2_vl"):
    """VLM 구성요소별 FSDP wrap policy."""

    if model_type == "qwen2_vl":
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VLDecoderLayer,
        )
        # LLM decoder layer 단위로 wrapping
        llm_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen2VLDecoderLayer},
        )
        return llm_policy

    raise ValueError(f"지원하지 않는 모델: {model_type}")
```

### 3. Vision Encoder Freeze + LLM만 분산 학습

OCR 도메인에서 자주 사용하는 패턴: Vision Encoder를 freeze하고 LLM만 학습.

```python
def setup_vlm_for_distributed(model, freeze_vision: bool = True):
    """VLM 분산 학습 설정.

    Vision Encoder를 freeze하면 해당 파라미터의 gradient가 계산되지 않으므로
    통신량이 줄고 메모리도 절약된다.
    """
    if freeze_vision:
        for name, param in model.named_parameters():
            if "visual" in name or "vision" in name:
                param.requires_grad = False

    # 학습 가능 파라미터 통계
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    logger.info(f"전체: {total:,} | 학습: {trainable:,} | Frozen: {frozen:,}")
    logger.info(f"학습 비율: {trainable / total * 100:.1f}%")

    return model
```

### 4. 이미지 크기 불균등 문제

VLM은 이미지 입력의 크기가 샘플마다 다를 수 있다. DDP에서 각 GPU의 배치 크기가 달라지면 All-Reduce에서 hang이 발생한다.

```python
def vlm_collate_fn(batch: list[dict], max_image_tokens: int = 1280) -> dict:
    """VLM용 collate function.

    이미지 토큰 수를 제한하여 GPU 간 배치 크기 불균형을 완화한다.
    """
    processed = []
    for item in batch:
        image_tokens = item.get("image_token_count", 0)
        if image_tokens > max_image_tokens:
            # 이미지 리사이즈 또는 크롭
            item = _resize_image_tokens(item, max_image_tokens)
        processed.append(item)

    # 패딩으로 길이 통일
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in processed],
        batch_first=True,
        padding_value=0,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in processed],
        batch_first=True,
        padding_value=0,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [item["labels"] for item in processed],
        batch_first=True,
        padding_value=-100,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
```

### 5. 분산 학습 방식 선택 가이드

| 조건 | 권장 방식 | 이유 |
|------|-----------|------|
| 7B 모델, 4x A100 80GB | DDP 또는 ZeRO Stage 1 | 메모리 충분, 통신 최소 |
| 7B 모델, 4x A100 40GB | ZeRO Stage 2 | gradient 분할로 메모리 확보 |
| 7B 모델, 2x RTX 4090 24GB | ZeRO Stage 3 + CPU Offload | 메모리 매우 부족 |
| 70B 모델, 8x A100 80GB | ZeRO Stage 3 + Tensor Parallel | 모델이 단일 GPU에 안 들어감 |
| 70B 모델, 멀티 노드 | 3D Parallelism | TP(노드 내) + PP + DP(노드 간) |

---

## 성능 최적화 팁

### 1. 통신 오버랩

gradient 계산과 All-Reduce를 동시에 수행하면 통신 지연을 숨길 수 있다.

```python
# DDP에서 자동으로 수행됨 (bucket 단위 통신)
# FSDP에서도 backward_prefetch로 제어 가능
from torch.distributed.fsdp import BackwardPrefetch

# BACKWARD_PRE: 현재 레이어 backward 시작 전에 다음 레이어의 파라미터를 prefetch
# BACKWARD_POST: 현재 레이어 backward 완료 후에 다음 레이어를 prefetch
# 일반적으로 BACKWARD_PRE가 더 빠르지만 메모리를 약간 더 사용
```

### 2. Gradient Checkpointing + FSDP

메모리가 극도로 부족할 때, activation을 저장하지 않고 backward에서 재계산한다:

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.checkpoint import checkpoint

def apply_gradient_checkpointing(model):
    """FSDP + Gradient Checkpointing 적용."""
    model.gradient_checkpointing_enable()
    # 또는 HuggingFace Trainer에서:
    # training_args.gradient_checkpointing = True
    return model
```

메모리 절약: activation 메모리를 $O(\sqrt{L})$로 줄임 ($L$: 레이어 수). 대신 연산량이 약 33% 증가한다.

### 3. NCCL 환경변수 튜닝

```bash
# 통신 성능 최적화
export NCCL_IB_DISABLE=0          # InfiniBand 사용 (해당 시)
export NCCL_NET_GDR_LEVEL=5       # GPUDirect RDMA 레벨
export NCCL_SOCKET_IFNAME=eth0    # 네트워크 인터페이스 지정
export NCCL_DEBUG=INFO            # 디버깅 시

# P2P 통신 최적화
export NCCL_P2P_DISABLE=0         # GPU 간 P2P 활성화
export NCCL_TREE_THRESHOLD=0      # Tree All-Reduce 사용
```

---

## 용어 체크리스트

학습 후 아래 용어를 설명할 수 있는지 점검해봐라:

- [ ] **DDP**: All-Reduce 기반 gradient 동기화 방식과 DistributedSampler의 역할을 설명할 수 있는가?
- [ ] **FSDP**: ZeRO Stage 3와의 관계, ShardingStrategy 옵션별 차이를 아는가?
- [ ] **ZeRO Stage 1/2/3**: 각 단계에서 무엇을 분할하는지, 메모리 절약량을 계산할 수 있는가?
- [ ] **All-Reduce**: Ring All-Reduce의 통신량 $2(N-1)/N \times M$을 유도할 수 있는가?
- [ ] **Pipeline Parallelism**: Bubble ratio 공식과 이를 줄이는 방법을 아는가?
- [ ] **Tensor Parallelism**: MLP와 Attention에서 행렬을 어떻게 분할하는지 설명할 수 있는가?
- [ ] **Gradient Checkpointing**: 메모리와 연산의 트레이드오프를 이해하는가?
- [ ] **VLM 분산 학습**: Vision Encoder와 LLM을 다르게 처리해야 하는 이유를 아는가?
- [ ] **통신 오버랩**: backward와 All-Reduce가 어떻게 동시에 진행되는지 설명할 수 있는가?
- [ ] **3D Parallelism**: DP, PP, TP를 어떻게 조합하고 각각의 역할이 무엇인지 아는가?
