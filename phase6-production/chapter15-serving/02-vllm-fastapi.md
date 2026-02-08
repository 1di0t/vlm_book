---
---

# 15.2 vLLM + FastAPI 프로덕션 서빙

vLLM은 PagedAttention 기반의 고성능 LLM 서빙 엔진이다. FastAPI와 결합하면 비동기 처리, OpenAI-compatible API, 스트리밍 응답을 모두 지원하는 프로덕션급 서빙 시스템을 구축할 수 있다.

---

> **핵심 용어 박스**
>
> | 용어 | 정의 |
> |------|------|
> | **AsyncLLMEngine** | vLLM의 비동기 추론 엔진. 이벤트 루프 기반으로 다수 요청을 동시 처리 |
> | **OpenAI-compatible API** | OpenAI의 `/v1/chat/completions` 포맷을 호환하는 API 인터페이스 |
> | **Streaming Response** | 토큰이 생성될 때마다 실시간으로 클라이언트에 전송하는 방식 (SSE) |
> | **PagedAttention** | KV Cache를 OS의 가상 메모리처럼 페이지 단위로 관리하여 메모리 낭비를 줄이는 기법 |
> | **SamplingParams** | temperature, top_p, max_tokens 등 텍스트 생성 파라미터를 캡슐화한 객체 |
> | **Multimodal Input** | 이미지 + 텍스트를 동시에 모델에 입력하는 방식. VLM 서빙의 핵심 |
> | **Tensor Parallelism** | 모델의 가중치를 여러 GPU에 분할하여 병렬 추론하는 기법 |

---

## 15.2.1 vLLM AsyncLLMEngine의 구조

### PagedAttention 수학적 원리

기존 방식에서 KV Cache는 시퀀스마다 연속된 메모리 블록을 할당한다. 최대 시퀀스 길이에 맞춰 미리 할당하므로 낭비가 심하다.

$$
\text{Memory}_{\text{naive}} = N_{\text{seq}} \times L_{\max} \times 2 \times d_{\text{model}} \times n_{\text{layers}} \times \text{sizeof(dtype)}
$$

PagedAttention은 고정 크기 블록(page)으로 KV Cache를 관리한다:

$$
\text{Memory}_{\text{paged}} = N_{\text{blocks\_used}} \times B_{\text{size}} \times 2 \times d_{\text{model}} \times n_{\text{layers}} \times \text{sizeof(dtype)}
$$

메모리 효율 개선:

$$
\text{Efficiency} = \frac{\text{Memory}_{\text{naive}} - \text{Memory}_{\text{paged}}}{\text{Memory}_{\text{naive}}} \approx 1 - \frac{\bar{L}}{L_{\max}}
$$

여기서 $\bar{L}$은 평균 시퀀스 길이, $L_{\max}$는 최대 시퀀스 길이다. 의료 문서 OCR에서 문서 길이 편차가 크면($\bar{L} \ll L_{\max}$) 효율이 극대화된다.

---

## 15.2.2 FastAPI + vLLM 통합 서버

### 프로덕션 서빙 코드

```python
"""
medical_ocr_server.py
의료 문서 OCR 프로덕션 서빙 서버 (vLLM + FastAPI)
"""

import os
import uuid
import time
import asyncio
import logging
import base64
from io import BytesIO
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs


# ============================================================
# 설정
# ============================================================

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("medical-ocr-server")


class ServerConfig:
    """환경 변수 기반 서버 설정"""
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
    TENSOR_PARALLEL_SIZE: int = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
    GPU_MEMORY_UTILIZATION: float = float(os.getenv("GPU_MEMORY_UTIL", "0.90"))
    MAX_MODEL_LEN: int = int(os.getenv("MAX_MODEL_LEN", "8192"))
    MAX_NUM_SEQS: int = int(os.getenv("MAX_NUM_SEQS", "32"))
    API_KEY: str = os.getenv("API_KEY", "")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


config = ServerConfig()


# ============================================================
# 요청/응답 스키마
# ============================================================

class OCRRequest(BaseModel):
    """OCR 요청 스키마"""
    image_base64: Optional[str] = Field(None, description="Base64 인코딩된 이미지")
    image_url: Optional[str] = Field(None, description="이미지 URL")
    prompt: str = Field(
        default="이 의료 문서의 모든 텍스트를 정확하게 추출하라.",
        description="OCR 프롬프트"
    )
    max_tokens: int = Field(default=4096, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = Field(default=False, description="스트리밍 응답 여부")
    request_id: Optional[str] = Field(default=None)


class OCRResponse(BaseModel):
    """OCR 응답 스키마"""
    request_id: str
    text: str
    processing_time_ms: float
    token_count: int
    model: str


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    model_loaded: bool
    gpu_available: bool
    active_requests: int


# ============================================================
# vLLM 엔진 초기화
# ============================================================

engine: Optional[AsyncLLMEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: 서버 시작/종료 시 엔진 관리"""
    global engine
    logger.info(f"모델 로딩 시작: {config.MODEL_NAME}")

    engine_args = AsyncEngineArgs(
        model=config.MODEL_NAME,
        tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        max_model_len=config.MAX_MODEL_LEN,
        max_num_seqs=config.MAX_NUM_SEQS,
        trust_remote_code=True,
        dtype="auto",
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info("모델 로딩 완료")

    yield

    logger.info("서버 종료, 리소스 해제")
    engine = None


# ============================================================
# FastAPI 앱
# ============================================================

app = FastAPI(
    title="Medical Document OCR API",
    version="1.0.0",
    description="의료 문서 OCR 서빙 API (vLLM 기반)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ============================================================
# 인증 미들웨어
# ============================================================

async def verify_api_key(authorization: Optional[str] = Header(None)):
    """API 키 검증"""
    if not config.API_KEY:
        return  # 개발 모드: API 키 없으면 패스

    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization 헤더 없음")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer 토큰 형식이 아님")

    token = authorization.split(" ", 1)[1]
    if token != config.API_KEY:
        raise HTTPException(status_code=403, detail="유효하지 않은 API 키")


# ============================================================
# 이미지 처리
# ============================================================

def decode_image(image_base64: str) -> Image.Image:
    """Base64 이미지 디코딩 및 검증"""
    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes))

        # 이미지 크기 제한 (의료 문서: 최대 4096x4096)
        max_dim = 4096
        if image.width > max_dim or image.height > max_dim:
            ratio = min(max_dim / image.width, max_dim / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            logger.info(f"이미지 리사이즈: {image.size}")

        return image.convert("RGB")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 디코딩 실패: {str(e)}")


# ============================================================
# OCR 엔드포인트
# ============================================================

@app.post("/v1/ocr", response_model=OCRResponse, dependencies=[Depends(verify_api_key)])
async def ocr_endpoint(request: OCRRequest):
    """
    동기 OCR 엔드포인트.
    전체 결과를 생성한 후 한 번에 반환한다.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로딩 중")

    if not request.image_base64 and not request.image_url:
        raise HTTPException(status_code=400, detail="image_base64 또는 image_url 필수")

    request_id = request.request_id or str(uuid.uuid4())
    start_time = time.time()

    # 이미지 디코딩
    image = decode_image(request.image_base64)

    # vLLM 멀티모달 입력 구성
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=0.95 if request.temperature > 0 else 1.0,
    )

    # Qwen2.5-VL 형식의 멀티모달 프롬프트
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": request.prompt},
            ],
        }
    ]

    # 비동기 추론
    results_generator = engine.generate(
        prompt={"messages": messages},
        sampling_params=sampling_params,
        request_id=request_id,
    )

    final_output = None
    async for output in results_generator:
        final_output = output

    if final_output is None:
        raise HTTPException(status_code=500, detail="추론 실패")

    generated_text = final_output.outputs[0].text
    token_count = len(final_output.outputs[0].token_ids)
    elapsed_ms = (time.time() - start_time) * 1000

    logger.info(
        f"[{request_id}] OCR 완료: {token_count} tokens, {elapsed_ms:.1f}ms"
    )

    return OCRResponse(
        request_id=request_id,
        text=generated_text,
        processing_time_ms=round(elapsed_ms, 2),
        token_count=token_count,
        model=config.MODEL_NAME,
    )


@app.post("/v1/ocr/stream", dependencies=[Depends(verify_api_key)])
async def ocr_stream_endpoint(request: OCRRequest):
    """
    스트리밍 OCR 엔드포인트.
    Server-Sent Events(SSE)로 토큰 단위 실시간 전송.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로딩 중")

    if not request.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 필수")

    request_id = request.request_id or str(uuid.uuid4())
    image = decode_image(request.image_base64)

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": request.prompt},
            ],
        }
    ]

    async def generate_stream():
        previous_text = ""
        async for output in engine.generate(
            prompt={"messages": messages},
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            current_text = output.outputs[0].text
            delta = current_text[len(previous_text):]
            if delta:
                import json
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": delta}, "index": 0}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                previous_text = current_text

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": request_id,
        },
    )


# ============================================================
# 파일 업로드 엔드포인트
# ============================================================

@app.post("/v1/ocr/upload", response_model=OCRResponse, dependencies=[Depends(verify_api_key)])
async def ocr_upload_endpoint(
    file: UploadFile = File(...),
    prompt: str = "이 의료 문서의 모든 텍스트를 정확하게 추출하라.",
    max_tokens: int = 4096,
):
    """파일 업로드 기반 OCR 엔드포인트"""
    allowed_types = {"image/png", "image/jpeg", "image/tiff"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식: {file.content_type}"
        )

    # 파일 크기 제한 (20MB)
    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="파일 크기 20MB 초과")

    image_base64 = base64.b64encode(contents).decode("utf-8")
    request = OCRRequest(
        image_base64=image_base64,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    return await ocr_endpoint(request)


# ============================================================
# 헬스 체크
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """기본 헬스 체크"""
    return HealthResponse(
        status="ok" if engine is not None else "loading",
        model_loaded=engine is not None,
        gpu_available=True,  # 실제로는 torch.cuda.is_available() 확인
        active_requests=0,   # 실제로는 엔진에서 조회
    )


# ============================================================
# OpenAI-compatible 엔드포인트
# ============================================================

class ChatMessage(BaseModel):
    role: str
    content: str | list


class ChatCompletionRequest(BaseModel):
    model: str = "medical-ocr"
    messages: list[ChatMessage]
    max_tokens: int = 4096
    temperature: float = 0.0
    stream: bool = False


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions 엔드포인트.
    기존 OpenAI SDK를 사용하는 클라이언트와 호환된다.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="모델 로딩 중")

    request_id = str(uuid.uuid4())
    start_time = time.time()

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    # 멀티모달 메시지 파싱
    parsed_messages = []
    for msg in request.messages:
        if isinstance(msg.content, str):
            parsed_messages.append({"role": msg.role, "content": msg.content})
        else:
            parsed_messages.append({"role": msg.role, "content": msg.content})

    results_generator = engine.generate(
        prompt={"messages": parsed_messages},
        sampling_params=sampling_params,
        request_id=request_id,
    )

    final_output = None
    async for output in results_generator:
        final_output = output

    if final_output is None:
        raise HTTPException(status_code=500, detail="추론 실패")

    generated_text = final_output.outputs[0].text
    token_count = len(final_output.outputs[0].token_ids)
    elapsed = time.time() - start_time

    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": config.MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": generated_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": token_count,
            "total_tokens": token_count,
        },
    }


# ============================================================
# 진입점
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "medical_ocr_server:app",
        host=config.HOST,
        port=config.PORT,
        workers=1,  # GPU 모델은 단일 워커
        log_level="info",
    )
```

---

## 15.2.3 Dockerfile

```dockerfile
# Dockerfile
# 의료 문서 OCR 서빙 이미지

# ---------- Stage 1: Base ----------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ---------- Stage 2: Dependencies ----------
FROM base AS deps

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ---------- Stage 3: Application ----------
FROM deps AS app

WORKDIR /app

# 비특권 사용자 생성
RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY --chown=appuser:appuser . /app/

# 헬스 체크 스크립트
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER appuser

EXPOSE 8000

# Gunicorn + Uvicorn 워커로 실행
CMD ["python", "-m", "uvicorn", "medical_ocr_server:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
```

### requirements.txt

```
# requirements.txt
vllm>=0.6.0
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
python-multipart>=0.0.9
Pillow>=10.0.0
pydantic>=2.0.0
prometheus-client>=0.20.0
```

---

## 15.2.4 Docker Compose

```yaml
# docker-compose.yml
# 의료 문서 OCR 서빙 스택

version: "3.9"

services:
  # ---- 모델 서버 ----
  ocr-server:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: medical-ocr-server
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}
      - TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
      - GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.90}
      - MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
      - MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
      - API_KEY=${API_KEY}
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - model-cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 180s
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
    networks:
      - ocr-network

  # ---- Nginx 리버스 프록시 ----
  nginx:
    image: nginx:1.27-alpine
    container_name: ocr-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      ocr-server:
        condition: service_healthy
    restart: unless-stopped
    networks:
      - ocr-network

  # ---- Redis (캐싱 + 큐) ----
  redis:
    image: redis:7-alpine
    container_name: ocr-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: >
      redis-server
      --maxmemory 2gb
      --maxmemory-policy allkeys-lru
      --requirepass ${REDIS_PASSWORD:-changeme}
    restart: unless-stopped
    networks:
      - ocr-network

  # ---- Prometheus (메트릭 수집) ----
  prometheus:
    image: prom/prometheus:v2.53.0
    container_name: ocr-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    restart: unless-stopped
    networks:
      - ocr-network

  # ---- Grafana (대시보드) ----
  grafana:
    image: grafana/grafana:11.1.0
    container_name: ocr-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
    volumes:
      - grafana-data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - ocr-network

volumes:
  model-cache:
  redis-data:
  prometheus-data:
  grafana-data:

networks:
  ocr-network:
    driver: bridge
```

---

## 15.2.5 Nginx 설정

```nginx
# nginx.conf
# 의료 문서 OCR API 리버스 프록시

worker_processes auto;
error_log /var/log/nginx/error.log warn;

events {
    worker_connections 1024;
    multi_accept on;
}

http {
    # 기본 설정
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;

    # 타임아웃 (OCR은 처리 시간이 길 수 있다)
    proxy_connect_timeout 10s;
    proxy_read_timeout 60s;
    proxy_send_timeout 30s;

    # 업로드 크기 제한
    client_max_body_size 25M;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=30r/m;

    # 업스트림 정의
    upstream ocr_backend {
        server ocr-server:8000;
        keepalive 32;
    }

    server {
        listen 80;
        server_name _;

        # 헬스 체크 (rate limit 제외)
        location /health {
            proxy_pass http://ocr_backend;
        }

        # OCR API
        location /v1/ {
            limit_req zone=api burst=10 nodelay;

            proxy_pass http://ocr_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Request-ID $request_id;

            # SSE 스트리밍 지원
            proxy_buffering off;
            proxy_cache off;
            proxy_set_header Connection '';
            proxy_http_version 1.1;
            chunked_transfer_encoding off;
        }

        # Prometheus 메트릭 (내부만)
        location /metrics {
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            deny all;
            proxy_pass http://ocr_backend;
        }
    }
}
```

---

## 15.2.6 클라이언트 사용 예시

### Python SDK

```python
"""
의료 문서 OCR 클라이언트 예시
"""

import base64
import httpx
from pathlib import Path


class MedicalOCRClient:
    """의료 문서 OCR API 클라이언트"""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def ocr_from_file(self, image_path: str, prompt: str = None) -> dict:
        """파일 경로로 OCR 요청"""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"파일 없음: {image_path}")

        image_bytes = path.read_bytes()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "image_base64": image_base64,
            "max_tokens": 4096,
            "temperature": 0.0,
        }
        if prompt:
            payload["prompt"] = prompt

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.base_url}/v1/ocr",
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()

    def ocr_stream(self, image_path: str, prompt: str = None):
        """스트리밍 OCR 요청"""
        path = Path(image_path)
        image_bytes = path.read_bytes()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "image_base64": image_base64,
            "stream": True,
        }
        if prompt:
            payload["prompt"] = prompt

        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/v1/ocr/stream",
                json=payload,
                headers=self.headers,
            ) as response:
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        import json
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta

    def ocr_openai_compatible(self, image_path: str, prompt: str = None):
        """OpenAI SDK 호환 형식으로 요청"""
        path = Path(image_path)
        image_bytes = path.read_bytes()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "model": "medical-ocr",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt or "이 의료 문서의 텍스트를 추출하라.",
                    },
                ],
            }],
            "max_tokens": 4096,
            "temperature": 0.0,
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()


# 사용 예시
if __name__ == "__main__":
    client = MedicalOCRClient(
        base_url="http://localhost:8000",
        api_key="your-api-key-here",
    )

    # 동기 요청
    result = client.ocr_from_file(
        "prescription.png",
        prompt="이 처방전에서 약품명, 용량, 복용법을 추출하라."
    )
    print(f"결과: {result['text']}")
    print(f"처리 시간: {result['processing_time_ms']:.0f}ms")

    # 스트리밍 요청
    print("\n--- 스트리밍 ---")
    for token in client.ocr_stream("medical_chart.png"):
        print(token, end="", flush=True)
    print()
```

---

## 수학적 원리: 서버 처리량 분석

### Throughput과 Latency의 관계

단일 서버의 이론적 최대 처리량:

$$
\text{Throughput}_{\max} = \frac{C}{\bar{t}} = \frac{\text{max\_num\_seqs}}{\text{avg\_latency}}
$$

GPU 메모리 제약 하에서 동시 처리 가능 수:

$$
C = \left\lfloor \frac{M_{\text{GPU}} \times U - M_{\text{model}}}{M_{\text{kv\_per\_seq}}} \right\rfloor
$$

여기서:
- $M_{\text{GPU}}$: 총 GPU 메모리 (예: A100 80GB)
- $U$: GPU 메모리 활용률 (기본 0.90)
- $M_{\text{model}}$: 모델 가중치 메모리 (7B FP16 ≈ 14GB)
- $M_{\text{kv\_per\_seq}}$: 시퀀스당 KV Cache 메모리

**예시 계산:**

$$
C = \left\lfloor \frac{80 \times 0.9 - 14}{0.5} \right\rfloor = \left\lfloor \frac{58}{0.5} \right\rfloor = 116
$$

실제로는 이미지 인코딩에 추가 메모리가 필요하므로, VLM의 경우 이 값은 더 작아진다.

---

## 용어 체크리스트

학습 후 아래 질문에 답할 수 있어야 한다:

- [ ] AsyncLLMEngine이 동기 엔진보다 서빙에 유리한 이유는?
- [ ] PagedAttention의 메모리 효율 공식에서 $\bar{L} / L_{\max}$ 비율이 작을수록 좋은 이유는?
- [ ] FastAPI의 lifespan 패턴으로 모델을 로딩하는 이유는?
- [ ] SSE(Server-Sent Events) 스트리밍의 동작 원리와 WebSocket과의 차이는?
- [ ] Docker 멀티스테이지 빌드로 이미지 크기를 줄이는 원리는?
- [ ] Nginx에서 `proxy_buffering off`가 스트리밍에 필요한 이유는?
- [ ] GPU 메모리 제약 하에서 동시 처리 수 $C$를 계산하는 공식을 설명할 수 있는가?
- [ ] OpenAI-compatible API의 장점은 무엇이며 왜 이 형식을 채택하는가?
