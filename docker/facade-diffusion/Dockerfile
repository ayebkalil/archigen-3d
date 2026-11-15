# Stable Diffusion material worker. Same CUDA base as the TripoSR worker, so the
# large torch layers are shared. Weights are NOT baked in: the one-shot
# facade-diffusion-init service downloads and SHA-256-verifies them into the
# persistent `facade_diffusion_models` volume once.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-facade-hybrid.txt /tmp/requirements-facade-hybrid.txt
RUN printf 'torch==2.6.0\ntorchvision==0.21.0\n' > /tmp/constraints.txt \
    && pip install -c /tmp/constraints.txt -r /tmp/requirements-facade-hybrid.txt \
    && python -c "import torch, diffusers; assert torch.__version__.startswith('2.6.0') and torch.version.cuda == '12.4', torch.__version__"

COPY configs/facade_hybrid.json /app/configs/facade_hybrid.json
COPY src /app/src
COPY backend/app /app/backend/app

RUN useradd -m appuser \
    && install -d -o appuser -g appuser /models /cache /run/archigen-gpu

ARG APP_REVISION=local
LABEL org.opencontainers.image.revision=$APP_REVISION
ENV ARCHIGEN_IMAGE_REVISION=$APP_REVISION \
    ARCHIGEN_SD_MODEL_DIR=/models \
    ARCHIGEN_SD_CACHE_DIR=/cache/materials \
    ARCHIGEN_GPU_LOCK=/run/archigen-gpu/gpu.lock \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2
USER appuser
EXPOSE 8002

CMD ["uvicorn", "backend.app.facade_diffusion_worker:app", "--host", "0.0.0.0", "--port", "8002"]
