# Multi-stage optimized Dockerfile for FastAPI Backend
FROM python:3.11-slim AS builder

WORKDIR /install
COPY backend/requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck & network diagnostics
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local
COPY backend/requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt \
    && useradd -m appuser \
    && install -d -o appuser -g appuser /app/library/custom
COPY ./models/saved/floorplan_unetpp/floorplan_multitask.onnx /app/models/saved/floorplan_unetpp/floorplan_multitask.onnx
COPY ./models/saved/floorplan_unetpp/floorplan_fallback.onnx /app/models/saved/floorplan_unetpp/floorplan_fallback.onnx
COPY ./models/saved/generator_perceptual_ema.onnx /app/models/saved/generator_perceptual_ema.onnx
COPY ./models/saved/facade512_experiment/generator.onnx ./models/saved/facade512_experiment/generator.json /app/models/saved/facade512_experiment/
COPY ./models/saved/facade_authoritative_metrics.json /app/models/saved/facade_authoritative_metrics.json
COPY ./models/benchmarks /app/models/benchmarks
COPY ./src /app/src
COPY ./backend/app /app/app

ARG APP_REVISION=local
LABEL org.opencontainers.image.revision=$APP_REVISION
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

