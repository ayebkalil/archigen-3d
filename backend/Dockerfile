# Multi-stage optimized Dockerfile for FastAPI Backend
FROM python:3.11-slim as builder

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
COPY ./src /app/src
COPY ./backend/app /app/app
COPY ./models/saved/floorplan_unetpp/floorplan_multitask.onnx /app/models/saved/floorplan_unetpp/floorplan_multitask.onnx
COPY ./models/saved/generator_perceptual_ema.onnx /app/models/saved/generator_perceptual_ema.onnx
COPY ./models/saved/generator.onnx /app/models/saved/generator.onnx
COPY ./models/saved/facade_authoritative_metrics.json /app/models/saved/facade_authoritative_metrics.json
COPY ./models/benchmarks /app/models/benchmarks

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

