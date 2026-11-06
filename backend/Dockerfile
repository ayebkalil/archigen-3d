# The CPU runtime dependencies are available as wheels on supported hosts.
FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck & network diagnostics
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements-docker.txt /tmp/
RUN pip install --no-cache-dir --only-binary=:all: -r /tmp/requirements.txt -r /tmp/requirements-docker.txt \
    && useradd -m appuser \
    && install -d -o appuser -g appuser /app/library /app/library/custom
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

