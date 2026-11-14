# Tracking server equivalent to scripts/start_mlflow.ps1 (same mlflow version, same store layout).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-tracking.txt /tmp/requirements-tracking.txt
COPY scripts/docker_data.py /app/scripts/docker_data.py
RUN pip install --no-cache-dir -r /tmp/requirements-tracking.txt \
    && useradd -m appuser \
    && install -d -o appuser -g appuser /mlflow /mlflow/artifacts

ARG APP_REVISION=local
LABEL org.opencontainers.image.revision=$APP_REVISION
USER appuser
WORKDIR /mlflow
EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--serve-artifacts", "--artifacts-destination", "/mlflow/artifacts", \
     "--default-artifact-root", "mlflow-artifacts:/", \
     "--host", "0.0.0.0", "--port", "5000", "--workers", "1", \
     "--allowed-hosts", "mlflow,mlflow:*,localhost,localhost:*,127.0.0.1,127.0.0.1:*"]
