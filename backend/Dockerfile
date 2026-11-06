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
COPY ./models/benchmarks /app/models/benchmarks
COPY ./src /app/src
COPY ./backend/app /app/app

# The CPU furniture path uses PyMCubes when TripoSR is unavailable.
RUN pip install --no-cache-dir --only-binary=:all: PyMCubes==0.1.6

COPY backend/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh

ARG APP_REVISION=local
LABEL org.opencontainers.image.revision=$APP_REVISION
USER appuser

EXPOSE 8000

ENTRYPOINT ["/bin/sh", "/app/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

