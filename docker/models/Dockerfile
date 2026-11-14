FROM python:3.11-slim
WORKDIR /app
COPY models/artifacts.json /app/models/artifacts.json
COPY scripts/prepare_docker_models.py scripts/docker_data.py /app/scripts/
CMD ["sh", "-c", "python scripts/prepare_docker_models.py && python scripts/docker_data.py seed"]
