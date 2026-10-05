import pytest
from fastapi.testclient import TestClient
import io
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    """Verify that GET /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model_loaded" in data

def test_model_metadata_endpoint():
    """Verify that GET /model/metadata returns architecture and metrics."""
    response = client.get("/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["project_name"] == "ArchiGen 3D (AI HomeByMe)"
    assert "supported_styles" in data
    assert len(data["supported_styles"]) > 0

def test_predict_floorplan_invalid_file():
    """Verify that uploading a non-image file returns HTTP 400 Bad Request."""
    file_content = b"this is a plain text file, not an image"
    response = client.post(
        "/predict/floorplan",
        files={"file": ("test.txt", file_content, "text/plain")}
    )
    assert response.status_code == 400
    assert "image format" in response.json()["detail"]

def test_predict_render_empty_payload():
    """Verify that uploading an empty image returns HTTP 400."""
    response = client.post(
        "/predict/render",
        files={"file": ("empty.png", b"", "image/png")},
        data={"style": "Modern Minimalist"}
    )
    assert response.status_code == 400
