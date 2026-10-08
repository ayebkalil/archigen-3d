import sys
from pathlib import Path

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

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

def test_predict_floorplan_valid_image():
    """Verify that uploading a valid floorplan image triggers Deep Learning segmentation & 3D extrusion."""
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new("RGB", (256, 256), color=(255, 255, 255))
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict/floorplan",
        files={"file": ("floorplan.png", buf.getvalue(), "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "total_area_m2" in data
    assert "num_rooms" in data
    assert "walls" in data
    assert "rooms" in data
    assert "threejs_scene" in data
    assert "X-Inference-Latency-ms" in response.headers

