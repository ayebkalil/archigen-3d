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
    """Verify that GET /health returns 200, healthy status, and fallback router status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model_loaded" in data
    assert "fallback_router_loaded" in data

def test_model_metadata_endpoint():
    """Verify that GET /model/metadata returns architecture, metrics, and fallback router metadata."""
    response = client.get("/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["project_name"] == "ArchiGen 3D (AI HomeByMe)"
    assert "supported_styles" in data
    assert len(data["supported_styles"]) > 0
    assert "fallback_router" in data

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


def test_design_room_endpoint():
    """Verify that POST /design returns deterministic placed items with plan coordinates and rotation."""
    payload = {
        "room_id": "room_bed_1",
        "room_type": "bedroom",
        "polygon_meters": [[0.0, 0.0], [4.5, 0.0], [4.5, 4.0], [0.0, 4.0]],
        "doors": [{"start_point": [0.0, 1.0], "end_point": [0.0, 1.9]}],
        "preset": "contemporary_minimal",
        "seed": 42
    }
    response = client.post("/design", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["room_id"] == "room_bed_1"
    assert data["room_type"] == "bedroom"
    assert data["is_valid"] is True
    assert len(data["items"]) >= 2
    # Check plan coordinate format and rotation
    for item in data["items"]:
        assert "position_m" in item
        assert len(item["position_m"]) == 2
        assert "rotation_deg" in item
        assert "dimensions_m" in item
        assert "glb" in item
        assert "wall_distance_cm" in item
    assert "validation" in data
    assert data["validation"]["is_valid"] is True


def test_design_layout_endpoint():
    """Verify that POST /design/layout furnishes rooms across a full floorplan layout."""
    layout_data = {
        "rooms": [
            {
                "id": "room_0",
                "room_type": "bedroom",
                "polygon_meters": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.5], [0.0, 3.5]]
            },
            {
                "id": "room_1",
                "room_type": "living_room",
                "polygon_meters": [[4.5, 0.0], [9.5, 0.0], [9.5, 5.0], [4.5, 5.0]]
            }
        ],
        "doors": [],
        "windows": [],
        "walls": [],
        "metadata": {"scale_meters_per_pixel": 0.03}
    }
    response = client.post("/design/layout", json={"layout": layout_data, "preset": "contemporary_minimal", "seed": 42})
    assert response.status_code == 200
    data = response.json()
    assert data["num_rooms_furnished"] == 2
    assert "room_0" in data["rooms"]
    assert "room_1" in data["rooms"]
    assert data["rooms"]["room_0"]["is_valid"] is True
    assert data["rooms"]["room_1"]["is_valid"] is True


