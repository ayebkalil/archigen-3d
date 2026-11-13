import sys
from pathlib import Path

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest
from fastapi.testclient import TestClient
import io
import json
import base64
import httpx
import app.main as api
from app.main import app

client = TestClient(app)


def test_custom_room_catalog_uses_saved_models_that_match_and_fit(monkeypatch, tmp_path):
    import trimesh

    monkeypatch.setattr(api, "CUSTOM_FURNITURE_DIR", tmp_path)
    monkeypatch.setattr(api, "CUSTOM_CATALOG_FILE", tmp_path / "catalog.json")
    asset = trimesh.creation.box(extents=[.55, .8, .58]).export(file_type="glb")
    entry = {"id": "custom_armchair", "name": "My chair", "category": "chair", "furniture_type": "armchair",
             "dimensions_m": [.55, .58, .8], "bounds": {}, "glb_base64": base64.b64encode(asset).decode()}
    saved = client.post("/furniture/save-to-library", json=entry)
    assert saved.status_code == 200
    assert saved.json()["item"]["furniture_type"] == "armchair"
    assert client.post("/furniture/save-to-library", json={**entry, "glb_base64": "AAA="}).status_code == 422
    assert (tmp_path / "custom_armchair.glb").read_bytes() == asset

    catalog = json.loads((tmp_path / "catalog.json").read_text())
    catalog["custom_legacy"] = {**catalog["custom_armchair"], "name": "Legacy dining chair", "furniture_type": None}
    catalog["custom_broken"] = {**catalog["custom_armchair"], "name": "Broken armchair"}
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    (tmp_path / "custom_legacy.glb").write_bytes(asset)
    (tmp_path / "custom_broken.glb").write_bytes(b"\0\0")

    room = {"room_type": "Living Room", "polygon_meters": [[0, 0], [6, 0], [6, 4.5], [0, 4.5]]}
    items = client.post("/design", json=room).json()["items"]
    chair = next(item for item in items if item["category"] == "armchair")
    payload = {**room, "items": items, "item_id": chair["item_id"]}
    options = client.post("/design/furniture-choices", json=payload).json()["choices"]
    custom = [item for item in options if item.get("source") == "custom"]
    assert len(custom) == 1
    assert custom[0]["catalog_id"] == "custom_armchair"
    assert custom[0]["category"] == "armchair"
    assert custom[0]["glb"] == "/models/custom/custom_armchair.glb"
    payload["items"] = [custom[0] if item is chair else item for item in items]
    restored = client.post("/design/furniture-choices", json=payload).json()["choices"]
    assert any(item["catalog_id"] == chair["catalog_id"] for item in restored)


@pytest.mark.parametrize("available", [True, False])
def test_furniture_uses_configured_gpu_worker(monkeypatch, available):
    monkeypatch.setattr('src.furniture.service.generate_furniture_response',
                        lambda *args, **kwargs: pytest.fail('Proxy must not reconstruct or create a second run'))
    monkeypatch.setenv("ARCHIGEN_FURNITURE_SERVICE_URL", "http://furniture-worker:8001")
    result = {"status": "success", "engine": "TripoSR Quality", "quality": {"checkpoint_loading": "strict"}}

    def worker(request):
        assert str(request.url) == "http://furniture-worker:8001/furniture/generate-3d"
        assert json.loads(request.content)["category"] == "chair"
        if not available:
            raise httpx.ConnectError("Worker stopped", request=request)
        return httpx.Response(200, json=result)

    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(worker), **kwargs))
    response = client.post("/furniture/generate-3d", json={"image_base64": "test image", "category": "chair"})
    if available:
        assert response.status_code == 200
        assert response.json() == result
    else:
        assert response.status_code == 503
        assert "GPU furniture worker" in response.json()["detail"]

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
        files={"file": ("floorplan.png", buf.getvalue(), "image/png")},
        data={"reference_rooms": json.dumps([
            {"id": "bedroom", "room_type": "Bedroom", "polygon_px": [[20, 20], [80, 20], [80, 80]]},
            {"id": "sauna", "room_type": "Sauna", "polygon_px": [[90, 90], [120, 90], [120, 120]]},
        ])},
    )
    assert response.status_code == 200
    data = response.json()
    assert "total_area_m2" in data
    assert "num_rooms" in data
    assert "walls" in data
    assert "rooms" in data
    assert "threejs_scene" in data
    assert "X-Inference-Latency-ms" in response.headers
    scores = data["reference_room_confidences"]
    assert scores.keys() == {"bedroom", "sauna"}
    assert scores["bedroom"] is None or 0 <= scores["bedroom"] <= 1
    assert scores["sauna"] is None


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


