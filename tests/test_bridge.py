import pytest
import numpy as np
from PIL import Image
from src.bridge.facade_bridge import FacadeBridge, CMP_PALETTE

def test_facade_bridge_sketch_synthesis():
    bridge = FacadeBridge()
    mock_wall = {
        "id": "wall_test",
        "is_exterior": True,
        "is_front": True,
        "polygon_meters": [[0.0, 0.0], [6.0, 0.0], [6.0, 0.25], [0.0, 0.25]],
        "openings": [
            {"id": "win_0", "type": "window", "position_along_wall_m": 1.5},
            {"id": "win_1", "type": "window", "position_along_wall_m": 4.5},
            {"id": "door_0", "type": "door", "position_along_wall_m": 3.0}
        ]
    }

    # 1. Single floor test
    sketch_1fl, uv_1fl = bridge.synthesize_facade_sketch(mock_wall, num_floors=1, is_front=True)
    assert isinstance(sketch_1fl, Image.Image)
    assert sketch_1fl.size == (256, 256)
    assert abs(uv_1fl["wall_length_m"] - 6.0) < 0.05
    assert uv_1fl["wall_height_m"] == 2.8
    assert uv_1fl["u_max"] > uv_1fl["u_min"]
    assert uv_1fl["v_max"] > uv_1fl["v_min"]

    # 2. Multi-floor test (2 floors)
    sketch_2fl, uv_2fl = bridge.synthesize_facade_sketch(mock_wall, num_floors=2, is_front=True)
    assert isinstance(sketch_2fl, Image.Image)
    assert uv_2fl["wall_height_m"] == 5.6
    assert abs(uv_2fl["aspect_ratio"] - (6.0 / 5.6)) < 0.05

    # 3. Check CMP palette usage (wall, window, door, cornice, base present)
    arr = np.array(sketch_2fl)
    wall_c = np.array(CMP_PALETTE["facade"])
    win_c = np.array(CMP_PALETTE["window"])
    door_c = np.array(CMP_PALETTE["door"])
    cornice_c = np.array(CMP_PALETTE["cornice"])

    assert np.any(np.all(arr == wall_c, axis=-1))
    assert np.any(np.all(arr == win_c, axis=-1))
    assert np.any(np.all(arr == door_c, axis=-1))
    assert np.any(np.all(arr == cornice_c, axis=-1))

def test_facade_bridge_layout_processing():
    bridge = FacadeBridge()
    mock_layout = {
        "metadata": {"scale_m_per_px": 0.03, "total_area_m2": 75.0, "is_building_shell_valid": True},
        "walls": [
            {
                "id": "wall_0",
                "is_exterior": True,
                "polygon_meters": [[0.0, 0.0], [5.0, 0.0], [5.0, 0.2], [0.0, 0.2]],
                "openings": [{"type": "door", "position_along_wall_m": 2.5}]
            },
            {
                "id": "wall_1",
                "is_exterior": False,
                "polygon_meters": [[2.5, 0.0], [2.5, 4.0], [2.7, 4.0], [2.7, 0.0]],
                "openings": []
            }
        ],
        "rooms": []
    }

    res = bridge.process_layout_facades(mock_layout, num_floors=1)
    assert res["front_wall_id"] == "wall_0"
    assert "facade_texture" in mock_layout["walls"][0]
    assert "texture_base64" in mock_layout["walls"][0]["facade_texture"]
    assert mock_layout["walls"][0]["facade_texture"]["texture_base64"].startswith("data:image/png;base64,")
    assert "facade_texture" not in mock_layout["walls"][1]

def test_facade_bridge_backends():
    """Verify that FacadeBridge synthesizes valid RGB textures under both native PyTorch and ONNX CPU fallback."""
    import io
    from PIL import Image
    from pathlib import Path

    bridge_default = FacadeBridge()
    sketch = Image.new("RGB", (256, 256), (0, 48, 255))
    buf = io.BytesIO()
    sketch.save(buf, format="PNG")
    sketch_bytes = buf.getvalue()

    if bridge_default.inference_engine is not None:
        out_bytes = bridge_default.inference_engine.generate(sketch_bytes)
        assert isinstance(out_bytes, bytes)
        assert len(out_bytes) > 0
        out_img = Image.open(io.BytesIO(out_bytes))
        assert out_img.size == (256, 256)

    # 2. Explicit ONNX CPU fallback test
    from backend.app.inference import ArchiGenONNXInference
    onnx_path = Path("models/saved/generator_perceptual.onnx")
    if onnx_path.exists():
        onnx_engine = ArchiGenONNXInference(str(onnx_path))
        onnx_bytes = onnx_engine.generate(sketch_bytes)
        assert isinstance(onnx_bytes, bytes)
        assert len(onnx_bytes) > 0
        onnx_img = Image.open(io.BytesIO(onnx_bytes))
        assert onnx_img.size == (256, 256)

def test_facade_bridge_batch_and_front_only_modes():
    """Verify that FacadeBridge executes front-only on-demand mode and batched GPU mode correctly."""
    bridge = FacadeBridge()
    mock_layout = {
        "metadata": {"scale_m_per_px": 0.03, "total_area_m2": 80.0, "is_building_shell_valid": True},
        "walls": [
            {
                "id": "wall_ext_1",
                "is_exterior": True,
                "polygon_meters": [[0.0, 0.0], [6.0, 0.0], [6.0, 0.25], [0.0, 0.25]],
                "openings": [{"type": "door", "position_along_wall_m": 2.0}]
            },
            {
                "id": "wall_ext_2",
                "is_exterior": True,
                "polygon_meters": [[6.0, 0.0], [6.0, 5.0], [5.75, 5.0], [5.75, 0.0]],
                "openings": [{"type": "window", "position_along_wall_m": 2.5}]
            }
        ],
        "rooms": []
    }

    # 1. Test front_only mode (default)
    res_front = bridge.process_layout_facades(mock_layout, front_only=True)
    assert res_front["front_wall_id"] == "wall_ext_1"
    assert mock_layout["walls"][0]["facade_texture"]["mode"] == "neural_generated"
    assert mock_layout["walls"][1]["facade_texture"]["mode"] == "procedural_exterior"

    # 2. Test batched mode across all walls
    res_batch = bridge.process_layout_facades(mock_layout, front_only=False)
    assert len(res_batch["facade_results"]) == 2
    assert mock_layout["walls"][0]["facade_texture"]["mode"] in ["neural_batch", "neural_generated"]
    assert mock_layout["walls"][1]["facade_texture"]["mode"] in ["neural_batch", "neural_generated"]

def test_enhance_facade_details():
    """Verify that enhance_facade_details enhances micro-contrast and edge sharpness."""
    from PIL import ImageDraw
    from src.bridge.facade_bridge import enhance_facade_details
    img = Image.new("RGB", (256, 256), (128, 128, 128))
    draw = ImageDraw.Draw(img)
    draw.line([(0, 128), (256, 128)], fill=(255, 255, 255), width=2)
    enhanced = enhance_facade_details(img)
    assert isinstance(enhanced, Image.Image)
    assert enhanced.size == (256, 256)
    arr_orig = np.array(img).astype(np.float32)
    arr_enh = np.array(enhanced).astype(np.float32)
    assert not np.array_equal(arr_orig, arr_enh)



