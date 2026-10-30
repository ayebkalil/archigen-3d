"""The semantic image and viewer must use one physical projection frame."""
import io

import numpy as np
import pytest
from PIL import Image

from src.bridge.facade_bridge import FacadeBridge, CMP_PALETTE


@pytest.mark.parametrize("angle", [0.0, 0.7, np.pi / 2])
def test_detected_openings_roundtrip_to_semantic_pixels(angle):
    bridge = FacadeBridge.__new__(FacadeBridge)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = np.array([[0, 0], [8, 0], [8, 0.22], [0, 0.22]]) @ rotation.T
    position = np.array([2.1, 0]) @ rotation.T
    wall = {
        "polygon_meters": points.tolist(),
        "openings": [{"id": "window_0", "type": "window", "position_meters": position.tolist(), "width_m": 1.6, "height_m": 1.2, "sill_m": 0.8}],
    }
    image, bounds, frame, openings = bridge._build_facade_sketch(wall, num_floors=2, is_front=True)
    assert len(openings) == 2  # no invented entrance or extra window bays
    assert all(op["type"] == "window" for op in openings)
    assert frame["length_m"] == pytest.approx(8.0)
    pixels = np.array(image)
    for op in openings:
        u0, v0, u1, v1 = op["uv_rect"]
        assert tuple(pixels[int((1 - (v0 + v1) / 2) * 256), int((u0 + u1) / 2 * 256)]) == CMP_PALETTE["window"]
        # The viewer's physical planar UV transform lands on the input rectangle.
        u = bounds["u_min"] + op["position_along_wall_m"] / frame["length_m"] * (bounds["u_max"] - bounds["u_min"])
        v = bounds["v_min"] + (op["sill_m"] + op["height_m"] / 2) / bounds["wall_height_m"] * (bounds["v_max"] - bounds["v_min"])
        assert u == pytest.approx((u0 + u1) / 2, abs=1e-4)
        assert v == pytest.approx((v0 + v1) / 2, abs=1e-4)
        reconstructed = np.array(frame["origin"]) + np.array(frame["direction"]) * op["position_along_wall_m"]
        assert np.linalg.norm(reconstructed - position) < 0.04  # half a pixel at 256px
    assert openings[1]["sill_m"] - openings[0]["sill_m"] == pytest.approx(2.8, abs=0.04)


def test_onnx_facade_all_walls_executes_inference_and_keeps_projection():
    class SingleImageEngine:
        def __init__(self):
            self.calls = 0

        def generate(self, image_bytes):
            self.calls += 1
            assert Image.open(io.BytesIO(image_bytes)).size == (256, 256)
            buffer = io.BytesIO()
            Image.new("RGB", (256, 256), (160, 160, 160)).save(buffer, format="PNG")
            return buffer.getvalue()

    bridge = FacadeBridge.__new__(FacadeBridge)
    bridge.inference_engine = SingleImageEngine()
    bridge.enhance_textures = False
    bridge.device_name = "ONNX test double"
    walls = [{"id": str(i), "is_exterior": True, "polygon_meters": [[0, 0], [6, 0], [6, 0.2], [0, 0.2]], "openings": []} for i in range(2)]
    bridge.process_layout_facades({"metadata": {}, "walls": walls}, front_only=False)
    assert bridge.inference_engine.calls == 2
    for wall in walls:
        assert wall["facade_texture"]["facade_frame"]["length_m"] == 6
        assert "openings" in wall["facade_texture"]
