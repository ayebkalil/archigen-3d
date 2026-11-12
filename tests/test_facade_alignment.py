"""The semantic image and viewer must use one physical projection frame."""
import io

import numpy as np
import pytest
from PIL import Image

from src.bridge.facade_bridge import FacadeBridge, CMP_PALETTE
from src.data.facade_palette import EXPERIMENT_CMP_PALETTE, FACADE_STYLES


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
def test_auto_facade_selection_prefers_an_elevation_to_a_short_door_fragment():
    from src.bridge.facade_bridge import FacadeBridge

    stub = {"id": "stub", "is_exterior": True, "polygon_meters": [[0, 0], [.55, 0], [.55, .2], [0, .2]],
            "openings": [{"type": "door"}]}
    elevation = {"id": "elevation", "is_exterior": True, "polygon_meters": [[0, 2], [8, 2], [8, 2.2], [0, 2.2]]}
    assert FacadeBridge.select_front_wall([stub, elevation]) is elevation


def test_blank_structural_wall_remains_blank_in_facade_sketch():
    bridge = FacadeBridge.__new__(FacadeBridge)
    wall = {"geometry_fixed": True, "polygon_meters": [[0, 0], [8, 0], [8, .2], [0, .2]], "openings": []}
    _, _, _, openings = bridge._build_facade_sketch(wall, is_front=True)
    assert openings == []


def test_neutral_checkpoint_finishes_change_only_wall_pixels():
    class NeutralEngine:
        neural_style = True
        resolution = 512

        def generate(self, sketch, style):
            buffer = io.BytesIO()
            Image.new("RGB", (512, 512), (160, 160, 160)).save(buffer, format="PNG")
            return buffer.getvalue()

        def describe(self, style):
            return {"style_conditioned": False, "supported_styles": []}

    bridge = FacadeBridge.__new__(FacadeBridge)
    bridge.inference_engine = NeutralEngine()
    bridge.enhance_textures = False
    wall = {
        "height_m": 2.5,
        "polygon_meters": [[0, 0], [6, 0], [6, .2], [0, .2]],
        "openings": [{"id": "window", "type": "window", "position_along_wall_m": 3, "width_m": 1.2}],
    }
    default = bridge.render_wall_facade(wall, style=None, material_finish=True)
    assert "finish" not in default["model_info"]
    assert np.all(np.asarray(default["photo_image"]) == 160)
    textures = []
    for style in FACADE_STYLES:
        result = bridge.render_wall_facade(wall, style=style, material_finish=True)
        assert result["wall_height_m"] == 2.5
        assert result["model_info"] == {"style_conditioned": False, "supported_styles": [], "finish": style}
        semantic = np.asarray(result["sketch_image"])
        photo = np.asarray(result["photo_image"])
        wall_pixels = np.all(semantic == EXPERIMENT_CMP_PALETTE["facade"], axis=-1)
        assert np.any(wall_pixels)
        assert np.all(photo[~wall_pixels] == 160)
        textures.append(photo[wall_pixels].tobytes())
    assert len(set(textures)) == len(FACADE_STYLES)
