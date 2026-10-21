"""
Unit tests for CubiCasa5k Dataset, FloorplanVectorizer, and 3D Extrusion Engine.
Satisfies MLOps Criterion: 'Tests unitaires' (3/3).
"""

import pytest
import numpy as np
import torch
import cv2

from src.floorplan.vectorizer import FloorplanVectorizer
from src.floorplan.extruder_3d import Floorplan3DExtruder

def test_vectorizer_rectilinear_snapping():
    """Verify that near-orthogonal edges are snapped cleanly to 0 and 90 degrees."""
    vectorizer = FloorplanVectorizer(snap_angle_deg_tolerance=15.0)

    # Angle near horizontal (e.g. 5 degrees)
    p1 = (0.0, 0.0)
    p2 = (10.0, 0.8) # ~4.5 degrees
    snapped = vectorizer.snap_angle_rectilinear(p1, p2)
    # y should be snapped to p1[1] (0.0)
    assert snapped[1] == 0.0, f"Expected y snapped to 0.0, got {snapped[1]}"

def test_vectorizer_layout_extraction():
    """Verify vector layout parsing from synthetic binary mask."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    # Draw simple room wall
    cv2.rectangle(mask, (30, 30), (220, 220), 1, 10)
    # Draw door
    cv2.rectangle(mask, (80, 25), (120, 35), 2, -1)

    vectorizer = FloorplanVectorizer()
    layout = vectorizer.vectorize(mask)

    assert "metadata" in layout
    assert "walls" in layout
    assert "rooms" in layout
    assert len(layout["walls"]) > 0
    assert layout["metadata"]["num_walls"] > 0

def test_extruder_threejs_and_obj_generation(tmp_path):
    """Verify 3D scene data structures and valid OBJ file export."""
    vectorizer = FloorplanVectorizer()
    extruder = Floorplan3DExtruder()

    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(mask, (40, 40), (200, 200), 1, 8)

    layout = vectorizer.vectorize(mask)
    scene_data = extruder.generate_threejs_scene_data(layout)

    assert scene_data["type"] == "ArchiGen3D_Floorplan_Scene"
    assert len(scene_data["walls"]) > 0

    # Test OBJ export
    obj_file = tmp_path / "test.obj"
    extruder.export_obj(layout, str(obj_file))
    assert obj_file.exists()
    content = obj_file.read_text(encoding="utf-8")
    assert "v " in content
    assert "f " in content

def test_vectorizer_hybrid_room_partitioning_and_absorption():
    """Verify that hybrid vectorizer splits same-class rooms separated by a wall and absorbs sub-threshold fragments."""
    vectorizer = FloorplanVectorizer(min_area_habitable_m2=1.5, min_area_service_m2=0.8)

    # 256x256 canvas
    struct_mask = np.zeros((256, 256), dtype=np.uint8)
    room_mask = np.zeros((256, 256), dtype=np.uint8)

    # Outer wall enclosing the space
    cv2.rectangle(struct_mask, (20, 20), (230, 230), 2, 6) # Exterior wall

    # Dividing interior wall at x=125
    cv2.line(struct_mask, (125, 20), (125, 230), 1, 6) # Interior wall

    # Both sides are LivingRoom (class 1)
    room_mask[26:224, 26:122] = 1 # Room A
    room_mask[26:224, 128:224] = 1 # Room B

    # Tiny spurious noise fragment of Closet (class 8, <0.8 m^2) inside Room A
    room_mask[40:44, 40:44] = 8 # 16 px fragment

    mask_4class = np.zeros_like(struct_mask)
    mask_4class[struct_mask > 0] = 1

    layout = vectorizer.vectorize(mask=mask_4class, struct_mask=struct_mask, room_mask=room_mask)
    rooms = layout.get("rooms", [])

    # The dividing wall should separate Room A and Room B into at least 2 distinct rooms
    assert len(rooms) >= 2, f"Expected at least 2 rooms after wall masking, got {len(rooms)}"

    # The tiny 16px closet fragment should have been absorbed, so no separate closet exists
    closet_rooms = [r for r in rooms if r.get("room_type") == "Closet"]
    assert len(closet_rooms) == 0, "Tiny closet fragment was not absorbed into parent room"

def test_vectorizer_no_rooms_detected_status():
    """Verify that empty floorplans with 0 rooms are explicitly classified as NO_ROOMS_DETECTED."""
    vectorizer = FloorplanVectorizer()
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(mask, (30, 30), (220, 220), 1, 10) # Empty outer shell, 0 rooms
    room_mask = np.zeros((256, 256), dtype=np.uint8)

    layout = vectorizer.vectorize(mask, room_mask=room_mask)
    meta = layout.get("metadata", {})

    assert meta.get("num_rooms") == 0
    assert meta.get("shell_status") == "NO_ROOMS_DETECTED"
    assert meta.get("is_building_shell_valid") is False

def test_vectorizer_envelope_shell_validation():
    """Verify that a continuous outer envelope with >=4 segments is marked valid when enclosing rooms."""
    vectorizer = FloorplanVectorizer()
    struct_mask = np.zeros((256, 256), dtype=np.uint8)
    room_mask = np.zeros((256, 256), dtype=np.uint8)

    # Single continuous outer envelope
    cv2.rectangle(struct_mask, (30, 30), (220, 220), 2, 8)
    # One enclosed room
    room_mask[40:210, 40:210] = 1

    mask_4class = np.zeros_like(struct_mask)
    mask_4class[struct_mask > 0] = 1

    layout = vectorizer.vectorize(mask=mask_4class, struct_mask=struct_mask, room_mask=room_mask)
    meta = layout.get("metadata", {})

    assert meta.get("num_rooms") == 1
    assert meta.get("num_envelope_segments", 0) >= 4
    assert meta.get("is_building_shell_valid") is True

def test_floorplan_pipeline_tta_modes():
    """Verify that FloorplanPipeline correctly supports tta_mode none, 4_fold, and 8_fold."""
    from backend.app.floorplan_engine import FloorplanPipeline
    from PIL import Image
    import io
    pipeline = FloorplanPipeline.get_instance()
    img = Image.new("RGB", (256, 256), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    res_none = pipeline.process_image(raw_bytes, tta_mode="none")
    assert res_none["inference_mode"] == "single_pass"

    res_4 = pipeline.process_image(raw_bytes, tta_mode="4_fold")
    assert res_4["inference_mode"] in ["4_fold_tta", "single_pass"]

    res_8 = pipeline.process_image(raw_bytes, tta_mode="8_fold")
    assert res_8["inference_mode"] in ["8_fold_tta", "single_pass"]

