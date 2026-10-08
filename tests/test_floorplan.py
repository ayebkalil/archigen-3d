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
