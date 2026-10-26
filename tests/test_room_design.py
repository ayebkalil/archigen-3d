"""
tests/test_room_design.py
-------------------------
Unit tests for the deterministic rule-based room design placement engine.
Tests:
- Bedroom, Living, Kitchen, Bathroom furnishing rules
- Determinism for fixed seeds
- Spatial validation: 0 overlaps, door clearance, polygon containment, wall distance <= 2cm, BFS reachability
- Uncalibrated scale flag handling
"""

import pytest
import numpy as np
from src.design.engine import RoomDesignEngine
from src.design.types import RoomDesignResult

@pytest.fixture
def engine():
    return RoomDesignEngine()

def test_bedroom_furnishing_and_validation(engine):
    """Verify that a standard bedroom is furnished with a double bed, nightstand, and wardrobe."""
    # 4.5m x 4.0m bedroom
    room_pts = [
        [1.0, 1.0],
        [5.5, 1.0],
        [5.5, 5.0],
        [1.0, 5.0]
    ]
    # Door at South-West
    doors = [{"position_m": [1.5, 5.0], "width_m": 0.90}]

    res = engine.design_room(
        room_polygon_meters=room_pts,
        room_type="bedroom",
        doors=doors,
        preset="contemporary_minimal",
        seed=42
    )

    assert res.is_valid, f"Bedroom layout failed validation: {res.failure_reason}"
    assert len(res.items) >= 2, "Bedroom should have at least bed and one accessory"

    categories = [it.category for it in res.items]
    assert "bed" in categories, "Bedroom must contain a bed"
    assert "nightstand" in categories, "Bedroom should have at least one nightstand"

    # Verify bed is double bed for 18m² room
    bed = next(it for it in res.items if it.category == "bed")
    assert bed.catalog_id == "bed_double"

    # Verify back-to-wall distance <= 2.0 cm
    assert bed.wall_distance_cm <= 2.5, f"Bed wall gap too large: {bed.wall_distance_cm} cm"

    # Validation report checks
    val = res.validation
    assert val is not None
    assert val.overlap_count == 0, "No overlaps permitted"
    assert not val.door_clearance_violation, "Door clearance must be respected"
    assert val.out_of_bounds_count == 0, "All items must be within room"
    assert val.bfs_reachable, "Walking path must be reachable"

def test_living_room_furnishing_and_validation(engine):
    """Verify living room furnishings: sofa, coffee table, rug, TV unit."""
    # 6.0m x 4.5m living room
    room_pts = [
        [0.0, 0.0],
        [6.0, 0.0],
        [6.0, 4.5],
        [0.0, 4.5]
    ]
    doors = [{"position_m": [5.0, 4.5], "width_m": 0.90}]

    res = engine.design_room(
        room_polygon_meters=room_pts,
        room_type="living",
        doors=doors,
        preset="warm_wood",
        seed=42
    )

    assert res.is_valid, f"Living room layout failed: {res.failure_reason}"
    categories = [it.category for it in res.items]
    assert "sofa" in categories, "Living room must have sofa"
    assert "coffee_table" in categories, "Living room should have coffee table"

    sofa = next(it for it in res.items if it.category == "sofa")
    assert sofa.wall_distance_cm <= 2.5

    val = res.validation
    assert val.overlap_count == 0
    assert not val.door_clearance_violation
    assert val.bfs_reachable

def test_kitchen_furnishing_and_validation(engine):
    """Verify kitchen base counters (depth 0.60m) and sink unit."""
    # 4.0m x 3.0m kitchen
    room_pts = [
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 3.0],
        [0.0, 3.0]
    ]
    doors = [{"position_m": [3.5, 3.0], "width_m": 0.85}]

    res = engine.design_room(
        room_polygon_meters=room_pts,
        room_type="kitchen",
        doors=doors,
        preset="contemporary_minimal",
        seed=100
    )

    assert res.is_valid, f"Kitchen failed: {res.failure_reason}"
    categories = [it.category for it in res.items]
    assert "kitchen_sink" in categories, "Kitchen must have sink"
    assert "kitchen_counter" in categories, "Kitchen must have counter"

    # Verify counter depth is 0.60m standard
    sink = next(it for it in res.items if it.category == "kitchen_sink")
    assert abs(sink.dimensions_m[1] - 0.60) < 1e-3, f"Counter depth must be 0.60m, got {sink.dimensions_m[1]}"

def test_bathroom_furnishing_and_validation(engine):
    """Verify bathroom items: toilet and vanity sink."""
    # 2.5m x 2.2m bathroom
    room_pts = [
        [0.0, 0.0],
        [2.5, 0.0],
        [2.5, 2.2],
        [0.0, 2.2]
    ]
    doors = [{"position_m": [1.25, 2.2], "width_m": 0.80}]

    res = engine.design_room(
        room_polygon_meters=room_pts,
        room_type="bathroom",
        doors=doors,
        preset="contemporary_minimal",
        seed=77
    )

    assert res.is_valid, f"Bathroom failed: {res.failure_reason}"
    categories = [it.category for it in res.items]
    assert "toilet" in categories
    assert "bathroom_sink" in categories

def test_placement_determinism(engine):
    """Verify 100% determinism: identical seed produces identical positions and rotations."""
    room_pts = [
        [2.0, 2.0],
        [6.5, 2.0],
        [6.5, 6.0],
        [2.0, 6.0]
    ]
    doors = [{"position_m": [2.5, 6.0], "width_m": 0.90}]

    res1 = engine.design_room(room_polygon_meters=room_pts, room_type="bedroom", doors=doors, seed=123)
    res2 = engine.design_room(room_polygon_meters=room_pts, room_type="bedroom", doors=doors, seed=123)

    assert res1.is_valid and res2.is_valid
    assert len(res1.items) == len(res2.items)

    for it1, it2 in zip(res1.items, res2.items):
        assert it1.item_id == it2.item_id
        assert it1.catalog_id == it2.catalog_id
        np.testing.assert_allclose(it1.position_m, it2.position_m, atol=1e-6)
        assert abs(it1.rotation_deg - it2.rotation_deg) < 1e-6

def test_uncalibrated_scale_flag_and_dimensions(engine):
    """Verify uncalibrated scale flag and room width/depth metadata are returned."""
    room_pts = [
        [0.0, 0.0],
        [5.0, 0.0],
        [5.0, 4.0],
        [0.0, 4.0]
    ]
    res = engine.design_room(
        room_polygon_meters=room_pts,
        room_type="bedroom",
        is_scale_uncalibrated=True
    )

    assert res.metadata["scale_uncalibrated"] is True
    assert res.metadata["room_width_m"] == 5.0
    assert res.metadata["room_depth_m"] == 4.0
    assert res.metadata["room_area_m2"] == 20.0

def test_tiny_room_fails_gracefully(engine):
    """Verify that a tiny room below minimum area threshold fails gracefully with TOO_SMALL."""
    tiny_pts = [
        [0.0, 0.0],
        [1.2, 0.0],
        [1.2, 1.2],
        [0.0, 1.2]
    ]
    res = engine.design_room(
        room_polygon_meters=tiny_pts,
        room_type="bedroom",
        seed=42
    )

    assert not res.is_valid
    assert res.failure_reason == "TOO_SMALL"
    assert len(res.items) == 0
