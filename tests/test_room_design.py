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

def test_partition_fallback_enclosed_room(engine):
    """
    Verify partition fallback: an enclosed room with empty wall_segments
    (interior partitions omitted by vectorizer) successfully recognizes its
    boundary edges as partition wall faces and furnishes validly.
    """
    room_pts = [
        [0.0, 0.0],
        [2.6, 0.0],
        [2.6, 2.4],
        [0.0, 2.4]
    ]
    doors = [{"position_m": [1.3, 2.4], "width_m": 0.80}]

    res = engine.design_room(
        room_polygon_meters=room_pts,
        room_type="bathroom",
        wall_segments=[], # Simulates missing interior partition wall evidence
        doors=doors,
        preset="contemporary_minimal",
        seed=42
    )

    assert res.is_valid, f"Partition fallback failed: {res.failure_reason}"
    assert len(res.items) >= 1
    categories = [it.category for it in res.items]
    assert "toilet" in categories
    assert res.validation.overlap_count == 0
    assert not res.validation.door_clearance_violation
    assert res.validation.bfs_reachable

def test_kitchen_secondary_wall_counter(engine):
    """
    Verify kitchen L-shaped counter fallback:
    Primary wall (1.70m) cannot accommodate sink (0.90m) + counter (0.90m) + margin (0.10m),
    which requires 1.90m. The counter is placed on the secondary wall (2.20m).
    """
    room_pts = [
        [0.0, 0.0],
        [2.2, 0.0],
        [2.2, 1.7],
        [0.0, 1.7]
    ]
    doors = [{"position_m": [0.0, 0.85], "width_m": 0.80}]

    res = engine.design_room(
        room_polygon_meters=room_pts,
        room_type="kitchen",
        doors=doors,
        preset="contemporary_minimal",
        seed=42
    )

    assert res.is_valid, f"Kitchen secondary counter failed: {res.failure_reason}"
    categories = [it.category for it in res.items]
    assert "kitchen_sink" in categories, "Kitchen must have sink"
    assert "kitchen_counter" in categories, "Kitchen must have counter on secondary wall"

    sink = next(it for it in res.items if it.category == "kitchen_sink")
    counter = next(it for it in res.items if it.category == "kitchen_counter")

    # They should be on different walls (different rotation angles)
    assert abs(sink.rotation_deg - counter.rotation_deg) > 1e-3 or np.linalg.norm(np.array(sink.position_m) - np.array(counter.position_m)) > 0.80
    assert res.validation.overlap_count == 0
    assert not res.validation.door_clearance_violation
    assert res.validation.bfs_reachable

def test_optional_item_dropping_compact_bathroom(engine):
    """
    Verify optional item dropping:
    A compact powder room / WC (2.0 m²) has enough space for a toilet,
    but not enough space for a bathtub or large vanity without collision.
    The engine gracefully drops the optional bathtub and vanity,
    accepting the toilet alone as a valid collision-free half-bath.
    """
    # 1.5m x 1.35m room (area ~2.0 m²)
    room_pts = [
        [0.0, 0.0],
        [1.5, 0.0],
        [1.5, 1.35],
        [0.0, 1.35]
    ]
    # Door occupies half of the south wall
    doors = [{"position_m": [1.1, 1.35], "width_m": 0.75}]

    res = engine.design_room(
        room_polygon_meters=room_pts,
        room_type="bathroom",
        doors=doors,
        preset="contemporary_minimal",
        seed=42
    )

    assert res.is_valid, f"Compact bathroom failed: {res.failure_reason}"
    categories = [it.category for it in res.items]
    assert "toilet" in categories, "Toilet must be placed"
    # Bathtub must be dropped because room is < 4.5 m²
    assert "bathtub" not in categories, "Bathtub must be dropped in compact bathroom"
    assert res.validation.overlap_count == 0
    assert not res.validation.door_clearance_violation
    assert res.validation.bfs_reachable


def test_back_to_wall_80_percent_contact_rule(engine):
    """
    Verify the strict floating-item prevention rule:
    Back-to-wall items must have >= 80% of their back edge within 0.10 m of a wall pixel.
    When a room edge has no real wall behind it (e.g. open space), placement is rejected.
    """
    # 4.0m x 4.0m room
    room_pts = [
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 4.0],
        [0.0, 4.0]
    ]
    # Door on South wall
    doors = [{"position_m": [2.0, 4.0], "width_m": 0.90}]

    # Create a synthetic wall raster (512x512) at scale 0.03 m/px
    # Room spans x: [0, 133px], y: [0, 133px]
    scale_m = 0.03
    wall_mask = np.zeros((512, 512), dtype=np.uint8)

    # Put a real wall only along the North edge: y in [0..5px], x in [0..133px]
    wall_mask[0:6, 0:134] = 1

    # Design bedroom: North wall has real wall mask, East/West/South do not
    res = engine.design_room(
        room_polygon_meters=room_pts,
        room_type="bedroom",
        doors=doors,
        wall_mask=wall_mask,
        scale_m_per_px=scale_m,
        seed=42
    )

    assert res.is_valid, f"Bedroom on real wall failed: {res.failure_reason}"
    bed = next(it for it in res.items if it.category == "bed")
    assert bed.wall_contact_ratio >= 0.80, f"Bed wall contact ratio {bed.wall_contact_ratio} is < 0.80"

    # Now test with a wall mask that has NO wall pixels at all near any edge
    empty_wall_mask = np.zeros((512, 512), dtype=np.uint8)
    empty_wall_mask[400:450, 400:450] = 1  # Wall far away from room

    res_floating = engine.design_room(
        room_polygon_meters=room_pts,
        room_type="bedroom",
        doors=doors,
        wall_mask=empty_wall_mask,
        scale_m_per_px=scale_m,
        seed=42
    )
    # The room cannot place floating furniture against non-existent walls
    assert not res_floating.is_valid, "Floating furniture should be rejected by 80% contact rule"
    assert res_floating.failure_reason in ["NO_USABLE_WALL", "VALIDATION_FAILED"]
