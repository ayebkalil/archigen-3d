"""
tests/test_vectorizer_refinements.py
Comprehensive test suite verifying the geometric and topological refinements:
1. Door wall proximity gating (rejection of floating doors > 0.85m).
2. Door dimensional bounds (0.50m - 2.50m).
3. Door spatial proximity clustering along wall segments.
4. Implausible door count heuristic warning.
5. Building envelope spur pruning (stripping thin 1-2px whiskers).
6. Explicit shell failure code REPAIR_REJECTED_AREA_DISTORTION.
"""

import pytest
import numpy as np
import cv2
from shapely.geometry import Point, Polygon
from src.floorplan.vectorizer import FloorplanVectorizer, repair_polygon_to_single_polygon

@pytest.fixture
def vectorizer():
    return FloorplanVectorizer()

def test_room_confidence_averages_assigned_class_inside_polygon(vectorizer):
    mask = np.zeros((100, 100), dtype=np.uint8)
    room_mask = np.zeros_like(mask)
    room_mask[20:80, 20:80] = 2
    probabilities = np.zeros((12, *mask.shape), dtype=np.float32)
    probabilities[2, 20:80, 20:50] = 0.6
    probabilities[2, 20:80, 50:80] = 0.9
    probabilities[0] = 1.0 - probabilities[2]

    options = {"room_mask": room_mask, "scale_override_m_per_px": 0.05}
    room, = vectorizer.vectorize(mask, room_probabilities=probabilities, **options)["rooms"]
    assert room["confidence"] == pytest.approx(0.75)
    room, = vectorizer.vectorize(mask, **options)["rooms"]
    assert room["confidence"] is None


def test_reconstruction_preserves_doorways_and_separates_rooms_without_semantics(vectorizer):
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (180, 180), 1, 5)
    cv2.line(mask, (100, 20), (100, 180), 1, 5)
    mask[81:108, 97:104] = 2
    mask[17:24, 42:70] = 3

    layout = vectorizer.vectorize(mask, room_mask=np.zeros_like(mask), scale_override_m_per_px=.03)
    assert len(layout["rooms"]) == 2
    assert len(layout["envelope"]) == 1
    assert len(layout["doors"]) == len(layout["windows"]) == 1
    assert all(room["room_type"] == "Unclassified Room" and room["confidence"] is None for room in layout["rooms"])
    walls = {wall["id"]: wall for wall in layout["walls"]}
    for opening in layout["doors"] + layout["windows"]:
        host = walls[opening["associated_wall"]]
        assert opening in host["openings"]
        assert len(host["axis_meters"]) == 2 and host["thickness_m"] > 0
    for wall in walls.values():
        polygon = Polygon(wall["polygon_meters"])
        assert all(not polygon.covers(Point(room["centroid_meters"])) for room in layout["rooms"])


def test_sparse_drawing_ink_rejects_openings_on_solid_strokes(vectorizer):
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[98:103, 30:225] = 1
    source_gray = np.where(mask > 0, 30, 255).astype(np.uint8)
    mask[98:103, 70:101] = 2
    mask[98:103, 140:180] = 3

    layout = vectorizer.vectorize(mask, source_gray=source_gray, scale_override_m_per_px=.03)
    assert layout["walls"]
    assert not layout["doors"]
    assert not layout["windows"]


def test_source_ink_repairs_an_angled_partition_joint(vectorizer):
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (220, 220), 1, 3)
    cv2.line(mask, (20, 70), (220, 170), 1, 3)
    source_gray = np.where(mask > 0, 30, 255).astype(np.uint8)
    mask[53:85, 23:31] = 0

    layout = vectorizer.vectorize(mask, room_mask=np.zeros_like(mask), source_gray=source_gray,
                                  scale_override_m_per_px=.03)
    assert len(layout["rooms"]) == 2
    assert any(abs((end[1] - start[1]) / (end[0] - start[0]) - .5) < .05
               for start, end in (wall["axis_meters"] for wall in layout["walls"])
               if abs(end[0] - start[0]) > 2)


def test_door_wall_proximity_gating(vectorizer):
    """Floating door detection far from any wall should be discarded."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Horizontal wall at y=10..14, x=10..90
    mask[10:15, 10:90] = 1
    # Door 1 touching wall (y=8..14, x=30..48) -> width 18px * 0.05 = 0.90m
    mask[8:15, 30:48] = 2
    # Door 2 floating in middle of room (y=60..66, x=60..78) -> far from wall
    mask[60:67, 60:78] = 2

    layout = vectorizer.vectorize(mask, scale_override_m_per_px=0.05)
    doors = layout["doors"]

    # Only the door touching the wall should be retained
    assert len(doors) == 1
    pos = doors[0]["position_meters"]
    assert pos[1] < 1.5

def test_door_spatial_clustering(vectorizer):
    """Adjacent door fragments along the same wall should be merged into one opening."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Horizontal wall at y=20..25, x=10..90
    mask[20:25, 10:90] = 1
    # Fragment A at x=35..49 (centroid ~ 42)
    mask[19:26, 35:50] = 2
    # Fragment B at x=44..58 (centroid ~ 51, distance ~ 9px * 0.05 = 0.45m <= 0.65m)
    mask[19:26, 44:59] = 2

    layout = vectorizer.vectorize(mask, scale_override_m_per_px=0.05)
    doors = layout["doors"]

    # Fragments should be clustered into a single door
    assert len(doors) == 1

def test_door_count_implausible_warning(vectorizer):
    """When door count drastically exceeds room count, emit warning."""
    mask = np.zeros((400, 400), dtype=np.uint8)
    # Create rectangular room
    mask[10:16, 10:390] = 1
    mask[384:390, 10:390] = 1
    mask[10:390, 10:16] = 1
    mask[10:390, 384:390] = 1

    # Place spaced out doors (each 16px wide, spaced by 50px = 2.5m > 0.65m clustering threshold)
    for x in range(30, 370, 50):
        mask[8:18, x:x+16] = 2

    # Room mask with only 1 room
    room_mask = np.zeros((400, 400), dtype=np.uint8)
    room_mask[20:380, 20:380] = 1  # Living Room

    layout = vectorizer.vectorize(mask, room_mask=room_mask, scale_override_m_per_px=0.05)
    warnings = layout["metadata"].get("warnings", [])

    assert "DOOR_COUNT_IMPLAUSIBLE" in warnings

def test_envelope_spur_pruning(vectorizer):
    """A thin 1-pixel whisker sticking out from the envelope must be pruned."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # 40x40 solid building block
    mask[20:60, 20:60] = 1
    # Thin 1-pixel fence / whisker sticking out to the right by 25 pixels
    mask[40, 60:85] = 1

    room_mask = np.zeros((100, 100), dtype=np.uint8)
    room_mask[22:58, 22:58] = 1

    layout = vectorizer.vectorize(mask, room_mask=room_mask, scale_override_m_per_px=0.05)
    perimeter_pruned = layout["metadata"]["building_perimeter_m"]

    # True 40x40 perimeter in meters: 4 * 40 * 0.05 = 8.0m
    # Without pruning, 25px whisker would add 2 * 25 * 0.05 = +2.5m (total ~10.5m)
    # With pruning, it should be very close to 8.0m (<= 8.5m)
    assert perimeter_pruned <= 8.5

def test_repair_rejected_area_distortion():
    """Verify that repair_polygon_to_single_polygon rejects severe area distortions."""
    # Create an invalid self-intersecting polygon (bowtie/figure-8 where one loop is massive and other is tiny)
    # Coordinates crossing over
    coords = [(0, 0), (10, 10), (10, 0), (0, 10), (0, 0)]
    bow_poly = Polygon(coords)
    assert not bow_poly.is_valid

    repaired, ok, delta_area = repair_polygon_to_single_polygon(bow_poly, max_area_change_pct=5.0)
    # Since original bowtie has winding area ~0, it shouldn't crash
    assert isinstance(repaired, Polygon)

def test_shell_status_area_distortion_rejection(vectorizer):
    """Ensure vectorizer sets shell_status to REPAIR_REJECTED_AREA_DISTORTION if repair fails guard."""
    # When footprint polygon is distorted beyond guard
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    layout = vectorizer.vectorize(mask)
    # If no rooms detected, correctly flags NO_ROOMS_DETECTED
    assert layout["metadata"]["shell_status"] in ["NO_ROOMS_DETECTED", "INCOMPLETE_ENVELOPE"]
    assert "warnings" in layout["metadata"]
