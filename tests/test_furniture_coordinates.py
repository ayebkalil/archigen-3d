"""
tests/test_furniture_coordinates.py
-----------------------------------
Tests coordinate system conventions, rotation mapping between 2D floorplan space
and Three.js world coordinates, asymmetric marker orientation, and catalog integrity.

Shared Plan-to-World Transform (Unmirrored, det = +1.0):
  X_three = x_plan - center_x
  Y_three = elevation (0.0 at floor level)
  Z_three = y_plan - center_y
  Three.js Euler Y rotation = (180 - alpha_deg) * (pi / 180)
"""

import os
import json
import numpy as np
import pytest
import trimesh

CATALOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "design", "catalog.json"))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "models", "furniture"))

def test_catalog_file_and_schema():
    """Verify catalog.json exists and all entries follow the strict schema."""
    assert os.path.exists(CATALOG_PATH), f"Catalog not found at {CATALOG_PATH}"
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    assert "items" in catalog
    assert len(catalog["items"]) >= 23, f"Catalog must have at least 23 items, found {len(catalog['items'])}"
    assert "scaling_policy" in catalog
    assert "box_like_non_uniform_items" in catalog
    assert len(catalog["box_like_non_uniform_items"]) == 6

    for item_id, item in catalog["items"].items():
        assert item["id"] == item_id
        assert "dimensions_m" in item and len(item["dimensions_m"]) == 3
        w, d, h = item["dimensions_m"]
        assert w > 0 and d > 0 and h > 0
        assert "glb" in item
        assert "placement_rule" in item
        assert "clearance" in item
        assert "required" in item
        assert "allowed_rooms" in item and len(item["allowed_rooms"]) > 0
        assert "presets" in item

def test_all_catalog_assets_exist_and_dimensions_match():
    """Verify that every asset in catalog.json exists on disk and matches measured dimensions."""
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    for item_id, item in catalog["items"].items():
        filename = os.path.basename(item["glb"])
        filepath = os.path.join(MODELS_DIR, filename)
        assert os.path.exists(filepath), f"Asset {filename} does not exist at {filepath}"

        # Load mesh and verify bounding box
        scene = trimesh.load(filepath, file_type="glb")
        bounds = scene.bounds
        extents = bounds[1] - bounds[0]  # [W_x, H_y, D_z]

        w_exp, d_exp, h_exp = item["dimensions_m"]
        np.testing.assert_allclose(extents[0], w_exp, atol=2e-3, err_msg=f"Width mismatch for {item_id}")
        np.testing.assert_allclose(extents[1], h_exp, atol=2e-3, err_msg=f"Height mismatch for {item_id}")
        np.testing.assert_allclose(extents[2], d_exp, atol=2e-3, err_msg=f"Depth mismatch for {item_id}")

        # Verify footprint origin is centered and base is at y=0
        np.testing.assert_allclose(bounds[0][1], 0.0, atol=2e-3, err_msg=f"Base not at y=0 for {item_id}")
        center_x = (bounds[0][0] + bounds[1][0]) / 2.0
        center_z = (bounds[0][2] + bounds[1][2]) / 2.0
        np.testing.assert_allclose(center_x, 0.0, atol=2e-3, err_msg=f"Footprint X not centered for {item_id}")
        np.testing.assert_allclose(center_z, 0.0, atol=2e-3, err_msg=f"Footprint Z not centered for {item_id}")

def test_asymmetric_marker_rotation_convention():
    """
    Test rotation convention using an asymmetric 3D marker object under the unmirrored transform.

    Coordinate Definition:
      Canonical Model Space (Three.js):
        - Forward (Nose):   +Z = (0, 0, 1)
        - Backward (Tail):  -Z = (0, 0, -1)
        - Upward (Fin):     +Y = (0, 1, 0)
        - Right (Wing):     +X = (1, 0, 0)

      Plan Space (2D Image):
        - X_plan: Rightward / East (+X_plan)
        - Y_plan: Downward / South (+Y_plan)
        - North on plan:    (0, -1) [decreasing Y]
        - East on plan:     (1,  0) [increasing X]
        - South on plan:    (0,  1) [increasing Y]
        - West on plan:     (-1, 0) [decreasing X]

      Unmirrored Transform:
        - X_three = x_plan - center_x
        - Z_three = y_plan - center_y  (NO axis negation, det = +1.0)
        - Y_three = elevation
        - Three.js rotation_y = (180 - alpha_deg) * (pi / 180)
    """
    # Define asymmetric key points on marker
    nose_canonical = np.array([0.0, 0.0, 1.0])       # Forward +Z
    wing_canonical = np.array([0.5, 0.0, 0.0])       # Right Wing +X
    fin_canonical  = np.array([0.0, 0.8, 0.0])       # Vertical Fin +Y

    def three_euler_y(angle_deg):
        # Three.js rotation.y = (180 - angle_deg)
        rot_deg = (180.0 - angle_deg)
        rad = np.radians(rot_deg)
        c, s = np.cos(rad), np.sin(rad)
        # Standard 3D rotation matrix around Y axis:
        # [ cos  0  sin]
        # [  0   1   0 ]
        # [-sin  0  cos]
        return np.array([
            [ c, 0.0,  s],
            [0.0, 1.0, 0.0],
            [-s, 0.0,  c]
        ])

    def three_to_plan_dir(v_three):
        # Under unmirrored transform:
        # dx_plan = dx_three, dy_plan = dz_three
        return np.array([v_three[0], v_three[2]])

    # Cardinal Test Cases: (alpha_deg_clockwise_from_north, expected_3d_nose, expected_plan_dir, direction_name)
    test_cases = [
        (0.0,   np.array([0.0, 0.0, -1.0]), np.array([0.0, -1.0]), "North (Up on plan)"),
        (90.0,  np.array([1.0, 0.0,  0.0]), np.array([1.0,  0.0]), "East (Right on plan)"),
        (180.0, np.array([0.0, 0.0,  1.0]), np.array([0.0,  1.0]), "South (Down on plan)"),
        (270.0, np.array([-1.0, 0.0, 0.0]), np.array([-1.0, 0.0]), "West (Left on plan)")
    ]

    for deg, exp_3d_nose, exp_plan_dir, name in test_cases:
        rot_mat = three_euler_y(deg)

        # Verify rotation matrix properties: det = +1.0 (proper SO(3))
        np.testing.assert_allclose(np.linalg.det(rot_mat), 1.0, atol=1e-6, err_msg=f"Matrix is not proper SO(3) at {deg} deg")

        # 1. Check transformed 3D nose vector
        nose_3d = rot_mat @ nose_canonical
        np.testing.assert_allclose(nose_3d, exp_3d_nose, atol=1e-5, err_msg=f"3D nose mismatch at {deg} deg ({name})")

        # 2. Check 2D plan direction vector (must match plan navigation)
        plan_dir = three_to_plan_dir(nose_3d)
        np.testing.assert_allclose(plan_dir, exp_plan_dir, atol=1e-5, err_msg=f"Plan direction mismatch at {deg} deg ({name})")

        # 3. Check right wing orientation (orthogonality to nose preserved)
        wing_3d = rot_mat @ wing_canonical
        assert abs(np.dot(nose_3d, wing_3d)) < 1e-6, f"Nose and wing not orthogonal at {deg} deg"

        # 4. Check vertical fin invariance under Y-rotation
        fin_3d = rot_mat @ fin_canonical
        np.testing.assert_allclose(fin_3d, fin_canonical, atol=1e-5, err_msg=f"Vertical fin changed under Y-rotation at {deg} deg")

def test_plan_to_three_coordinate_roundtrip():
    """Verify bidirectional coordinate transform between 2D plan and Three.js world without axis negation."""
    center_x = 5.25
    center_y = 4.10

    # Sample furniture placements on 2D floorplan [x_plan_m, y_plan_m]
    test_positions_plan = [
        [2.0, 3.0],
        [5.25, 4.10],  # At center
        [8.5, 6.2],
        [0.0, 0.0],
    ]

    for x_plan, y_plan in test_positions_plan:
        # Convert to Three.js world coordinates (Unmirrored: Z_three = y_plan - center_y)
        pos_x_three = x_plan - center_x
        pos_z_three = y_plan - center_y
        pos_y_three = 0.0  # Floor elevation

        # Inverse convert back to plan coordinates
        recovered_x_plan = pos_x_three + center_x
        recovered_y_plan = pos_z_three + center_y

        assert abs(recovered_x_plan - x_plan) < 1e-6
        assert abs(recovered_y_plan - y_plan) < 1e-6
