"""
Geometric Vectorization and Rectilinear Polygon Post-Processing Pipeline.
Converts raw 2D pixel segmentation masks (walls, doors, windows, rooms) into
clean, simplified vector geometry and structured 3D extrusion JSON specifications.
"""

import math
from typing import Dict, List, Any, Tuple
import numpy as np
import cv2
from shapely.geometry import Polygon, LineString, Point

ROOM_NAMES = {
    0: "Background", 1: "Living Room", 2: "Bedroom", 3: "Kitchen",
    4: "Bathroom", 5: "Entry / Hall", 6: "Balcony / Terrace",
    7: "Storage", 8: "Closet", 9: "Garage", 10: "Office", 11: "Other"
}

class FloorplanVectorizer:


    """
    Transforms multi-class floorplan probability masks into vector geometry.
    Classes:
      0: Background / Exterior
      1: Wall
      2: Door
      3: Window
    """
    def __init__(
        self,
        min_wall_area: int = 40,
        polygon_epsilon_ratio: float = 0.015,
        snap_angle_deg_tolerance: float = 12.0,
        default_door_width_m: float = 0.90,
        wall_height_m: float = 2.80
    ):
        self.min_wall_area = min_wall_area
        self.polygon_epsilon_ratio = polygon_epsilon_ratio
        self.snap_angle_deg_tolerance = snap_angle_deg_tolerance
        self.default_door_width_m = default_door_width_m
        self.wall_height_m = wall_height_m

    def snap_angle_rectilinear(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
        """Snaps an edge to rectilinear axes (0, 90, 180, 270 deg) if within tolerance."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-4:
            return p2

        angle = math.degrees(math.atan2(dy, dx)) % 360

        # Snap to horizontal (0 / 180 deg)
        if abs(angle - 0) <= self.snap_angle_deg_tolerance or abs(angle - 360) <= self.snap_angle_deg_tolerance:
            return (p1[0] + dist, p1[1])
        elif abs(angle - 180) <= self.snap_angle_deg_tolerance:
            return (p1[0] - dist, p1[1])
        # Snap to vertical (90 / 270 deg)
        elif abs(angle - 90) <= self.snap_angle_deg_tolerance:
            return (p1[0], p1[1] + dist)
        elif abs(angle - 270) <= self.snap_angle_deg_tolerance:
            return (p1[0], p1[1] - dist)

        return p2

    def estimate_scale(self, door_mask: np.ndarray) -> float:
        """
        Estimates meters per pixel scale factor using median door width.
        Falls back to 0.025 m/pixel (~40 px per meter) if no doors are detected.
        """
        contours, _ = cv2.findContours(door_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        door_widths = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 15:
                continue
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            short_dim = min(w, h)
            if short_dim > 3:
                door_widths.append(short_dim)

        if len(door_widths) > 0:
            median_door_px = float(np.median(door_widths))
            return self.default_door_width_m / median_door_px
        return 0.025  # Standard fallback: 1 pixel ~ 2.5 cm

    def vectorize(

        self,
        mask: np.ndarray,
        room_mask: Any = None,
        struct_mask: Any = None,
        scale_override_m_per_px: Any = None
    ) -> Dict[str, Any]:
        """
        Main pipeline: Takes (H, W) class index mask and extracts structured vector layout.
        Supports optional semantic room_mask and struct_mask (with Wall_External).
        """
        h, w = mask.shape
        wall_binary = (mask == 1).astype(np.uint8)
        door_binary = (mask == 2).astype(np.uint8)
        window_binary = (mask == 3).astype(np.uint8)

        # 1. Morphological filtering to clean noise and bridge small wall breaks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        wall_clean = cv2.morphologyEx(wall_binary, cv2.MORPH_CLOSE, kernel)
        wall_clean = cv2.morphologyEx(wall_clean, cv2.MORPH_OPEN, kernel)

        # 2. Scale estimation with manual override support
        if scale_override_m_per_px is not None and scale_override_m_per_px > 0:
            scale_m_per_px = float(scale_override_m_per_px)
            scale_source = "user_override"
        else:
            scale_m_per_px = self.estimate_scale(door_binary)
            scale_source = "door_width_estimation"

        # 3. Outer Building Shell (Watertight Exterior Envelope)
        ext_contours, _ = cv2.findContours(wall_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shell_valid = False
        building_perimeter_m = 0.0

        if len(ext_contours) > 0:
            largest_ext = max(ext_contours, key=cv2.contourArea)
            if cv2.contourArea(largest_ext) > 100:
                ext_poly_approx = cv2.approxPolyDP(largest_ext, 3.0, True).reshape(-1, 2)
                if len(ext_poly_approx) >= 3:
                    shell_poly = Polygon(ext_poly_approx * scale_m_per_px)
                    shell_valid = shell_poly.is_valid
                    building_perimeter_m = round(float(shell_poly.length), 2)

        # 4. Wall Contour extraction and Rectilinear Polygon simplification
        wall_contours, _ = cv2.findContours(wall_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        walls = []
        wall_segments = []

        wall_id = 0
        for cnt in wall_contours:
            area = cv2.contourArea(cnt)
            if area < self.min_wall_area:
                continue

            # Simplify polygon using Douglas-Peucker
            peri = cv2.arcLength(cnt, True)
            epsilon = max(1.5, self.polygon_epsilon_ratio * peri)
            approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)

            if len(approx) < 3:
                continue

            # Rectilinear snapping
            snapped_poly = [approx[0].tolist()]
            for i in range(1, len(approx)):
                p_snap = self.snap_angle_rectilinear(snapped_poly[-1], approx[i].tolist())
                snapped_poly.append([round(p_snap[0], 2), round(p_snap[1], 2)])

            # Convert to meters
            poly_meters = [[round(pt[0] * scale_m_per_px, 3), round(pt[1] * scale_m_per_px, 3)] for pt in snapped_poly]

            # Determine if Exterior Wall:
            # Method A: from struct_mask (class 2 = Wall_External)
            # Method B: from building envelope proximity
            is_exterior = False
            if struct_mask is not None:
                # Sample pixels under contour
                wall_patch = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(wall_patch, [cnt], -1, 1, thickness=-1)
                ext_pixels = np.logical_and(wall_patch == 1, struct_mask == 2).sum()
                if ext_pixels > 20:
                    is_exterior = True
            elif len(ext_contours) > 0:
                # Fallback: check distance to outer contour
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    c_pt = (M["m10"] / M["m00"], M["m01"] / M["m00"])
                    dist_to_ext = cv2.pointPolygonTest(largest_ext, c_pt, True)
                    if abs(dist_to_ext) < 15.0:
                        is_exterior = True

            # Register wall
            walls.append({
                "id": f"wall_{wall_id}",
                "is_exterior": is_exterior,
                "polygon_px": snapped_poly,
                "polygon_meters": poly_meters,
                "area_m2": round(area * (scale_m_per_px ** 2), 2),
                "height_m": self.wall_height_m,
                "openings": []
            })

            # Break into segments for door/window association
            for i in range(len(poly_meters)):
                p1 = poly_meters[i]
                p2 = poly_meters[(i + 1) % len(poly_meters)]
                wall_segments.append({
                    "wall_id": f"wall_{wall_id}",
                    "line": LineString([p1, p2]),
                    "start": p1,
                    "end": p2
                })

            wall_id += 1

        # 5. Doors & Windows extraction and wall projection
        def extract_openings(bin_mask: np.ndarray, opening_type: str):
            openings = []
            cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            op_idx = 0
            for c in cnts:
                if cv2.contourArea(c) < 10:
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = (M["m10"] / M["m00"]) * scale_m_per_px
                cy = (M["m01"] / M["m00"]) * scale_m_per_px
                center_pt = Point(cx, cy)

                rect = cv2.minAreaRect(c)
                dim_m = max(rect[1]) * scale_m_per_px

                # Find closest wall segment
                best_wall_id = None
                min_dist = float("inf")
                for seg in wall_segments:
                    dist = seg["line"].distance(center_pt)
                    if dist < min_dist:
                        min_dist = dist
                        best_wall_id = seg["wall_id"]

                op_data = {
                    "id": f"{opening_type}_{op_idx}",
                    "type": opening_type,
                    "position_meters": [round(cx, 3), round(cy, 3)],
                    "width_m": round(max(dim_m, 0.6), 2),
                    "height_m": 2.1 if opening_type == "door" else 1.4,
                    "associated_wall": best_wall_id,
                    "distance_to_wall_m": round(min_dist, 3)
                }
                openings.append(op_data)

                # Attach to wall openings list
                if best_wall_id is not None and min_dist < 1.0:
                    for w_item in walls:
                        if w_item["id"] == best_wall_id:
                            w_item["openings"].append(op_data)
                            break
                op_idx += 1
            return openings

        doors = extract_openings(door_binary, "door")
        windows = extract_openings(window_binary, "window")

        # 6. Extract Room Interior Spaces with Semantic Room Typing
        rooms = []
        room_idx = 0
        total_interior_area_m2 = 0.0

        if room_mask is not None and np.max(room_mask) > 0:
            # Semantic rooms extracted directly from room_mask
            for r_id in range(1, 12):
                r_bin = (room_mask == r_id).astype(np.uint8)
                r_cnts, _ = cv2.findContours(r_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in r_cnts:
                    area_px = cv2.contourArea(c)
                    if area_px < 50:
                        continue
                    area_m2 = round(area_px * (scale_m_per_px ** 2), 2)
                    total_interior_area_m2 += area_m2
                    r_approx = cv2.approxPolyDP(c, 2.5, True).reshape(-1, 2)
                    r_poly_m = [[round(pt[0] * scale_m_per_px, 3), round(pt[1] * scale_m_per_px, 3)] for pt in r_approx]

                    M = cv2.moments(c)
                    cx = (M["m10"] / max(M["m00"], 1e-4)) * scale_m_per_px
                    cy = (M["m01"] / max(M["m00"], 1e-4)) * scale_m_per_px

                    rooms.append({
                        "id": f"room_{room_idx}",
                        "room_type": ROOM_NAMES.get(r_id, "Other Room"),
                        "surface_m2": area_m2,
                        "centroid_meters": [round(cx, 3), round(cy, 3)],
                        "polygon_meters": r_poly_m
                    })
                    room_idx += 1
        else:
            # Geometric Connected Components fallback
            interior_mask = (wall_clean == 0).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(interior_mask)
            for lbl in range(1, num_labels):
                area_px = stats[lbl, cv2.CC_STAT_AREA]
                if area_px < 200 or area_px > (h * w * 0.7):
                    continue

                area_m2 = round(area_px * (scale_m_per_px ** 2), 2)
                total_interior_area_m2 += area_m2
                room_bin = (labels == lbl).astype(np.uint8)
                r_cnts, _ = cv2.findContours(room_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(r_cnts) > 0:
                    largest_c = max(r_cnts, key=cv2.contourArea)
                    r_approx = cv2.approxPolyDP(largest_c, 3.0, True).reshape(-1, 2)
                    r_poly_m = [[round(pt[0] * scale_m_per_px, 3), round(pt[1] * scale_m_per_px, 3)] for pt in r_approx]
                    r_type = "Living Room / Common" if area_m2 > 25 else "Bedroom" if area_m2 > 12 else "Bathroom / Storage"

                    rooms.append({
                        "id": f"room_{room_idx}",
                        "room_type": r_type,
                        "surface_m2": area_m2,
                        "centroid_meters": [round(centroids[lbl][0] * scale_m_per_px, 3), round(centroids[lbl][1] * scale_m_per_px, 3)],
                        "polygon_meters": r_poly_m
                    })
                    room_idx += 1

        # 7. Refine Exterior Walls using Room Union Outer Boundary (handles L and U-shaped buildings)
        try:
            from shapely.ops import unary_union
            valid_room_polys = [
                Polygon(r["polygon_meters"])
                for r in rooms
                if len(r.get("polygon_meters", [])) >= 3 and Polygon(r["polygon_meters"]).is_valid
            ]
            if len(valid_room_polys) > 0:
                room_envelope = unary_union(valid_room_polys)
                envelope_boundary = room_envelope.boundary
                for w_item in walls:
                    poly_pts = w_item.get("polygon_meters", [])
                    if len(poly_pts) >= 3:
                      w_poly = Polygon(poly_pts)
                      if w_poly.is_valid:
                        if envelope_boundary.distance(w_poly) < 0.45:
                          w_item["is_exterior"] = True
        except Exception:
            pass

        return {
            "metadata": {
                "canvas_size": [w, h],
                "scale_meters_per_pixel": round(scale_m_per_px, 5),
                "scale_source": scale_source,
                "building_perimeter_m": building_perimeter_m,
                "is_building_shell_valid": shell_valid,
                "total_surface_m2": round(total_interior_area_m2, 2),
                "num_walls": len(walls),
                "num_exterior_walls": sum(1 for w in walls if w.get("is_exterior", False)),
                "num_rooms": len(rooms),
                "num_doors": len(doors),
                "num_windows": len(windows)
            },
            "walls": walls,
            "rooms": rooms,
            "doors": doors,
            "windows": windows
        }

