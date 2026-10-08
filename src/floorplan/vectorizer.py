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

    def vectorize(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Main pipeline: Takes (H, W) class index mask and extracts structured vector layout.
        """
        h, w = mask.shape
        wall_binary = (mask == 1).astype(np.uint8)
        door_binary = (mask == 2).astype(np.uint8)
        window_binary = (mask == 3).astype(np.uint8)

        # 1. Morphological filtering to clean noise and bridge small wall breaks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        wall_clean = cv2.morphologyEx(wall_binary, cv2.MORPH_CLOSE, kernel)
        wall_clean = cv2.morphologyEx(wall_clean, cv2.MORPH_OPEN, kernel)

        # 2. Scale estimation
        scale_m_per_px = self.estimate_scale(door_binary)

        # 3. Wall Contour extraction and Rectilinear Polygon simplification
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

            # Register wall
            walls.append({
                "id": f"wall_{wall_id}",
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

        # 4. Doors & Windows extraction and wall projection
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

        # 5. Extract Room Interior Spaces
        # Rooms are the connected components of interior spaces (inverse of walls)
        interior_mask = (wall_clean == 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(interior_mask)
        rooms = []
        room_idx = 0
        total_interior_area_m2 = 0.0

        for lbl in range(1, num_labels):
            area_px = stats[lbl, cv2.CC_STAT_AREA]
            # Ignore background border and tiny noise
            if area_px < 200 or area_px > (h * w * 0.7):
                continue

            area_m2 = round(area_px * (scale_m_per_px ** 2), 2)
            total_interior_area_m2 += area_m2

            # Extract room boundary polygon
            room_bin = (labels == lbl).astype(np.uint8)
            r_cnts, _ = cv2.findContours(room_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(r_cnts) > 0:
                largest_c = max(r_cnts, key=cv2.contourArea)
                r_approx = cv2.approxPolyDP(largest_c, 3.0, True).reshape(-1, 2)
                r_poly_m = [[round(pt[0] * scale_m_per_px, 3), round(pt[1] * scale_m_per_px, 3)] for pt in r_approx]

                # Heuristic room type assignment
                r_type = "Living Room / Common" if area_m2 > 25 else "Bedroom" if area_m2 > 12 else "Bathroom / Storage"

                rooms.append({
                    "id": f"room_{room_idx}",
                    "room_type": r_type,
                    "surface_m2": area_m2,
                    "centroid_meters": [round(centroids[lbl][0] * scale_m_per_px, 3), round(centroids[lbl][1] * scale_m_per_px, 3)],
                    "polygon_meters": r_poly_m
                })
                room_idx += 1

        return {
            "metadata": {
                "canvas_size": [w, h],
                "scale_meters_per_pixel": round(scale_m_per_px, 5),
                "total_surface_m2": round(total_interior_area_m2, 2),
                "num_walls": len(walls),
                "num_rooms": len(rooms),
                "num_doors": len(doors),
                "num_windows": len(windows)
            },
            "walls": walls,
            "rooms": rooms,
            "doors": doors,
            "windows": windows
        }
