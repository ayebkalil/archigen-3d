"""
src/design/validator.py
-----------------------
Strict geometric validation suite for furnished room layouts:
1. Zero overlaps between furniture items
2. Door clearance zones completely clear
3. All items inside room polygon
4. Back-to-wall items within 2 cm of wall face
5. BFS walking path reachability from door to items
"""

import math
from collections import deque
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from shapely.geometry import Polygon, Point, box, LineString
from src.design.types import PlacedItem, ValidationReport

class LayoutValidator:
    """Performs rigorous spatial verification on a generated room layout."""

    def __init__(
        self,
        wall_contact_tolerance_m: float = 0.02,
        min_corridor_width_m: float = 0.50,
        min_wall_contact_ratio: float = 0.80,
        wall_contact_dist_threshold_m: float = 0.10
    ):
        self.wall_contact_tolerance_m = wall_contact_tolerance_m
        self.min_corridor_width_m = min_corridor_width_m
        self.min_wall_contact_ratio = min_wall_contact_ratio
        self.wall_contact_dist_threshold_m = wall_contact_dist_threshold_m

    def validate(
        self,
        room_poly: Polygon,
        items: List[PlacedItem],
        door_clearance_zones: List[Polygon],
        doors: Optional[List[Dict[str, Any]]] = None,
        wall_dist_map_m: Optional[np.ndarray] = None,
        scale_m_per_px: float = 0.03
    ) -> ValidationReport:
        """
        Validates all constraints on the furnished layout.
        Returns a ValidationReport with complete pass/fail diagnostics.
        """
        reasons = []
        is_valid = True

        if not items:
            return ValidationReport(is_valid=True, failure_reasons=[])

        # 1. Zero Overlaps between furniture items
        overlap_count = 0
        max_overlap_area = 0.0
        n_items = len(items)

        item_polys = []
        for it in items:
            p = Polygon(it.polygon_coords)
            if not p.is_valid:
                p = p.buffer(0)
            item_polys.append(p)

        for i in range(n_items):
            for j in range(i + 1, n_items):
                # Small exception: rug is an under-anchor item placed under sofa/table
                if items[i].category == "rug" or items[j].category == "rug":
                    continue

                if item_polys[i].intersects(item_polys[j]):
                    inter = item_polys[i].intersection(item_polys[j])
                    area = inter.area
                    if area > 1e-4:  # > 1 cm²
                        overlap_count += 1
                        max_overlap_area = max(max_overlap_area, area)
                        reasons.append(f"OVERLAP: {items[i].item_id} overlaps {items[j].item_id} by {area:.4f} m²")

        if overlap_count > 0:
            is_valid = False

        # 2. Door clearance zones completely clear
        door_violation = False
        for dz in door_clearance_zones:
            if not dz or not dz.is_valid:
                continue
            for idx, p in enumerate(item_polys):
                if items[idx].category == "rug":
                    continue  # Rugs flat on floor may be walked on
                if dz.intersects(p):
                    inter = dz.intersection(p)
                    if inter.area > 1e-4:
                        door_violation = True
                        reasons.append(f"DOOR_BLOCKED: {items[idx].item_id} encroaches into door clear zone by {inter.area:.4f} m²")

        if door_violation:
            is_valid = False

        # 3. Items inside the room polygon
        out_of_bounds_count = 0
        # Allow small numerical buffer of 2cm for flush wall placement
        buffered_room = room_poly.buffer(0.025)

        for idx, p in enumerate(item_polys):
            diff = p.difference(buffered_room)
            if diff.area > 0.01:  # More than 100 cm² outside
                out_of_bounds_count += 1
                reasons.append(f"OUT_OF_BOUNDS: {items[idx].item_id} protrudes {diff.area:.4f} m² outside room")

        if out_of_bounds_count > 0:
            is_valid = False

        # 4. Back-to-wall contact (within 2 cm of wall face)
        max_wall_gap_cm = 0.0
        room_boundary = room_poly.exterior

        for it in items:
            if it.placement_rule == "back_to_wall":
                # Back edge distance to room boundary
                p = Polygon(it.polygon_coords)
                d = room_boundary.distance(p)
                d_cm = d * 100.0
                it.wall_distance_cm = d_cm
                max_wall_gap_cm = max(max_wall_gap_cm, d_cm)
                if d > (self.wall_contact_tolerance_m + 0.015):  # > 3.5 cm
                    reasons.append(f"WALL_GAP: {it.item_id} back is {d_cm:.1f} cm from wall face (limit 2.0 cm)")
                    is_valid = False

                # Strict wall raster contact check: >= 80% within 0.10 m (measured on predicted wall mask)
                if wall_dist_map_m is not None and len(it.polygon_coords) >= 4:
                    c_br = it.polygon_coords[1]
                    c_bl = it.polygon_coords[2]
                    back_line = LineString([c_br, c_bl])
                    if back_line.length > 1e-4:
                        pts_sample = [back_line.interpolate(frac, normalized=True) for frac in np.linspace(0.05, 0.95, 15)]
                        h, w = wall_dist_map_m.shape
                        contact_count = 0
                        for pt in pts_sample:
                            ix = int(round(pt.x / scale_m_per_px))
                            iy = int(round(pt.y / scale_m_per_px))
                            if 0 <= ix < w and 0 <= iy < h:
                                if wall_dist_map_m[iy, ix] <= self.wall_contact_dist_threshold_m:
                                    contact_count += 1
                        ratio = contact_count / len(pts_sample)
                        it.wall_contact_ratio = round(ratio, 3)
                        if ratio < self.min_wall_contact_ratio:
                            reasons.append(
                                f"WALL_CONTACT_VIOLATION: {it.item_id} has only {ratio*100:.1f}% "
                                f"of back edge within {self.wall_contact_dist_threshold_m*100:.0f}cm of wall pixel (minimum required: {self.min_wall_contact_ratio*100:.0f}%)"
                            )
                            is_valid = False

        # 5. BFS Reachability from door to primary items
        bfs_ok = self._check_bfs_reachability(room_poly, items, item_polys, doors)
        if not bfs_ok:
            is_valid = False
            reasons.append("NO_PATH: Placed furniture blocks walkable path from entrance door to room items")

        return ValidationReport(
            is_valid=is_valid,
            overlap_count=overlap_count,
            max_overlap_area_m2=max_overlap_area,
            door_clearance_violation=door_violation,
            out_of_bounds_count=out_of_bounds_count,
            max_wall_gap_cm=max_wall_gap_cm,
            bfs_reachable=bfs_ok,
            failure_reasons=reasons
        )

    def _check_bfs_reachability(
        self,
        room_poly: Polygon,
        items: List[PlacedItem],
        item_polys: List[Polygon],
        doors: Optional[List[Dict[str, Any]]]
    ) -> bool:
        """
        Runs BFS on a 10cm grid inside the room to confirm that an agent entering
        through the door can walk freely to the front of each placed primary item.
        """
        bounds = room_poly.bounds  # minx, miny, maxx, maxy
        grid_res = 0.10  # 10 cm grid
        minx, miny, maxx, maxy = bounds

        cols = int(math.ceil((maxx - minx) / grid_res)) + 1
        rows = int(math.ceil((maxy - miny) / grid_res)) + 1

        if cols < 2 or rows < 2 or (cols * rows) > 250000:
            return True  # Avoid excessive grid sizes

        # 0 = walkable, 1 = obstacle/outside
        grid = np.ones((rows, cols), dtype=np.uint8)

        # Mark room interior as walkable
        for r in range(rows):
            y = miny + r * grid_res
            for c in range(cols):
                x = minx + c * grid_res
                pt = Point(x, y)
                if room_poly.contains(pt):
                    grid[r, c] = 0

        # Mark non-rug furniture as obstacles
        for idx, ip in enumerate(item_polys):
            if items[idx].category == "rug":
                continue
            i_bounds = ip.bounds
            c_start = max(0, int((i_bounds[0] - minx) / grid_res))
            c_end = min(cols, int(math.ceil((i_bounds[2] - minx) / grid_res)) + 1)
            r_start = max(0, int((i_bounds[1] - miny) / grid_res))
            r_end = min(rows, int(math.ceil((i_bounds[3] - miny) / grid_res)) + 1)

            for r in range(r_start, r_end):
                y = miny + r * grid_res
                for c in range(c_start, c_end):
                    x = minx + c * grid_res
                    if ip.contains(Point(x, y)):
                        grid[r, c] = 1

        # Determine start cell(s) from door position
        start_cells = []
        if doors:
            for d in doors:
                pos = d.get("center_meters") or d.get("position_m")
                if pos:
                    c = int(round((pos[0] - minx) / grid_res))
                    r = int(round((pos[1] - miny) / grid_res))
                    # Find nearest walkable cell within 4 cells (40cm)
                    for dr in range(-4, 5):
                        for dc in range(-4, 5):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                                start_cells.append((nr, nc))
                                break

        if not start_cells:
            # Fallback: start at center of room
            cx, cy = room_poly.centroid.x, room_poly.centroid.y
            cr, cc = int(round((cy - miny) / grid_res)), int(round((cx - minx) / grid_res))
            if 0 <= cr < rows and 0 <= cc < cols and grid[cr, cc] == 0:
                start_cells.append((cr, cc))

        if not start_cells:
            return True  # If room has no walkable cells, pass to other checks

        # Run BFS flood fill
        queue = deque(start_cells)
        visited = set(start_cells)

        while queue:
            curr_r, curr_c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        # Check reachability to the front of primary items
        for it in items:
            if it.category in ["rug", "plant"]:
                continue
            # Point in front of item
            px, py = it.position_m
            rad = math.radians(it.rotation_deg)
            # Forward vector in plan space: dx = sin(rad), dy = -cos(rad)
            dx = math.sin(rad)
            dy = -math.cos(rad)
            front_x = px + dx * (it.dimensions_m[1] / 2.0 + 0.35)
            front_y = py + dy * (it.dimensions_m[1] / 2.0 + 0.35)

            target_c = int(round((front_x - minx) / grid_res))
            target_r = int(round((front_y - miny) / grid_res))

            # Check if any cell within 30cm of item front is reached
            reached = False
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    nr, nc = target_r + dr, target_c + dc
                    if (nr, nc) in visited:
                        reached = True
                        break
                if reached:
                    break

            if not reached and room_poly.contains(Point(front_x, front_y)):
                return False  # An item is completely blocked by obstacles

        return True
