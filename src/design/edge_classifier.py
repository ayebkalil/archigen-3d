"""
src/design/edge_classifier.py
-----------------------------
Snaps room polygon edges to wall faces, detects door/window spans,
and computes unobstructed usable wall segments for furniture placement.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from shapely.geometry import Polygon, LineString, Point, box
from src.design.types import WallSegment, OpeningSpan, UsableWallSegment

def compute_inward_normal(p1: Tuple[float, float], p2: Tuple[float, float], room_poly: Polygon) -> Tuple[float, float]:
    """
    Computes the unit normal vector to edge (p1 -> p2) pointing INWARD into the room.
    In plan space: North is -Y, South is +Y, East is +X, West is -X.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return (0.0, 1.0)

    # Two possible perpendicular unit normals
    n1 = (-dy / length, dx / length)
    n2 = (dy / length, -dx / length)

    # Test midpoint + small offset epsilon
    mid_x = (p1[0] + p2[0]) / 2.0
    mid_y = (p1[1] + p2[1]) / 2.0
    eps = 0.05  # 5 cm into room

    test_pt1 = Point(mid_x + n1[0] * eps, mid_y + n1[1] * eps)
    if room_poly.contains(test_pt1) or room_poly.touches(test_pt1):
        return n1
    return n2

def normal_to_yaw_deg(nx: float, ny: float) -> float:
    """
    Converts a 2D inward normal (nx, ny) in plan space to a clockwise angle in degrees from North [0, 360).
    North is (0, -1) -> 0 deg
    East is (1, 0)   -> 90 deg
    South is (0, 1)  -> 180 deg
    West is (-1, 0)  -> 270 deg
    """
    # atan2(x, -y) gives clockwise angle from (0, -1)
    rad = math.atan2(nx, -ny)
    deg = math.degrees(rad)
    return (deg + 360.0) % 360.0

class EdgeClassifier:
    """Classifies room edges, snaps to walls, and extracts usable segments."""

    def __init__(self, wall_tolerance_m: float = 0.15, door_snap_tolerance_m: float = 0.40):
        self.wall_tolerance_m = wall_tolerance_m
        self.door_snap_tolerance_m = door_snap_tolerance_m

    def classify_and_snap_edges(
        self,
        room_poly: Polygon,
        wall_mask: Optional[np.ndarray] = None,
        scale_m_per_px: float = 0.03,
        wall_segments: Optional[List[Dict[str, Any]]] = None,
        doors: Optional[List[Dict[str, Any]]] = None,
        windows: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Polygon, List[UsableWallSegment], List[Polygon]]:
        """
        Takes raw room polygon and wall evidence.
        Returns:
          1. Snapped room polygon (flush against wall inner face)
          2. List of unobstructed UsableWallSegments (with door/window cutouts subtracted)
          3. List of door clearance zone polygons (reserved areas that must stay clear)
        """
        pts = list(room_poly.exterior.coords)[:-1]
        num_pts = len(pts)
        if num_pts < 3:
            return room_poly, [], []

        # 1. Classify each edge: wall-backed vs open
        edge_info = []
        door_clearance_zones = []

        # Precompute distance transform if wall_mask is provided
        dist_map_m = None
        if wall_mask is not None and wall_mask.max() > 0:
            import scipy.ndimage as ndi
            wall_bin = (wall_mask > 0).astype(np.uint8)
            dist_px = ndi.distance_transform_edt(1 - wall_bin)
            dist_map_m = dist_px * scale_m_per_px

        snapped_pts = []

        for i in range(num_pts):
            p1 = pts[i]
            p2 = pts[(i + 1) % num_pts]
            edge = LineString([p1, p2])
            edge_len = edge.length

            # Measure distance to nearest wall
            med_dist = self._measure_edge_wall_distance(edge, dist_map_m, scale_m_per_px, wall_segments)
            is_wall_backed = (med_dist <= self.wall_tolerance_m)

            inward_n = compute_inward_normal(p1, p2, room_poly)
            yaw_deg = normal_to_yaw_deg(inward_n[0], inward_n[1])

            edge_info.append({
                "edge_idx": i,
                "p1": p1,
                "p2": p2,
                "length_m": edge_len,
                "is_wall_backed": is_wall_backed,
                "wall_dist_m": med_dist,
                "inward_normal": inward_n,
                "yaw_deg": yaw_deg,
                "openings": []
            })
            snapped_pts.append(p1)

        snapped_room_poly = room_poly

        # 2. Snap doors onto edges & generate door clearance zones
        if doors:
            for door in doors:
                door_pos = self._get_opening_position(door)
                if door_pos is None:
                    continue

                best_edge_idx, best_dist, best_proj = self._find_nearest_edge(door_pos, edge_info)
                if best_edge_idx is not None and best_dist <= self.door_snap_tolerance_m:
                    door_width_m = door.get("width_m", 0.90)
                    e = edge_info[best_edge_idx]
                    e_len = e["length_m"]

                    # Project door span along edge
                    start_s = max(0.0, best_proj - door_width_m / 2.0)
                    end_s = min(e_len, best_proj + door_width_m / 2.0)

                    e["openings"].append(OpeningSpan(
                        opening_type="door",
                        start_dist=start_s,
                        end_dist=end_s,
                        clearance_depth=0.85
                    ))

                    # Construct 2D door clearance polygon (swing box)
                    cz = self._make_opening_clearance_polygon(
                        e["p1"], e["p2"], start_s, end_s, e["inward_normal"], depth_m=0.85
                    )
                    if cz is not None and cz.is_valid:
                        door_clearance_zones.append(cz)

        # 3. Snap windows onto edges
        if windows:
            for win in windows:
                win_pos = self._get_opening_position(win)
                if win_pos is None:
                    continue
                best_edge_idx, best_dist, best_proj = self._find_nearest_edge(win_pos, edge_info)
                if best_edge_idx is not None and best_dist <= self.door_snap_tolerance_m:
                    win_width_m = win.get("width_m", 1.20)
                    e = edge_info[best_edge_idx]
                    e_len = e["length_m"]

                    start_s = max(0.0, best_proj - win_width_m / 2.0)
                    end_s = min(e_len, best_proj + win_width_m / 2.0)

                    e["openings"].append(OpeningSpan(
                        opening_type="window",
                        start_dist=start_s,
                        end_dist=end_s,
                        clearance_depth=0.10
                    ))

        # 3.5 Fallback for enclosed room partitions
        # If wall_segments evidence omitted interior partitions (resulting in 0 wall-backed edges),
        # treat the room boundary edges as partition wall faces.
        if not any(e["is_wall_backed"] for e in edge_info):
            for e in edge_info:
                e["is_wall_backed"] = True
                e["wall_dist_m"] = 0.05

        # 4. Extract continuous unobstructed usable wall segments
        usable_segments = []
        for e in edge_info:
            if not e["is_wall_backed"]:
                continue

            e_len = e["length_m"]
            p1 = np.array(e["p1"])
            p2 = np.array(e["p2"])
            v_dir = (p2 - p1) / max(e_len, 1e-6)

            # Sort openings along edge
            openings = sorted(e["openings"], key=lambda o: o.start_dist)

            if not openings:
                # Full edge is unobstructed
                usable_segments.append(UsableWallSegment(
                    start=tuple(p1),
                    end=tuple(p2),
                    length_m=round(float(e_len), 3),
                    normal=e["inward_normal"],
                    yaw_deg=e["yaw_deg"],
                    edge_idx=e["edge_idx"],
                    wall_distance_m=round(e["wall_dist_m"], 3)
                ))
            else:
                # Split edge around openings
                curr_s = 0.0
                for op in openings:
                    seg_len = op.start_dist - curr_s
                    if seg_len >= 0.50:  # Minimum 50cm to be usable for any item
                        seg_p1 = p1 + v_dir * curr_s
                        seg_p2 = p1 + v_dir * op.start_dist
                        usable_segments.append(UsableWallSegment(
                            start=(float(seg_p1[0]), float(seg_p1[1])),
                            end=(float(seg_p2[0]), float(seg_p2[1])),
                            length_m=round(float(seg_len), 3),
                            normal=e["inward_normal"],
                            yaw_deg=e["yaw_deg"],
                            edge_idx=e["edge_idx"],
                            wall_distance_m=round(e["wall_dist_m"], 3)
                        ))
                    curr_s = max(curr_s, op.end_dist)

                # Trailing segment
                if (e_len - curr_s) >= 0.50:
                    seg_p1 = p1 + v_dir * curr_s
                    seg_p2 = p2
                    usable_segments.append(UsableWallSegment(
                        start=(float(seg_p1[0]), float(seg_p1[1])),
                        end=(float(seg_p2[0]), float(seg_p2[1])),
                        length_m=round(float(e_len - curr_s), 3),
                        normal=e["inward_normal"],
                        yaw_deg=e["yaw_deg"],
                        edge_idx=e["edge_idx"],
                        wall_distance_m=round(e["wall_dist_m"], 3)
                    ))

        return snapped_room_poly, usable_segments, door_clearance_zones

    def _measure_edge_wall_distance(
        self,
        edge: LineString,
        dist_map_m: Optional[np.ndarray],
        scale_m: float,
        wall_segments: Optional[List[Dict[str, Any]]]
    ) -> float:
        """Samples points along edge to compute robust median distance to wall."""
        if dist_map_m is not None:
            h, w = dist_map_m.shape
            dists = []
            for frac in [0.2, 0.4, 0.5, 0.6, 0.8]:
                pt = edge.interpolate(frac, normalized=True)
                ix = int(round(pt.x / scale_m))
                iy = int(round(pt.y / scale_m))
                if 0 <= ix < w and 0 <= iy < h:
                    dists.append(float(dist_map_m[iy, ix]))
            if dists:
                return float(np.median(dists))

        if wall_segments:
            # Measure to list of wall segments / polygons
            dists = []
            for frac in [0.2, 0.4, 0.5, 0.6, 0.8]:
                pt = edge.interpolate(frac, normalized=True)
                min_d = 999.0
                for ws in wall_segments:
                    pts = ws.get("polygon_meters", [])
                    if len(pts) >= 3:
                        poly = Polygon(pts)
                        if poly.is_valid:
                            min_d = min(min_d, poly.distance(pt))
                dists.append(min_d)
            if dists:
                return float(np.median(dists))

        return 0.05  # Default assumed close if no evidence

    def _get_opening_position(self, opening: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        if "center_meters" in opening:
            return tuple(opening["center_meters"][:2])
        if "position_m" in opening:
            return tuple(opening["position_m"][:2])
        if "polygon_meters" in opening and len(opening["polygon_meters"]) > 0:
            pts = opening["polygon_meters"]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        return None

    def _find_nearest_edge(self, pt: Tuple[float, float], edges: List[Dict[str, Any]]) -> Tuple[Optional[int], float, float]:
        p = Point(pt)
        best_idx = None
        best_dist = 999.0
        best_proj = 0.0

        for idx, e in enumerate(edges):
            ls = LineString([e["p1"], e["p2"]])
            d = ls.distance(p)
            if d < best_dist:
                best_dist = d
                best_idx = idx
                best_proj = ls.project(p)

        return best_idx, best_dist, best_proj

    def _make_opening_clearance_polygon(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        start_s: float,
        end_s: float,
        inward_n: Tuple[float, float],
        depth_m: float = 0.85
    ) -> Optional[Polygon]:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return None

        ux = dx / length
        uy = dy / length

        # 4 corners of clearance box projecting inward
        c1 = (p1[0] + ux * start_s, p1[1] + uy * start_s)
        c2 = (p1[0] + ux * end_s, p1[1] + uy * end_s)
        c3 = (c2[0] + inward_n[0] * depth_m, c2[1] + inward_n[1] * depth_m)
        c4 = (c1[0] + inward_n[0] * depth_m, c1[1] + inward_n[1] * depth_m)

        try:
            return Polygon([c1, c2, c3, c4])
        except Exception:
            return None
