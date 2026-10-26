"""
src/design/rules.py
-------------------
Deterministic rule-based furnishing logic for each room type:
- Bedroom: Double/Single Bed (back-to-wall), Nightstands (adjacent), Wardrobe (secondary wall)
- Living Room: Sofa (back-to-wall), Coffee Table (in front), TV Console (facing wall), Rug (underneath), Armchair
- Kitchen: Base counter runs (depth 0.60m), Sink counter, Refrigerator
- Bathroom: Toilet (back-to-wall), Vanity Sink, Bathtub
- Dining (User Relabel): Dining Table, Surrounding Chairs, Sideboard
- Office (User Relabel): Desk, Swivel Chair, Bookcase
"""

import math
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from shapely.geometry import Polygon, Point, box, LineString
from src.design.types import PlacedItem, UsableWallSegment

def compute_footprint_polygon(
    cx: float, cy: float, width_m: float, depth_m: float, yaw_deg: float
) -> List[Tuple[float, float]]:
    """
    Computes the 4 corners of a furniture footprint in plan space.
    cx, cy: Footprint center in plan meters
    width_m: Lateral width (transverse)
    depth_m: Front-to-back depth (longitudinal)
    yaw_deg: Clockwise angle in degrees from North [0, 360)
    """
    rad = math.radians(yaw_deg)
    # Forward vector pointing in direction of yaw
    fx = math.sin(rad)
    fy = -math.cos(rad)
    # Right lateral vector perpendicular to forward
    rx = math.cos(rad)
    ry = math.sin(rad)

    hw = width_m / 2.0
    hd = depth_m / 2.0

    # 4 corners: Front-Right, Back-Right, Back-Left, Front-Left
    c_fr = (cx + hw * rx + hd * fx, cy + hw * ry + hd * fy)
    c_br = (cx + hw * rx - hd * fx, cy + hw * ry - hd * fy)
    c_bl = (cx - hw * rx - hd * fx, cy - hw * ry - hd * fy)
    c_fl = (cx - hw * rx + hd * fx, cy - hw * ry + hd * fy)

    return [c_fr, c_br, c_bl, c_fl]

class RoomRuleEngine:
    """Applies architectural furnishing rules based on room type and preset."""

    def __init__(self, catalog: Dict[str, Any]):
        self.catalog = catalog["items"]

    def furnish_room(
        self,
        room_type: str,
        room_poly: Polygon,
        usable_segments: List[UsableWallSegment],
        door_clearance_zones: List[Polygon],
        preset: str = "contemporary_minimal",
        seed: int = 42
    ) -> Tuple[List[PlacedItem], Optional[str]]:
        """
        Generates candidate furniture placements according to room type rules.
        Returns: (List[PlacedItem], failure_reason_if_any)
        """
        rng = random.Random(seed)
        norm_type = self._normalize_room_type(room_type)
        area = room_poly.area

        if norm_type == "bedroom":
            return self._furnish_bedroom(room_poly, usable_segments, door_clearance_zones, preset, area, rng)
        elif norm_type == "living":
            return self._furnish_living(room_poly, usable_segments, door_clearance_zones, preset, area, rng)
        elif norm_type == "kitchen":
            return self._furnish_kitchen(room_poly, usable_segments, door_clearance_zones, preset, area, rng)
        elif norm_type == "bathroom":
            return self._furnish_bathroom(room_poly, usable_segments, door_clearance_zones, preset, area, rng)
        elif norm_type == "dining":
            return self._furnish_dining(room_poly, usable_segments, door_clearance_zones, preset, area, rng)
        elif norm_type == "office":
            return self._furnish_office(room_poly, usable_segments, door_clearance_zones, preset, area, rng)
        else:
            # Service room / balcony
            return [], None

    def _normalize_room_type(self, rtype: str) -> str:
        s = rtype.lower().strip()
        if "bed" in s: return "bedroom"
        if "liv" in s: return "living"
        if "kitch" in s: return "kitchen"
        if "bath" in s or "wc" in s: return "bathroom"
        if "din" in s: return "dining"
        if "off" in s or "study" in s: return "office"
        if "stor" in s or "clos" in s: return "storage"
        if "balc" in s or "terr" in s: return "balcony"
        return "other"

    def _get_item_spec(self, item_id: str) -> Dict[str, Any]:
        return self.catalog.get(item_id, {})

    # =========================================================================
    # 1. BEDROOM
    # =========================================================================
    def _furnish_bedroom(
        self,
        room_poly: Polygon,
        usable_segments: List[UsableWallSegment],
        door_zones: List[Polygon],
        preset: str,
        area: float,
        rng: random.Random
    ) -> Tuple[List[PlacedItem], Optional[str]]:
        if area < 4.0:
            return [], "TOO_SMALL"

        placed = []
        # Sort usable wall segments by length descending
        sorted_segs = sorted(usable_segments, key=lambda s: s.length_m, reverse=True)
        if not sorted_segs:
            return [], "NO_USABLE_WALL"

        # Determine bed size
        use_double = (area >= 7.5 and sorted_segs[0].length_m >= 1.80)
        bed_id = "bed_double" if use_double else "bed_single"
        bed_spec = self._get_item_spec(bed_id)
        bw, bd, bh = bed_spec["dimensions_m"]

        # 1. Place Primary Bed on longest usable wall segment
        bed_seg = None
        for seg in sorted_segs:
            if seg.length_m >= (bw + 0.10):  # Must fit bed width
                bed_seg = seg
                break

        if bed_seg is None:
            return [], "NO_USABLE_WALL"

        # Position bed centered or slightly offset on the segment
        seg_len = bed_seg.length_m
        p1 = np.array(bed_seg.start)
        p2 = np.array(bed_seg.end)
        u_dir = (p2 - p1) / max(seg_len, 1e-6)
        n_in = np.array(bed_seg.normal)

        # Center on segment
        center_s = seg_len / 2.0
        back_pt = p1 + u_dir * center_s
        # Item center = back_pt + (depth / 2) * n_in
        bed_center = back_pt + n_in * (bd / 2.0)

        bed_poly_coords = compute_footprint_polygon(
            float(bed_center[0]), float(bed_center[1]), bw, bd, bed_seg.yaw_deg
        )
        bed_poly = Polygon(bed_poly_coords)

        # Collision check with room polygon and doors
        if not room_poly.buffer(0.02).contains(bed_poly) or any(dz.intersects(bed_poly) for dz in door_zones):
            # Try shifting left or right
            shifted = False
            for shift in [-0.4, 0.4, -0.7, 0.7]:
                trial_s = center_s + shift
                if (bw / 2.0) <= trial_s <= (seg_len - bw / 2.0):
                    b_pt = p1 + u_dir * trial_s
                    b_c = b_pt + n_in * (bd / 2.0)
                    t_poly = Polygon(compute_footprint_polygon(float(b_c[0]), float(b_c[1]), bw, bd, bed_seg.yaw_deg))
                    if room_poly.buffer(0.02).contains(t_poly) and not any(dz.intersects(t_poly) for dz in door_zones):
                        bed_center = b_c
                        bed_poly_coords = list(t_poly.exterior.coords)[:-1]
                        bed_poly = t_poly
                        center_s = trial_s
                        shifted = True
                        break
            if not shifted and any(dz.intersects(bed_poly) for dz in door_zones):
                return [], "DOOR_BLOCKED"

        bed_item = PlacedItem(
            item_id="bed_0",
            catalog_id=bed_id,
            name=bed_spec["name"],
            category="bed",
            position_m=(float(bed_center[0]), float(bed_center[1])),
            rotation_deg=bed_seg.yaw_deg,
            dimensions_m=(bw, bd, bh),
            glb=bed_spec["glb"],
            polygon_coords=bed_poly_coords,
            placement_rule="back_to_wall"
        )
        placed.append(bed_item)

        # 2. Flanking Nightstands (if space permits on sides)
        ns_spec = self._get_item_spec("nightstand")
        nw, nd, nh = ns_spec["dimensions_m"]
        left_space = center_s - (bw / 2.0)
        right_space = seg_len - (center_s + bw / 2.0)

        if left_space >= (nw + 0.05):
            ns_s = center_s - (bw / 2.0) - (nw / 2.0) - 0.04
            ns_back = p1 + u_dir * ns_s
            ns_c = ns_back + n_in * (nd / 2.0)
            ns_poly = Polygon(compute_footprint_polygon(float(ns_c[0]), float(ns_c[1]), nw, nd, bed_seg.yaw_deg))
            if room_poly.buffer(0.02).contains(ns_poly) and not any(dz.intersects(ns_poly) for dz in door_zones):
                placed.append(PlacedItem(
                    item_id="nightstand_left",
                    catalog_id="nightstand",
                    name=ns_spec["name"],
                    category="nightstand",
                    position_m=(float(ns_c[0]), float(ns_c[1])),
                    rotation_deg=bed_seg.yaw_deg,
                    dimensions_m=(nw, nd, nh),
                    glb=ns_spec["glb"],
                    polygon_coords=list(ns_poly.exterior.coords)[:-1],
                    placement_rule="adjacent_anchor"
                ))

        if right_space >= (nw + 0.05):
            ns_s = center_s + (bw / 2.0) + (nw / 2.0) + 0.04
            ns_back = p1 + u_dir * ns_s
            ns_c = ns_back + n_in * (nd / 2.0)
            ns_poly = Polygon(compute_footprint_polygon(float(ns_c[0]), float(ns_c[1]), nw, nd, bed_seg.yaw_deg))
            if room_poly.buffer(0.02).contains(ns_poly) and not any(dz.intersects(ns_poly) for dz in door_zones):
                placed.append(PlacedItem(
                    item_id="nightstand_right",
                    catalog_id="nightstand",
                    name=ns_spec["name"],
                    category="nightstand",
                    position_m=(float(ns_c[0]), float(ns_c[1])),
                    rotation_deg=bed_seg.yaw_deg,
                    dimensions_m=(nw, nd, nh),
                    glb=ns_spec["glb"],
                    polygon_coords=list(ns_poly.exterior.coords)[:-1],
                    placement_rule="adjacent_anchor"
                ))

        # 3. Wardrobe on Secondary Wall
        ward_spec = self._get_item_spec("wardrobe")
        ww, wd, wh = ward_spec["dimensions_m"]

        placed_polys = [Polygon(p.polygon_coords) for p in placed]

        for seg in sorted_segs:
            if seg == bed_seg:
                continue
            if seg.length_m >= (ww + 0.10):
                w_p1 = np.array(seg.start)
                w_u = (np.array(seg.end) - w_p1) / seg.length_m
                w_nin = np.array(seg.normal)
                w_center_s = ww / 2.0 + 0.10  # Placed near start of segment
                w_back = w_p1 + w_u * w_center_s
                w_c = w_back + w_nin * (wd / 2.0)

                w_poly = Polygon(compute_footprint_polygon(float(w_c[0]), float(w_c[1]), ww, wd, seg.yaw_deg))
                if room_poly.buffer(0.02).contains(w_poly) and not any(dz.intersects(w_poly) for dz in door_zones):
                    if not any(w_poly.intersects(pp) for pp in placed_polys):
                        placed.append(PlacedItem(
                            item_id="wardrobe_0",
                            catalog_id="wardrobe",
                            name=ward_spec["name"],
                            category="wardrobe",
                            position_m=(float(w_c[0]), float(w_c[1])),
                            rotation_deg=seg.yaw_deg,
                            dimensions_m=(ww, wd, wh),
                            glb=ward_spec["glb"],
                            polygon_coords=list(w_poly.exterior.coords)[:-1],
                            placement_rule="back_to_wall"
                        ))
                        break

        return placed, None

    # =========================================================================
    # 2. LIVING ROOM
    # =========================================================================
    def _furnish_living(
        self,
        room_poly: Polygon,
        usable_segments: List[UsableWallSegment],
        door_zones: List[Polygon],
        preset: str,
        area: float,
        rng: random.Random
    ) -> Tuple[List[PlacedItem], Optional[str]]:
        if area < 6.0:
            return [], "TOO_SMALL"

        placed = []
        sorted_segs = sorted(usable_segments, key=lambda s: s.length_m, reverse=True)
        if not sorted_segs:
            return [], "NO_USABLE_WALL"

        # Sofa selection based on preset
        sofa_id = "sofa_modern" if preset == "contemporary_minimal" else ("sofa_classic" if preset == "tailored_classic" else "sofa_3seater")
        sofa_spec = self._get_item_spec(sofa_id)
        sw, sd, sh = sofa_spec["dimensions_m"]

        sofa_seg = None
        for seg in sorted_segs:
            if seg.length_m >= (sw + 0.10):
                sofa_seg = seg
                break

        if sofa_seg is None:
            # Fallback to smaller 3-seater if modern was too wide
            sofa_spec = self._get_item_spec("sofa_3seater")
            sw, sd, sh = sofa_spec["dimensions_m"]
            for seg in sorted_segs:
                if seg.length_m >= (sw + 0.10):
                    sofa_seg = seg
                    break

        if sofa_seg is None:
            return [], "NO_USABLE_WALL"

        # 1. Place Sofa centered on primary wall
        seg_len = sofa_seg.length_m
        p1 = np.array(sofa_seg.start)
        p2 = np.array(sofa_seg.end)
        u_dir = (p2 - p1) / max(seg_len, 1e-6)
        n_in = np.array(sofa_seg.normal)

        center_s = seg_len / 2.0
        back_pt = p1 + u_dir * center_s
        sofa_center = back_pt + n_in * (sd / 2.0)

        sofa_poly_coords = compute_footprint_polygon(
            float(sofa_center[0]), float(sofa_center[1]), sw, sd, sofa_seg.yaw_deg
        )
        sofa_poly = Polygon(sofa_poly_coords)

        # Shift if colliding with doors
        if any(dz.intersects(sofa_poly) for dz in door_zones):
            shifted = False
            for shift in [-0.5, 0.5, -0.9, 0.9]:
                trial_s = center_s + shift
                if (sw / 2.0) <= trial_s <= (seg_len - sw / 2.0):
                    b_pt = p1 + u_dir * trial_s
                    s_c = b_pt + n_in * (sd / 2.0)
                    t_poly = Polygon(compute_footprint_polygon(float(s_c[0]), float(s_c[1]), sw, sd, sofa_seg.yaw_deg))
                    if not any(dz.intersects(t_poly) for dz in door_zones) and room_poly.buffer(0.02).contains(t_poly):
                        sofa_center = s_c
                        sofa_poly_coords = list(t_poly.exterior.coords)[:-1]
                        sofa_poly = t_poly
                        center_s = trial_s
                        shifted = True
                        break
            if not shifted:
                return [], "DOOR_BLOCKED"

        placed.append(PlacedItem(
            item_id="sofa_0",
            catalog_id=sofa_spec.get("id", sofa_id),
            name=sofa_spec["name"],
            category="sofa",
            position_m=(float(sofa_center[0]), float(sofa_center[1])),
            rotation_deg=sofa_seg.yaw_deg,
            dimensions_m=(sw, sd, sh),
            glb=sofa_spec["glb"],
            polygon_coords=sofa_poly_coords,
            placement_rule="back_to_wall"
        ))

        # 2. Coffee Table (in front of sofa at distance ~0.45m)
        ct_id = "coffee_table_glass" if preset == "contemporary_minimal" else "coffee_table"
        ct_spec = self._get_item_spec(ct_id)
        cw, cd, ch = ct_spec["dimensions_m"]

        ct_dist = (sd / 2.0) + 0.45 + (cd / 2.0)
        ct_center = sofa_center + n_in * ct_dist
        ct_poly = Polygon(compute_footprint_polygon(float(ct_center[0]), float(ct_center[1]), cw, cd, sofa_seg.yaw_deg))

        if room_poly.buffer(0.02).contains(ct_poly) and not any(dz.intersects(ct_poly) for dz in door_zones):
            placed.append(PlacedItem(
                item_id="coffee_table_0",
                catalog_id=ct_id,
                name=ct_spec["name"],
                category="coffee_table",
                position_m=(float(ct_center[0]), float(ct_center[1])),
                rotation_deg=sofa_seg.yaw_deg,
                dimensions_m=(cw, cd, ch),
                glb=ct_spec["glb"],
                polygon_coords=list(ct_poly.exterior.coords)[:-1],
                placement_rule="in_front_of_anchor"
            ))

        # 3. Area Rug (under sofa front and coffee table)
        rug_spec = self._get_item_spec("rug_living")
        rw, rd, rh = rug_spec["dimensions_m"]
        rug_dist = (sd / 2.0) + 0.35
        rug_center = sofa_center + n_in * rug_dist
        rug_poly = Polygon(compute_footprint_polygon(float(rug_center[0]), float(rug_center[1]), rw, rd, sofa_seg.yaw_deg))

        if room_poly.buffer(0.02).contains(rug_poly):
            placed.append(PlacedItem(
                item_id="rug_0",
                catalog_id="rug_living",
                name=rug_spec["name"],
                category="rug",
                position_m=(float(rug_center[0]), float(rug_center[1])),
                rotation_deg=sofa_seg.yaw_deg,
                dimensions_m=(rw, rd, rh),
                glb=rug_spec["glb"],
                polygon_coords=list(rug_poly.exterior.coords)[:-1],
                placement_rule="under_anchor"
            ))

        # 4. TV Unit on facing or adjacent wall
        tv_spec = self._get_item_spec("tv_unit")
        tw, td, th = tv_spec["dimensions_m"]
        placed_polys = [Polygon(p.polygon_coords) for p in placed if p.category != "rug"]

        for seg in sorted_segs:
            if seg == sofa_seg:
                continue
            if seg.length_m >= (tw + 0.10):
                # Check distance from sofa
                seg_mid = (np.array(seg.start) + np.array(seg.end)) / 2.0
                dist_to_sofa = np.linalg.norm(seg_mid - sofa_center)
                if dist_to_sofa >= 2.0:  # Minimum 2m viewing distance
                    t_p1 = np.array(seg.start)
                    t_u = (np.array(seg.end) - t_p1) / seg.length_m
                    t_nin = np.array(seg.normal)
                    t_back = t_p1 + t_u * (seg.length_m / 2.0)
                    t_c = t_back + t_nin * (td / 2.0)
                    tv_poly = Polygon(compute_footprint_polygon(float(t_c[0]), float(t_c[1]), tw, td, seg.yaw_deg))

                    if room_poly.buffer(0.02).contains(tv_poly) and not any(dz.intersects(tv_poly) for dz in door_zones):
                        if not any(tv_poly.intersects(pp) for pp in placed_polys):
                            placed.append(PlacedItem(
                                item_id="tv_unit_0",
                                catalog_id="tv_unit",
                                name=tv_spec["name"],
                                category="tv_unit",
                                position_m=(float(t_c[0]), float(t_c[1])),
                                rotation_deg=seg.yaw_deg,
                                dimensions_m=(tw, td, th),
                                glb=tv_spec["glb"],
                                polygon_coords=list(tv_poly.exterior.coords)[:-1],
                                placement_rule="back_to_wall"
                            ))
                            break

        return placed, None

    # =========================================================================
    # 3. KITCHEN
    # =========================================================================
    def _furnish_kitchen(
        self,
        room_poly: Polygon,
        usable_segments: List[UsableWallSegment],
        door_zones: List[Polygon],
        preset: str,
        area: float,
        rng: random.Random
    ) -> Tuple[List[PlacedItem], Optional[str]]:
        if area < 3.5:
            return [], "TOO_SMALL"

        placed = []
        sorted_segs = sorted(usable_segments, key=lambda s: s.length_m, reverse=True)
        if not sorted_segs:
            return [], "NO_USABLE_WALL"

        # Kitchen run components (depth 0.60m)
        counter_spec = self._get_item_spec("kitchen_counter")
        sink_spec = self._get_item_spec("kitchen_sink_unit")
        fridge_spec = self._get_item_spec("refrigerator")

        cw, cd, ch = counter_spec["dimensions_m"]
        sw, sd, sh = sink_spec["dimensions_m"]
        fw, fd, fh = fridge_spec["dimensions_m"]

        primary_seg = sorted_segs[0]
        seg_len = primary_seg.length_m

        if seg_len < (sw + 0.10):
            return [], "NO_USABLE_WALL"

        p1 = np.array(primary_seg.start)
        p2 = np.array(primary_seg.end)
        u_dir = (p2 - p1) / max(seg_len, 1e-6)
        n_in = np.array(primary_seg.normal)

        curr_s = 0.10  # 10 cm corner margin

        # 1. Kitchen Sink Unit
        if (curr_s + sw) <= seg_len:
            sink_center_s = curr_s + sw / 2.0
            sink_back = p1 + u_dir * sink_center_s
            sink_c = sink_back + n_in * (sd / 2.0)
            sink_poly = Polygon(compute_footprint_polygon(float(sink_c[0]), float(sink_c[1]), sw, sd, primary_seg.yaw_deg))

            if room_poly.buffer(0.02).contains(sink_poly) and not any(dz.intersects(sink_poly) for dz in door_zones):
                placed.append(PlacedItem(
                    item_id="kitchen_sink_0",
                    catalog_id="kitchen_sink_unit",
                    name=sink_spec["name"],
                    category="kitchen_sink",
                    position_m=(float(sink_c[0]), float(sink_c[1])),
                    rotation_deg=primary_seg.yaw_deg,
                    dimensions_m=(sw, sd, sh),
                    glb=sink_spec["glb"],
                    polygon_coords=list(sink_poly.exterior.coords)[:-1],
                    placement_rule="back_to_wall"
                ))
                curr_s += sw

        # 2. Adjacent Base Counter Unit
        if (curr_s + cw) <= seg_len:
            cnt_center_s = curr_s + cw / 2.0
            cnt_back = p1 + u_dir * cnt_center_s
            cnt_c = cnt_back + n_in * (cd / 2.0)
            cnt_poly = Polygon(compute_footprint_polygon(float(cnt_c[0]), float(cnt_c[1]), cw, cd, primary_seg.yaw_deg))

            if room_poly.buffer(0.02).contains(cnt_poly) and not any(dz.intersects(cnt_poly) for dz in door_zones):
                placed.append(PlacedItem(
                    item_id="kitchen_counter_0",
                    catalog_id="kitchen_counter",
                    name=counter_spec["name"],
                    category="kitchen_counter",
                    position_m=(float(cnt_c[0]), float(cnt_c[1])),
                    rotation_deg=primary_seg.yaw_deg,
                    dimensions_m=(cw, cd, ch),
                    glb=counter_spec["glb"],
                    polygon_coords=list(cnt_poly.exterior.coords)[:-1],
                    placement_rule="back_to_wall"
                ))
                curr_s += cw

        # 3. Refrigerator (at end of counter run or on secondary wall)
        fridge_placed = False
        placed_polys = [Polygon(p.polygon_coords) for p in placed]

        if (curr_s + fw) <= seg_len:
            fr_center_s = curr_s + fw / 2.0
            fr_back = p1 + u_dir * fr_center_s
            fr_c = fr_back + n_in * (fd / 2.0)
            fr_poly = Polygon(compute_footprint_polygon(float(fr_c[0]), float(fr_c[1]), fw, fd, primary_seg.yaw_deg))

            if room_poly.buffer(0.02).contains(fr_poly) and not any(dz.intersects(fr_poly) for dz in door_zones):
                placed.append(PlacedItem(
                    item_id="refrigerator_0",
                    catalog_id="refrigerator",
                    name=fridge_spec["name"],
                    category="refrigerator",
                    position_m=(float(fr_c[0]), float(fr_c[1])),
                    rotation_deg=primary_seg.yaw_deg,
                    dimensions_m=(fw, fd, fh),
                    glb=fridge_spec["glb"],
                    polygon_coords=list(fr_poly.exterior.coords)[:-1],
                    placement_rule="back_to_wall"
                ))
                fridge_placed = True

        if not fridge_placed and len(sorted_segs) > 1:
            sec_seg = sorted_segs[1]
            if sec_seg.length_m >= (fw + 0.10):
                s2_p1 = np.array(sec_seg.start)
                s2_u = (np.array(sec_seg.end) - s2_p1) / sec_seg.length_m
                s2_nin = np.array(sec_seg.normal)
                fr_back = s2_p1 + s2_u * (fw / 2.0 + 0.10)
                fr_c = fr_back + s2_nin * (fd / 2.0)
                fr_poly = Polygon(compute_footprint_polygon(float(fr_c[0]), float(fr_c[1]), fw, fd, sec_seg.yaw_deg))

                if room_poly.buffer(0.02).contains(fr_poly) and not any(dz.intersects(fr_poly) for dz in door_zones):
                    if not any(fr_poly.intersects(pp) for pp in placed_polys):
                        placed.append(PlacedItem(
                            item_id="refrigerator_0",
                            catalog_id="refrigerator",
                            name=fridge_spec["name"],
                            category="refrigerator",
                            position_m=(float(fr_c[0]), float(fr_c[1])),
                            rotation_deg=sec_seg.yaw_deg,
                            dimensions_m=(fw, fd, fh),
                            glb=fridge_spec["glb"],
                            polygon_coords=list(fr_poly.exterior.coords)[:-1],
                            placement_rule="back_to_wall"
                        ))

        if not placed:
            return [], "NO_USABLE_WALL"

        return placed, None

    # =========================================================================
    # 4. BATHROOM
    # =========================================================================
    def _furnish_bathroom(
        self,
        room_poly: Polygon,
        usable_segments: List[UsableWallSegment],
        door_zones: List[Polygon],
        preset: str,
        area: float,
        rng: random.Random
    ) -> Tuple[List[PlacedItem], Optional[str]]:
        if area < 1.8:
            return [], "TOO_SMALL"

        placed = []
        sorted_segs = sorted(usable_segments, key=lambda s: s.length_m, reverse=True)
        if not sorted_segs:
            return [], "NO_USABLE_WALL"

        toilet_spec = self._get_item_spec("toilet")
        tw, td, th = toilet_spec["dimensions_m"]

        vanity_spec = self._get_item_spec("bathroom_vanity")
        vw, vd, vh = vanity_spec["dimensions_m"]

        # 1. Place Toilet
        t_seg = sorted_segs[0]
        p1 = np.array(t_seg.start)
        u_dir = (np.array(t_seg.end) - p1) / max(t_seg.length_m, 1e-6)
        n_in = np.array(t_seg.normal)

        t_center_s = tw / 2.0 + 0.15
        t_back = p1 + u_dir * t_center_s
        t_c = t_back + n_in * (td / 2.0)
        t_poly = Polygon(compute_footprint_polygon(float(t_c[0]), float(t_c[1]), tw, td, t_seg.yaw_deg))

        if any(dz.intersects(t_poly) for dz in door_zones):
            # Try other side of segment
            t_center_s = t_seg.length_m - (tw / 2.0 + 0.15)
            t_back = p1 + u_dir * t_center_s
            t_c = t_back + n_in * (td / 2.0)
            t_poly = Polygon(compute_footprint_polygon(float(t_c[0]), float(t_c[1]), tw, td, t_seg.yaw_deg))

        if not any(dz.intersects(t_poly) for dz in door_zones) and room_poly.buffer(0.02).contains(t_poly):
            placed.append(PlacedItem(
                item_id="toilet_0",
                catalog_id="toilet",
                name=toilet_spec["name"],
                category="toilet",
                position_m=(float(t_c[0]), float(t_c[1])),
                rotation_deg=t_seg.yaw_deg,
                dimensions_m=(tw, td, th),
                glb=toilet_spec["glb"],
                polygon_coords=list(t_poly.exterior.coords)[:-1],
                placement_rule="back_to_wall"
            ))

        # 2. Place Vanity Sink (on secondary wall or further down segment)
        placed_polys = [Polygon(p.polygon_coords) for p in placed]
        vanity_placed = False

        if len(sorted_segs) > 1:
            v_seg = sorted_segs[1]
            if v_seg.length_m >= (vw + 0.10):
                vp1 = np.array(v_seg.start)
                vu = (np.array(v_seg.end) - vp1) / v_seg.length_m
                vnin = np.array(v_seg.normal)
                v_back = vp1 + vu * (v_seg.length_m / 2.0)
                v_c = v_back + vnin * (vd / 2.0)
                v_poly = Polygon(compute_footprint_polygon(float(v_c[0]), float(v_c[1]), vw, vd, v_seg.yaw_deg))

                if room_poly.buffer(0.02).contains(v_poly) and not any(dz.intersects(v_poly) for dz in door_zones):
                    if not any(v_poly.intersects(pp) for pp in placed_polys):
                        placed.append(PlacedItem(
                            item_id="vanity_0",
                            catalog_id="bathroom_vanity",
                            name=vanity_spec["name"],
                            category="bathroom_sink",
                            position_m=(float(v_c[0]), float(v_c[1])),
                            rotation_deg=v_seg.yaw_deg,
                            dimensions_m=(vw, vd, vh),
                            glb=vanity_spec["glb"],
                            polygon_coords=list(v_poly.exterior.coords)[:-1],
                            placement_rule="back_to_wall"
                        ))
                        vanity_placed = True

        if not vanity_placed and t_seg.length_m >= (tw + vw + 0.60):
            v_center_s = t_center_s + (tw / 2.0) + 0.35 + (vw / 2.0)
            if v_center_s + vw / 2.0 <= t_seg.length_m:
                v_back = p1 + u_dir * v_center_s
                v_c = v_back + n_in * (vd / 2.0)
                v_poly = Polygon(compute_footprint_polygon(float(v_c[0]), float(v_c[1]), vw, vd, t_seg.yaw_deg))
                if room_poly.buffer(0.02).contains(v_poly) and not any(dz.intersects(v_poly) for dz in door_zones):
                    placed.append(PlacedItem(
                        item_id="vanity_0",
                        catalog_id="bathroom_vanity",
                        name=vanity_spec["name"],
                        category="bathroom_sink",
                        position_m=(float(v_c[0]), float(v_c[1])),
                        rotation_deg=t_seg.yaw_deg,
                        dimensions_m=(vw, vd, vh),
                        glb=vanity_spec["glb"],
                        polygon_coords=list(v_poly.exterior.coords)[:-1],
                        placement_rule="back_to_wall"
                    ))

        # 3. Bathtub if room is spacious (>= 4.5 m²)
        if area >= 4.5:
            bt_spec = self._get_item_spec("bathtub")
            bw, bd, bh = bt_spec["dimensions_m"]
            placed_polys = [Polygon(p.polygon_coords) for p in placed]

            for seg in sorted_segs:
                if seg.length_m >= (bw + 0.10):
                    bp1 = np.array(seg.start)
                    bu = (np.array(seg.end) - bp1) / seg.length_m
                    bnin = np.array(seg.normal)
                    b_back = bp1 + bu * (bw / 2.0 + 0.10)
                    b_c = b_back + bnin * (bd / 2.0)
                    b_poly = Polygon(compute_footprint_polygon(float(b_c[0]), float(b_c[1]), bw, bd, seg.yaw_deg))

                    if room_poly.buffer(0.02).contains(b_poly) and not any(dz.intersects(b_poly) for dz in door_zones):
                        if not any(b_poly.intersects(pp) for pp in placed_polys):
                            placed.append(PlacedItem(
                                item_id="bathtub_0",
                                catalog_id="bathtub",
                                name=bt_spec["name"],
                                category="bathtub",
                                position_m=(float(b_c[0]), float(b_c[1])),
                                rotation_deg=seg.yaw_deg,
                                dimensions_m=(bw, bd, bh),
                                glb=bt_spec["glb"],
                                polygon_coords=list(b_poly.exterior.coords)[:-1],
                                placement_rule="corner_or_wall"
                            ))
                            break

        if not placed:
            return [], "NO_USABLE_WALL"

        return placed, None

    # =========================================================================
    # 5. DINING (User Relabel Only)
    # =========================================================================
    def _furnish_dining(
        self,
        room_poly: Polygon,
        usable_segments: List[UsableWallSegment],
        door_zones: List[Polygon],
        preset: str,
        area: float,
        rng: random.Random
    ) -> Tuple[List[PlacedItem], Optional[str]]:
        if area < 5.0:
            return [], "TOO_SMALL"

        placed = []
        table_id = "dining_table_round" if preset == "warm_wood" else ("dining_table_cloth" if preset == "tailored_classic" else "dining_table")
        t_spec = self._get_item_spec(table_id)
        tw, td, th = t_spec["dimensions_m"]

        # Center table in room centroid
        cx, cy = room_poly.centroid.x, room_poly.centroid.y
        t_poly = Polygon(compute_footprint_polygon(cx, cy, tw, td, 0.0))

        if not room_poly.buffer(0.02).contains(t_poly) or any(dz.intersects(t_poly) for dz in door_zones):
            return [], "TOO_SMALL"

        placed.append(PlacedItem(
            item_id="dining_table_0",
            catalog_id=table_id,
            name=t_spec["name"],
            category="dining_table",
            position_m=(cx, cy),
            rotation_deg=0.0,
            dimensions_m=(tw, td, th),
            glb=t_spec["glb"],
            polygon_coords=list(t_poly.exterior.coords)[:-1],
            placement_rule="center"
        ))

        # Dining Chairs (4 chairs around table)
        chair_id = "dining_chair_scandi" if preset == "warm_wood" else ("dining_chair_classic" if preset == "tailored_classic" else "dining_chair")
        c_spec = self._get_item_spec(chair_id)
        cw, cd, ch = c_spec["dimensions_m"]

        # 4 chair positions: North (facing South 180 deg), South (facing North 0 deg), West (facing East 90 deg), East (facing West 270 deg)
        chair_configs = [
            ("chair_north", (cx, cy - td / 2.0 - cd / 2.0 - 0.08), 180.0),
            ("chair_south", (cx, cy + td / 2.0 + cd / 2.0 + 0.08), 0.0),
            ("chair_west", (cx - tw / 2.0 - cd / 2.0 - 0.08, cy), 90.0),
            ("chair_east", (cx + tw / 2.0 + cd / 2.0 + 0.08, cy), 270.0)
        ]

        for cid, c_pos, c_yaw in chair_configs:
            c_poly = Polygon(compute_footprint_polygon(c_pos[0], c_pos[1], cw, cd, c_yaw))
            if room_poly.buffer(0.02).contains(c_poly) and not any(dz.intersects(c_poly) for dz in door_zones):
                placed.append(PlacedItem(
                    item_id=cid,
                    catalog_id=chair_id,
                    name=c_spec["name"],
                    category="dining_chair",
                    position_m=c_pos,
                    rotation_deg=c_yaw,
                    dimensions_m=(cw, cd, ch),
                    glb=c_spec["glb"],
                    polygon_coords=list(c_poly.exterior.coords)[:-1],
                    placement_rule="around_anchor"
                ))

        return placed, None

    # =========================================================================
    # 6. OFFICE (User Relabel Only)
    # =========================================================================
    def _furnish_office(
        self,
        room_poly: Polygon,
        usable_segments: List[UsableWallSegment],
        door_zones: List[Polygon],
        preset: str,
        area: float,
        rng: random.Random
    ) -> Tuple[List[PlacedItem], Optional[str]]:
        if area < 4.0:
            return [], "TOO_SMALL"

        placed = []
        sorted_segs = sorted(usable_segments, key=lambda s: s.length_m, reverse=True)
        if not sorted_segs:
            return [], "NO_USABLE_WALL"

        desk_id = "desk_corner" if (preset == "tailored_classic" and area >= 9.0) else "desk"
        desk_spec = self._get_item_spec(desk_id)
        dw, dd, dh = desk_spec["dimensions_m"]

        d_seg = None
        for seg in sorted_segs:
            if seg.length_m >= (dw + 0.10):
                d_seg = seg
                break

        if d_seg is None:
            return [], "NO_USABLE_WALL"

        p1 = np.array(d_seg.start)
        u_dir = (np.array(d_seg.end) - p1) / max(d_seg.length_m, 1e-6)
        n_in = np.array(d_seg.normal)

        d_center_s = d_seg.length_m / 2.0
        d_back = p1 + u_dir * d_center_s
        d_c = d_back + n_in * (dd / 2.0)
        d_poly = Polygon(compute_footprint_polygon(float(d_c[0]), float(d_c[1]), dw, dd, d_seg.yaw_deg))

        if not room_poly.buffer(0.02).contains(d_poly) or any(dz.intersects(d_poly) for dz in door_zones):
            return [], "DOOR_BLOCKED"

        placed.append(PlacedItem(
            item_id="desk_0",
            catalog_id=desk_id,
            name=desk_spec["name"],
            category="desk",
            position_m=(float(d_c[0]), float(d_c[1])),
            rotation_deg=d_seg.yaw_deg,
            dimensions_m=(dw, dd, dh),
            glb=desk_spec["glb"],
            polygon_coords=list(d_poly.exterior.coords)[:-1],
            placement_rule="back_to_wall"
        ))

        # Desk Chair (facing desk)
        chair_spec = self._get_item_spec("office_chair")
        cw, cd, ch = chair_spec["dimensions_m"]
        chair_c = d_c + n_in * (dd / 2.0 + cd / 2.0 + 0.15)
        # Chair faces desk, so yaw is opposite to inward normal (yaw + 180)
        chair_yaw = (d_seg.yaw_deg + 180.0) % 360.0
        chair_poly = Polygon(compute_footprint_polygon(float(chair_c[0]), float(chair_c[1]), cw, cd, chair_yaw))

        if room_poly.buffer(0.02).contains(chair_poly) and not any(dz.intersects(chair_poly) for dz in door_zones):
            placed.append(PlacedItem(
                item_id="office_chair_0",
                catalog_id="office_chair",
                name=chair_spec["name"],
                category="office_chair",
                position_m=(float(chair_c[0]), float(chair_c[1])),
                rotation_deg=chair_yaw,
                dimensions_m=(cw, cd, ch),
                glb=chair_spec["glb"],
                polygon_coords=list(chair_poly.exterior.coords)[:-1],
                placement_rule="facing_anchor"
            ))

        return placed, None
