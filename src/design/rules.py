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
from src.design.types import PlacedItem, UsableWallSegment, MIN_PLAUSIBLE_ROOM_AREAS

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

    def _check_back_edge_contact(
        self,
        poly_coords: List[Tuple[float, float]],
        wall_dist_map_m: Optional[np.ndarray],
        scale_m_per_px: float = 0.03,
        threshold_m: float = 0.10,
        required_ratio: float = 0.80
    ) -> bool:
        """
        Checks if >= 80% of back edge sample points are within 0.10m of a wall pixel.
        Floating items are exempt by not passing through back-to-wall checks.
        """
        if wall_dist_map_m is None or len(poly_coords) < 3:
            return True
        c_br = poly_coords[1]
        c_bl = poly_coords[2]
        back_line = LineString([c_br, c_bl])
        if back_line.length < 1e-4:
            return True
        pts = [back_line.interpolate(frac, normalized=True) for frac in np.linspace(0.05, 0.95, 15)]
        h, w = wall_dist_map_m.shape
        in_contact = 0
        for pt in pts:
            ix = int(round(pt.x / scale_m_per_px))
            iy = int(round(pt.y / scale_m_per_px))
            if 0 <= ix < w and 0 <= iy < h:
                if wall_dist_map_m[iy, ix] <= threshold_m:
                    in_contact += 1
        return (in_contact / len(pts)) >= required_ratio

    def furnish_room(
        self,
        room_type: str,
        room_poly: Polygon,
        usable_segments: List[UsableWallSegment],
        door_clearance_zones: List[Polygon],
        preset: str = "contemporary_minimal",
        seed: int = 42,
        wall_dist_map_m: Optional[np.ndarray] = None,
        scale_m_per_px: float = 0.03,
        force_furnish: bool = False
    ) -> Tuple[List[PlacedItem], Optional[str]]:
        """
        Generates candidate furniture placements according to room type rules.
        Returns: (List[PlacedItem], failure_reason_if_any)
        """
        rng = random.Random(seed)
        norm_type = self._normalize_room_type(room_type)
        area = room_poly.area

        if norm_type == "bedroom":
            return self._furnish_bedroom(room_poly, usable_segments, door_clearance_zones, preset, area, rng, wall_dist_map_m, scale_m_per_px, force_furnish=force_furnish)
        elif norm_type == "living":
            return self._furnish_living(room_poly, usable_segments, door_clearance_zones, preset, area, rng, wall_dist_map_m, scale_m_per_px, force_furnish=force_furnish)
        elif norm_type == "kitchen":
            return self._furnish_kitchen(room_poly, usable_segments, door_clearance_zones, preset, area, rng, wall_dist_map_m, scale_m_per_px, force_furnish=force_furnish)
        elif norm_type == "bathroom":
            return self._furnish_bathroom(room_poly, usable_segments, door_clearance_zones, preset, area, rng, wall_dist_map_m, scale_m_per_px, force_furnish=force_furnish)
        elif norm_type == "dining":
            return self._furnish_dining(room_poly, usable_segments, door_clearance_zones, preset, area, rng, wall_dist_map_m, scale_m_per_px, force_furnish=force_furnish)
        elif norm_type == "office":
            return self._furnish_office(room_poly, usable_segments, door_clearance_zones, preset, area, rng, wall_dist_map_m, scale_m_per_px, force_furnish=force_furnish)
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

    STANDARD_REF_AREAS = {
        "bedroom": 8.5,
        "living": 13.0,
        "kitchen": 6.0,
        "bathroom": 2.5,
        "dining": 8.0,
        "office": 6.0,
    }

    def _get_compact_scale(self, norm_type: str, area: float, force_furnish: bool) -> float:
        """
        Computes proportional scaling factor for furniture in sub-threshold / compact rooms.
        When door estimation compresses pixel dimensions or a compact room is furnished,
        furniture is scaled down proportionally to prevent oversized items crowding the space.
        """
        ref_area = self.STANDARD_REF_AREAS.get(norm_type, 8.0)
        min_cutoff = MIN_PLAUSIBLE_ROOM_AREAS.get(norm_type, 4.0)
        if force_furnish or area < min_cutoff:
            if area < ref_area:
                scale = float(np.sqrt(area / ref_area))
                return float(np.clip(scale, 0.45, 0.90))
        return 1.0

    def _synthesize_fallback_segments(self, room_poly: Polygon) -> List[UsableWallSegment]:
        coords = list(room_poly.exterior.coords)[:-1]
        n = len(coords)
        segs = []
        for i in range(n):
            p1 = np.array(coords[i])
            p2 = np.array(coords[(i + 1) % n])
            edge_v = p2 - p1
            edge_len = float(np.linalg.norm(edge_v))
            if edge_len < 0.4:
                continue
            tangent = edge_v / edge_len
            inward_normal = (-float(tangent[1]), float(tangent[0]))
            yaw_deg = float(np.degrees(np.arctan2(inward_normal[0], inward_normal[1])) % 360)
            segs.append(UsableWallSegment(
                start=(float(p1[0]), float(p1[1])),
                end=(float(p2[0]), float(p2[1])),
                length_m=round(edge_len, 3),
                normal=inward_normal,
                yaw_deg=yaw_deg,
                edge_idx=i,
                wall_distance_m=0.0
            ))
        segs.sort(key=lambda s: s.length_m, reverse=True)
        return segs

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
        rng: random.Random,
        wall_dist_map_m: Optional[np.ndarray] = None,
        scale_m_per_px: float = 0.03,
        force_furnish: bool = False
    ) -> Tuple[List[PlacedItem], Optional[str]]:
        min_area = 1.8 if force_furnish else 4.0
        if area < min_area:
            return [], "TOO_SMALL"

        placed = []
        # Sort usable wall segments by length descending
        sorted_segs = sorted(usable_segments, key=lambda s: s.length_m, reverse=True)
        if not sorted_segs and force_furnish:
            sorted_segs = self._synthesize_fallback_segments(room_poly)
        if not sorted_segs:
            return [], "NO_USABLE_WALL"

        # Determine bed size and compact scale
        c_scale = self._get_compact_scale("bedroom", area, force_furnish)
        scale_tuple = (round(c_scale, 3), round(c_scale, 3), round(c_scale, 3))

        use_double = (area >= 7.5 and sorted_segs[0].length_m >= 1.80 and c_scale >= 0.95)
        bed_id = "bed_double" if use_double else "bed_single"
        bed_spec = self._get_item_spec(bed_id)
        raw_bw, raw_bd, raw_bh = bed_spec["dimensions_m"]
        bw, bd, bh = raw_bw * c_scale, raw_bd * c_scale, raw_bh * c_scale

        # 1. Place Primary Bed on longest usable wall segment that satisfies >=80% wall contact
        bed_seg = None
        bed_center = None
        bed_poly = None
        bed_poly_coords = None

        for seg in sorted_segs:
            if seg.length_m < (bw + 0.10):
                continue
            seg_len = seg.length_m
            p1 = np.array(seg.start)
            p2 = np.array(seg.end)
            u_dir = (p2 - p1) / max(seg_len, 1e-6)
            n_in = np.array(seg.normal)

            center_s = seg_len / 2.0
            found_candidate = False

            offsets = [0.0] + ([-0.4, 0.4, -0.7, 0.7] if seg_len >= (bw + 0.8) else [-0.2, 0.2])
            for shift in offsets:
                trial_s = center_s + shift
                if (bw / 2.0) <= trial_s <= (seg_len - bw / 2.0):
                    b_pt = p1 + u_dir * trial_s
                    b_c = b_pt + n_in * (bd / 2.0)
                    t_poly_coords = compute_footprint_polygon(float(b_c[0]), float(b_c[1]), bw, bd, seg.yaw_deg)
                    t_poly = Polygon(t_poly_coords)
                    if room_poly.buffer(0.02).contains(t_poly) and not any(dz.intersects(t_poly) for dz in door_zones):
                        if self._check_back_edge_contact(t_poly_coords, wall_dist_map_m, scale_m_per_px):
                            bed_seg = seg
                            bed_center = b_c
                            bed_poly = t_poly
                            bed_poly_coords = t_poly_coords
                            center_s = trial_s
                            found_candidate = True
                            break
            if found_candidate:
                break

        if bed_seg is None and force_furnish and sorted_segs:
            seg = sorted_segs[0]
            seg_len = seg.length_m
            p1 = np.array(seg.start)
            p2 = np.array(seg.end)
            u_dir = (p2 - p1) / max(seg_len, 1e-6)
            n_in = np.array(seg.normal)
            center_s = seg_len / 2.0
            b_c = (p1 + p2) / 2.0 + n_in * (bd / 2.0)
            t_poly_coords = compute_footprint_polygon(float(b_c[0]), float(b_c[1]), bw, bd, seg.yaw_deg)
            bed_seg = seg
            bed_center = b_c
            bed_poly = Polygon(t_poly_coords)
            bed_poly_coords = t_poly_coords

        if bed_seg is None:
            return [], "NO_USABLE_WALL"

        bed_item = PlacedItem(
            item_id="bed_0",
            catalog_id=bed_id,
            name=bed_spec["name"],
            category="bed",
            position_m=(float(bed_center[0]), float(bed_center[1])),
            rotation_deg=bed_seg.yaw_deg,
            dimensions_m=(round(bw, 3), round(bd, 3), round(bh, 3)),
            glb=bed_spec["glb"],
            polygon_coords=bed_poly_coords,
            placement_rule="back_to_wall",
            scale=scale_tuple
        )
        placed.append(bed_item)

        # 2. Flanking Nightstands (if space permits on sides)
        ns_spec = self._get_item_spec("nightstand")
        raw_nw, raw_nd, raw_nh = ns_spec["dimensions_m"]
        nw, nd, nh = raw_nw * c_scale, raw_nd * c_scale, raw_nh * c_scale
        left_space = center_s - (bw / 2.0)
        right_space = seg_len - (center_s + bw / 2.0)

        if left_space >= (nw + 0.05):
            ns_s = center_s - (bw / 2.0) - (nw / 2.0) - 0.04
            ns_back = p1 + u_dir * ns_s
            ns_c = ns_back + n_in * (nd / 2.0)
            ns_poly = Polygon(compute_footprint_polygon(float(ns_c[0]), float(ns_c[1]), nw, nd, bed_seg.yaw_deg))
            if not ns_poly.intersects(bed_poly) and room_poly.buffer(0.02).contains(ns_poly) and not any(dz.intersects(ns_poly) for dz in door_zones):
                placed.append(PlacedItem(
                    item_id="nightstand_left",
                    catalog_id="nightstand",
                    name=ns_spec["name"],
                    category="nightstand",
                    position_m=(float(ns_c[0]), float(ns_c[1])),
                    rotation_deg=bed_seg.yaw_deg,
                    dimensions_m=(round(nw, 3), round(nd, 3), round(nh, 3)),
                    glb=ns_spec["glb"],
                    polygon_coords=list(ns_poly.exterior.coords)[:-1],
                    placement_rule="adjacent_anchor",
                    scale=scale_tuple
                ))

        if right_space >= (nw + 0.05):
            ns_s = center_s + (bw / 2.0) + (nw / 2.0) + 0.04
            ns_back = p1 + u_dir * ns_s
            ns_c = ns_back + n_in * (nd / 2.0)
            ns_poly = Polygon(compute_footprint_polygon(float(ns_c[0]), float(ns_c[1]), nw, nd, bed_seg.yaw_deg))
            if not ns_poly.intersects(bed_poly) and room_poly.buffer(0.02).contains(ns_poly) and not any(dz.intersects(ns_poly) for dz in door_zones):
                placed.append(PlacedItem(
                    item_id="nightstand_right",
                    catalog_id="nightstand",
                    name=ns_spec["name"],
                    category="nightstand",
                    position_m=(float(ns_c[0]), float(ns_c[1])),
                    rotation_deg=bed_seg.yaw_deg,
                    dimensions_m=(round(nw, 3), round(nd, 3), round(nh, 3)),
                    glb=ns_spec["glb"],
                    polygon_coords=list(ns_poly.exterior.coords)[:-1],
                    placement_rule="adjacent_anchor",
                    scale=scale_tuple
                ))

        # 3. Wardrobe on Secondary Wall
        ward_spec = self._get_item_spec("wardrobe")
        raw_ww, raw_wd, raw_wh = ward_spec["dimensions_m"]
        ww, wd, wh = raw_ww * c_scale, raw_wd * c_scale, raw_wh * c_scale

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

                w_poly_coords = compute_footprint_polygon(float(w_c[0]), float(w_c[1]), ww, wd, seg.yaw_deg)
                w_poly = Polygon(w_poly_coords)
                if room_poly.buffer(0.02).contains(w_poly) and not any(dz.intersects(w_poly) for dz in door_zones):
                    if self._check_back_edge_contact(w_poly_coords, wall_dist_map_m, scale_m_per_px):
                        if not any(w_poly.intersects(pp) for pp in placed_polys):
                            placed.append(PlacedItem(
                                item_id="wardrobe_0",
                                catalog_id="wardrobe",
                                name=ward_spec["name"],
                                category="wardrobe",
                                position_m=(float(w_c[0]), float(w_c[1])),
                                rotation_deg=seg.yaw_deg,
                                dimensions_m=(round(ww, 3), round(wd, 3), round(wh, 3)),
                                glb=ward_spec["glb"],
                                polygon_coords=w_poly_coords,
                                placement_rule="back_to_wall",
                                scale=scale_tuple
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
        rng: random.Random,
        wall_dist_map_m: Optional[np.ndarray] = None,
        scale_m_per_px: float = 0.03,
        force_furnish: bool = False
    ) -> Tuple[List[PlacedItem], Optional[str]]:
        min_area = 2.0 if force_furnish else 6.0
        if area < min_area:
            return [], "TOO_SMALL"

        placed = []
        sorted_segs = sorted(usable_segments, key=lambda s: s.length_m, reverse=True)
        if not sorted_segs and force_furnish:
            sorted_segs = self._synthesize_fallback_segments(room_poly)
        if not sorted_segs:
            return [], "NO_USABLE_WALL"

        # Determine sofa scale for compact mode
        c_scale = self._get_compact_scale("living", area, force_furnish)
        scale_tuple = (round(c_scale, 3), round(c_scale, 3), round(c_scale, 3))

        # Sofa selection based on preset
        sofa_id = "sofa_modern" if preset == "contemporary_minimal" else ("sofa_classic" if preset == "tailored_classic" else "sofa_3seater")
        sofa_spec = self._get_item_spec(sofa_id)
        raw_sw, raw_sd, raw_sh = sofa_spec["dimensions_m"]
        sw, sd, sh = raw_sw * c_scale, raw_sd * c_scale, raw_sh * c_scale

        sofa_seg = None
        sofa_center = None
        sofa_poly = None
        sofa_poly_coords = None

        candidate_sofas = [sofa_spec]
        fallback_3s = self._get_item_spec("sofa_3seater")
        if fallback_3s.get("id") != sofa_spec.get("id"):
            candidate_sofas.append(fallback_3s)

        for current_sofa in candidate_sofas:
            raw_csw, raw_csd, raw_csh = current_sofa["dimensions_m"]
            c_sw, c_sd, c_sh = raw_csw * c_scale, raw_csd * c_scale, raw_csh * c_scale
            for seg in sorted_segs:
                if seg.length_m < (c_sw + 0.10):
                    continue
                seg_len = seg.length_m
                p1 = np.array(seg.start)
                p2 = np.array(seg.end)
                u_dir = (p2 - p1) / max(seg_len, 1e-6)
                n_in = np.array(seg.normal)

                center_s = seg_len / 2.0
                offsets = [0.0] + ([-0.5, 0.5, -0.9, 0.9] if seg_len >= (c_sw + 1.0) else [-0.3, 0.3])
                for shift in offsets:
                    trial_s = center_s + shift
                    if (c_sw / 2.0) <= trial_s <= (seg_len - c_sw / 2.0):
                        b_pt = p1 + u_dir * trial_s
                        s_c = b_pt + n_in * (c_sd / 2.0)
                        t_coords = compute_footprint_polygon(float(s_c[0]), float(s_c[1]), c_sw, c_sd, seg.yaw_deg)
                        t_poly = Polygon(t_coords)
                        if room_poly.buffer(0.02).contains(t_poly) and not any(dz.intersects(t_poly) for dz in door_zones):
                            if self._check_back_edge_contact(t_coords, wall_dist_map_m, scale_m_per_px):
                                sofa_spec = current_sofa
                                sw, sd, sh = c_sw, c_sd, c_sh
                                sofa_seg = seg
                                sofa_center = s_c
                                sofa_poly = t_poly
                                sofa_poly_coords = t_coords
                                break
                if sofa_seg is not None:
                    break
            if sofa_seg is not None:
                break

        if sofa_seg is None and force_furnish and sorted_segs:
            seg = sorted_segs[0]
            p1 = np.array(seg.start)
            p2 = np.array(seg.end)
            u_dir = (p2 - p1) / max(seg.length_m, 1e-6)
            n_in = np.array(seg.normal)
            s_c = (p1 + p2) / 2.0 + n_in * (sd / 2.0)
            t_coords = compute_footprint_polygon(float(s_c[0]), float(s_c[1]), sw, sd, seg.yaw_deg)
            sofa_seg = seg
            sofa_center = s_c
            sofa_poly = Polygon(t_coords)
            sofa_poly_coords = t_coords

        if sofa_seg is None:
            return [], "NO_USABLE_WALL"

        placed.append(PlacedItem(
            item_id="sofa_0",
            catalog_id=sofa_spec.get("id", sofa_id),
            name=sofa_spec["name"],
            category="sofa",
            position_m=(float(sofa_center[0]), float(sofa_center[1])),
            rotation_deg=sofa_seg.yaw_deg,
            dimensions_m=(round(sw, 3), round(sd, 3), round(sh, 3)),
            glb=sofa_spec["glb"],
            polygon_coords=sofa_poly_coords,
            placement_rule="back_to_wall",
            scale=scale_tuple
        ))

        # 2. Coffee Table (in front of sofa at distance ~0.45m)
        ct_id = "coffee_table_glass" if preset == "contemporary_minimal" else "coffee_table"
        ct_spec = self._get_item_spec(ct_id)
        raw_cw, raw_cd, raw_ch = ct_spec["dimensions_m"]
        cw, cd, ch = raw_cw * c_scale, raw_cd * c_scale, raw_ch * c_scale

        ct_dist = (sd / 2.0) + (0.40 * c_scale) + (cd / 2.0)
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
                dimensions_m=(round(cw, 3), round(cd, 3), round(ch, 3)),
                glb=ct_spec["glb"],
                polygon_coords=list(ct_poly.exterior.coords)[:-1],
                placement_rule="in_front_of_anchor",
                scale=scale_tuple
            ))

        # 3. Area Rug (under sofa front and coffee table)
        rug_spec = self._get_item_spec("rug_living")
        raw_rw, raw_rd, raw_rh = rug_spec["dimensions_m"]
        rw, rd, rh = raw_rw * c_scale, raw_rd * c_scale, raw_rh * c_scale
        rug_dist = (sd / 2.0) + (0.30 * c_scale)
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
                dimensions_m=(round(rw, 3), round(rd, 3), round(rh, 3)),
                glb=rug_spec["glb"],
                polygon_coords=list(rug_poly.exterior.coords)[:-1],
                placement_rule="under_anchor",
                scale=scale_tuple
            ))

        # 4. TV Unit on facing or adjacent wall
        tv_spec = self._get_item_spec("tv_unit")
        raw_tw, raw_td, raw_th = tv_spec["dimensions_m"]
        tw, td, th = raw_tw * c_scale, raw_td * c_scale, raw_th * c_scale
        placed_polys = [Polygon(p.polygon_coords) for p in placed if p.category != "rug"]

        for seg in sorted_segs:
            if seg == sofa_seg:
                continue
            if seg.length_m >= (tw + 0.10):
                # Check distance from sofa
                seg_mid = (np.array(seg.start) + np.array(seg.end)) / 2.0
                dist_to_sofa = np.linalg.norm(seg_mid - sofa_center)
                min_dist = 1.4 if force_furnish else 2.0
                if dist_to_sofa >= min_dist:
                    t_p1 = np.array(seg.start)
                    t_u = (np.array(seg.end) - t_p1) / seg.length_m
                    t_nin = np.array(seg.normal)
                    t_back = t_p1 + t_u * (seg.length_m / 2.0)
                    t_c = t_back + t_nin * (td / 2.0)
                    t_poly_coords = compute_footprint_polygon(float(t_c[0]), float(t_c[1]), tw, td, seg.yaw_deg)
                    tv_poly = Polygon(t_poly_coords)

                    if room_poly.buffer(0.02).contains(tv_poly) and not any(dz.intersects(tv_poly) for dz in door_zones):
                        if self._check_back_edge_contact(t_poly_coords, wall_dist_map_m, scale_m_per_px):
                            if not any(tv_poly.intersects(pp) for pp in placed_polys):
                                placed.append(PlacedItem(
                                    item_id="tv_unit_0",
                                    catalog_id="tv_unit",
                                    name=tv_spec["name"],
                                    category="tv_unit",
                                    position_m=(float(t_c[0]), float(t_c[1])),
                                    rotation_deg=seg.yaw_deg,
                                    dimensions_m=(round(tw, 3), round(td, 3), round(th, 3)),
                                    glb=tv_spec["glb"],
                                    polygon_coords=t_poly_coords,
                                    placement_rule="back_to_wall",
                                    scale=scale_tuple
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
        rng: random.Random,
        wall_dist_map_m: Optional[np.ndarray] = None,
        scale_m_per_px: float = 0.03,
        force_furnish: bool = False
    ) -> Tuple[List[PlacedItem], Optional[str]]:
        min_area = 1.0 if force_furnish else 3.5
        if area < min_area:
            return [], "TOO_SMALL"

        placed = []
        sorted_segs = sorted(usable_segments, key=lambda s: s.length_m, reverse=True)
        if not sorted_segs and force_furnish:
            sorted_segs = self._synthesize_fallback_segments(room_poly)
        if not sorted_segs:
            return [], "NO_USABLE_WALL"

        # Determine kitchen scale for compact mode
        c_scale = self._get_compact_scale("kitchen", area, force_furnish)
        scale_tuple = (round(c_scale, 3), round(c_scale, 3), round(c_scale, 3))

        # Kitchen run components (depth 0.60m)
        counter_spec = self._get_item_spec("kitchen_counter")
        sink_spec = self._get_item_spec("kitchen_sink_unit")
        fridge_spec = self._get_item_spec("refrigerator")

        raw_cw, raw_cd, raw_ch = counter_spec["dimensions_m"]
        raw_sw, raw_sd, raw_sh = sink_spec["dimensions_m"]
        raw_fw, raw_fd, raw_fh = fridge_spec["dimensions_m"]

        cw, cd, ch = raw_cw * c_scale, raw_cd * c_scale, raw_ch * c_scale
        sw, sd, sh = raw_sw * c_scale, raw_sd * c_scale, raw_sh * c_scale
        fw, fd, fh = raw_fw * c_scale, raw_fd * c_scale, raw_fh * c_scale

        primary_seg = None
        sink_placed = False
        curr_s = 0.10
        p1, p2, u_dir, n_in = None, None, None, None
        seg_len = 0.0

        for cand_seg in sorted_segs:
            if cand_seg.length_m < (sw + 0.10):
                continue
            cand_len = cand_seg.length_m
            cand_p1 = np.array(cand_seg.start)
            cand_p2 = np.array(cand_seg.end)
            cand_u = (cand_p2 - cand_p1) / max(cand_len, 1e-6)
            cand_nin = np.array(cand_seg.normal)

            for s_trial in [0.10, max(0.0, cand_len - (sw + 0.10))]:
                if (s_trial + sw) <= cand_len and s_trial >= 0.0:
                    sink_center_s = s_trial + sw / 2.0
                    sink_back = cand_p1 + cand_u * sink_center_s
                    sink_c = sink_back + cand_nin * (sd / 2.0)
                    sink_coords = compute_footprint_polygon(float(sink_c[0]), float(sink_c[1]), sw, sd, cand_seg.yaw_deg)
                    sink_poly = Polygon(sink_coords)

                    if room_poly.buffer(0.02).contains(sink_poly) and not any(dz.intersects(sink_poly) for dz in door_zones):
                        if self._check_back_edge_contact(sink_coords, wall_dist_map_m, scale_m_per_px):
                            primary_seg = cand_seg
                            seg_len = cand_len
                            p1, p2, u_dir, n_in = cand_p1, cand_p2, cand_u, cand_nin
                            placed.append(PlacedItem(
                                item_id="kitchen_sink_0",
                                catalog_id="kitchen_sink_unit",
                                name=sink_spec["name"],
                                category="kitchen_sink",
                                position_m=(float(sink_c[0]), float(sink_c[1])),
                                rotation_deg=cand_seg.yaw_deg,
                                dimensions_m=(round(sw, 3), round(sd, 3), round(sh, 3)),
                                glb=sink_spec["glb"],
                                polygon_coords=sink_coords,
                                placement_rule="back_to_wall",
                                scale=scale_tuple
                            ))
                            curr_s = s_trial + sw
                            sink_placed = True
                            break
            if sink_placed:
                break

        if (not sink_placed or primary_seg is None) and force_furnish and sorted_segs:
            primary_seg = sorted_segs[0]
            cand_p1 = np.array(primary_seg.start)
            cand_p2 = np.array(primary_seg.end)
            cand_u = (cand_p2 - cand_p1) / max(primary_seg.length_m, 1e-6)
            cand_nin = np.array(primary_seg.normal)
            sink_c = (cand_p1 + cand_p2) / 2.0 + cand_nin * (sd / 2.0)
            sink_coords = compute_footprint_polygon(float(sink_c[0]), float(sink_c[1]), sw, sd, primary_seg.yaw_deg)
            placed.append(PlacedItem(
                item_id="kitchen_sink_0",
                catalog_id="kitchen_sink_unit",
                name=sink_spec["name"],
                category="kitchen_sink",
                position_m=(float(sink_c[0]), float(sink_c[1])),
                rotation_deg=primary_seg.yaw_deg,
                dimensions_m=(round(sw, 3), round(sd, 3), round(sh, 3)),
                glb=sink_spec["glb"],
                polygon_coords=sink_coords,
                placement_rule="back_to_wall",
                scale=scale_tuple
            ))
            sink_placed = True

        if not sink_placed or primary_seg is None:
            return [], "NO_USABLE_WALL"

        # 2. Adjacent Base Counter Unit
        counter_placed = False
        if (curr_s + cw) <= seg_len:
            cnt_center_s = curr_s + cw / 2.0
            cnt_back = p1 + u_dir * cnt_center_s
            cnt_c = cnt_back + n_in * (cd / 2.0)
            cnt_coords = compute_footprint_polygon(float(cnt_c[0]), float(cnt_c[1]), cw, cd, primary_seg.yaw_deg)
            cnt_poly = Polygon(cnt_coords)

            if room_poly.buffer(0.02).contains(cnt_poly) and not any(dz.intersects(cnt_poly) for dz in door_zones):
                if self._check_back_edge_contact(cnt_coords, wall_dist_map_m, scale_m_per_px):
                    placed.append(PlacedItem(
                        item_id="kitchen_counter_0",
                        catalog_id="kitchen_counter",
                        name=counter_spec["name"],
                        category="kitchen_counter",
                        position_m=(float(cnt_c[0]), float(cnt_c[1])),
                        rotation_deg=primary_seg.yaw_deg,
                        dimensions_m=(round(cw, 3), round(cd, 3), round(ch, 3)),
                        glb=counter_spec["glb"],
                        polygon_coords=cnt_coords,
                        placement_rule="back_to_wall",
                        scale=scale_tuple
                    ))
                    curr_s += cw
                    counter_placed = True

        # Fallback: if counter didn't fit on primary wall, try secondary wall
        if not counter_placed and len(sorted_segs) > 1:
            sec_cand = sorted_segs[1]
            if sec_cand.length_m >= (cw + 0.10):
                cp1 = np.array(sec_cand.start)
                cu = (np.array(sec_cand.end) - cp1) / sec_cand.length_m
                cnin = np.array(sec_cand.normal)
                c_back = cp1 + cu * (cw / 2.0 + 0.10)
                c_c = c_back + cnin * (cd / 2.0)
                c_coords = compute_footprint_polygon(float(c_c[0]), float(c_c[1]), cw, cd, sec_cand.yaw_deg)
                c_poly = Polygon(c_coords)
                placed_polys = [Polygon(p.polygon_coords) for p in placed]
                if room_poly.buffer(0.02).contains(c_poly) and not any(dz.intersects(c_poly) for dz in door_zones):
                    if not any(c_poly.intersects(pp) for pp in placed_polys):
                        if self._check_back_edge_contact(c_coords, wall_dist_map_m, scale_m_per_px):
                            placed.append(PlacedItem(
                                item_id="kitchen_counter_0",
                                catalog_id="kitchen_counter",
                                name=counter_spec["name"],
                                category="kitchen_counter",
                                position_m=(float(c_c[0]), float(c_c[1])),
                                rotation_deg=sec_cand.yaw_deg,
                                dimensions_m=(round(cw, 3), round(cd, 3), round(ch, 3)),
                                glb=counter_spec["glb"],
                                polygon_coords=c_coords,
                                placement_rule="back_to_wall",
                                scale=scale_tuple
                            ))
                            counter_placed = True

        # 3. Refrigerator (at end of counter run or on secondary wall)
        fridge_placed = False
        placed_polys = [Polygon(p.polygon_coords) for p in placed]

        if (curr_s + fw) <= seg_len:
            fr_center_s = curr_s + fw / 2.0
            fr_back = p1 + u_dir * fr_center_s
            fr_c = fr_back + n_in * (fd / 2.0)
            fr_coords = compute_footprint_polygon(float(fr_c[0]), float(fr_c[1]), fw, fd, primary_seg.yaw_deg)
            fr_poly = Polygon(fr_coords)

            if room_poly.buffer(0.02).contains(fr_poly) and not any(dz.intersects(fr_poly) for dz in door_zones):
                if self._check_back_edge_contact(fr_coords, wall_dist_map_m, scale_m_per_px):
                    placed.append(PlacedItem(
                        item_id="refrigerator_0",
                        catalog_id="refrigerator",
                        name=fridge_spec["name"],
                        category="refrigerator",
                        position_m=(float(fr_c[0]), float(fr_c[1])),
                        rotation_deg=primary_seg.yaw_deg,
                        dimensions_m=(round(fw, 3), round(fd, 3), round(fh, 3)),
                        glb=fridge_spec["glb"],
                        polygon_coords=fr_coords,
                        placement_rule="back_to_wall",
                        scale=scale_tuple
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
                fr_coords = compute_footprint_polygon(float(fr_c[0]), float(fr_c[1]), fw, fd, sec_seg.yaw_deg)
                fr_poly = Polygon(fr_coords)

                if room_poly.buffer(0.02).contains(fr_poly) and not any(dz.intersects(fr_poly) for dz in door_zones):
                    if not any(fr_poly.intersects(pp) for pp in placed_polys):
                        if self._check_back_edge_contact(fr_coords, wall_dist_map_m, scale_m_per_px):
                            placed.append(PlacedItem(
                                item_id="refrigerator_0",
                                catalog_id="refrigerator",
                                name=fridge_spec["name"],
                                category="refrigerator",
                                position_m=(float(fr_c[0]), float(fr_c[1])),
                                rotation_deg=sec_seg.yaw_deg,
                                dimensions_m=(round(fw, 3), round(fd, 3), round(fh, 3)),
                                glb=fridge_spec["glb"],
                                polygon_coords=fr_coords,
                                placement_rule="back_to_wall",
                                scale=scale_tuple
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
        rng: random.Random,
        wall_dist_map_m: Optional[np.ndarray] = None,
        scale_m_per_px: float = 0.03,
        force_furnish: bool = False
    ) -> Tuple[List[PlacedItem], Optional[str]]:
        min_area = 0.5 if force_furnish else 1.8
        if area < min_area:
            return [], "TOO_SMALL"

        placed = []
        sorted_segs = sorted(usable_segments, key=lambda s: s.length_m, reverse=True)
        if not sorted_segs and force_furnish:
            sorted_segs = self._synthesize_fallback_segments(room_poly)
        if not sorted_segs:
            return [], "NO_USABLE_WALL"

        # Determine bathroom scale for compact mode
        c_scale = self._get_compact_scale("bathroom", area, force_furnish)
        scale_tuple = (round(c_scale, 3), round(c_scale, 3), round(c_scale, 3))

        toilet_spec = self._get_item_spec("toilet")
        raw_tw, raw_td, raw_th = toilet_spec["dimensions_m"]
        tw, td, th = raw_tw * c_scale, raw_td * c_scale, raw_th * c_scale

        vanity_spec = self._get_item_spec("bathroom_vanity")
        raw_vw, raw_vd, raw_vh = vanity_spec["dimensions_m"]
        vw, vd, vh = raw_vw * c_scale, raw_vd * c_scale, raw_vh * c_scale

        # 1. Place Toilet with Multi-Wall and Adaptive Corner Clearance Fallbacks
        toilet_placed = False
        toilet_seg_idx = -1

        for s_idx, t_seg in enumerate(sorted_segs):
            if t_seg.length_m < (tw + 0.05):
                continue

            p1 = np.array(t_seg.start)
            seg_len = t_seg.length_m
            u_dir = (np.array(t_seg.end) - p1) / max(seg_len, 1e-6)
            n_in = np.array(t_seg.normal)

            candidate_offsets = [
                tw / 2.0 + 0.15,
                seg_len - (tw / 2.0 + 0.15),
                seg_len / 2.0,
                tw / 2.0 + 0.05,
                seg_len - (tw / 2.0 + 0.05)
            ]

            for t_center_s in candidate_offsets:
                if (tw / 2.0) <= t_center_s <= (seg_len - tw / 2.0):
                    t_back = p1 + u_dir * t_center_s
                    t_c = t_back + n_in * (td / 2.0)
                    t_coords = compute_footprint_polygon(float(t_c[0]), float(t_c[1]), tw, td, t_seg.yaw_deg)
                    t_poly = Polygon(t_coords)

                    if not any(dz.intersects(t_poly) for dz in door_zones) and room_poly.buffer(0.02).contains(t_poly):
                        if self._check_back_edge_contact(t_coords, wall_dist_map_m, scale_m_per_px):
                            placed.append(PlacedItem(
                                item_id="toilet_0",
                                catalog_id="toilet",
                                name=toilet_spec["name"],
                                category="toilet",
                                position_m=(float(t_c[0]), float(t_c[1])),
                                rotation_deg=t_seg.yaw_deg,
                                dimensions_m=(round(tw, 3), round(td, 3), round(th, 3)),
                                glb=toilet_spec["glb"],
                                polygon_coords=t_coords,
                                placement_rule="back_to_wall",
                                scale=scale_tuple
                            ))
                            toilet_placed = True
                            toilet_seg_idx = s_idx
                            break

            if toilet_placed:
                break

        if not toilet_placed and force_furnish and sorted_segs:
            t_seg = sorted_segs[0]
            p1 = np.array(t_seg.start)
            seg_len = t_seg.length_m
            n_in = np.array(t_seg.normal)
            t_c = (p1 + np.array(t_seg.end)) / 2.0 + n_in * (td / 2.0)
            t_coords = compute_footprint_polygon(float(t_c[0]), float(t_c[1]), tw, td, t_seg.yaw_deg)
            placed.append(PlacedItem(
                item_id="toilet_0",
                catalog_id="toilet",
                name=toilet_spec["name"],
                category="toilet",
                position_m=(float(t_c[0]), float(t_c[1])),
                rotation_deg=t_seg.yaw_deg,
                dimensions_m=(round(tw, 3), round(td, 3), round(th, 3)),
                glb=toilet_spec["glb"],
                polygon_coords=t_coords,
                placement_rule="back_to_wall",
                scale=scale_tuple
            ))
            toilet_placed = True
            toilet_seg_idx = 0

        # 2. Place Vanity Sink (on remaining walls or shared wall)
        placed_polys = [Polygon(p.polygon_coords) for p in placed]
        vanity_placed = False

        for s_idx, v_seg in enumerate(sorted_segs):
            if s_idx == toilet_seg_idx:
                # Same wall as toilet
                seg_p1 = np.array(v_seg.start)
                seg_len = v_seg.length_m
                if seg_len >= (tw + vw + 0.40):
                    u_dir = (np.array(v_seg.end) - seg_p1) / max(seg_len, 1e-6)
                    n_in = np.array(v_seg.normal)
                    for v_s in [vw / 2.0 + 0.10, seg_len - (vw / 2.0 + 0.10)]:
                        v_back = seg_p1 + u_dir * v_s
                        v_c = v_back + n_in * (vd / 2.0)
                        v_coords = compute_footprint_polygon(float(v_c[0]), float(v_c[1]), vw, vd, v_seg.yaw_deg)
                        v_poly = Polygon(v_coords)
                        if room_poly.buffer(0.02).contains(v_poly) and not any(dz.intersects(v_poly) for dz in door_zones):
                            if not any(v_poly.intersects(pp) for pp in placed_polys):
                                if self._check_back_edge_contact(v_coords, wall_dist_map_m, scale_m_per_px):
                                    placed.append(PlacedItem(
                                        item_id="vanity_0",
                                        catalog_id="bathroom_vanity",
                                        name=vanity_spec["name"],
                                        category="bathroom_sink",
                                        position_m=(float(v_c[0]), float(v_c[1])),
                                        rotation_deg=v_seg.yaw_deg,
                                        dimensions_m=(round(vw, 3), round(vd, 3), round(vh, 3)),
                                        glb=vanity_spec["glb"],
                                        polygon_coords=v_coords,
                                        placement_rule="back_to_wall",
                                        scale=scale_tuple
                                    ))
                                    vanity_placed = True
                                    break
            else:
                # Different wall from toilet
                if v_seg.length_m >= (vw + 0.05):
                    vp1 = np.array(v_seg.start)
                    v_len = v_seg.length_m
                    vu = (np.array(v_seg.end) - vp1) / max(v_len, 1e-6)
                    vnin = np.array(v_seg.normal)
                    for v_s in [vw / 2.0 + 0.10, v_len / 2.0, v_len - (vw / 2.0 + 0.10)]:
                        if (vw / 2.0) <= v_s <= (v_len - vw / 2.0):
                            v_back = vp1 + vu * v_s
                            v_c = v_back + vnin * (vd / 2.0)
                            v_coords = compute_footprint_polygon(float(v_c[0]), float(v_c[1]), vw, vd, v_seg.yaw_deg)
                            v_poly = Polygon(v_coords)
                            if room_poly.buffer(0.02).contains(v_poly) and not any(dz.intersects(v_poly) for dz in door_zones):
                                if not any(v_poly.intersects(pp) for pp in placed_polys):
                                    if self._check_back_edge_contact(v_coords, wall_dist_map_m, scale_m_per_px):
                                        placed.append(PlacedItem(
                                            item_id="vanity_0",
                                            catalog_id="bathroom_vanity",
                                            name=vanity_spec["name"],
                                            category="bathroom_sink",
                                            position_m=(float(v_c[0]), float(v_c[1])),
                                            rotation_deg=v_seg.yaw_deg,
                                            dimensions_m=(round(vw, 3), round(vd, 3), round(vh, 3)),
                                            glb=vanity_spec["glb"],
                                            polygon_coords=v_coords,
                                            placement_rule="back_to_wall",
                                            scale=scale_tuple
                                        ))
                                        vanity_placed = True
                                        break
            if vanity_placed:
                break

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
                    b_coords = compute_footprint_polygon(float(b_c[0]), float(b_c[1]), bw, bd, seg.yaw_deg)
                    b_poly = Polygon(b_coords)

                    if room_poly.buffer(0.02).contains(b_poly) and not any(dz.intersects(b_poly) for dz in door_zones):
                        if not any(b_poly.intersects(pp) for pp in placed_polys):
                            if self._check_back_edge_contact(b_coords, wall_dist_map_m, scale_m_per_px):
                                placed.append(PlacedItem(
                                    item_id="bathtub_0",
                                    catalog_id="bathtub",
                                    name=bt_spec["name"],
                                    category="bathtub",
                                    position_m=(float(b_c[0]), float(b_c[1])),
                                    rotation_deg=seg.yaw_deg,
                                    dimensions_m=(bw, bd, bh),
                                    glb=bt_spec["glb"],
                                    polygon_coords=b_coords,
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
        rng: random.Random,
        wall_dist_map_m: Optional[np.ndarray] = None,
        scale_m_per_px: float = 0.03,
        force_furnish: bool = False
    ) -> Tuple[List[PlacedItem], Optional[str]]:
        min_area = 1.5 if force_furnish else 5.0
        if area < min_area:
            return [], "TOO_SMALL"

        placed = []
        c_scale = self._get_compact_scale("dining", area, force_furnish)
        scale_tuple = (round(c_scale, 3), round(c_scale, 3), round(c_scale, 3))

        table_id = "dining_table_round" if preset == "warm_wood" else ("dining_table_cloth" if preset == "tailored_classic" else "dining_table")
        t_spec = self._get_item_spec(table_id)
        raw_tw, raw_td, raw_th = t_spec["dimensions_m"]
        tw, td, th = raw_tw * c_scale, raw_td * c_scale, raw_th * c_scale

        # Center table in room centroid
        cx, cy = room_poly.centroid.x, room_poly.centroid.y
        t_poly = Polygon(compute_footprint_polygon(cx, cy, tw, td, 0.0))

        if not force_furnish:
            if not room_poly.buffer(0.02).contains(t_poly) or any(dz.intersects(t_poly) for dz in door_zones):
                return [], "TOO_SMALL"

        placed.append(PlacedItem(
            item_id="dining_table_0",
            catalog_id=table_id,
            name=t_spec["name"],
            category="dining_table",
            position_m=(cx, cy),
            rotation_deg=0.0,
            dimensions_m=(round(tw, 3), round(td, 3), round(th, 3)),
            glb=t_spec["glb"],
            polygon_coords=list(t_poly.exterior.coords)[:-1],
            placement_rule="center",
            scale=scale_tuple
        ))

        # Dining Chairs (4 chairs around table)
        chair_id = "dining_chair_scandi" if preset == "warm_wood" else ("dining_chair_classic" if preset == "tailored_classic" else "dining_chair")
        c_spec = self._get_item_spec(chair_id)
        raw_cw, raw_cd, raw_ch = c_spec["dimensions_m"]
        cw, cd, ch = raw_cw * c_scale, raw_cd * c_scale, raw_ch * c_scale

        # 4 chair positions: North (facing South 180 deg), South (facing North 0 deg), West (facing East 90 deg), East (facing West 270 deg)
        chair_configs = [
            ("chair_north", (cx, cy - td / 2.0 - cd / 2.0 - 0.08 * c_scale), 180.0),
            ("chair_south", (cx, cy + td / 2.0 + cd / 2.0 + 0.08 * c_scale), 0.0),
            ("chair_west", (cx - tw / 2.0 - cd / 2.0 - 0.08 * c_scale, cy), 90.0),
            ("chair_east", (cx + tw / 2.0 + cd / 2.0 + 0.08 * c_scale, cy), 270.0)
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
                    dimensions_m=(round(cw, 3), round(cd, 3), round(ch, 3)),
                    glb=c_spec["glb"],
                    polygon_coords=list(c_poly.exterior.coords)[:-1],
                    placement_rule="around_anchor",
                    scale=scale_tuple
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
        rng: random.Random,
        wall_dist_map_m: Optional[np.ndarray] = None,
        scale_m_per_px: float = 0.03,
        force_furnish: bool = False
    ) -> Tuple[List[PlacedItem], Optional[str]]:
        min_area = 1.2 if force_furnish else 4.0
        if area < min_area:
            return [], "TOO_SMALL"

        placed = []
        sorted_segs = sorted(usable_segments, key=lambda s: s.length_m, reverse=True)
        if not sorted_segs and force_furnish:
            sorted_segs = self._synthesize_fallback_segments(room_poly)
        if not sorted_segs:
            return [], "NO_USABLE_WALL"

        c_scale = self._get_compact_scale("office", area, force_furnish)
        scale_tuple = (round(c_scale, 3), round(c_scale, 3), round(c_scale, 3))

        desk_id = "desk_corner" if (preset == "tailored_classic" and area >= 9.0) else "desk"
        desk_spec = self._get_item_spec(desk_id)
        raw_dw, raw_dd, raw_dh = desk_spec["dimensions_m"]
        dw, dd, dh = raw_dw * c_scale, raw_dd * c_scale, raw_dh * c_scale

        d_seg = None
        d_c = None
        d_poly = None
        d_coords = None

        for seg in sorted_segs:
            if seg.length_m >= (dw + 0.10):
                p1 = np.array(seg.start)
                u_dir = (np.array(seg.end) - p1) / max(seg.length_m, 1e-6)
                n_in = np.array(seg.normal)
                d_center_s = seg.length_m / 2.0
                d_back = p1 + u_dir * d_center_s
                cand_c = d_back + n_in * (dd / 2.0)
                cand_coords = compute_footprint_polygon(float(cand_c[0]), float(cand_c[1]), dw, dd, seg.yaw_deg)
                cand_poly = Polygon(cand_coords)

                if room_poly.buffer(0.02).contains(cand_poly) and not any(dz.intersects(cand_poly) for dz in door_zones):
                    if self._check_back_edge_contact(cand_coords, wall_dist_map_m, scale_m_per_px):
                        d_seg = seg
                        d_c = cand_c
                        d_poly = cand_poly
                        d_coords = cand_coords
                        break

        if d_seg is None and force_furnish and sorted_segs:
            seg = sorted_segs[0]
            p1 = np.array(seg.start)
            p2 = np.array(seg.end)
            u_dir = (p2 - p1) / max(seg.length_m, 1e-6)
            n_in = np.array(seg.normal)
            cand_c = (p1 + p2) / 2.0 + n_in * (dd / 2.0)
            cand_coords = compute_footprint_polygon(float(cand_c[0]), float(cand_c[1]), dw, dd, seg.yaw_deg)
            d_seg = seg
            d_c = cand_c
            d_poly = Polygon(cand_coords)
            d_coords = cand_coords

        if d_seg is None:
            return [], "NO_USABLE_WALL"

        placed.append(PlacedItem(
            item_id="desk_0",
            catalog_id=desk_id,
            name=desk_spec["name"],
            category="desk",
            position_m=(float(d_c[0]), float(d_c[1])),
            rotation_deg=d_seg.yaw_deg,
            dimensions_m=(round(dw, 3), round(dd, 3), round(dh, 3)),
            glb=desk_spec["glb"],
            polygon_coords=d_coords,
            placement_rule="back_to_wall",
            scale=scale_tuple
        ))

        # Desk Chair (facing desk)
        chair_spec = self._get_item_spec("office_chair")
        raw_cw, raw_cd, raw_ch = chair_spec["dimensions_m"]
        cw, cd, ch = raw_cw * c_scale, raw_cd * c_scale, raw_ch * c_scale
        chair_c = d_c + n_in * (dd / 2.0 + cd / 2.0 + 0.15 * c_scale)
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
                dimensions_m=(round(cw, 3), round(cd, 3), round(ch, 3)),
                glb=chair_spec["glb"],
                polygon_coords=list(chair_poly.exterior.coords)[:-1],
                placement_rule="facing_anchor",
                scale=scale_tuple
            ))

        return placed, None
