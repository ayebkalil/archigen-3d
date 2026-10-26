import os
import sys
import glob
import json
import numpy as np
from pathlib import Path
from PIL import Image
import io
from typing import List, Tuple, Dict, Any, Optional
from shapely.geometry import Polygon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.design.engine import RoomDesignEngine
from src.design.edge_classifier import EdgeClassifier
from src.design.rules import RoomRuleEngine, compute_footprint_polygon
from src.design.types import PlacedItem, UsableWallSegment
from backend.app.floorplan_engine import FloorplanPipeline

EXCLUDED_PLANS = {"colorful_11261.npz", "colorful_1148.npz"}

class FallbackBathroomRuleEngine(RoomRuleEngine):
    """
    Enhanced bathroom placement rule engine with 3-tier fallbacks:
    1. Multi-wall search (second-best & third-best wall) for toilet and vanity
    2. Adaptive corner clearance (sliding search along wall segment)
    3. Optional item dropping (toilet alone in powder rooms/WCs, bathtub dropped if no wall fits)
    """

    def _furnish_bathroom(
        self,
        room_poly: Polygon,
        usable_segments: List[UsableWallSegment],
        door_zones: List[Polygon],
        preset: str,
        area: float,
        rng
    ):
        if area < 1.8:
            return [], "TOO_SMALL"

        placed = []
        # Sort usable segments by length descending
        sorted_segs = sorted(usable_segments, key=lambda s: s.length_m, reverse=True)
        if not sorted_segs:
            return [], "NO_USABLE_WALL"

        toilet_spec = self._get_item_spec("toilet")
        tw, td, th = toilet_spec["dimensions_m"]

        vanity_spec = self._get_item_spec("bathroom_vanity")
        vw, vd, vh = vanity_spec["dimensions_m"]

        toilet_placed = False
        toilet_seg_idx = -1

        # -------------------------------------------------------------
        # FALLBACK 1 & 2: Search ALL usable wall segments for toilet,
        # trying both ends and adaptive positions along each segment.
        # -------------------------------------------------------------
        for seg_idx, seg in enumerate(sorted_segs):
            if seg.length_m < (tw + 0.05):
                continue

            seg_p1 = np.array(seg.start)
            seg_len = seg.length_m
            u_dir = (np.array(seg.end) - seg_p1) / max(seg_len, 1e-6)
            n_in = np.array(seg.normal)

            # Candidate positions along this segment
            candidate_offsets = [
                tw / 2.0 + 0.15,                      # Start end with standard clearance
                seg_len - (tw / 2.0 + 0.15),          # Far end with standard clearance
                seg_len / 2.0,                         # Center of segment
                tw / 2.0 + 0.05,                      # Start end compact clearance
                seg_len - (tw / 2.0 + 0.05),          # Far end compact clearance
            ]

            for c_s in candidate_offsets:
                if (tw / 2.0) <= c_s <= (seg_len - tw / 2.0):
                    t_back = seg_p1 + u_dir * c_s
                    t_c = t_back + n_in * (td / 2.0)
                    t_poly = Polygon(compute_footprint_polygon(float(t_c[0]), float(t_c[1]), tw, td, seg.yaw_deg))

                    if not any(dz.intersects(t_poly) for dz in door_zones) and room_poly.buffer(0.02).contains(t_poly):
                        placed.append(PlacedItem(
                            item_id="toilet_0",
                            catalog_id="toilet",
                            name=toilet_spec["name"],
                            category="toilet",
                            position_m=(float(t_c[0]), float(t_c[1])),
                            rotation_deg=seg.yaw_deg,
                            dimensions_m=(tw, td, th),
                            glb=toilet_spec["glb"],
                            polygon_coords=list(t_poly.exterior.coords)[:-1],
                            placement_rule="back_to_wall"
                        ))
                        toilet_placed = True
                        toilet_seg_idx = seg_idx
                        break

            if toilet_placed:
                break

        # -------------------------------------------------------------
        # 2. Place Vanity Sink (on remaining walls or shared wall)
        # -------------------------------------------------------------
        vanity_placed = False
        placed_polys = [Polygon(p.polygon_coords) for p in placed]

        for seg_idx, seg in enumerate(sorted_segs):
            if seg_idx == toilet_seg_idx:
                # Same wall as toilet: check if remaining length accommodates vanity
                seg_p1 = np.array(seg.start)
                seg_len = seg.length_m
                if seg_len >= (tw + vw + 0.40):
                    u_dir = (np.array(seg.end) - seg_p1) / max(seg_len, 1e-6)
                    n_in = np.array(seg.normal)
                    # Try both opposite ends from toilet
                    for v_s in [vw / 2.0 + 0.10, seg_len - (vw / 2.0 + 0.10)]:
                        v_back = seg_p1 + u_dir * v_s
                        v_c = v_back + n_in * (vd / 2.0)
                        v_poly = Polygon(compute_footprint_polygon(float(v_c[0]), float(v_c[1]), vw, vd, seg.yaw_deg))

                        if room_poly.buffer(0.02).contains(v_poly) and not any(dz.intersects(v_poly) for dz in door_zones):
                            if not any(v_poly.intersects(pp) for pp in placed_polys):
                                placed.append(PlacedItem(
                                    item_id="vanity_0",
                                    catalog_id="bathroom_vanity",
                                    name=vanity_spec["name"],
                                    category="bathroom_sink",
                                    position_m=(float(v_c[0]), float(v_c[1])),
                                    rotation_deg=seg.yaw_deg,
                                    dimensions_m=(vw, vd, vh),
                                    glb=vanity_spec["glb"],
                                    polygon_coords=list(v_poly.exterior.coords)[:-1],
                                    placement_rule="back_to_wall"
                                ))
                                vanity_placed = True
                                break
            else:
                # Different wall from toilet
                if seg.length_m >= (vw + 0.05):
                    seg_p1 = np.array(seg.start)
                    seg_len = seg.length_m
                    u_dir = (np.array(seg.end) - seg_p1) / max(seg_len, 1e-6)
                    n_in = np.array(seg.normal)

                    for v_s in [vw / 2.0 + 0.10, seg_len / 2.0, seg_len - (vw / 2.0 + 0.10)]:
                        if (vw / 2.0) <= v_s <= (seg_len - vw / 2.0):
                            v_back = seg_p1 + u_dir * v_s
                            v_c = v_back + n_in * (vd / 2.0)
                            v_poly = Polygon(compute_footprint_polygon(float(v_c[0]), float(v_c[1]), vw, vd, seg.yaw_deg))

                            if room_poly.buffer(0.02).contains(v_poly) and not any(dz.intersects(v_poly) for dz in door_zones):
                                if not any(v_poly.intersects(pp) for pp in placed_polys):
                                    placed.append(PlacedItem(
                                        item_id="vanity_0",
                                        catalog_id="bathroom_vanity",
                                        name=vanity_spec["name"],
                                        category="bathroom_sink",
                                        position_m=(float(v_c[0]), float(v_c[1])),
                                        rotation_deg=seg.yaw_deg,
                                        dimensions_m=(vw, vd, vh),
                                        glb=vanity_spec["glb"],
                                        polygon_coords=list(v_poly.exterior.coords)[:-1],
                                        placement_rule="back_to_wall"
                                    ))
                                    vanity_placed = True
                                    break
            if vanity_placed:
                break

        # -------------------------------------------------------------
        # 3. Bathtub (Optional, only if area >= 4.5m2 and space allows)
        # -------------------------------------------------------------
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

        # FALLBACK 3: If toilet or vanity is placed alone, accept as valid WC / powder room!
        if not placed:
            return [], "NO_USABLE_WALL"

        return placed, None

def evaluate_bathroom_fallbacks():
    print("=" * 80)
    print("EVALUATING BATHROOM FALLBACK ENGINE (ON VALIDATION PLANS)")
    print(f"Excluding screenshot plans: {EXCLUDED_PLANS}")
    print("=" * 80)

    val_files = sorted(glob.glob("data/CubiCasa5k/cached_512/val/*.npz"))
    eval_files = [f for f in val_files if os.path.basename(f) not in EXCLUDED_PLANS][:80]
    print(f"Evaluating across {len(eval_files)} validation plans.")

    pipeline = FloorplanPipeline.get_instance()

    # Base Engine
    base_engine = RoomDesignEngine()
    
    # Fallback Engine
    fallback_engine = RoomDesignEngine()
    fallback_engine.rules = FallbackBathroomRuleEngine(fallback_engine.catalog)

    base_results = {"total": 0, "valid": 0, "TOO_SMALL": 0, "NO_USABLE_WALL": 0, "OTHER": 0, "collisions": 0, "door_violations": 0}
    fallback_results = {"total": 0, "valid": 0, "TOO_SMALL": 0, "NO_USABLE_WALL": 0, "OTHER": 0, "collisions": 0, "door_violations": 0}

    for fpath in eval_files:
        fname = os.path.basename(fpath)
        d = np.load(fpath)
        img_np = d["image"] if "image" in d else np.stack([d["mask"] * 50] * 3, axis=-1)
        buf = io.BytesIO()
        Image.fromarray(img_np).save(buf, format="PNG")
        proc = pipeline.process_image(buf.getvalue(), front_only=True)
        pred_layout = proc["layout"]

        walls = pred_layout.get("walls", [])
        doors = pred_layout.get("doors", [])
        windows = pred_layout.get("windows", [])

        for r_idx, r in enumerate(pred_layout.get("rooms", [])):
            raw_type = r.get("room_type", "other")
            norm_type = raw_type.lower().replace(" ", "_")
            if norm_type != "bathroom":
                continue

            pts = r.get("polygon_meters", [])
            if len(pts) < 3:
                continue

            base_results["total"] += 1
            fallback_results["total"] += 1

            # 1. Base placement
            res_base = base_engine.design_room(
                room_polygon_meters=pts,
                room_type="bathroom",
                room_id=f"base_{fname}_{r_idx}",
                wall_segments=walls,
                doors=doors,
                windows=windows,
                preset="contemporary_minimal",
                seed=42
            )
            if res_base.is_valid:
                base_results["valid"] += 1
                if res_base.validation.overlap_count > 0: base_results["collisions"] += 1
                if res_base.validation.door_clearance_violation: base_results["door_violations"] += 1
            else:
                reason = res_base.failure_reason or "OTHER"
                if reason in base_results: base_results[reason] += 1
                else: base_results["OTHER"] += 1

            # 2. Fallback placement
            res_fb = fallback_engine.design_room(
                room_polygon_meters=pts,
                room_type="bathroom",
                room_id=f"fb_{fname}_{r_idx}",
                wall_segments=walls,
                doors=doors,
                windows=windows,
                preset="contemporary_minimal",
                seed=42
            )
            if res_fb.is_valid:
                fallback_results["valid"] += 1
                if res_fb.validation.overlap_count > 0: fallback_results["collisions"] += 1
                if res_fb.validation.door_clearance_violation: fallback_results["door_violations"] += 1
            else:
                reason = res_fb.failure_reason or "OTHER"
                if reason in fallback_results: fallback_results[reason] += 1
                else: fallback_results["OTHER"] += 1

    print("\n=== BATHROOM FALLBACK EVALUATION RESULTS ===")
    print(f"Base Bathroom Engine:")
    print(f"  Total: {base_results['total']}")
    print(f"  Valid: {base_results['valid']} ({base_results['valid']/base_results['total']*100:.2f}%)")
    print(f"  NO_USABLE_WALL: {base_results['NO_USABLE_WALL']} ({base_results['NO_USABLE_WALL']/base_results['total']*100:.2f}%)")
    print(f"  TOO_SMALL: {base_results['TOO_SMALL']} ({base_results['TOO_SMALL']/base_results['total']*100:.2f}%)")
    print(f"  Collisions: {base_results['collisions']}")
    print(f"  Door Violations: {base_results['door_violations']}")

    print(f"\nFallback Bathroom Engine (Second-best wall, adaptive clearance, optional items):")
    print(f"  Total: {fallback_results['total']}")
    print(f"  Valid: {fallback_results['valid']} ({fallback_results['valid']/fallback_results['total']*100:.2f}%)")
    print(f"  NO_USABLE_WALL: {fallback_results['NO_USABLE_WALL']} ({fallback_results['NO_USABLE_WALL']/fallback_results['total']*100:.2f}%)")
    print(f"  TOO_SMALL: {fallback_results['TOO_SMALL']} ({fallback_results['TOO_SMALL']/fallback_results['total']*100:.2f}%)")
    print(f"  Collisions: {fallback_results['collisions']}")
    print(f"  Door Violations: {fallback_results['door_violations']}")

    gain = fallback_results['valid'] - base_results['valid']
    print(f"\nValidity Gain: +{gain} rooms (+{gain/base_results['total']*100:.2f} percentage points)")
    print(f"New Collisions: {fallback_results['collisions'] - base_results['collisions']}")
    print(f"New Door Violations: {fallback_results['door_violations'] - base_results['door_violations']}")

    # Save to outputs
    out_dir = PROJECT_ROOT / "outputs" / "design_stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bathroom_fallback_experiment.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "plans_count": len(eval_files),
            "excluded_plans": list(EXCLUDED_PLANS),
            "base_results": base_results,
            "fallback_results": fallback_results,
            "validity_gain_count": gain,
            "validity_gain_pct": round(gain / base_results['total'] * 100, 2),
            "new_collisions": fallback_results['collisions'] - base_results['collisions'],
            "new_door_violations": fallback_results['door_violations'] - base_results['door_violations']
        }, f, indent=2)
    print(f"Saved results to: {out_file}")

if __name__ == "__main__":
    evaluate_bathroom_fallbacks()
