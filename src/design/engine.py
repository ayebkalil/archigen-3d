"""
src/design/engine.py
--------------------
Orchestrates the deterministic rule-based room design placement engine:
1. Loads catalog.json
2. Snaps edges and extracts usable wall segments
3. Applies room-specific furnishing rules
4. Strictly validates spatial constraints
5. Returns plan coordinates (meters) + rotation (degrees clockwise from North)
"""

import os
import json
import time
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from shapely.geometry import Polygon, box

from src.design.types import RoomDesignResult, PlacedItem, ValidationReport
from src.design.edge_classifier import EdgeClassifier
from src.design.rules import RoomRuleEngine
from src.design.validator import LayoutValidator

CATALOG_DEFAULT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "catalog.json"))

class RoomDesignEngine:
    """Main entrypoint for offline deterministic room design synthesis."""

    def __init__(self, catalog_path: str = CATALOG_DEFAULT_PATH):
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog file not found at: {catalog_path}")
        with open(catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        self.classifier = EdgeClassifier(wall_tolerance_m=0.15, door_snap_tolerance_m=0.40)
        self.rules = RoomRuleEngine(self.catalog)
        self.validator = LayoutValidator(wall_contact_tolerance_m=0.02, min_corridor_width_m=0.50)

    def design_room(
        self,
        room_polygon_meters: List[List[float]],
        room_type: str,
        room_id: str = "room_0",
        wall_mask: Optional[np.ndarray] = None,
        scale_m_per_px: float = 0.03,
        wall_segments: Optional[List[Dict[str, Any]]] = None,
        doors: Optional[List[Dict[str, Any]]] = None,
        windows: Optional[List[Dict[str, Any]]] = None,
        preset: str = "contemporary_minimal",
        seed: int = 42,
        is_scale_uncalibrated: bool = False
    ) -> RoomDesignResult:
        """
        Designs a furnished layout for a single room.
        All coordinates returned are in 2D plan meters.
        """
        t0 = time.perf_counter()

        # Handle preset aliases
        aliases = self.catalog.get("preset_aliases", {})
        norm_preset = aliases.get(preset, preset)
        if norm_preset not in self.catalog.get("preset_support", {}):
            norm_preset = "contemporary_minimal"

        if len(room_polygon_meters) < 3:
            return RoomDesignResult(
                room_id=room_id,
                room_type=room_type,
                preset=norm_preset,
                seed=seed,
                is_valid=False,
                failure_reason="INVALID_POLYGON",
                metadata={"runtime_ms": 0.0}
            )

        room_poly = Polygon(room_polygon_meters)
        if not room_poly.is_valid:
            room_poly = room_poly.buffer(0)

        # Compute room dimensions
        minx, miny, maxx, maxy = room_poly.bounds
        room_width = round(maxx - minx, 2)
        room_depth = round(maxy - miny, 2)
        room_area = round(room_poly.area, 2)

        # 1. Edge classification and snapping
        snapped_poly, usable_segs, door_zones = self.classifier.classify_and_snap_edges(
            room_poly=room_poly,
            wall_mask=wall_mask,
            scale_m_per_px=scale_m_per_px,
            wall_segments=wall_segments,
            doors=doors,
            windows=windows
        )

        # 2. Rule-based furnishing
        placed_items, fail_reason = self.rules.furnish_room(
            room_type=room_type,
            room_poly=snapped_poly,
            usable_segments=usable_segs,
            door_clearance_zones=door_zones,
            preset=norm_preset,
            seed=seed
        )

        if fail_reason:
            runtime_ms = (time.perf_counter() - t0) * 1000.0
            return RoomDesignResult(
                room_id=room_id,
                room_type=room_type,
                preset=norm_preset,
                seed=seed,
                is_valid=False,
                failure_reason=fail_reason,
                items=[],
                metadata={
                    "room_width_m": room_width,
                    "room_depth_m": room_depth,
                    "room_area_m2": room_area,
                    "usable_wall_segments_count": len(usable_segs),
                    "scale_m_per_px": round(scale_m_per_px, 4),
                    "scale_source": "2-Pt Calibrated" if not is_scale_uncalibrated else "Model Estimated",
                    "scale_uncalibrated": is_scale_uncalibrated,
                    "runtime_ms": round(runtime_ms, 2)
                }
            )

        # 3. Spatial validation
        val_report = self.validator.validate(
            room_poly=snapped_poly,
            items=placed_items,
            door_clearance_zones=door_zones,
            doors=doors
        )

        runtime_ms = (time.perf_counter() - t0) * 1000.0

        return RoomDesignResult(
            room_id=room_id,
            room_type=room_type,
            preset=norm_preset,
            seed=seed,
            is_valid=val_report.is_valid,
            failure_reason="VALIDATION_FAILED" if not val_report.is_valid else None,
            items=placed_items,
            metadata={
                "room_width_m": room_width,
                "room_depth_m": room_depth,
                "room_area_m2": room_area,
                "usable_wall_segments_count": len(usable_segs),
                "scale_m_per_px": round(scale_m_per_px, 4),
                "scale_source": "2-Pt Calibrated" if not is_scale_uncalibrated else "Model Estimated",
                "scale_uncalibrated": is_scale_uncalibrated,
                "runtime_ms": round(runtime_ms, 2)
            },
            validation=val_report
        )

    def design_layout(
        self,
        layout: Dict[str, Any],
        preset: str = "contemporary_minimal",
        seed: int = 42,
        wall_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Designs all eligible rooms in a floorplan layout.
        """
        rooms = layout.get("rooms", [])
        doors = layout.get("doors", [])
        windows = layout.get("windows", [])
        walls = layout.get("walls", [])
        scale_m = layout.get("metadata", {}).get("scale_meters_per_pixel", 0.03)

        results = {}
        for r in rooms:
            rid = r.get("id", "room_0")
            rtype = r.get("room_type", "other")
            pts = r.get("polygon_meters", [])

            res = self.design_room(
                room_polygon_meters=pts,
                room_type=rtype,
                room_id=rid,
                wall_mask=wall_mask,
                scale_m_per_px=scale_m,
                wall_segments=walls,
                doors=doors,
                windows=windows,
                preset=preset,
                seed=seed
            )
            results[rid] = res.to_dict()

        return {
            "version": "1.0.0",
            "preset": preset,
            "seed": seed,
            "num_rooms_furnished": sum(1 for r in results.values() if r["is_valid"] and len(r["items"]) > 0),
            "rooms": results
        }
