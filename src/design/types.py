"""
src/design/types.py
-------------------
Data models and type definitions for the deterministic rule-based placement engine.
All spatial coordinates are in metric meters in 2D floorplan space (plan coordinates).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from shapely.geometry import Polygon, Point, LineString

@dataclass
class WallSegment:
    """A linear wall segment or snapped wall face in plan meters."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    length_m: float
    is_exterior: bool = False
    wall_id: Optional[str] = None
    normal: Tuple[float, float] = (0.0, 0.0)  # Inward unit normal pointing into room

@dataclass
class OpeningSpan:
    """A door or window opening located along an edge."""
    opening_type: str  # "door" or "window"
    start_dist: float  # distance along edge from edge start in meters
    end_dist: float    # distance along edge from edge start in meters
    clearance_depth: float = 0.85  # Door swing clearance box depth into room

@dataclass
class UsableWallSegment:
    """An unobstructed sub-segment of a wall-backed room edge, after subtracting openings."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    length_m: float
    normal: Tuple[float, float]  # Inward normal pointing into room (dx, dy)
    yaw_deg: float               # Clockwise angle in degrees from North for back-to-wall items
    edge_idx: int
    wall_distance_m: float

@dataclass
class PlacedItem:
    """A placed 3D furniture item represented in 2D plan space."""
    item_id: str                # Unique instance ID, e.g. "bed_double_0"
    catalog_id: str            # Key in catalog.json, e.g. "bed_double"
    name: str                  # Human-readable name
    category: str              # Category, e.g. "bed", "sofa", "nightstand"
    position_m: Tuple[float, float]   # Base center position [x, y] in plan meters
    rotation_deg: float        # Rotation angle in degrees clockwise from North [0, 360)
    dimensions_m: Tuple[float, float, float]  # [width_x, depth_z, height_y]
    glb: str                   # Asset path, e.g. "/models/furniture/bed_double.glb"
    polygon_coords: List[Tuple[float, float]]  # Footprint polygon vertices in plan meters
    placement_rule: str        # "back_to_wall", "in_front_of_anchor", "center", etc.
    wall_distance_cm: float = 0.0  # Distance from back edge to wall face in cm (target <= 2.0 cm)

@dataclass
class ValidationReport:
    """Strict geometric validation report for a furnished room layout."""
    is_valid: bool
    overlap_count: int = 0
    max_overlap_area_m2: float = 0.0
    door_clearance_violation: bool = False
    out_of_bounds_count: int = 0
    max_wall_gap_cm: float = 0.0
    bfs_reachable: bool = True
    failure_reasons: List[str] = field(default_factory=list)

@dataclass
class RoomDesignResult:
    """Complete output of the rule-based room design engine for a single room."""
    room_id: str
    room_type: str
    preset: str
    seed: int
    is_valid: bool
    failure_reason: Optional[str]
    items: List[PlacedItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation: Optional[ValidationReport] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_id": self.room_id,
            "room_type": self.room_type,
            "preset": self.preset,
            "seed": self.seed,
            "is_valid": self.is_valid,
            "failure_reason": self.failure_reason,
            "items": [
                {
                    "item_id": it.item_id,
                    "catalog_id": it.catalog_id,
                    "name": it.name,
                    "category": it.category,
                    "position_m": [round(it.position_m[0], 3), round(it.position_m[1], 3)],
                    "rotation_deg": round(it.rotation_deg, 1),
                    "dimensions_m": [round(d, 3) for d in it.dimensions_m],
                    "glb": it.glb,
                    "placement_rule": it.placement_rule,
                    "wall_distance_cm": round(it.wall_distance_cm, 1),
                    "footprint_polygon": [[round(p[0], 3), round(p[1], 3)] for p in it.polygon_coords]
                }
                for it in self.items
            ],
            "metadata": self.metadata,
            "validation": {
                "is_valid": self.validation.is_valid if self.validation else self.is_valid,
                "overlap_count": self.validation.overlap_count if self.validation else 0,
                "door_clearance_violation": self.validation.door_clearance_violation if self.validation else False,
                "out_of_bounds_count": self.validation.out_of_bounds_count if self.validation else 0,
                "max_wall_gap_cm": round(self.validation.max_wall_gap_cm, 1) if self.validation else 0.0,
                "bfs_reachable": self.validation.bfs_reachable if self.validation else True,
                "failure_reasons": self.validation.failure_reasons if self.validation else []
            } if self.validation else {}
        }
