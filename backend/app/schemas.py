from pydantic import BaseModel, Field, FiniteFloat
from typing import Annotated, List, Dict, Any, Optional
from src.floorplan.dimensions import STOREY_HEIGHT_M

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    gpu_available: bool = Field(..., example=False)
    version: str = Field(..., example="1.0.0")
    fallback_router_loaded: bool = Field(default=False, example=True)
    fallback_loaded: bool = Field(default=False, example=True)
    fallback_model_name: Optional[str] = Field(default=None, example="multitask_resnet34_blend_alpha_0_3")
    engine_mode: Optional[str] = Field(default=None, example="ONNX Runtime CPU")

class ModelMetadataResponse(BaseModel):
    project_name: str = Field(..., example="ArchiGen 3D (AI HomeByMe)")
    major_axis: str = Field(..., example="Axe 2 - Generative AI (Pix2Pix / cGAN)")
    upstream_axis: str = Field(..., example="Axe 4 - Floorplan Segmentation (SegFormer)")
    framework: str = Field(..., example="PyTorch + ONNX Runtime")
    target_resolution: List[int] = Field(default=[256, 256], example=[256, 256])
    supported_styles: List[str] = Field(
        default=["Modern Minimalist", "Mediterranean Stone", "Traditional Brick", "Lush Vegetation"],
        example=["Modern Minimalist", "Mediterranean Stone"]
    )
    metrics: Dict[str, float] = Field(
        default={"target_fid": 45.0, "target_lpips": 0.28, "target_miou": 0.78}
    )
    fallback_router: Dict[str, Any] = Field(default={})
    fallback_loaded: bool = Field(default=False)
    checkpoint_hashes: Dict[str, Any] = Field(default={})

class RoomSegment(BaseModel):
    id: Optional[str] = Field(default=None)
    room_type: str = Field(...)
    surface_m2: float = Field(...)
    confidence: Optional[float] = Field(
        default=None, ge=0, le=1, description="Mean model probability of the assigned room type inside its polygon."
    )
    polygon_points: List[List[float]] = Field(...)
    edges: Optional[List[Dict[str, Any]]] = Field(default=None)
    wall_backed_edge_count: Optional[int] = Field(default=None)

CanvasCoordinate = Annotated[float, Field(ge=0, le=512)]

class RoomScoreRegion(BaseModel):
    id: str
    room_type: str
    polygon_px: List[tuple[CanvasCoordinate, CanvasCoordinate]] = Field(min_length=3, max_length=256)

class WallSegment(BaseModel):
    id: str = Field(...)
    polygon_meters: List[List[float]] = Field(...)
    axis_meters: Optional[List[List[float]]] = None
    thickness_m: Optional[float] = None
    facade_frame: Optional[Dict[str, Any]] = None
    height_m: float = Field(default=STOREY_HEIGHT_M)
    is_exterior: bool = Field(default=False)
    is_front: bool = Field(default=False)
    openings: List[Dict[str, Any]] = Field(default=[])
    facade_texture: Optional[Dict[str, Any]] = Field(default=None)


class FloorPlanAnalysisResponse(BaseModel):
    total_area_m2: float = Field(...)
    num_rooms: int = Field(...)
    rooms: List[RoomSegment]
    reference_room_confidences: Optional[Dict[str, Optional[float]]] = None
    walls: List[WallSegment] = Field(default=[])
    doors: List[Dict[str, Any]] = Field(default=[])
    windows: List[Dict[str, Any]] = Field(default=[])
    envelope: Optional[List[List[List[float]]]] = Field(default=[])
    has_garden_or_terrace: bool = Field(default=False)
    threejs_scene: Optional[Dict[str, Any]] = Field(default=None)


class RoomDesignRequest(BaseModel):
    room_id: Optional[str] = Field(default="room_0")
    room_type: str = Field(...)
    polygon_meters: List[List[float]] = Field(...)
    doors: Optional[List[Dict[str, Any]]] = Field(default=[])
    windows: Optional[List[Dict[str, Any]]] = Field(default=[])
    wall_segments: Optional[List[Dict[str, Any]]] = Field(default=[])
    preset: str = Field(default="contemporary_minimal")
    seed: Optional[int] = Field(default=42)
    room_edges: Optional[List[Dict[str, Any]]] = Field(default=None)
    force_furnish: Optional[bool] = Field(default=False)
    scale_m_per_px: float = Field(default=0.03, gt=0)


class PlacedFurnitureSnapshot(BaseModel):
    item_id: str
    catalog_id: str
    category: Optional[str] = None
    position_m: tuple[FiniteFloat, FiniteFloat]
    rotation_deg: FiniteFloat
    dimensions_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    scale: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    footprint_polygon: List[tuple[FiniteFloat, FiniteFloat]] = Field(min_length=4, max_length=4)
    placement_rule: str


class FurnitureChoicesRequest(RoomDesignRequest):
    item_id: str
    items: List[PlacedFurnitureSnapshot] = Field(min_length=1, max_length=64)


class LayoutDesignRequest(BaseModel):
    layout: Dict[str, Any] = Field(...)
    preset: str = Field(default="contemporary_minimal")
    seed: Optional[int] = Field(default=42)


class FurnitureGenerateRequest(BaseModel):
    image_base64: str
    name: Optional[str] = "Custom Furniture"
    category: Optional[str] = "chair"
    remove_bg: Optional[bool] = True
    target_height_m: Optional[float] = None


