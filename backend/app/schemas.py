from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    gpu_available: bool = Field(..., example=False)
    version: str = Field(..., example="1.0.0")

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

class RoomSegment(BaseModel):
    room_type: str = Field(..., example="Living Room")
    surface_m2: float = Field(..., example=32.5)
    polygon_points: List[List[float]] = Field(..., example=[[0.0, 0.0], [5.0, 0.0], [5.0, 6.5], [0.0, 6.5]])

class WallSegment(BaseModel):
    id: str = Field(..., example="wall_0")
    polygon_meters: List[List[float]] = Field(...)
    height_m: float = Field(default=2.8)
    is_exterior: bool = Field(default=False)
    openings: List[Dict[str, Any]] = Field(default=[])


class FloorPlanAnalysisResponse(BaseModel):
    total_area_m2: float = Field(..., example=115.0)
    num_rooms: int = Field(..., example=4)
    rooms: List[RoomSegment]
    walls: List[WallSegment] = Field(default=[])
    doors: List[Dict[str, Any]] = Field(default=[])
    windows: List[Dict[str, Any]] = Field(default=[])
    has_garden_or_terrace: bool = Field(default=False)
    threejs_scene: Optional[Dict[str, Any]] = Field(default=None)

