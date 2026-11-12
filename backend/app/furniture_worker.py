"""Native CUDA furniture worker used by the Docker API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .schemas import FurnitureGenerateRequest
from src.furniture.service import generate_furniture_response


@asynccontextmanager
async def lifespan(app):
    from src.furniture.quality_engine import QualityFurnitureEngine

    generator = QualityFurnitureEngine()
    if generator.device.type != "cuda":
        raise RuntimeError("The furniture worker requires the native CUDA environment.")
    app.state.generator = generator
    yield


app = FastAPI(title="ArchiGen GPU Furniture Worker", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "healthy", "engine": "TripoSR Quality", "device": app.state.generator.device_name}


@app.post("/furniture/generate-3d")
def generate(request: FurnitureGenerateRequest):
    try:
        return generate_furniture_response(request, app.state.generator)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
