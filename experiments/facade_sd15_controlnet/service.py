"""Run in an isolated diffusion venv, bound to localhost on port 8002."""
import base64
import io
import os
import json
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .runtime import FacadeControlNetRuntime

app = FastAPI(title='ArchiGen experimental facade ControlNet worker')


@lru_cache(maxsize=1)
def runtime():
    root = os.getenv('ARCHIGEN_CONTROLNET_MODEL')
    if not root: raise RuntimeError('Set ARCHIGEN_CONTROLNET_MODEL to the downloaded export folder.')
    return FacadeControlNetRuntime(root, allow_download=os.getenv('ARCHIGEN_CONTROLNET_ALLOW_DOWNLOAD') == '1')


class GenerateRequest(BaseModel):
    image_base64: str = Field(max_length=12000000)
    style: str | None = None


@app.get('/health')
def health():
    try: return {'status': 'ready', **runtime().describe()}
    except (OSError, ValueError, RuntimeError) as error: raise HTTPException(503, str(error)) from error


@app.post('/generate')
def generate(request: GenerateRequest):
    try:
        raw = base64.b64decode(request.image_base64.split(',')[-1], validate=True)
        image = runtime().generate_sketch(raw, request.style)
        buffer = io.BytesIO(); image.save(buffer, format='PNG')
        return Response(buffer.getvalue(), media_type='image/png',
            headers={'X-ArchiGen-Profile': json.dumps(runtime().last_profile, separators=(',', ':'))})
    except ValueError as error: raise HTTPException(422, str(error)) from error
    except (RuntimeError, OSError) as error: raise HTTPException(503, str(error)) from error
