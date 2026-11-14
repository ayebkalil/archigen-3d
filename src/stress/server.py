"""Dedicated local test service; production API implementation and schemas are reused."""
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import re
import time
import traceback
from functools import lru_cache

from fastapi import FastAPI, APIRouter, HTTPException, Request, UploadFile, File
from fastapi.routing import APIRoute
from backend.app.schemas import FurnitureGenerateRequest
from src.furniture.service import generate_furniture_response
from .trace import CURRENT, install_furniture, stage, write_trace


def architecture_router(production):
    # StaticFiles mounted at '/' has an empty route path in recent Starlette.
    # Only API routes belong in the isolated service, never the frontend mounts.
    return APIRouter(routes=[route for route in production.routes if isinstance(route, APIRoute)])


@asynccontextmanager
async def lifespan(app):
    if os.getenv('ARCHIGEN_STRESS_MODEL') == 'furniture_preprocessing':
        app.state.engine = None
        import onnxruntime as ort
        original_run = ort.InferenceSession.run
        def run(session, names, feed, *args, **kwargs):
            trace = CURRENT.get()
            if trace is not None:
                trace.setdefault('onnx_inputs', []).append({k:list(v.shape) for k,v in feed.items()})
                trace.setdefault('onnx_input_dtypes', []).append({k:str(v.dtype) for k,v in feed.items()})
            with stage('background_inference'):
                return original_run(session, names, feed, *args, **kwargs)
        ort.InferenceSession.run = run
        import rembg
        original_remove = rembg.remove
        def remove(*args, **kwargs):
            result = original_remove(*args, **kwargs)
            trace = CURRENT.get()
            if trace is not None and hasattr(result,'getchannel'):
                trace['_raw_mask'] = result.getchannel('A')
            return result
        rembg.remove = remove
        yield
        return
    if os.getenv('ARCHIGEN_STRESS_MODEL', 'furniture') != 'furniture':
        from backend.app.main import app as production
        from app.floorplan_engine import FloorplanPipeline
        pipeline = FloorplanPipeline.get_instance()
        model = os.environ['ARCHIGEN_STRESS_MODEL']
        if model == 'floorplan' and not (pipeline.multitask_model or pipeline.baseline_model or pipeline.onnx_multitask):
            raise RuntimeError('No real floorplan weights loaded; mock segmentation excluded')
        if model == 'facade' and pipeline.bridge.inference_engine is None:
            raise RuntimeError('No real facade weights loaded; mock facade excluded')
        from .trace import install_architecture
        install_architecture(pipeline)
        app.state.pipeline = pipeline
        app.include_router(architecture_router(production))
        app.state.engine = None
        yield
        return
    from src.furniture.quality_engine import QualityFurnitureEngine
    import torch
    import sys
    import faulthandler
    faulthandler.enable()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'TripoSR'))
    import tsr.system as system
    # Resolve imports before entering the meta device context: torchvision's import
    # probes execute scalar tensor operations which are not valid on meta tensors.
    import importlib
    from omegaconf import OmegaConf
    config = OmegaConf.load(Path(__file__).resolve().parents[2] / 'TripoSR/config.yaml')
    for key, value in config.items():
        if key.endswith('_cls'):
            importlib.import_module(str(value).rsplit('.', 1)[0])
    original_tsr = system.TSR
    class StartupHalfTSR(original_tsr):
        def __init__(self, *args, **kwargs):
            previous_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float16)
            try:
                with torch.device('meta'):
                    super().__init__(*args, **kwargs)
            finally:
                torch.set_default_dtype(previous_dtype)
            # The deployed engine keeps its field decoder FP32. Restore its dtype
            # before loading so those checkpoint values are never rounded to FP16.
            self.decoder.float()
        def load_state_dict(self, state_dict, strict=True, **kwargs):
            # Allocate checkpoint values once at their deployed dtype; meta avoids
            # allocating and random-initializing a second, soon-discarded model.
            targets = self.state_dict()
            for key in list(state_dict):
                if key in targets:
                    state_dict[key] = state_dict[key].to(dtype=targets[key].dtype)
            result = super().load_state_dict(state_dict, strict=strict, assign=True)
            self.image_tokenizer.image_mean = torch.tensor([.485,.456,.406],dtype=torch.float16).reshape(1,1,3,1,1)
            self.image_tokenizer.image_std = torch.tensor([.229,.224,.225],dtype=torch.float16).reshape(1,1,3,1,1)
            return result
    system.TSR = StartupHalfTSR
    try:
        app.state.engine = QualityFurnitureEngine()
    finally:
        system.TSR = original_tsr
    if app.state.engine.device.type != 'cuda':
        raise RuntimeError('This campaign requires the configured TripoSR CUDA engine')
    install_furniture(app.state.engine)
    lock = app.state.engine.lock
    class ObservedLock:
        def __enter__(self):
            started = time.perf_counter()
            lock.acquire()
            trace = CURRENT.get()
            if trace is not None:
                trace['queue_seconds'] = time.perf_counter() - started
            return self
        def __exit__(self, *args):
            lock.release()
    app.state.engine.lock = ObservedLock()
    yield


app = FastAPI(lifespan=lifespan)


@lru_cache(maxsize=1)
def runtime_versions():
    import importlib.metadata
    import platform
    result = {'python':platform.python_version()}
    for name in ('torch','onnxruntime','numpy','pillow','transformers','trimesh','mlflow-skinny'):
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return result


@app.post('/stress/facade-semantic')
def facade_semantic(file: UploadFile = File(...)):
    import base64
    bridge = app.state.pipeline.bridge
    with stage('facade_semantic_total'):
        photo = bridge.inference_engine.generate(file.file.read())
    return {'texture_base64':'data:image/png;base64,'+base64.b64encode(photo).decode(),
            'sketch_base64':'data:image/png;base64,'+base64.b64encode(photo).decode(),
            'evaluation_note':'direct semantic generator test; no material finishing; sketch artifact is input.bin'}


@app.middleware('http')
async def tracing(request: Request, call_next):
    run_id = request.headers.get('x-archigen-stress-id', '')
    if not re.fullmatch(r'[a-f0-9]{32}', run_id):
        return await call_next(request)
    directory = Path(os.environ['ARCHIGEN_STRESS_DIRECTORY']) / run_id
    directory.mkdir(exist_ok=True, parents=True)
    trace = {'directory': str(directory), 'stages_seconds': {}, 'last_stage': 'api_validation'}
    token = CURRENT.set(trace)
    started = time.perf_counter()
    try:
        response = await call_next(request)
        trace['http_status'] = response.status_code
        return response
    except Exception as exc:
        trace.setdefault('error_stage', trace['last_stage'])
        trace['error'] = str(exc)
        trace['traceback'] = traceback.format_exc()
        raise
    finally:
        raw_mask = trace.pop('_raw_mask',None)
        if raw_mask is not None: raw_mask.save(directory/'raw-mask.png')
        trace['service_total_seconds'] = time.perf_counter() - started
        write_trace(trace)
        CURRENT.reset(token)


@app.get('/health')
def health():
    if app.state.engine is None:
        result = {'status': 'healthy', 'stress_only': True, 'model': os.environ['ARCHIGEN_STRESS_MODEL'],
                  'runtime_versions':runtime_versions(),'process_id':os.getpid()}
        if hasattr(app.state,'pipeline'):
            pipeline = app.state.pipeline
            result['weights_metadata'] = pipeline.checkpoint_hashes
            result['device'] = pipeline.bridge.device_name if result['model']=='facade' else str(pipeline.device)
            result['onnx_providers'] = pipeline.onnx_multitask.get_providers() if pipeline.onnx_multitask is not None else []
        else: result.update(device='cpu',engine='u2netp and image preprocessing only')
        return result
    return {'status': 'healthy', 'engine': 'TripoSR Quality', 'device': app.state.engine.device_name,
            'checkpoint_loading': 'strict', 'stress_only': True,'runtime_versions':runtime_versions(),
            'process_id':os.getpid(),
            'initialization_profile':'meta_assign_strict_no_mmap'}


@app.post('/furniture/generate-3d')
def furniture(request: FurnitureGenerateRequest):
    try:
        if os.getenv('ARCHIGEN_STRESS_MODEL') == 'furniture_preprocessing':
            import base64
            import io
            from PIL import Image
            from src.furniture.quality_geometry import prepare_input
            with stage('decode'):
                image = Image.open(io.BytesIO(base64.b64decode(request.image_base64.split(',')[-1])))
                image.load()
            trace = CURRENT.get()
            if trace is not None: trace['source_dimensions'] = list(image.size)
            with stage('preprocessing'):
                prepared, preview, mask_source = prepare_input(image,request.remove_bg)
            if trace is not None:
                trace['prepared_dimensions'] = list(prepared.size)
                trace['mask_source'] = mask_source
                prepared.save(Path(trace['directory'])/'prepared.png')
                preview.save(Path(trace['directory'])/'mask-preview.png')
            return {'status':'success','component_only':True,'prepared_dimensions':list(prepared.size),
                    'mask_source':mask_source,'device':'cpu','neural_reconstruction_executed':False}
        with stage('reconstruction_total'):
            result = generate_furniture_response(request, app.state.engine)
        trace = CURRENT.get()
        if trace is not None:
            trace['quality'] = result.get('quality')
        return result
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
