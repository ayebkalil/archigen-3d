"""Request-scoped stage observations for the dedicated test process only."""
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
import json
from pathlib import Path
import time
import traceback

CURRENT = ContextVar('stress_trace', default=None)


def sync():
    import sys
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def stage(name):
    trace = CURRENT.get()
    if trace is None:
        yield
        return
    trace['last_stage'] = name
    active = trace.setdefault('_active_stages', [])
    if name in active:
        yield
        return
    active.append(name)
    if 'directory' in trace:
        try:
            write_trace(trace)
        except OSError as exc:
            trace['trace_storage_warning'] = str(exc)
    sync()
    started = time.perf_counter()
    try:
        yield
    except Exception as exc:
        trace.setdefault('error_stage', name)
        trace['exception_type'] = type(exc).__name__
        trace['traceback'] = traceback.format_exc()
        raise
    finally:
        sync()
        trace['stages_seconds'][name] = trace['stages_seconds'].get(name, 0) + time.perf_counter() - started
        active.remove(name)
        trace['last_stage'] = active[-1] if active else name
        if 'directory' in trace:
            try: write_trace(trace)
            except OSError as exc: trace['trace_storage_warning'] = str(exc)


def wrap(owner, name, label):
    original = getattr(owner, name)
    @wraps(original)
    def measured(*args, **kwargs):
        with stage(label):
            return original(*args, **kwargs)
    setattr(owner, name, measured)


def install_furniture(engine):
    import src.furniture.quality_engine as quality
    from PIL import Image
    original = quality.prepare_input

    def prepare(image, remove_bg=True):
        trace = CURRENT.get()
        if trace is not None:
            trace['source_dimensions'] = list(image.size)
        with stage('preprocessing'):
            result = original(image, remove_bg)
        if trace is not None:
            trace['prepared_dimensions'] = list(result[0].size)
            trace['mask_source'] = result[2]
            # Artifact I/O is deliberately outside the timed stage.
            result[0].save(Path(trace['directory']) / 'prepared.png')
            result[1].save(Path(trace['directory']) / 'mask-preview.png')
        return result
    quality.prepare_input = prepare
    # Decoder loads can occur lazily, separately from header parsing.
    original_open = Image.open
    def opened(*args, **kwargs):
        with stage('decode_header'):
            return original_open(*args, **kwargs)
    Image.open = opened
    wrap(Image.Image, 'load', 'decode_pixels')
    wrap(engine.model, 'forward', 'inference')
    wrap(engine, 'density_volume', 'density_field')
    wrap(quality, 'extract_surface', 'mesh_extraction')
    wrap(quality, 'bake_neural_atlas', 'texture_baking')
    wrap(quality.trimesh.exchange.gltf, 'export_glb', 'export')

    def tensor_input(module, inputs):
        trace = CURRENT.get()
        if trace is not None:
            trace.setdefault('model_inputs', []).append(list(inputs[0].shape))
    engine.model.image_tokenizer.register_forward_pre_hook(tensor_input)

    # Observe u2netp's actual ONNX tensor, rather than inferring it from the photo size.
    import onnxruntime as ort
    original_run = ort.InferenceSession.run
    def run(session, output_names, input_feed, *args, **kwargs):
        trace = CURRENT.get()
        if trace is not None:
            trace.setdefault('onnx_inputs', []).append({k: list(v.shape) for k, v in input_feed.items()})
            trace.setdefault('onnx_input_dtypes', []).append({k: str(v.dtype) for k, v in input_feed.items()})
        return original_run(session, output_names, input_feed, *args, **kwargs)
    ort.InferenceSession.run = run


def write_trace(trace):
    path = Path(trace['directory']) / 'trace.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({k:v for k,v in trace.items() if not k.startswith('_')}, indent=2), encoding='utf-8')
    temporary.replace(path)


def install_architecture(pipeline):
    """Instrumentation for restored architectural weights (no synthetic fallback)."""
    import onnxruntime as ort
    from PIL import Image
    wrap(pipeline, 'process_image', 'pipeline_total')
    original_open = Image.open
    def opened(*args, **kwargs):
        with stage('decode_header'):
            return original_open(*args, **kwargs)
    Image.open = opened
    wrap(Image.Image, 'load', 'decode_pixels')
    wrap(pipeline.vectorizer, 'vectorize', 'vectorization')
    wrap(pipeline.extruder, 'generate_threejs_scene_data', 'construction_3d')
    wrap(pipeline.bridge, '_finish_texture', 'facade_finishing')
    wrap(pipeline.bridge, '_build_facade_sketch', 'facade_sketch')
    original_run = ort.InferenceSession.run
    def run(session, outputs, feed, *args, **kwargs):
        trace = CURRENT.get()
        if trace is not None:
            trace.setdefault('onnx_inputs', []).append({k: list(v.shape) for k,v in feed.items()})
            trace.setdefault('onnx_input_dtypes', []).append({k: str(v.dtype) for k,v in feed.items()})
        with stage('inference'):
            return original_run(session, outputs, feed, *args, **kwargs)
    ort.InferenceSession.run = run
    for model in (pipeline.multitask_model, pipeline.baseline_model, pipeline.fallback_model,
                  getattr(pipeline.bridge.inference_engine, 'generator', None)):
        if model is not None:
            original = model.forward
            def forward(*args, _original=original, **kwargs):
                trace = CURRENT.get()
                if trace is not None:
                    trace.setdefault('model_inputs', []).append(list(args[0].shape))
                with stage('inference'):
                    return _original(*args, **kwargs)
            model.forward = forward
    original_resize = Image.Image.resize
    def resize(image, size, *args, **kwargs):
        with stage('resize'):
            result = original_resize(image, size, *args, **kwargs)
        trace = CURRENT.get()
        if trace is not None:
            trace.setdefault('resize_shapes', []).append({'from': list(image.size), 'to': list(size)})
        return result
    Image.Image.resize = resize
