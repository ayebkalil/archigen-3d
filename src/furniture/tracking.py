"""Explicit, request-scoped MLflow runs for furniture inference (no autologging)."""
import base64
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = 'furniture-reconstruction'
LOGGER = logging.getLogger(__name__)


def tracking_enabled():
    return os.getenv('ARCHIGEN_MLFLOW_ENABLED', '0') == '1'


def get_client(uri=None):
    # A failed local tracking service must not hold up inference for minutes.
    os.environ.setdefault('MLFLOW_HTTP_REQUEST_TIMEOUT', '3')
    os.environ.setdefault('MLFLOW_HTTP_REQUEST_MAX_RETRIES', '0')
    # MLflow's decorative URL output cannot be encoded by redirected Windows
    # cp1252 streams. The UI and ordinary run-ID logs provide the same links.
    os.environ.setdefault('MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT', 'true')
    from mlflow import MlflowClient
    return MlflowClient(tracking_uri=uri or os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))


def get_experiment_id(client):
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment:
        return experiment.experiment_id
    try:
        return client.create_experiment(EXPERIMENT)
    except Exception:
        # Two first requests can race to create the same experiment.
        experiment = client.get_experiment_by_name(EXPERIMENT)
        if experiment:
            return experiment.experiment_id
        raise


@lru_cache(maxsize=8)
def _file_hash(path, size, modified_ns):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@lru_cache(maxsize=1)
def code_version():
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, timeout=5).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True, timeout=5).strip())
        return {'git_revision': revision, 'git_dirty_at_process_start': str(dirty).lower()}
    except (OSError, subprocess.SubprocessError):
        return {'git_revision': 'unavailable'}


def generation_parameters(request, generator):
    target = getattr(generator, '_quality_engine', None) or generator
    quality = type(target).__name__ == 'QualityFurnitureEngine'
    neural = quality or getattr(target, 'tsr_model', None) is not None
    device = str(getattr(target, 'device_name', 'unknown'))
    params = {
        'engine_family': 'triposr' if neural else 'volumetric',
        'device_type': 'cuda' if 'cuda' in device.lower() else 'cpu',
        'device': device, 'category': request.category or 'chair',
        'target_height_m': request.target_height_m if request.target_height_m is not None else 'category_default',
        'remove_bg': bool(request.remove_bg), 'python_version': platform.python_version(),
        **code_version(),
    }
    for package in ('torch', 'numpy', 'trimesh', 'transformers', 'mlflow-skinny'):
        try:
            params['version_' + package.replace('-', '_')] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if neural:
        checkpoint = ROOT / 'TripoSR/model.ckpt'
        if checkpoint.is_file():
            stat = checkpoint.stat()
            params['checkpoint_sha256'] = _file_hash(str(checkpoint), stat.st_size, stat.st_mtime_ns)
    if params['device_type'] == 'cuda':
        import torch
        params['gpu_name'] = torch.cuda.get_device_name(0)
        params['cuda_runtime'] = torch.version.cuda
    if quality:
        preset = os.getenv('ARCHIGEN_FURNITURE_DETAIL', 'balanced')
        params.update(preset=preset, density_resolution=384 if preset == 'high' else 256,
                      texture_resolution=2048 if preset == 'high' else 1024)
    elif not neural:
        params.update(preset='volumetric', precision='FP32', texture_resolution='source_crop')
        if hasattr(target, 'volumetric_resolution'):
            params['density_resolution'] = target.volumetric_resolution
    return params


class GenerationRun:
    """All MLflow errors are warnings; reconstruction errors retain their type."""

    def __init__(self, request, generator, tags=None):
        self.client = None
        self.run_id = None
        self.params = {}
        if not tracking_enabled():
            return
        try:
            self.client = get_client()
            run = self.client.create_run(get_experiment_id(self.client), tags={
                'mlflow.runName': 'furniture-generation', 'run_kind': 'studio_generation',
                **(tags or {}),
            })
            self.run_id = run.info.run_id
            self.params = generation_parameters(request, generator)
            self._params(self.params)
            LOGGER.info('Furniture MLflow run: %s', self.run_id)
        except Exception as exc:
            self._warning(exc)

    def _warning(self, exc):
        LOGGER.warning('Furniture MLflow tracking incomplete (run=%s): %s', self.run_id, exc)

    def _params(self, params):
        from mlflow.entities import Param
        self.client.log_batch(self.run_id, params=[Param(k, str(v)) for k, v in params.items()])

    def _metrics(self, metrics):
        from mlflow.entities import Metric
        now = int(time.time() * 1000)
        self.client.log_batch(self.run_id, metrics=[
            Metric(k, float(v), now, 0) for k, v in metrics.items()
            if isinstance(v, (int, float)) and math.isfinite(v)
        ])

    def input(self, image_bytes, image):
        if not self.run_id:
            return
        try:
            self._params({'input_sha256': hashlib.sha256(image_bytes).hexdigest(),
                          'input_width_px': image.width, 'input_height_px': image.height})
            with tempfile.TemporaryDirectory(prefix='furniture-mlflow-') as directory:
                path = Path(directory) / 'input.png'
                image.save(path, format='PNG')
                self.client.log_artifact(self.run_id, str(path), 'input')
        except Exception as exc:
            self._warning(exc)

    def finish(self, result, duration):
        if not self.run_id:
            return
        try:
            quality = result.get('quality') or {}
            self.client.set_tag(self.run_id, 'furniture_id', result['id'])
            self._params({'engine': result['engine'],
                          **{key: quality[key] for key in ('precision', 'mask_source', 'texture_source', 'checkpoint_loading') if key in quality}})
            glb = base64.b64decode(result['glb_base64'].split(',')[-1])
            metrics = {
                'generation_seconds': duration, 'num_vertices': result['num_vertices'],
                'num_faces': result['num_faces'], 'glb_bytes': len(glb),
                **dict(zip(('width_m', 'depth_m', 'height_m'), result['dimensions_m'])),
            }
            for key in ('atlas_coverage', 'peak_device_sampled_mib', 'peak_allocated_mib', 'peak_reserved_mib', 'components', 'watertight'):
                if key in quality:
                    metrics[key] = quality[key]
            self._metrics(metrics)
            with tempfile.TemporaryDirectory(prefix='furniture-mlflow-') as directory:
                directory = Path(directory)
                (directory / 'model.glb').write_bytes(glb)
                (directory / 'preview.png').write_bytes(base64.b64decode(result['preview_image_base64'].split(',')[-1]))
                metadata = {k: v for k, v in result.items() if k not in ('glb_bytes', 'glb_base64', 'preview_image_base64')}
                metadata.update(generation_seconds=duration, mlflow_run_id=self.run_id, parameters=self.params)
                (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
                self.client.log_artifacts(self.run_id, str(directory), 'output')
        except Exception as exc:
            self._warning(exc)
        finally:
            self._terminate('FINISHED')

    def fail(self, exc):
        if not self.run_id:
            return
        try:
            self.client.set_tag(self.run_id, 'error_type', type(exc).__name__)
            with tempfile.TemporaryDirectory(prefix='furniture-mlflow-') as directory:
                path = Path(directory) / 'error.txt'
                path.write_text(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)), encoding='utf-8')
                self.client.log_artifact(self.run_id, str(path), 'diagnostics')
        except Exception as tracking_exc:
            self._warning(tracking_exc)
        finally:
            self._terminate('FAILED')

    def _terminate(self, status):
        try:
            self.client.set_terminated(self.run_id, status)
        except Exception as exc:
            self._warning(exc)
