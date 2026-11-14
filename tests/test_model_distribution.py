import asyncio
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.models.artifacts import ModelArtifactError, inspect_artifacts, require_artifacts, verify_model


def test_inventory_rejects_missing_and_equal_size_corruption(tmp_path):
    content = b'verified model bytes'
    model = tmp_path / 'models/example.onnx'
    model.parent.mkdir()
    manifest = {'schema_version': 1, 'profiles': {'runtime': ['models/example.onnx']}, 'artifacts': [
        {'path': 'models/example.onnx', 'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}]}
    (tmp_path / 'models/artifacts.json').write_text(json.dumps(manifest))
    assert inspect_artifacts(root=tmp_path)[0]['status'] == 'missing'
    with pytest.raises(ModelArtifactError, match='scripts/models.py pull'):
        require_artifacts(root=tmp_path)
    model.write_bytes(b'X' * len(content))
    assert inspect_artifacts(root=tmp_path)[0]['status'] == 'corrupt'
    with pytest.raises(ModelArtifactError, match='checksum mismatch'):
        verify_model(model, root=tmp_path)
    model.write_bytes(content)
    assert require_artifacts(root=tmp_path)[0]['status'] == 'ok'


def test_inventory_rejects_path_escape(tmp_path):
    (tmp_path / 'models').mkdir()
    (tmp_path / 'models/artifacts.json').write_text(json.dumps({'schema_version': 1,
        'profiles': {'runtime': []}, 'artifacts': [{'path': '../outside.onnx'}]}))
    with pytest.raises(ModelArtifactError, match='escapes'):
        inspect_artifacts(root=tmp_path)


def test_health_reports_unavailable_models_as_503(monkeypatch):
    from backend.app import main
    from app.floorplan_engine import FloorplanPipeline

    def unavailable():
        raise ModelArtifactError('Missing champion. Run python scripts/models.py pull')

    monkeypatch.setattr(FloorplanPipeline, 'get_instance', unavailable)
    result = asyncio.run(main.health_check())
    assert result.status_code == 503
    content = json.loads(result.body)
    assert content['model_loaded'] is False
    assert content['status'] == 'not_ready'
    assert 'scripts/models.py pull' in content['readiness_errors'][0]


def test_health_requires_facade_and_fallback_too(monkeypatch):
    from backend.app import main
    from app.floorplan_engine import FloorplanPipeline
    pipeline = SimpleNamespace(multitask_model=None, onnx_multitask=object(),
        fallback_model=None, onnx_fallback=None, checkpoint_hashes={}, bridge=SimpleNamespace(inference_engine=None))
    monkeypatch.setattr(FloorplanPipeline, 'get_instance', lambda: pipeline)
    result = asyncio.run(main.health_check())
    assert result.status_code == 503
    content = json.loads(result.body)
    assert content['model_loaded'] is True
    assert content['facade_model_loaded'] is False
    assert len(content['readiness_errors']) == 2


def test_missing_facade_does_not_return_input_as_generated_texture():
    from src.bridge.facade_bridge import FacadeBridge
    bridge = FacadeBridge.__new__(FacadeBridge)
    bridge.inference_engine, bridge.enhance_textures = None, False
    with pytest.raises(ModelArtifactError, match='Facade model is unavailable'):
        bridge.render_wall_facade({'polygon_meters': [[0, 0], [6, 0], [6, .2], [0, .2]], 'openings': []})


def test_missing_triposr_does_not_return_procedural_furniture(monkeypatch):
    from src.furniture.image_to_3d import ImageTo3DGenerator
    monkeypatch.delenv('ARCHIGEN_ALLOW_PROCEDURAL_FURNITURE', raising=False)
    generator = ImageTo3DGenerator.__new__(ImageTo3DGenerator)
    generator.tsr_model = None
    with pytest.raises(ModelArtifactError, match='setup_furniture.py pull'):
        generator.generate_from_image(Image.new('RGB', (16, 16)))


def test_render_api_does_not_echo_upload_on_missing_weights(monkeypatch):
    from backend.app import main
    import app.inference as inference
    from fastapi.testclient import TestClient
    def missing():
        raise ModelArtifactError('Missing facade weights')
    monkeypatch.setattr(inference, 'ArchiGenONNXInference', missing)
    content = io.BytesIO(); Image.new('RGB', (8, 8)).save(content, format='PNG')
    result = TestClient(main.app).post('/predict/render', files={'file': ('test.png', content.getvalue(), 'image/png')})
    assert result.status_code == 503
    assert result.json()['error'] == 'model_artifact_unavailable'


def test_runtime_dvc_sources_are_portable_and_match_manifest():
    import yaml
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'models/artifacts.json').read_text())
    for item in manifest['artifacts']:
        if item['path'] not in manifest['profiles']['runtime']:
            continue
        pointer = yaml.safe_load((root / item['dvc']).read_text())
        assert pointer['deps'][0]['path'] == item['url']
        assert pointer['outs'][0].get('md5', item['md5']) == item['md5']
        assert pointer['deps'][0]['size'] == item['size']
        assert item['url'].startswith('https://github.com/ayebkalil/archigen-3d/releases/download/model-weights-v1/')


def test_onnx_eight_fold_inverts_all_views_before_averaging():
    import numpy as np
    from backend.app.floorplan_engine import FloorplanPipeline

    class StopAfterInference(Exception):
        pass

    class Session:
        calls = 0

        def run(self, names, inputs):
            self.calls += 1
            pixels = inputs['input'][:, :1]
            return [np.concatenate([pixels * factor for factor in range(count)], axis=1) for count in (5, 12)]

    observed = []
    def capture(**values):
        observed.append(values['room_probabilities'])
        raise StopAfterInference()

    pipeline = FloorplanPipeline.__new__(FloorplanPipeline)
    pipeline.device = pipeline.multitask_model = pipeline.baseline_model = None
    pipeline.prefer_onnx = True
    pipeline.onnx_multitask = Session()
    pipeline.mean_np = np.zeros((3, 1, 1), np.float32)
    pipeline.std_np = np.ones((3, 1, 1), np.float32)
    pipeline.vectorizer = SimpleNamespace(vectorize=capture)
    # Asymmetric pattern catches transpose/flip inverses applied in the wrong order.
    pixels = np.zeros((512, 512, 3), np.uint8)
    pixels[20:90, 130:290] = 190
    pixels[300:440, 30:100] = 75
    stream = io.BytesIO(); Image.fromarray(pixels).save(stream, format='PNG')
    for mode in ('none', '8_fold'):
        with pytest.raises(StopAfterInference):
            pipeline.process_image(stream.getvalue(), tta_mode=mode)
    assert pipeline.onnx_multitask.calls == 9
    np.testing.assert_allclose(observed[0], observed[1], atol=1e-6)
