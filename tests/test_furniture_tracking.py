import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from types import SimpleNamespace
import uuid

from PIL import Image
import pytest
import trimesh

from src.furniture import tracking
from src.furniture.service import generate_furniture_response


@pytest.fixture(scope='module')
def store(tmp_path_factory):
    from mlflow import MlflowClient
    directory = tmp_path_factory.mktemp('tracking')
    client = MlflowClient(tracking_uri='sqlite:///' + (directory / 'mlflow.db').as_posix())
    client.create_experiment(tracking.EXPERIMENT, artifact_location=(directory / 'artifacts').as_uri())
    return client


@pytest.fixture
def tracked(monkeypatch, store):
    monkeypatch.setenv('ARCHIGEN_MLFLOW_ENABLED', '1')
    monkeypatch.setattr(tracking, 'get_client', lambda: store)
    return store


def request(name='test chair'):
    buffer = io.BytesIO()
    Image.new('RGBA', (16, 16), (80, 100, 120, 255)).save(buffer, format='PNG')
    return SimpleNamespace(image_base64=base64.b64encode(buffer.getvalue()).decode(), name=name,
                           category='chair', target_height_m=.85, remove_bg=True)


class Generator:
    device_name = 'CPU'

    def generate_from_image(self, image, **kwargs):
        glb = trimesh.creation.box(extents=[.5, .85, .5]).export(file_type='glb')
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return dict(id=uuid.uuid4().hex, name=kwargs['name'], category=kwargs['category'],
                    engine='test volumetric', device='CPU', dimensions_m=[.5, .5, .85], bounds={},
                    num_vertices=8, num_faces=12, glb_base64=base64.b64encode(glb).decode(),
                    preview_image_base64=base64.b64encode(buffer.getvalue()).decode())


def run_for(client, item_id):
    experiment_id = tracking.get_experiment_id(client)
    runs = client.search_runs([experiment_id], filter_string=f"tags.furniture_id = '{item_id}'")
    assert len(runs) == 1
    return runs[0]


def test_success_records_metrics_input_and_downloadable_glb(tracked):
    req = request()
    result = generate_furniture_response(req, Generator())
    run = run_for(tracked, result['id'])
    assert run.info.status == 'FINISHED'
    assert run.data.params['input_sha256'] == hashlib.sha256(base64.b64decode(req.image_base64)).hexdigest()
    assert run.data.params['engine'] == 'test volumetric'
    assert run.data.params['device_type'] == 'cpu'
    assert run.data.metrics['generation_seconds'] > 0
    assert run.data.metrics['num_faces'] == 12
    assert 'peak_allocated_mib' not in run.data.metrics
    path = tracked.download_artifacts(run.info.run_id, 'output/model.glb')
    assert open(path, 'rb').read(4) == b'glTF'
    metadata_path = tracked.download_artifacts(run.info.run_id, 'output/metadata.json')
    assert json.load(open(metadata_path))['id'] == result['id']
    assert tracked.list_artifacts(run.info.run_id, 'input')[0].path == 'input/input.png'
    assert 'mlflow_run_id' not in result  # Existing public response is unchanged.


def test_reconstruction_failure_records_failed_run_and_original_exception(tracked):
    class Broken(Generator):
        def generate_from_image(self, *args, **kwargs):
            raise ValueError('invalid furniture foreground')

    before = {r.info.run_id for r in tracked.search_runs([tracking.get_experiment_id(tracked)])}
    with pytest.raises(ValueError, match='invalid furniture foreground'):
        generate_furniture_response(request(), Broken())
    runs = [r for r in tracked.search_runs([tracking.get_experiment_id(tracked)]) if r.info.run_id not in before]
    assert len(runs) == 1 and runs[0].info.status == 'FAILED'
    assert runs[0].data.tags['error_type'] == 'ValueError'
    path = tracked.download_artifacts(runs[0].info.run_id, 'diagnostics/error.txt')
    assert 'invalid furniture foreground' in open(path).read()


def test_concurrent_generations_keep_distinct_runs_and_artifacts(tracked):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda name: generate_furniture_response(request(name), Generator()), ['one', 'two']))
    runs = [run_for(tracked, result['id']) for result in results]
    assert runs[0].info.run_id != runs[1].info.run_id
    for result, run in zip(results, runs):
        path = tracked.download_artifacts(run.info.run_id, 'output/metadata.json')
        assert json.load(open(path))['name'] == result['name']


def test_tracking_outage_does_not_break_reconstruction(monkeypatch, caplog):
    monkeypatch.setenv('ARCHIGEN_MLFLOW_ENABLED', '1')
    monkeypatch.setattr(tracking, 'get_client', lambda: (_ for _ in ()).throw(ConnectionError('MLflow unavailable')))
    assert generate_furniture_response(request(), Generator())['status'] == 'success'
    assert 'MLflow tracking incomplete' in caplog.text


def test_artifact_outage_still_finishes_successful_reconstruction(tracked, monkeypatch, caplog):
    def unavailable(*args, **kwargs):
        raise ConnectionError('artifact upload unavailable')
    monkeypatch.setattr(tracked, 'log_artifact', unavailable)
    monkeypatch.setattr(tracked, 'log_artifacts', unavailable)
    result = generate_furniture_response(request(), Generator())
    assert run_for(tracked, result['id']).info.status == 'FINISHED'
    assert 'artifact upload unavailable' in caplog.text


def test_disabled_tracking_never_contacts_mlflow(monkeypatch):
    monkeypatch.setenv('ARCHIGEN_MLFLOW_ENABLED', '0')
    monkeypatch.setattr(tracking, 'get_client', lambda: pytest.fail('Tracking is disabled'))
    assert generate_furniture_response(request(), Generator())['status'] == 'success'


def test_explicit_cpu_baseline_cannot_load_triposr(monkeypatch):
    from src.furniture.image_to_3d import ImageTo3DGenerator
    monkeypatch.setenv('ARCHIGEN_FURNITURE_QUALITY', '1')
    monkeypatch.setattr(ImageTo3DGenerator, '_init_triposr', lambda self: pytest.fail('Neural loader called'))
    model = ImageTo3DGenerator(use_gpu=False, engine_mode='volumetric')
    assert model.tsr_model is None and model._quality_engine is None and model.device_name == 'CPU'
