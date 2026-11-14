"""Safety checks for container bootstrap (no Docker daemon/network required)."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(root, content=b'correct'):
    path = root / 'models/artifacts.json'
    path.parent.mkdir(parents=True)
    item = {'path': 'models/saved/model.onnx', 'size': len(content),
            'sha256': hashlib.sha256(content).hexdigest(), 'url': 'https://example.invalid/model'}
    path.write_text(json.dumps({'schema_version': 1, 'profiles': {'runtime': [item['path']]}, 'artifacts': [item]}))
    return root / item['path']


def test_bootstrap_missing_and_verified_reuse(tmp_path):
    target = fixture(tmp_path)
    module = load('prepare_docker_models')
    module.prepare(tmp_path, fetch=lambda *a, **k: io.BytesIO(b'correct'))
    assert target.read_bytes() == b'correct'
    module.prepare(tmp_path, fetch=lambda *a, **k: pytest.fail('Verified files must not download'))


def test_bad_download_preserves_previous_file(tmp_path):
    target = fixture(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b'old')
    with pytest.raises(ValueError, match='mismatch'):
        load('prepare_docker_models').prepare(tmp_path, fetch=lambda *a, **k: io.BytesIO(b'incorrect'[:7]))
    assert target.read_bytes() == b'old'
    assert list(target.parent.iterdir()) == [target]


def test_download_repairs_corruption(tmp_path):
    target = fixture(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b'corrupt')
    load('prepare_docker_models').prepare(tmp_path, fetch=lambda *a, **k: io.BytesIO(b'correct'))
    assert target.read_bytes() == b'correct'


def test_manifest_cannot_escape_weight_directory(tmp_path):
    fixture(tmp_path)
    path = tmp_path / 'models/artifacts.json'
    manifest = json.loads(path.read_text())
    manifest['profiles']['runtime'] = ['../../escape']
    manifest['artifacts'][0]['path'] = '../../escape'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='escapes'):
        load('prepare_docker_models').prepare(tmp_path)


def test_sqlite_backup_preserves_runs(tmp_path):
    database = tmp_path / 'mlflow.db'
    with sqlite3.connect(database) as connection:
        connection.execute('create table runs (id text)')
        connection.execute("insert into runs values ('historical')")
    load('docker_data').backup(database, tmp_path / 'backups')
    with sqlite3.connect(next((tmp_path / 'backups').glob('*.db'))) as connection:
        assert connection.execute('select id from runs').fetchall() == [('historical',)]


def test_library_import_is_once_and_does_not_overwrite(tmp_path, monkeypatch):
    module = load('docker_data')
    monkeypatch.setattr(module.os, 'chown', lambda *args: None, raising=False)
    source, destination = tmp_path / 'native', tmp_path / 'library'
    (source / 'custom').mkdir(parents=True)
    destination.mkdir()
    (source / 'custom/chair.glb').write_bytes(b'original')
    (source / 'custom_furniture.json').write_text('[]')
    module.seed(source, destination)
    (destination / 'custom/chair.glb').write_bytes(b'edited')
    module.seed(source, destination)
    assert (destination / 'custom/chair.glb').read_bytes() == b'edited'


def test_native_transition_rejects_foreign_process_before_any_stop(tmp_path, monkeypatch):
    pytest.importorskip('psutil')
    module = load('docker_native_transition')
    monkeypatch.setattr(module, 'STATE', tmp_path / 'snapshot.json')
    def foreign():
        raise RuntimeError('refusing unverified process')
    monkeypatch.setattr(module, 'inspect', foreign)
    with pytest.raises(RuntimeError, match='unverified'):
        module.suspend()
    assert not module.STATE.exists()


def test_native_transition_restores_after_partial_stop_failure(tmp_path, monkeypatch):
    pytest.importorskip('psutil')
    module = load('docker_native_transition')
    monkeypatch.setattr(module, 'STATE', tmp_path / 'snapshot.json')
    monkeypatch.setattr(module, 'inspect', lambda: {
        'frontend': {'pid': 123, 'created': 1, 'stopped': False}})
    class Process:
        def create_time(self): return 1
        def children(self, recursive): return []
        def kill(self): raise OSError('simulated stop failure')
    monkeypatch.setattr(module.psutil, 'Process', lambda pid: Process())
    monkeypatch.setattr(module, 'owned', lambda process, role: True)
    recovered = []
    monkeypatch.setattr(module, 'restore', lambda: recovered.append(True))
    with pytest.raises(OSError, match='stop failure'):
        module.suspend()
    assert recovered == [True]
    snapshot = json.loads(module.STATE.read_text())
    assert snapshot['suspended'] and snapshot['services']['frontend']['stopped']
