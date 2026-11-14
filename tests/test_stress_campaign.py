import io
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import psutil
import pytest

from src.stress.inputs import cases, create
from src.stress.isolation import Job
from src.stress.trace import CURRENT, stage
from src.stress.report import render
from src.stress.manage_stack import owned

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source(tmp_path):
    path = tmp_path/'source.png'
    Image.new('RGBA',(80,100),(100,150,200,255)).save(path)
    return path


def test_transform_determinism_and_source_dimensions(source):
    case = {'transform':'noise','value':24}
    assert create(case,source,123)==create(case,source,123)
    assert create(case,source,123)!=create(case,source,124)
    data = create({'transform':'resize','value':2048},source,123)
    with Image.open(io.BytesIO(data)) as image: assert image.size==(1638,2048)


def test_cases_have_unique_ids_and_distinct_facade_interfaces():
    config = json.loads((ROOT/'configs/stress_campaign.json').read_text())
    for model in ('furniture','floorplan','facade'):
        generated = cases(config,model)
        assert len({c['id'] for c in generated})==len(generated)
        if model=='facade': assert all('payload' in c or c.get('semantic') for c in generated)
    from src.stress.inputs import plan_geometry
    geometry = plan_geometry(128)
    assert len(geometry['room_polygons_source_px'])==128
    assert len(geometry['door_segments_source_px'])==128
    assert len(geometry['window_segments_source_px'])==128


def test_invalid_inputs_are_not_repaired(source):
    assert create({'transform':'empty'},source,0)==b''
    for name in ('unsupported','zero_dimension','extreme_header'):
        with pytest.raises(Exception): Image.open(io.BytesIO(create({'transform':name},source,0)))


def test_concurrent_trace_isolation_and_failure_stage(monkeypatch):
    monkeypatch.setattr('src.stress.trace.sync',lambda:None)
    def task(name):
        trace = {'stages_seconds':{}}
        token = CURRENT.set(trace)
        try:
            with pytest.raises(ValueError), stage('outer'), stage(name):
                time.sleep(.01)
                raise ValueError(name)
            return trace
        finally: CURRENT.reset(token)
    with ThreadPoolExecutor(2) as pool: results = list(pool.map(task,['decode','inference']))
    assert [r['error_stage'] for r in results]==['decode','inference']
    assert 'inference' not in results[0]['stages_seconds']


@pytest.mark.skipif(os.name!='nt',reason='Windows Job Objects')
def test_job_kills_entire_process_tree(tmp_path):
    # Child starts a grandchild; close must kill both without touching other processes.
    child_module = tmp_path/'spawn_child.py'
    child_module.write_text('import subprocess,sys,time\np=subprocess.Popen([sys.executable,"-c","import time;time.sleep(90)"])\nprint(p.pid,flush=True)\ntime.sleep(90)\n')
    env = {**os.environ,'PYTHONPATH':str(tmp_path)}
    log_path = tmp_path/'log.txt'
    job = Job(512*1024**2)
    with log_path.open('wb') as log:
        child = job.start(sys.executable,'spawn_child',[],cwd=tmp_path,env=env,stdout=log)
        deadline = time.monotonic()+10
        while not log_path.read_text().strip() and time.monotonic()<deadline: time.sleep(.05)
        grandchild = int(log_path.read_text().strip())
        assert psutil.pid_exists(grandchild)
        assert job.contains(grandchild)
        job.close()
        deadline = time.monotonic()+3
        while psutil.pid_exists(grandchild) and time.monotonic()<deadline: time.sleep(.05)
        assert child.poll() is not None
        assert not psutil.pid_exists(grandchild)


def test_tracking_failure_is_nonfatal(monkeypatch,tmp_path):
    from src.stress.campaign import Tracking
    monkeypatch.setattr('src.stress.campaign.get_client',lambda uri: (_ for _ in ()).throw(OSError('offline')))
    tracking = Tracking('http://unavailable')
    run_id = tracking.start('failure-test')
    assert len(run_id)==32
    tracking.finish(run_id,{},tmp_path)
    assert 'offline' in tracking.warnings[0]


def test_no_production_tracking_in_test_environment(monkeypatch):
    from src.furniture.tracking import GenerationRun
    monkeypatch.setenv('ARCHIGEN_MLFLOW_ENABLED','0')
    monkeypatch.setattr('src.furniture.tracking.get_client',lambda: pytest.fail('duplicate run'))
    assert GenerationRun(None,None).run_id is None


def test_ownership_requires_exact_module_and_checkout():
    class Process:
        def cmdline(self): return ['powershell.exe','-Command','uvicorn app.main:app']
        def cwd(self): return str(ROOT)
    assert not owned(Process(),'backend')


def test_report_excludes_warmups_and_does_not_claim_accuracy():
    record = {'model':'furniture','case':{'id':'reference'},'run_kind':'warmup','metrics':{'http_total_seconds':999}}
    report = render({'records':[record],'run_id':'test','skipped':[]})
    assert '999' not in report
    assert '0 measured attempts' in report
    assert 'CPU/GPU comparison was not repeated' in report


def test_glb_checks_do_not_claim_visual_quality(tmp_path):
    import trimesh,base64
    from src.stress.evaluate import evaluate
    mesh = trimesh.creation.box()
    data = mesh.export(file_type='glb')
    response = {'model_glb_base64':base64.b64encode(data).decode(),'quality':{}}
    result = evaluate('furniture',response,tmp_path)
    assert result['quality']=='unreviewed'
    assert result['metrics']['faces']==12
    assert (tmp_path/'model.glb').read_bytes()==data


def test_mask_metrics_require_matching_coordinates(tmp_path):
    from src.stress.evaluate import mask_metrics
    expected = Image.new('L',(32,32),0)
    from PIL import ImageDraw
    ImageDraw.Draw(expected).rectangle((8,8,24,24),fill=255)
    expected.save(tmp_path/'expected.png'); expected.save(tmp_path/'actual.png')
    metrics = mask_metrics(tmp_path/'expected.png',tmp_path/'actual.png',tmp_path)
    assert metrics['foreground_iou']==1
    assert metrics['mask_boundary_f1_2px']==1
    Image.new('L',(16,16)).save(tmp_path/'actual.png')
    with pytest.raises(ValueError): mask_metrics(tmp_path/'expected.png',tmp_path/'actual.png',tmp_path)


@pytest.mark.skipif(os.name!='nt',reason='Windows Job Objects')
def test_job_memory_limit_prevents_large_allocation(tmp_path):
    module = tmp_path/'allocate_memory.py'
    module.write_text('import sys\ntry:\n x=bytearray(512*1024**2)\nexcept MemoryError:\n sys.exit(0)\nsys.exit(9)\n')
    with Job(96*1024**2) as job, (tmp_path/'memory.log').open('wb') as log:
        child = job.start(sys.executable,'allocate_memory',[],cwd=tmp_path,
                          env={**os.environ,'PYTHONPATH':str(tmp_path)},stdout=log)
        assert child.wait(timeout=10)==0


def test_local_weight_recovery_verifies_dvc_digest(tmp_path):
    import hashlib
    from src.stress.assets import recover_local_weights
    data = b'checkpoint-fixture'
    digest = hashlib.md5(data).hexdigest()
    model = tmp_path/'models/saved'; model.mkdir(parents=True)
    manifest = model/'fixture.onnx.dvc'
    manifest.write_text(f'outs:\n- md5: {digest}\n  size: {len(data)}\n  path: fixture.onnx\n')
    cache = tmp_path/'.dvc/cache/files/md5'/digest[:2]; cache.mkdir(parents=True)
    (cache/digest[2:]).write_bytes(b'x'*len(data))
    result = recover_local_weights(tmp_path)
    assert not result['recovered'] and result['warnings']
    assert not (model/'fixture.onnx').exists()
    (cache/digest[2:]).write_bytes(data)
    result = recover_local_weights(tmp_path)
    assert len(result['recovered'])==1
    assert (model/'fixture.onnx').read_bytes()==data
