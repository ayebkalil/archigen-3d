"""Suspend only ownership-verified native services; leave frontend and MLflow running."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT/'outputs/stress/stack-state.json'
NATIVE_KEYS = ('ARCHIGEN_FURNITURE_DETAIL','ARCHIGEN_FURNITURE_QUALITY','ARCHIGEN_FURNITURE_SERVICE_URL',
               'ARCHIGEN_FACADE_EXPERIMENT','ARCHIGEN_MLFLOW_ENABLED','MLFLOW_TRACKING_URI',
               'MLFLOW_HTTP_REQUEST_TIMEOUT','MLFLOW_HTTP_REQUEST_MAX_RETRIES',
               'HF_HOME','U2NET_HOME','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
               'OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONUNBUFFERED','CUDA_VISIBLE_DEVICES')


def owned(process, role):
    try:
        command = process.cmdline()
        cwd = Path(process.cwd()).resolve()
        expected = 'backend.app.furniture_worker:app' if role=='worker' else 'app.main:app'
        # Exact argv members, never matching this controller's shell command text.
        python = ROOT/('.venv-furniture-quality' if role=='worker' else '.venv')/'Scripts/python.exe'
        return bool(command) and cwd == ROOT.resolve() and expected in command and 'uvicorn' in command and Path(command[0]).resolve()==python.resolve()
    except psutil.Error: return False


def suspend():
    if STATE.exists():
        previous = json.loads(STATE.read_text())
        if previous.get('suspended'):
            raise RuntimeError('An earlier suspension has not been restored; restore it first')
    metadata = json.loads((ROOT/'outputs/native_gpu/processes.json').read_text())
    snapshot = {'suspended':False,'services':{},'detail':metadata.get('detail','high'),
                'owner_pid':int(os.getenv('ARCHIGEN_STRESS_OWNER_PID','0'))}
    for role in ('backend','worker'):
        try: process = psutil.Process(metadata[role])
        except psutil.NoSuchProcess: continue
        if not owned(process,role): raise RuntimeError(f'Refusing to stop unverified {role} PID {process.pid}')
        snapshot['services'][role] = {'pid':process.pid,'created':process.create_time(),
                                      'cmdline':process.cmdline(),'cwd':process.cwd(),
                                      'runtime_env':{k:v for k,v in process.environ().items() if k in NATIVE_KEYS}}
    # Save intent before stopping anything so a partial suspension remains recoverable.
    STATE.parent.mkdir(parents=True,exist_ok=True)
    snapshot['suspended'] = True
    STATE.write_text(json.dumps(snapshot,indent=2))
    for role, saved in snapshot['services'].items():
        process = psutil.Process(saved['pid'])
        if process.create_time()!=saved['created'] or not owned(process,role): raise RuntimeError('PID ownership changed')
        children = process.children(recursive=True)
        for child in reversed(children): child.kill()
        process.kill()
        psutil.wait_procs([process,*children],timeout=10)
    print('Verified native API and worker suspended; frontend and MLflow preserved.')


def restore():
    if not STATE.exists(): return
    snapshot = json.loads(STATE.read_text())
    if not snapshot.get('suspended'): return
    env = {**os.environ,'ARCHIGEN_FURNITURE_DETAIL':snapshot['detail'],
           'ARCHIGEN_MLFLOW_ENABLED':'1','MLFLOW_TRACKING_URI':'http://127.0.0.1:5000',
           'MLFLOW_HTTP_REQUEST_TIMEOUT':'3','MLFLOW_HTTP_REQUEST_MAX_RETRIES':'0',
           'HF_HOME':str(ROOT/'outputs/runtime/huggingface'),'U2NET_HOME':str(ROOT/'outputs/runtime/rembg'),
           'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}
    metadata = json.loads((ROOT/'outputs/native_gpu/processes.json').read_text())
    for role in ('worker','backend'):
        if role not in snapshot['services']: continue
        port = 8001 if role=='worker' else 8000
        restored_pid = snapshot['services'][role].get('restored_pid')
        if restored_pid:
            try:
                restored = psutil.Process(restored_pid)
                if restored.create_time()==snapshot['services'][role].get('restored_created') and owned(restored,role):
                    health = httpx.get(f'http://127.0.0.1:{port}/health',timeout=3,trust_env=False)
                    if health.is_success:
                        metadata[role] = restored.pid
                        continue
            except (psutil.Error,httpx.HTTPError): pass
        import socket
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',port))==0: raise RuntimeError(f'Restoration port {port} occupied; no existing process stopped')
        if role=='backend': env['ARCHIGEN_FURNITURE_SERVICE_URL']='http://127.0.0.1:8001'
        else: env.pop('ARCHIGEN_FURNITURE_SERVICE_URL',None)
        if 'runtime_env' in snapshot['services'][role]:
            for key in NATIVE_KEYS: env.pop(key,None)
            env.update(snapshot['services'][role]['runtime_env'])
        env.pop('ARCHIGEN_STRESS_OWNER_PID',None)
        with (ROOT/f'outputs/native_gpu/{role}.stdout.log').open('ab') as log:
            process = subprocess.Popen(snapshot['services'][role]['cmdline'],cwd=snapshot['services'][role]['cwd'],
                                        env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        metadata[role] = process.pid
        snapshot['services'][role]['restored_pid'] = process.pid
        snapshot['services'][role]['restored_created'] = psutil.Process(process.pid).create_time()
        STATE.write_text(json.dumps(snapshot,indent=2))
        deadline = time.monotonic()+180
        while time.monotonic()<deadline:
            if process.poll() is not None: raise RuntimeError(f'{role} exited during restoration')
            try:
                health = httpx.get(f'http://127.0.0.1:{port}/health',timeout=2,trust_env=False)
                if health.is_success: break
            except httpx.HTTPError: pass
            time.sleep(1)
        else: raise TimeoutError(f'{role} restoration readiness deadline')
        STATE.write_text(json.dumps(snapshot,indent=2))
    (ROOT/'outputs/native_gpu/processes.json').write_text(json.dumps(metadata,indent=2))
    snapshot['suspended'] = False
    STATE.write_text(json.dumps(snapshot,indent=2))
    print('Original native service commands restored and health checked.')


if __name__=='__main__':
    {'suspend':suspend,'restore':restore}[sys.argv[1]]()
