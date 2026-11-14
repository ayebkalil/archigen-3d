"""Optional migration of an ownership-verified Windows native stack to Docker."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import psutil

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'outputs/docker/native-state.json'
ROLES = {'frontend': (3000, '/'), 'backend': (8000, '/health'),
         'worker': (8001, '/health'), 'mlflow': (5000, '/health')}


def owned(process, role):
    try:
        command = process.cmdline()
        cwd = Path(process.cwd()).resolve()
        if cwd not in (ROOT, ROOT / 'frontend'):
            return False
        if role == 'frontend':
            return Path(command[0]).name.lower() == 'node.exe' and any('vite' in part for part in command[1:])
        expected = ROOT / ('.venv-furniture-quality' if role == 'worker' else '.venv') / 'Scripts/python.exe'
        interpreter = Path(command[0]).resolve() == expected.resolve()
        if role == 'mlflow':
            return interpreter and command[1:4] == ['-m', 'mlflow', 'server']
        module = 'backend.app.furniture_worker:app' if role == 'worker' else 'app.main:app'
        return interpreter and 'uvicorn' in command and module in command
    except (psutil.Error, IndexError):
        return False


def inspect():
    services = {}
    listeners = {row.laddr.port: row.pid for row in psutil.net_connections('tcp')
                 if row.status == psutil.CONN_LISTEN and row.laddr and row.pid}
    for role, (port, _) in ROLES.items():
        if port not in listeners:
            continue
        process = psutil.Process(listeners[port])
        # Venv launchers and MLflow both spawn child interpreters/listeners.
        process = next((p for p in [process, *process.parents()[:6]] if owned(p, role)), process)
        if not owned(process, role):
            raise RuntimeError(f'Port {port}: refusing unverified process PID {process.pid}')
        environment = {k: v for k, v in process.environ().items() if k.startswith(
            ('ARCHIGEN_', 'MLFLOW_', 'HF_', 'U2NET_', 'TRANSFORMERS_', 'OMP_', 'MKL_', 'CUDA_'))}
        services[role] = {'pid': process.pid, 'created': process.create_time(),
                          'command': process.cmdline(), 'cwd': process.cwd(),
                          'environment': environment, 'stopped': False}
    return services


def save(snapshot):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(snapshot, indent=2))
    temporary.replace(STATE)


def suspend():
    if STATE.exists() and json.loads(STATE.read_text()).get('suspended'):
        raise RuntimeError('Unrestored native snapshot exists; restore it before another migration')
    snapshot = {'suspended': True, 'services': inspect()}
    save(snapshot)
    try:
        for role in ('frontend', 'backend', 'worker', 'mlflow'):
            saved = snapshot['services'].get(role)
            if not saved:
                continue
            process = psutil.Process(saved['pid'])
            if process.create_time() != saved['created'] or not owned(process, role):
                raise RuntimeError('Process ownership changed; stopping migration')
            saved['stopped'] = True
            save(snapshot)
            children = process.children(recursive=True)
            for child in reversed(children):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
            _, alive = psutil.wait_procs([process, *children], timeout=15)
            if alive:
                raise RuntimeError(f'{role} did not stop')
        print('Verified native services stopped; recoverable snapshot saved.')
    except BaseException:
        restore()
        raise


def restore():
    if not STATE.exists():
        return
    snapshot = json.loads(STATE.read_text())
    if not snapshot.get('suspended'):
        return
    for role in ('mlflow', 'worker', 'backend', 'frontend'):
        saved = snapshot['services'].get(role)
        if not saved or not saved['stopped']:
            continue
        if any(row.laddr and row.laddr.port == ROLES[role][0] and row.status == psutil.CONN_LISTEN
               for row in psutil.net_connections('tcp')):
            raise RuntimeError(f'Restore port for {role} occupied; no process stopped')
        env = {**os.environ, **saved['environment']}
        with (STATE.parent / f'{role}-restored.log').open('ab') as log:
            process = subprocess.Popen(saved['command'], cwd=saved['cwd'], env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
        saved['restored_pid'] = process.pid
        save(snapshot)
        deadline = time.monotonic() + 240
        port, path = ROLES[role]
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f'{role} exited during restoration; inspect outputs/docker')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=3):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError(f'{role} restore deadline')
        saved['stopped'] = False
        save(snapshot)
    snapshot['suspended'] = False
    save(snapshot)
    print('Native services restored and ready.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=['inspect', 'suspend', 'restore'])
    args = parser.parse_args()
    if args.command == 'inspect':
        print('Verified native services: ' + ', '.join(inspect()))
    else:
        globals()[args.command]()
