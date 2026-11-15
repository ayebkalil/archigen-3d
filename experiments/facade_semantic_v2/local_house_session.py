"""Pause only recognized project GPU services, run the test, restore them in finally."""
import argparse
import os
from pathlib import Path
import subprocess
import time

import psutil


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--module', choices=['experiments.facade_semantic_v2.house_multiface',
                        'experiments.facade_reference_v3.run'], default='experiments.facade_semantic_v2.house_multiface')
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    logs = output / 'services'; logs.mkdir(parents=True, exist_ok=True)
    services = []
    for connection in psutil.net_connections(kind='tcp'):
        if connection.status != psutil.CONN_LISTEN or connection.laddr.port not in {8000, 8001}:
            continue
        process = psutil.Process(connection.pid); command = process.cmdline()
        if Path(process.cwd()).resolve() != project or not any('uvicorn' in arg for arg in command):
            raise RuntimeError('Unrecognized listener: no project services were paused')
        if not any(item['port'] == connection.laddr.port for item in services):
            services.append({'port': connection.laddr.port, 'process': process, 'command': command, 'environment': process.environ()})
    stopped = []
    try:
        for service in services:
            service['process'].terminate(); stopped.append(service); service['process'].wait(timeout=20)
            print('Paused project GPU service:', service['port'], flush=True)
        time.sleep(3)
        print('Available RAM GiB:', round(psutil.virtual_memory().available / 1024**3, 2), flush=True)
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'USE_TF': '0', 'USE_FLAX': '0',
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        subprocess.run([str(project / '.venv-facade-diffusion/Scripts/python.exe'), '-m',
            args.module, 'generate', '--model', str(args.model.resolve()),
            '--output', str(output)], cwd=Path(__file__).resolve().parents[2], env=env, check=True)
    finally:
        for service in stopped:
            port = service['port']
            if any(c.status == psutil.CONN_LISTEN and c.laddr.port == port for c in psutil.net_connections(kind='tcp')):
                print('Port already restored:', port, flush=True); continue
            command = list(service['command'])
            if port == 8001 and any('furniture_worker:app' in arg for arg in command):
                command[0] = str(project / '.venv-furniture-quality/Scripts/python.exe')
            with (logs / f'{port}-stdout.log').open('w', encoding='utf-8') as out, (logs / f'{port}-stderr.log').open('w', encoding='utf-8') as err:
                child = subprocess.Popen(command, cwd=project, env=service['environment'], stdout=out, stderr=err,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            print(f'Restored project service {port}, PID {child.pid}', flush=True)


if __name__ == '__main__': main()
