"""Temporarily release the project's GPU services, then restore them in finally."""
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
    args = parser.parse_args()
    project = args.project.resolve()
    experiment = Path(__file__).resolve().parents[2]
    logs = project / 'outputs/facade_house_test/services'
    logs.mkdir(parents=True, exist_ok=True)
    services = []
    listeners = {connection.laddr.port: connection.pid for connection in psutil.net_connections(kind='tcp')
                 if connection.status == psutil.CONN_LISTEN and connection.laddr.port in {8000, 8001}}
    for port, pid in listeners.items():
        process = psutil.Process(pid)
        command = process.cmdline()
        if Path(process.cwd()).resolve() != project or not any('uvicorn' in item for item in command):
            raise RuntimeError(f'Port {port} is not a recognized project service; no processes were stopped.')
        services.append({'port': port, 'process': process, 'command': command, 'environment': process.environ()})
    stopped = []
    try:
        for service in services:
            service['process'].terminate()
            stopped.append(service)
            service['process'].wait(timeout=20)
            print('Paused project service on port', service['port'], flush=True)
        time.sleep(3)
        print('Available RAM GiB after pausing:', round(psutil.virtual_memory().available / 1024**3, 2), flush=True)
        environment = os.environ.copy()
        environment.update({'PYTHONIOENCODING': 'utf-8', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'USE_TF': '0', 'USE_FLAX': '0'})
        python = project / '.venv-facade-diffusion/Scripts/python.exe'
        subprocess.run([str(python), '-m', 'experiments.facade_sd15_controlnet.run_house_wall',
                        '--project', str(project), '--model', str(args.model.resolve())],
                       cwd=experiment, env=environment, check=True)
    finally:
        for service in stopped:
            port = service['port']
            # No service should be replaced if another process acquired the port.
            if any(connection.status == psutil.CONN_LISTEN and connection.laddr.port == port
                   for connection in psutil.net_connections(kind='tcp')):
                print(f'Port {port} is already listening; did not launch a duplicate.', flush=True)
                continue
            with (logs / f'{port}-stdout.log').open('w', encoding='utf-8') as out, (logs / f'{port}-stderr.log').open('w', encoding='utf-8') as err:
                command = list(service['command'])
                # Windows venv launchers can expose their base interpreter in
                # process.cmdline(); preserve the known isolated furniture runtime.
                furniture_python = project / '.venv-furniture-quality/Scripts/python.exe'
                if port == 8001 and furniture_python.is_file() and any('furniture_worker:app' in item for item in command):
                    command[0] = str(furniture_python)
                process = subprocess.Popen(command, cwd=project, env=service['environment'],
                    stdout=out, stderr=err, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            print(f'Restored service on port {port}; PID {process.pid}', flush=True)


if __name__ == '__main__':
    main()
