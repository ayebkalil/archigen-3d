"""Run inside the API container: exercise the actual frontend proxy and CUDA API."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--url', default='http://frontend')
    parser.add_argument('--tracking-uri', default='http://mlflow:5000')
    parser.add_argument('--output', default='/tmp/docker-validation')
    parser.add_argument('--skip-furniture', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    def get(path):
        with urllib.request.urlopen(args.url + path, timeout=30) as response:
            return response.read()

    def post(path, body, content_type='application/json'):
        request = urllib.request.Request(args.url + path, body, headers={'Content-Type': content_type})
        with urllib.request.urlopen(request, timeout=840) as response:
            return json.load(response)

    summary = {'health': json.loads(get('/health')), 'metadata': json.loads(get('/model/metadata'))}
    hashes = summary['metadata']['checkpoint_hashes']
    assert all(hashes.get(key) for key in ('primary_sha256', 'fallback_sha256', 'facade_sha256'))
    plan = get('/sample_floorplan.png')
    boundary = 'ArchiGenDockerValidation'
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="plan.png"\r\n'
            'Content-Type: image/png\r\n\r\n').encode() + plan + f'\r\n--{boundary}--\r\n'.encode()
    started = time.monotonic()
    segmentation = post('/predict/floorplan', body, f'multipart/form-data; boundary={boundary}')
    (output / 'floorplan.json').write_text(json.dumps(segmentation))
    summary['floorplan_seconds'] = time.monotonic() - started
    summary['floorplan_rooms'] = len(segmentation.get('rooms', []))
    assert summary['floorplan_rooms'] > 0, 'No rooms in reference segmentation'
    facade = post('/facade/generate', json.dumps({'style': 'Modern Minimalist', 'num_floors': 1}).encode())
    assert facade['status'] == 'success'
    assert facade.get('model_info'), 'Missing facade model provenance'
    summary['facade_model'] = facade['model_info']
    (output / 'facade.png').write_bytes(base64.b64decode(facade['texture_base64'].split(',')[-1]))
    summary['library'] = json.loads(get('/furniture/custom-library'))
    for entry in summary['library'].values():
        assert get(entry['glb'])[:4] == b'glTF', 'Imported library GLB is unreadable'
    if not args.skip_furniture:
        from mlflow import MlflowClient
        import trimesh
        client = MlflowClient(tracking_uri=args.tracking_uri)
        image = get('/samples/furniture/chair.png')
        since = int(time.time() * 1000)
        request = {'image_base64': base64.b64encode(image).decode(), 'name': 'Docker validation chair',
                   'category': 'chair', 'target_height_m': 0.85, 'remove_bg': True}
        started = time.monotonic()
        result = post('/furniture/generate-3d', json.dumps(request).encode())
        summary['furniture_seconds'] = time.monotonic() - started
        glb = base64.b64decode(result['model_glb_base64'].split(',')[-1])
        (output / 'chair.glb').write_bytes(glb)
        scene = trimesh.load(io.BytesIO(glb), file_type='glb', force='scene')
        assert scene.geometry and sum(len(mesh.faces) for mesh in scene.geometry.values()) > 0
        assert result.get('quality', {}).get('checkpoint_loading') == 'strict'
        summary['furniture'] = {k: v for k, v in result.items() if 'base64' not in k}
        experiment = client.get_experiment_by_name('furniture-reconstruction')
        runs = client.search_runs([experiment.experiment_id], filter_string=f'attributes.start_time >= {since}')
        assert len(runs) == 1, f'Expected one reconstruction run, got {len(runs)}'
        run = runs[0]
        assert run.info.status == 'FINISHED', run.info.status
        assert run.data.params['device_type'] == 'cuda'
        assert run.data.params['preset'] == 'high'
        summary['mlflow'] = {'run_id': run.info.run_id, 'params': run.data.params, 'metrics': run.data.metrics}
        artifacts = []
        pending = ['']
        while pending:
            for item in client.list_artifacts(run.info.run_id, pending.pop()):
                if item.is_dir:
                    pending.append(item.path)
                else:
                    artifacts.append(item.path)
        model = next(path for path in artifacts if path.endswith('.glb'))
        downloaded = Path(client.download_artifacts(run.info.run_id, model, str(output)))
        assert hashlib.sha256(downloaded.read_bytes()).digest() == hashlib.sha256(glb).digest()
        summary['mlflow']['artifacts'] = artifacts
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ('library', 'metadata')}, indent=2))


if __name__ == '__main__':
    main()
