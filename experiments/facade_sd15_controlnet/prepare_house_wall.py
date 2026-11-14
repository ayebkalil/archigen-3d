"""Capture the existing sample wall and the running production facade baseline."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import httpx

from src.bridge.facade_bridge import FacadeBridge


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--api', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    project = args.project.resolve()
    output = project / 'outputs/facade_house_test'
    output.mkdir(parents=True, exist_ok=True)
    command = """import {createSampleHouse,SAMPLE_FRONT_WALL} from './frontend/src/geometry/sampleHousePlan.js';
import fs from 'node:fs'; const layout=createSampleHouse();
fs.writeFileSync('outputs/facade_house_test/house.json',JSON.stringify({layout,wall:layout.walls.find(w=>w.id===SAMPLE_FRONT_WALL)},null,2));"""
    subprocess.run(['node', '--input-type=module', '-e', command], cwd=project, check=True)
    case = json.loads((output / 'house.json').read_text(encoding='utf-8'))
    wall = {**case['wall'], 'geometry_fixed': True}
    with httpx.Client(timeout=90, trust_env=False) as client:
        response = client.post(args.api.rstrip('/') + '/facade/generate', json={'wall': wall, 'num_floors': 1, 'style': None})
        response.raise_for_status()
    baseline = response.json()
    (output / 'baseline.json').write_text(json.dumps(baseline), encoding='utf-8')
    (output / 'baseline.png').write_bytes(base64.b64decode(baseline['texture_base64'].split(',')[-1]))
    bridge = FacadeBridge.__new__(FacadeBridge)
    bridge.inference_engine = SimpleNamespace(resolution=512, neural_style=True)
    bridge.enhance_textures = False
    image, bounds, frame, openings = bridge._build_facade_sketch(wall, num_floors=1, is_front=True, target_size=512)
    image.save(output / 'sketch.png')
    probe = {'wall': wall, 'uv_bounds': bounds, 'facade_frame': frame, 'openings': openings,
             'wall_length_m': bounds['wall_length_m'], 'wall_height_m': bounds['wall_height_m'],
             'is_front': True, 'style': None, 'num_floors': 1, 'wall_id': wall['id']}
    (output / 'probe.json').write_text(json.dumps(probe, indent=2), encoding='utf-8')
    print(f'Prepared {wall["id"]}: {bounds["wall_length_m"]:.2f} × {bounds["wall_height_m"]:.2f} m; {len(openings)} openings')


if __name__ == '__main__':
    main()
