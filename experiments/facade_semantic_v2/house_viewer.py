"""Stage a reversible preview using the production Three.js viewer unchanged."""
import argparse
import json
from pathlib import Path
import shutil

from experiments.facade_sd15_controlnet.data import write_json


def stage(project, output):
    output = Path(output).resolve()
    target = Path(project).resolve() / 'frontend/tmp/facade-multiface-v2'
    target.mkdir(parents=True, exist_ok=True)
    for filename in ('house.json', 'panels.json', 'profile.json', 'plain.png'):
        shutil.copyfile(output / filename, target / filename)
    panels = json.loads((output / 'panels.json').read_text())
    for panel in panels:
        folder = target / panel['wall_id']; folder.mkdir(exist_ok=True)
        for filename in ('sketch.png', 'baseline.png', 'baseline.json', 'candidate.png', 'candidate.json'):
            shutil.copyfile(output / panel['wall_id'] / filename, folder / filename)
    baselines = {panel['wall_id']: json.loads((output / panel['wall_id'] / 'baseline.json').read_text()) for panel in panels}
    write_json(target / 'baselines.json', baselines)
    source = Path(__file__).with_name('house_viewer.jsx')
    shutil.copyfile(source, target / 'app.jsx')
    (target / 'index.html').write_text('''<!doctype html><html><head><meta charset="utf-8"><title>ArchiGen · V2 facade house test</title>
<meta name="viewport" content="width=device-width,initial-scale=1"></head><body><div id="root"></div>
<script type="module" src="./app.jsx"></script></body></html>''', encoding='utf-8')
    print('Preview: http://127.0.0.1:3001/tmp/facade-multiface-v2/', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--project', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); stage(args.project, args.output)
