"""Assemble a reviewed neural material with known facade opening regions."""
import argparse
import base64
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image
from src.bridge.facade_materials import compose_material_facade
from src.models.facade_diffusion import sha256


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--material', type=Path, required=True)
    parser.add_argument('--opening-source', type=Path, required=True)
    parser.add_argument('--style', required=True)
    parser.add_argument('--tile-meters', type=float, default=2.0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    with Image.open(args.material) as material, Image.open(args.opening_source) as source:
        texture, mask = compose_material_facade(fixture, material, source, args.tile_meters)
    args.output.mkdir(parents=True, exist_ok=False)
    texture.save(args.output / 'facade.png')
    mask.save(args.output / 'opening_mask.png')
    fixture['texture_base64'] = 'data:image/png;base64,' + base64.b64encode((args.output / 'facade.png').read_bytes()).decode()
    # Diagnostic semantics remain the original layout, at the new texture resolution.
    sketch = Image.open(io.BytesIO(base64.b64decode(fixture['sketch_base64'].split(',', 1)[1])))
    buffer = io.BytesIO()
    sketch.resize(texture.size, Image.Resampling.NEAREST).save(buffer, format='PNG')
    fixture['sketch_base64'] = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
    fixture['style'] = args.style
    fixture['device'] = 'Neural material + geometry-locked composition (experimental)'
    fixture['model_info'] = {
        'model': 'sdxl_material_composite_experiment', 'resolution': texture.width,
        'requested_style': args.style, 'effective_style': args.style + ' (neural material / fixed geometry)',
        'style_conditioned': True, 'supported_styles': [], 'cmp_style_training': False,
        'assembly': 'mirrored material tiling plus exact opening masks; not end-to-end prediction',
    }
    (args.output / 'viewer_fixture.json').write_text(json.dumps(fixture))
    report = {
        'kind': 'hybrid_neural_material_composite', 'status': 'pending_visual_review',
        'style': args.style, 'tile_meters': args.tile_meters, 'opening_count': len(fixture['openings']),
        'resolution': list(texture.size), 'output_sha256': sha256(args.output / 'facade.png'),
        'material_sha256': sha256(args.material), 'opening_source_sha256': sha256(args.opening_source),
        'fixture_sha256': sha256(args.fixture),
        'limitations': ['Material swatch requires visual review for objects and tiling artifacts',
                        'Opening interiors remain synthesized, not metrically reconstructed',
                        'Mirrored tiling and rectangular masking are deterministic graphics operations',
                        'No learned normal/roughness maps or CMP style labels'],
    }
    (args.output / 'composition.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
