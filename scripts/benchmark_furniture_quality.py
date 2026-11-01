"""Reconstruct a local image and save inspectable GLB + provenance; never save to user catalog."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--image', type=Path, default=ROOT/'TripoSR/examples/chair.png')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['baseline', 'balanced', 'high'], default='balanced')
    parser.add_argument('--remove-bg', action='store_true')
    parser.add_argument('--category', default='chair')
    parser.add_argument('--height', type=float, default=.85)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == 'baseline':
        os.environ.pop('ARCHIGEN_FURNITURE_QUALITY', None)
        from src.furniture.image_to_3d import ImageTo3DGenerator
        engine = ImageTo3DGenerator()
    else:
        os.environ['ARCHIGEN_FURNITURE_DETAIL'] = args.mode
        from src.furniture.quality_engine import QualityFurnitureEngine
        engine = QualityFurnitureEngine()
    with Image.open(args.image) as image:
        result = engine.generate_from_image(image, category=args.category, remove_bg=args.remove_bg, target_height_m=args.height)
    (args.output/'model.glb').write_bytes(result.pop('glb_bytes'))
    result.pop('glb_base64'); result.pop('preview_image_base64')
    result['input_sha256'] = hashlib.sha256(args.image.read_bytes()).hexdigest()
    with (ROOT/'TripoSR/model.ckpt').open('rb') as stream:
        result['checkpoint_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (args.output/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
