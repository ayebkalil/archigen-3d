"""Lay out unaltered Three.js screenshots for visual comparison."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 26)
        small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
    except OSError:
        font = ImageFont.load_default(size=26)
        small = ImageFont.load_default(size=20)
    comparison = Image.new('RGB', (1200, 690), '#e8e8e5')
    draw = ImageDraw.Draw(comparison)
    for x, directory, label in [(0, args.baseline, 'Before: projected front photograph'),
                                 (600, args.candidate, 'After: neural surface atlas')]:
        draw.text((x+20, 18), label, font=font, fill='#202020')
        comparison.paste(Image.open(directory/'three-quarter.png').convert('RGB'), (x, 60))
    draw.text((20, 663), 'Same input, camera and lighting. Hidden geometry is inferred; height assumed 0.85 m.', font=small, fill='#333333')
    comparison.save(args.output/'comparison.png')
    sheet = Image.new('RGB', (2400, 1290), '#e8e8e5')
    draw = ImageDraw.Draw(sheet)
    for y, directory, label in [(0, args.baseline, 'Before'), (645, args.candidate, 'After')]:
        for x, view in enumerate(['front', 'three-quarter', 'side', 'back']):
            draw.text((x*600+20, y+10), f'{label} / {view}', font=font, fill='#202020')
            sheet.paste(Image.open(directory/f'{view}.png').convert('RGB'), (x*600, y+45))
    sheet.save(args.output/'multiview.png')


if __name__ == '__main__':
    main()
