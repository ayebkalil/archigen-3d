"""Summarize actual runs and compare identical-camera viewer captures without retouching."""
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'outputs/phase3'
    candidate = output / 'composite_brick'
    baseline = Image.open(ROOT / 'outputs/phase2/viewer/facade-neural.png').convert('RGB')
    enhanced = Image.open(candidate / 'viewer/facade-neural.png').convert('RGB')
    if baseline.size != (1600, 1100) or enhanced.size != baseline.size:
        raise ValueError('Expected identical-camera 1600 x 1100 viewer screenshots')
    canvas = Image.new('RGB', (1730, 624), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate([
        (baseline, 'Previous 512 GAN - CMP neutral'),
        (enhanced, '1024 hybrid experiment - neural brick material + fixed opening masks'),
    ]):
        canvas.paste(im.crop((530, 195, 1395, 775)), (865*i, 44))
        draw.text((865*i+12, 16), label, fill='black')
    canvas.save(candidate / 'comparison_detail.png')
    fields = ('status', 'task', 'seed', 'style', 'style_status', 'prompt', 'negative_prompt',
              'prompt_token_lengths', 'prompt_truncation_observed', 'positive_tokens_before_truncation',
              'resolution', 'steps', 'control_scale', 'guidance_scale', 'offload', 'quality_review',
              'elapsed_seconds', 'sampled_peak_device_mib', 'peak_allocated_mib', 'peak_reserved_mib',
              'output_sha256', 'control_image_sha256', 'packages')
    runs = []
    for file in sorted(output.rglob('run.json')):
        data = json.loads(file.read_text())
        record = {'artifact': file.parent.relative_to(ROOT).as_posix(),
                  **{key: data[key] for key in fields if key in data}}
        record['case'] = {key: data['case'][key] for key in ('id', 'cmp_id', 'split', 'control_sha256')
                          if data.get('case') and key in data['case']}
        runs.append(record)
    report = {
        'branch': 'experiment/facade-diffusion-control',
        'comparison': 'Qualitative comparison of different materials; not a paired perceptual benchmark',
        'models': json.loads((ROOT / 'configs/facade_diffusion.json').read_text()),
        'runs': runs,
        'composition': json.loads((candidate / 'composition.json').read_text()),
        'viewer': json.loads((candidate / 'viewer/verification.json').read_text()),
        'scope': 'One CMP validation pilot run; no new held-out benchmark or four-style training claim',
    }
    (output / 'experiment_results.json').write_text(json.dumps(report, indent=2))
    print(candidate / 'comparison_detail.png')


if __name__ == '__main__':
    main()
