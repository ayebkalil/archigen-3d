"""Build side-by-side *actual viewer captures* and experiment curves, without retouching."""
import json
from pathlib import Path

from PIL import Image, ImageDraw


def report():
    root = Path('outputs/phase2')
    baseline = Image.open('outputs/phase1/facade-neural.png').convert('RGB')
    enhanced = Image.open(root/'viewer/facade-neural.png').convert('RGB')
    labels = ['Baseline 256 - Phase 1 viewer capture', 'Experimental 512 - CMP neutral (styles untrained)']
    images = [baseline, enhanced]
    full = Image.new('RGB', (1920, 704), 'white')
    draw = ImageDraw.Draw(full)
    for i, (im, label) in enumerate(zip(images, labels)):
        full.paste(im.resize((960, 660), Image.Resampling.LANCZOS), (i*960, 44))
        draw.text((i*960+16, 16), label, fill='black')
    full.save(root/'comparison.png')
    # The same pixel rectangle in identical 1600x1100 facade camera screenshots.
    if baseline.size == enhanced.size == (1600, 1100):
        rectangle = (530, 195, 1395, 775)
        detail = Image.new('RGB', (1730, 624), 'white')
        draw = ImageDraw.Draw(detail)
        for i, (im, label) in enumerate(zip(images, labels)):
            detail.paste(im.crop(rectangle), (i*865, 44))
            draw.text((i*865+12, 16), label, fill='black')
        detail.save(root/'comparison_detail.png')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for folder, label in [('training', 'Frozen / VGG=1'), ('training_vgg10', 'Frozen / VGG=10'),
                          ('training_decoder', 'Late decoder tuning')]:
        file = root/folder/'history.jsonl'
        if not file.exists(): continue
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        epochs = [r['epoch'] for r in rows]
        axes[0].plot(epochs, [r['val_vgg'] for r in rows], marker='o', label=label)
        axes[1].plot(epochs, [r['train_losses_g_d_l1_vgg_edges'][4] for r in rows], marker='o', label=label)
        axes[2].plot(epochs, [r['peak_device_sampled_mib'] for r in rows], marker='o', label=label)
    for ax, title, ylabel in zip(axes, ['Validation perceptual loss', 'Training opening-edge loss', 'Sampled device memory peak'],
                                ['VGG-19 loss (lower better)', 'Masked edge MAE', 'MiB']):
        ax.set_title(title); ax.set_xlabel('Epoch within each controlled run'); ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
    axes[2].axhline(3500, color='red', linestyle='--', label='Budget')
    axes[0].legend(fontsize=8)
    fig.suptitle('ArchiGen 512: validation-selected runs; later runs initialize from previous best checkpoints')
    fig.tight_layout(); fig.savefig(root/'training_curves.png', dpi=160); plt.close(fig)
    print(root/'comparison.png')


if __name__ == '__main__': report()
