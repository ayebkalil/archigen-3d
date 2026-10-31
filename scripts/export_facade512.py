"""Export experimental inference weights and verify ONNX parity; no production overwrite."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import onnxruntime as ort
from src.models.facade_experiment import load_generator


def export(checkpoint, output):
    torch.set_num_threads(4)
    model, state = load_generator(checkpoint)
    model.gradient_checkpointing = False
    dest = Path(output); dest.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(123)
    sketch = torch.rand(1, 3, 512, 512) * 2 - 1
    style = torch.tensor([4])
    torch.onnx.export(model, (sketch, style), str(dest), opset_version=17,
                      input_names=['sketch', 'style_id'], output_names=['photo'],
                      dynamic_axes={'sketch': {0: 'batch'}, 'style_id': {0: 'batch'}, 'photo': {0: 'batch'}})
    session = ort.InferenceSession(str(dest), providers=['CPUExecutionProvider'])
    errors = []
    for style_id in (0, 3, 4):
        style.fill_(style_id)
        with torch.no_grad(): expected = model(sketch, style).numpy()
        actual = session.run(['photo'], {'sketch': sketch.numpy(), 'style_id': style.numpy()})[0]
        error = float(np.max(np.abs(actual - expected))); errors.append(error)
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
    metadata = {k: v for k, v in state.items() if k != 'generator'}
    metadata['onnx_max_abs_error'] = max(errors)
    dest.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'onnx': str(dest), 'max_abs_error': max(errors)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', default='outputs/phase2/training/best.pth')
    parser.add_argument('--output', default='models/saved/facade512_experiment/generator.onnx')
    args = parser.parse_args(); export(args.checkpoint, args.output)
