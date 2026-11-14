"""Verify a real clean clone, without this PC's DVC cache or local configuration."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--repo', default='https://github.com/ayebkalil/archigen-3d.git')
    parser.add_argument('--ref', default='main')
    parser.add_argument('--keep', action='store_true')
    args = parser.parse_args()
    temporary_root = (ROOT / 'tmp').resolve()
    temporary_root.mkdir(exist_ok=True)
    clone = Path(tempfile.mkdtemp(prefix='team-install-', dir=temporary_root)).resolve()
    if not clone.is_relative_to(temporary_root):
        raise ValueError('Temporary clone escaped its intended directory')
    try:
        subprocess.run(['git', 'clone', '--no-local', '--depth', '1', '--branch', args.ref, args.repo, str(clone)], check=True)
        if (clone / '.dvc/config.local').exists():
            raise AssertionError('Clean clone unexpectedly inherited local DVC settings')
        result = subprocess.run([sys.executable, 'scripts/models.py', 'check'], cwd=clone)
        if result.returncode == 0:
            raise AssertionError('Weights unexpectedly present before restoring')
        subprocess.run([sys.executable, 'scripts/models.py', 'pull'], cwd=clone, check=True)
        subprocess.run([sys.executable, 'scripts/models.py', 'check', '--json'], cwd=clone, check=True)
        # Exercise all actual ONNX graphs after restoring them over the network.
        smoke = """
import io
import numpy as np
import onnxruntime as ort
from PIL import Image
for name in ['floorplan_multitask', 'floorplan_fallback']:
    model = ort.InferenceSession('models/saved/floorplan_unetpp/' + name + '.onnx', providers=['CPUExecutionProvider'])
    output = model.run(None, {model.get_inputs()[0].name: np.zeros((1,3,512,512),np.float32)})
    assert sorted(value.shape[1] for value in output) == [5,12]
    assert all(np.isfinite(value).all() for value in output)
    print(name, [value.shape for value in output])
from backend.app.inference import ArchiGenONNXInference
buffer = io.BytesIO(); Image.new('RGB',(256,256),(0,48,255)).save(buffer,format='PNG')
image = ArchiGenONNXInference().generate(buffer.getvalue())
assert Image.open(io.BytesIO(image)).size == (256,256)
print('Production facade graph generated a real PNG')
"""
        subprocess.run([sys.executable, '-c', smoke], cwd=clone, check=True)
        print('PASS: clean clone, empty cache, public DVC restore, SHA-256 checks and real ONNX inference')
    finally:
        if args.keep:
            print('Kept clean clone at', clone)
        else:
            # Python handles read-only Git files; resolved containment was checked above.
            import shutil
            import os
            import stat
            def writable(function, path, error):
                os.chmod(path, stat.S_IWRITE); function(path)
            shutil.rmtree(clone, onerror=writable)


if __name__ == '__main__':
    main()
