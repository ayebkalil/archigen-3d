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
    parser.add_argument('--profile', choices=['runtime', 'facade512'], default='runtime')
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
        result = subprocess.run([sys.executable, 'scripts/models.py', 'check', '--profile', args.profile], cwd=clone)
        if result.returncode == 0:
            raise AssertionError('Weights unexpectedly present before restoring')
        subprocess.run([sys.executable, 'scripts/models.py', 'pull', '--profile', args.profile], cwd=clone, check=True)
        subprocess.run([sys.executable, 'scripts/models.py', 'check', '--profile', args.profile, '--json'], cwd=clone, check=True)
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
        if args.profile == 'facade512':
            smoke = """
import io
import numpy as np
from PIL import Image, ImageDraw
from src.bridge.facade_experimental import ExperimentalFacadeInference
from src.data.facade_palette import EXPERIMENT_CMP_PALETTE
engine = ExperimentalFacadeInference('models/saved/facade512_experiment/generator.onnx')
image = Image.new('RGB', (512,512), EXPERIMENT_CMP_PALETTE['facade'])
draw = ImageDraw.Draw(image)
draw.rectangle((80,120,160,310), fill=EXPERIMENT_CMP_PALETTE['window'])
draw.rectangle((300,110,390,511), fill=EXPERIMENT_CMP_PALETTE['door'])
array = np.asarray(image,np.float32).transpose(2,0,1)[None] / 127.5 - 1
output = engine.session.run(['photo'], {'sketch':array, 'style_id':np.array([4],np.int64)})[0]
assert output.shape == (1,3,512,512) and np.isfinite(output).all()
buffer = io.BytesIO(); image.save(buffer,format='PNG')
photo = engine.generate(buffer.getvalue(),style=None)
assert Image.open(io.BytesIO(photo)).size == (512,512)
assert photo != buffer.getvalue()
assert engine.describe('Traditional Brick')['style_conditioned'] is False
print('Experimental ONNX + verified JSON generated a real 512 PNG; styles remain unlabelled')
"""
        subprocess.run([sys.executable, '-c', smoke], cwd=clone, check=True)
        status = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=clone,
                                capture_output=True, text=True, check=True)
        if status.stdout.strip():
            raise AssertionError('Restoring models changed tracked Git metadata: ' + status.stdout)
        print('PASS: clean clone, empty cache, public restore, SHA-256 checks, real ONNX inference and unchanged tracked metadata; profile=' + args.profile)
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
