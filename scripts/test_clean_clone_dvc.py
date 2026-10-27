import shutil
import subprocess
import sys
import hashlib
from pathlib import Path

import shutil
import subprocess
import sys
import hashlib
import tempfile
from pathlib import Path

TEMP_DIR = Path(tempfile.mkdtemp(prefix="archigen_dvc_test_"))
SOURCE_REPO = Path("C:/Users/ayebk/OneDrive/Desktop/deep/archigen-3d")

try:
    print(f"[1/6] Cloning repo from {SOURCE_REPO} to {TEMP_DIR}...")
    subprocess.run(["git", "clone", str(SOURCE_REPO), str(TEMP_DIR)], check=True)

    print("[2/6] Checking out feature/room-design...")
    subprocess.run(["git", "checkout", "feature/room-design"], cwd=str(TEMP_DIR), check=True)

    mt_onnx = TEMP_DIR / "models/saved/floorplan_unetpp/floorplan_multitask.onnx"
    fb_onnx = TEMP_DIR / "models/saved/floorplan_unetpp/floorplan_fallback.onnx"

    print(f"[3/6] Verifying models are absent before dvc pull:")
    print(f"  floorplan_multitask.onnx exists: {mt_onnx.exists()}")
    print(f"  floorplan_fallback.onnx exists:  {fb_onnx.exists()}")
    assert not mt_onnx.exists(), "ONNX champion should NOT exist in fresh git clone before dvc pull!"

    print("[4/6] Executing dvc pull for model artifacts in clean clone...")
    pull_targets = [
        "models/saved/floorplan_unetpp/floorplan_multitask.onnx.dvc",
        "models/saved/floorplan_unetpp/floorplan_fallback.onnx.dvc",
        "models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth.dvc",
        "models/saved/floorplan_unetpp/multitask_resnet34_blend_alpha_0_3.pth.dvc"
    ]
    pull_res = subprocess.run(["dvc", "pull"] + pull_targets, cwd=str(TEMP_DIR), capture_output=True, text=True)
    print(pull_res.stdout)
    if pull_res.stderr:
        print(pull_res.stderr)

    print(f"[5/6] Verifying restored models exist after dvc pull:")
    print(f"  floorplan_multitask.onnx exists: {mt_onnx.exists()}")
    print(f"  floorplan_fallback.onnx exists:  {fb_onnx.exists()}")
    assert mt_onnx.exists(), "floorplan_multitask.onnx was NOT restored by dvc pull!"
    assert fb_onnx.exists(), "floorplan_fallback.onnx was NOT restored by dvc pull!"

    def sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    h1 = sha256(mt_onnx)
    h2 = sha256(fb_onnx)
    print(f"  floorplan_multitask.onnx SHA256: {h1}")
    print(f"  floorplan_fallback.onnx  SHA256: {h2}")

    exp1 = "538ade1935a9624e19047c910ed595f0e9a62aa08d872715216d3b323dfa034c"
    exp2 = "7188af6fc15f11a8fc1979a02656e2bb32d200e7d2d7bb8b8d07b43296fff809"
    assert h1 == exp1, f"SHA mismatch on restored mt_onnx: {h1} != {exp1}"
    assert h2 == exp2, f"SHA mismatch on restored fb_onnx: {h2} != {exp2}"
    print("[SUCCESS] Hashes match expected champion and fallback values perfectly!")

    print("[6/6] Running pytest tests/test_model_registry.py in clean clone...")
    py_res = subprocess.run(["pytest", "tests/test_model_registry.py", "-v"], cwd=str(TEMP_DIR), check=True)

    print("\n[COMPLETE] Clean clone DVC pull restore test PASSED 100%!")
finally:
    subprocess.run(f'cmd /c rmdir /s /q "{TEMP_DIR}"', shell=True)
    print("Cleaned up temporary clone directory.")
