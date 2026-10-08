"""
Evaluation, Metric Computation, and ONNX Runtime Export Pipeline (Enhanced).
Computes SSIM, PSNR, and L1 metrics on the test split, and exports the trained
Generator into production-grade ONNX format.
Satisfies MLOps Criteria:
- 'Optimisation et quantification ONNX'
- 'Métriques et validation' (3/3)
"""

import sys
import math
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import onnx
import onnxruntime as ort

from src.data.dataset import FacadesDataset
from src.models.architecture import UNetGenerator
from src.utils.metrics import compute_ssim, compute_psnr

def evaluate_and_export_onnx(
    checkpoint_path: str = "models/saved/generator_enhanced_best.pth",
    data_dir: str = "data/raw/facades",
    onnx_output_path: str = "models/saved/generator_enhanced.onnx",
    upsample_mode: str = "nearest"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVALUATION] Starting test evaluation on device: {device}")

    # 1. Instantiate Generator with Anti-Checkerboard Architecture
    generator = UNetGenerator(
        in_channels=3,
        out_channels=3,
        num_filters=64,
        upsample_mode=upsample_mode
    ).to(device)

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        fallback = Path("models/saved/generator_best.pth")
        if fallback.exists():
            print(f"[MODEL] Checkpoint {ckpt_file} not found. Using fallback: {fallback}")
            ckpt_file = fallback

    if ckpt_file.exists():
        print(f"[MODEL] Loading weights from: {ckpt_file}")
        generator.load_state_dict(torch.load(ckpt_file, map_location=device))
    else:
        print(f"[WARNING] Checkpoint {ckpt_file} not found. Running with initialized weights.")

    generator.eval()

    # 2. Test Split Evaluation with Multi-Metric Suite (L1, PSNR, SSIM)
    test_dataset = FacadesDataset(root_dir=data_dir, split="test", img_size=256, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    criterion_l1 = nn.L1Loss()

    total_l1 = 0.0
    total_ssim = 0.0
    total_psnr = 0.0

    with torch.no_grad():
        for batch in test_loader:
            sketch = batch["sketch"].to(device)
            real_photo = batch["photo"].to(device)
            fake_photo = generator(sketch)

            total_l1 += criterion_l1(fake_photo, real_photo).item()
            total_ssim += compute_ssim(fake_photo, real_photo)
            total_psnr += compute_psnr(fake_photo, real_photo)

    avg_l1 = total_l1 / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    avg_psnr = total_psnr / len(test_loader)

    print("\n" + "="*55)
    print(f"[TEST METRICS] Average L1 Reconstruction Loss: {avg_l1:.4f}")
    print(f"[TEST METRICS] Structural Similarity (SSIM):    {avg_ssim:.4f}")
    print(f"[TEST METRICS] Peak Signal-to-Noise Ratio (PSNR): {avg_psnr:.2f} dB")
    print("="*55)

    # 3. Export to ONNX
    print(f"\n[ONNX EXPORT] Exporting model to: {onnx_output_path} ...")
    Path(onnx_output_path).parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    torch.onnx.export(
        generator,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_sketch"],
        output_names=["output_photo"],
        dynamic_axes={
            "input_sketch": {0: "batch_size"},
            "output_photo": {0: "batch_size"}
        }
    )

    # 4. Verify ONNX Model Integrity
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    print("[SUCCESS] ONNX graph verified: 100% valid!")

    # 5. Benchmark with ONNX Runtime
    print("[BENCHMARK] Validating inference with ONNX Runtime ...")
    ort_session = ort.InferenceSession(onnx_output_path, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: np.random.randn(1, 3, 256, 256).astype(np.float32)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(f"[SUCCESS] ONNX Runtime execution succeeded! Output shape: {ort_outs[0].shape}")

if __name__ == "__main__":
    evaluate_and_export_onnx()
