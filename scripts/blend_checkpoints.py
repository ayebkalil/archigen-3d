"""
Weight Blending for Floorplan UNet++ Models
Interpolates weights between Champion (v1.1) and Challenger (Style-Adapted):
    theta_blend = (1 - alpha) * theta_champ + alpha * theta_chal
Generates checkpoints for alpha in [0.3, 0.5, 0.7].
"""

import sys
from pathlib import Path
import copy
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def blend_models(champ_path: Path, chal_path: Path, alphas=[0.3, 0.5, 0.7], output_dir: Path = None):
    if output_dir is None:
        output_dir = champ_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[BLEND] Loading Champion: {champ_path.name}")
    ckpt_champ = torch.load(champ_path, map_location="cpu", weights_only=False)
    sd_champ = ckpt_champ.get("model_state_dict", ckpt_champ)

    print(f"[BLEND] Loading Challenger: {chal_path.name}")
    ckpt_chal = torch.load(chal_path, map_location="cpu", weights_only=False)
    sd_chal = ckpt_chal.get("model_state_dict", ckpt_chal)

    # Key validation
    keys_champ = set(sd_champ.keys())
    keys_chal = set(sd_chal.keys())
    assert keys_champ == keys_chal, f"Key mismatch! Diff: {keys_champ ^ keys_chal}"
    print(f"[BLEND] Key verification passed: {len(keys_champ)} identical parameters.")

    blended_paths = {}

    for alpha in alphas:
        alpha_str = str(alpha).replace(".", "_")
        out_name = f"multitask_resnet34_blend_alpha_{alpha_str}.pth"
        out_path = output_dir / out_name

        print(f"\n[BLEND] Computing blended weights for alpha = {alpha} (Champ: {1 - alpha:.1f}, Chal: {alpha:.1f})...")
        blended_sd = {}
        for k in sd_champ.keys():
            t_champ = sd_champ[k]
            t_chal = sd_chal[k]

            if torch.is_floating_point(t_champ):
                blended_sd[k] = (1.0 - alpha) * t_champ.float() + alpha * t_chal.float()
                # Maintain original dtype
                if blended_sd[k].dtype != t_champ.dtype:
                    blended_sd[k] = blended_sd[k].to(t_champ.dtype)
            else:
                # Integer tensors like num_batches_tracked
                blended_sd[k] = copy.deepcopy(t_champ)

        save_dict = {
            "model_state_dict": blended_sd,
            "blend_meta": {
                "alpha": alpha,
                "champ_source": champ_path.name,
                "chal_source": chal_path.name,
                "description": f"Linear weight interpolation: {1-alpha:.1f}*v1.1 + {alpha:.1f}*challenger"
            }
        }
        torch.save(save_dict, out_path)
        print(f"[BLEND] Saved {out_name} ({out_path.stat().st_size / (1024*1024):.2f} MB)")
        blended_paths[alpha] = out_path

    return blended_paths

if __name__ == "__main__":
    champ_p = PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth"
    chal_p = PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_restyled_challenger.pth"
    blend_models(champ_p, chal_p, alphas=[0.3, 0.5, 0.7])
