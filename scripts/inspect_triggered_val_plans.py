"""
Inspect the 16 validation plans that trigger fallback routing.
Compares ground truth, Champion prediction, Blend prediction, and adoption outcome.
"""

import sys
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset
from src.floorplan.models import MultiTaskFloorplanUNet
from src.floorplan.vectorizer import FloorplanVectorizer

def count_gt_rooms(gt_room_mask):
    total = 0
    for c in range(1, 12):
        c_bin = (gt_room_mask == c).astype(np.uint8)
        num_c, _ = cv2.connectedComponents(c_bin)
        total += max(0, num_c - 1)
    return total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INSPECT] Running on {device}...")

    ds = CubiCasaDataset(
        root_dir=str(PROJECT_ROOT / "data/CubiCasa5k/cubicasa5k/cubicasa5k"),
        split="val",
        img_size=512,
        augment=False,
        cache_dir=str(PROJECT_ROOT / "data/CubiCasa5k/cached_512")
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    champ_p = PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth"
    blend_p = PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_blend_alpha_0_3.pth"

    def load_m(p):
        ckpt = torch.load(p, map_location=device, weights_only=False)
        m = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
        m.load_state_dict(ckpt.get("model_state_dict", ckpt))
        m.eval()
        return m

    m_c = load_m(champ_p)
    m_b = load_m(blend_p)
    vec = FloorplanVectorizer(min_area_habitable_m2=2.0)

    triggered = []
    idx = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            gt_r = batch["room_mask"].numpy()
            sample_ids = batch["sample_id"]

            out_c = m_c(imgs)
            out_b = m_b(imgs)

            pr_s_c = torch.argmax(out_c["struct"], dim=1).cpu().numpy().astype(np.uint8)
            pr_r_c = torch.argmax(out_c["room"], dim=1).cpu().numpy().astype(np.uint8)

            pr_s_b = torch.argmax(out_b["struct"], dim=1).cpu().numpy().astype(np.uint8)
            pr_r_b = torch.argmax(out_b["room"], dim=1).cpu().numpy().astype(np.uint8)

            for i in range(len(imgs)):
                m4_c = np.zeros_like(pr_s_c[i])
                m4_c[(pr_s_c[i] == 1) | (pr_s_c[i] == 2)] = 1
                m4_c[pr_s_c[i] == 3] = 2
                m4_c[pr_s_c[i] == 4] = 3
                lay_c = vec.vectorize(m4_c, room_mask=pr_r_c[i], struct_mask=pr_s_c[i])

                n_c = len(lay_c.get("rooms", []))
                area_c = lay_c.get("metadata", {}).get("total_surface_m2", 0.0)
                valid_c = lay_c.get("metadata", {}).get("is_building_shell_valid", False)

                is_trig = (n_c == 0) or (n_c < 3 and area_c < 25.0)

                if is_trig:
                    m4_b = np.zeros_like(pr_s_b[i])
                    m4_b[(pr_s_b[i] == 1) | (pr_s_b[i] == 2)] = 1
                    m4_b[pr_s_b[i] == 3] = 2
                    m4_b[pr_s_b[i] == 4] = 3
                    lay_b = vec.vectorize(m4_b, room_mask=pr_r_b[i], struct_mask=pr_s_b[i])

                    n_b = len(lay_b.get("rooms", []))
                    area_b = lay_b.get("metadata", {}).get("total_surface_m2", 0.0)
                    valid_b = lay_b.get("metadata", {}).get("is_building_shell_valid", False)

                    gt_rooms = count_gt_rooms(gt_r[i])

                    reason = "zero_rooms" if n_c == 0 else "under_3_rooms_and_area_under_25m2"
                    adopted = (n_b > n_c)

                    triggered.append({
                        "index": idx,
                        "sample_id": sample_ids[i],
                        "reason": reason,
                        "gt_rooms": gt_rooms,
                        "champ_rooms": n_c,
                        "champ_area_m2": round(area_c, 1),
                        "champ_valid": valid_c,
                        "blend_rooms": n_b,
                        "blend_area_m2": round(area_b, 1),
                        "blend_valid": valid_b,
                        "adopted": adopted
                    })

                idx += 1

    print("\n" + "=" * 115)
    print(f"ANALYSIS OF THE {len(triggered)} TRIGGERED VALIDATION PLANS (OUT OF {idx} TOTAL PLANS)")
    print("=" * 115)
    print(f"{'Idx':<4} | {'Sample Path':<32} | {'GT':<3} | {'Champion (Rooms, Area)':<24} | {'Blend (Rooms, Area)':<24} | {'Adopted?'}")
    print("-" * 115)
    for t in triggered:
        c_str = f"{t['champ_rooms']} rms ({t['champ_area_m2']}m2, {'V' if t['champ_valid'] else 'X'})"
        b_str = f"{t['blend_rooms']} rms ({t['blend_area_m2']}m2, {'V' if t['blend_valid'] else 'X'})"
        ad_str = "YES (Blend)" if t["adopted"] else "NO (Kept Champ)"
        print(f"{t['index']:<4} | {t['sample_id']:<32} | {t['gt_rooms']:<3} | {c_str:<24} | {b_str:<24} | {ad_str}")
    print("=" * 115)

    out_json = PROJECT_ROOT / "outputs" / "triggered_val_plans_analysis.json"
    with open(out_json, "w") as f:
        json.dump(triggered, f, indent=2)
    print(f"\n[INSPECT] Results saved to {out_json}")

if __name__ == "__main__":
    main()
