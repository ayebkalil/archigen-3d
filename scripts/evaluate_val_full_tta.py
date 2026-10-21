"""
Authoritative 400-Plan Validation Evaluation: Baseline vs 4-Fold TTA.
Evaluates the production checkpoint (multitask_resnet34_finetuned_best.pth) on the FULL
validation split (N=400 plans) with:
1. Exact pixel IoUs (Structure & Rooms strictly aligned with ROOM_CLASSES).
2. True SVG Instance-Level Metrics:
   - Ground Truth Room instances parsed directly from SVG <g class="Space ..."> elements.
   - Ground Truth Door instances parsed from SVG <g class="Door ..."> elements.
   - Instance matching (IoU >= 0.5 and identical class label) -> Precision, Recall, F1.
   - Room Count MAE & Mean Signed Error.
   - Building shell validity rate.
3. 1,000 Paired Bootstrap Resamples over plans:
   - 95% Confidence Intervals for Baseline and TTA.
   - 95% Confidence Interval for Delta (TTA - Baseline).
   - Empirical p-value testing H0: Delta <= 0.
4. Runtime and Latency profiling (per-stage ms).
"""

import sys
import time
import json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from shapely.geometry import Polygon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset, ROOM_CLASSES, STRUCTURE_CLASSES, parse_svg_points
from src.floorplan.models import MultiTaskFloorplanUNet
from src.floorplan.vectorizer import FloorplanVectorizer

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha.update(chunk)
    return sha.hexdigest()

def extract_svg_gt_instances(svg_path: Path, img_size: int = 512) -> Tuple[List[Dict[str, Any]], int]:
    """
    Parses individual architectural room spaces and door elements directly from model.svg.
    Preserves adjacent room separation that merges in flat raster masks.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Determine scale and padding from SVG viewbox or image size
    # In CubiCasaDataset, scale = min(512/iw, 512/ih)
    # We can read the viewBox or width/height attributes
    vb = root.attrib.get("viewBox", "")
    if vb:
        parts = [float(p) for p in vb.replace(",", " ").split()]
        iw, ih = parts[2], parts[3]
    else:
        iw = float(root.attrib.get("width", 512))
        ih = float(root.attrib.get("height", 512))

    scale = min(img_size / iw, img_size / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    pad_x = (img_size - nw) // 2
    pad_y = (img_size - nh) // 2

    # 1. Parse individual room instances
    room_instances = []
    for g in root.findall(".//{http://www.w3.org/2000/svg}g"):
        c = g.attrib.get("class", "")
        if c.startswith("Space"):
            matched_cls = None
            for r_name, r_id in ROOM_CLASSES.items():
                if r_name in c:
                    matched_cls = (r_name, r_id)
                    break
            if matched_cls is None:
                continue

            r_name, r_id = matched_cls
            poly_mask = np.zeros((img_size, img_size), dtype=np.uint8)
            poly_count = 0
            for poly in g.findall(".//{http://www.w3.org/2000/svg}polygon"):
                if "points" in poly.attrib:
                    pts = parse_svg_points(poly.attrib["points"], scale, pad_x, pad_y)
                    if len(pts) >= 3:
                        cv2.fillPoly(poly_mask, [pts], 1)
                        poly_count += 1
            if poly_count > 0 and poly_mask.sum() > 25:
                room_instances.append({
                    "class_name": r_name,
                    "class_id": r_id,
                    "mask": (poly_mask > 0)
                })

    # 2. Count door elements
    door_count = 0
    for g in root.findall(".//{http://www.w3.org/2000/svg}g"):
        c = g.attrib.get("class", "")
        if c.startswith("Door"):
            door_count += 1

    return room_instances, door_count

def match_room_instances(pred_rooms: List[Dict[str, Any]], gt_rooms: List[Dict[str, Any]], iou_thresh: float = 0.5):
    """
    Computes exact instance-level Precision, Recall, and F1 between predicted room polygons
    and ground-truth SVG Space instances (matching IoU >= 0.5 and same class label).
    """
    tp = 0
    matched_gt = set()

    for p in pred_rooms:
        p_mask = p.get("mask")
        p_cls = p.get("class_id")
        if p_mask is None:
            continue

        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, g in enumerate(gt_rooms):
            if g_idx in matched_gt:
                continue
            # Must match class
            if g["class_id"] != p_cls:
                continue

            g_mask = g["mask"]
            inter = np.logical_and(p_mask, g_mask).sum()
            union = np.logical_or(p_mask, g_mask).sum()
            iou = inter / max(union, 1)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_thresh and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(pred_rooms) - tp
    fn = len(gt_rooms) - len(matched_gt)
    return tp, fp, fn

def run_evaluation(ckpt_path: str, split: str = "val"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_file}")

    ckpt_hash = compute_sha256(ckpt_file)
    print("=" * 80)
    print(f"[FULL-400 EVALUATION] Evaluating on '{split}' split (Device: {device})")
    print(f"  Checkpoint: {ckpt_file.name}")
    print(f"  SHA256:     {ckpt_hash}")
    print("=" * 80)

    dataset = CubiCasaDataset(split=split, img_size=512, augment=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    num_plans = len(dataset)

    model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    vectorizer = FloorplanVectorizer()

    # Storage for per-plan stats for Paired Bootstrap
    plans_base = []
    plans_tta = []

    # Latency timers
    t_base_infer = []
    t_tta_infer = []

    print(f"[*] Iterating over {num_plans} floorplans with Baseline & 4-Fold TTA...")

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            imgs = batch["image"].to(device)
            t_s = batch["struct_mask"].numpy()
            t_r = batch["room_mask"].numpy()
            B = imgs.size(0)

            # ----------------------------------------------------
            # 1. BASELINE SINGLE-PASS INFERENCE
            # ----------------------------------------------------
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out_base = model(imgs)
            prob_s_base = F.softmax(out_base["struct"], dim=1)
            prob_r_base = F.softmax(out_base["room"], dim=1)
            p_s_base = torch.argmax(prob_s_base, dim=1).cpu().numpy().astype(np.uint8)
            p_r_base = torch.argmax(prob_r_base, dim=1).cpu().numpy().astype(np.uint8)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_base_infer.append((time.perf_counter() - t0) / B)

            # ----------------------------------------------------
            # 2. 4-FOLD TEST-TIME AUGMENTATION (TTA)
            # ----------------------------------------------------
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            # A. Original
            prob_s_tta = prob_s_base.clone()
            prob_r_tta = prob_r_base.clone()

            # B. Horizontal Flip
            imgs_h = torch.flip(imgs, dims=[3])
            out_h = model(imgs_h)
            prob_s_tta += torch.flip(F.softmax(out_h["struct"], dim=1), dims=[3])
            prob_r_tta += torch.flip(F.softmax(out_h["room"], dim=1), dims=[3])

            # C. Vertical Flip
            imgs_v = torch.flip(imgs, dims=[2])
            out_v = model(imgs_v)
            prob_s_tta += torch.flip(F.softmax(out_v["struct"], dim=1), dims=[2])
            prob_r_tta += torch.flip(F.softmax(out_v["room"], dim=1), dims=[2])

            # D. 180 Rotation (H+V flip)
            imgs_hv = torch.flip(imgs, dims=[2, 3])
            out_hv = model(imgs_hv)
            prob_s_tta += torch.flip(F.softmax(out_hv["struct"], dim=1), dims=[2, 3])
            prob_r_tta += torch.flip(F.softmax(out_hv["room"], dim=1), dims=[2, 3])

            prob_s_tta /= 4.0
            prob_r_tta /= 4.0

            p_s_tta = torch.argmax(prob_s_tta, dim=1).cpu().numpy().astype(np.uint8)
            p_r_tta = torch.argmax(prob_r_tta, dim=1).cpu().numpy().astype(np.uint8)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_tta_infer.append((time.perf_counter() - t0) / B)

            # ----------------------------------------------------
            # 3. PER-PLAN COMPUTATION & VECTORIZATION
            # ----------------------------------------------------
            for b in range(B):
                plan_idx = b_idx * loader.batch_size + b
                rel_sample = dataset.valid_samples[plan_idx]
                svg_path = dataset.root_dir / rel_sample / "model.svg"

                gt_room_instances, gt_door_count = extract_svg_gt_instances(svg_path)
                n_gt_rooms = len(gt_room_instances)

                # Process BASELINE
                stats_b = compute_plan_record(
                    p_s_base[b], p_r_base[b], t_s[b], t_r[b],
                    vectorizer, gt_room_instances, gt_door_count, n_gt_rooms
                )
                plans_base.append(stats_b)

                # Process TTA
                stats_t = compute_plan_record(
                    p_s_tta[b], p_r_tta[b], t_s[b], t_r[b],
                    vectorizer, gt_room_instances, gt_door_count, n_gt_rooms
                )
                plans_tta.append(stats_t)

            if (b_idx + 1) % 20 == 0:
                print(f"  Processed {min((b_idx+1)*loader.batch_size, num_plans)}/{num_plans} floorplans...")

    # 4. Bootstrap Analysis
    run_comparative_bootstrap(plans_base, plans_tta, t_base_infer, t_tta_infer, ckpt_file.name, ckpt_hash)

def compute_plan_record(
    p_s: np.ndarray, p_r: np.ndarray, t_s: np.ndarray, t_r: np.ndarray,
    vectorizer: FloorplanVectorizer, gt_room_instances: List[Dict], gt_door_count: int, n_gt_rooms: int
) -> Dict[str, Any]:
    # 1. Structure Pixel Intersections / Unions
    p_mw = ((p_s == 1) | (p_s == 2))
    t_mw = ((t_s == 1) | (t_s == 2))
    mw_i = int(np.logical_and(p_mw, t_mw).sum())
    mw_u = int(np.logical_or(p_mw, t_mw).sum())

    s_i = np.zeros(5, dtype=np.int64)
    s_u = np.zeros(5, dtype=np.int64)
    for c in range(1, 5):
        s_i[c] = int(np.logical_and(p_s == c, t_s == c).sum())
        s_u[c] = int(np.logical_or(p_s == c, t_s == c).sum())

    # 2. Room Pixel Accuracy
    non_bg = (t_r > 0)
    r_corr = int(np.logical_and(p_r == t_r, non_bg).sum())
    r_tot = int(non_bg.sum())

    # 3. 10 Room Classes (strictly matching ROOM_CLASSES 1..10)
    r_i = np.zeros(12, dtype=np.int64)
    r_u = np.zeros(12, dtype=np.int64)
    r_gt = np.zeros(12, dtype=np.int64)
    for c in range(1, 11):
        r_i[c] = int(np.logical_and(p_r == c, t_r == c).sum())
        r_u[c] = int(np.logical_or(p_r == c, t_r == c).sum())
        r_gt[c] = int((t_r == c).sum())

    # 4. Downstream Vectorization
    mask_4c = np.zeros_like(p_s)
    mask_4c[(p_s == 1) | (p_s == 2)] = 1
    mask_4c[p_s == 3] = 2
    mask_4c[p_s == 4] = 3

    layout = vectorizer.vectorize(mask=mask_4c, struct_mask=p_s, room_mask=p_r)
    is_shell_valid = layout.get("metadata", {}).get("is_building_shell_valid", False)

    pred_rooms = layout.get("rooms", [])
    n_pred_rooms = len(pred_rooms)

    # Format pred rooms with binary masks for matching
    scale_m = layout.get("metadata", {}).get("scale_m_per_px", 0.03)
    formatted_pred = []
    for pr in pred_rooms:
        r_type = pr.get("room_type", "")
        c_id = ROOM_CLASSES.get(r_type, 0)
        poly_pts_m = pr.get("polygon_meters", [])
        if len(poly_pts_m) >= 3:
            poly_pts_px = [[int(round(pt[0] / scale_m)), int(round(pt[1] / scale_m))] for pt in poly_pts_m]
            r_mask = np.zeros((512, 512), dtype=np.uint8)
            cv2.fillPoly(r_mask, [np.array(poly_pts_px, dtype=np.int32)], 1)
            formatted_pred.append({
                "class_id": c_id,
                "mask": (r_mask > 0)
            })

    # Instance F1 matching
    tp, fp, fn = match_room_instances(formatted_pred, gt_room_instances, iou_thresh=0.5)

    # Count doors detected in layout
    pred_doors = layout.get("doors", [])
    n_pred_doors = len(pred_doors)

    return {
        "mw_i": mw_i, "mw_u": mw_u,
        "s_i": s_i, "s_u": s_u,
        "r_corr": r_corr, "r_tot": r_tot,
        "r_i": r_i, "r_u": r_u, "r_gt": r_gt,
        "shell_valid": 1 if is_shell_valid else 0,
        "n_gt_rooms": n_gt_rooms,
        "n_pred_rooms": n_pred_rooms,
        "room_count_err": abs(n_pred_rooms - n_gt_rooms),
        "room_count_signed": (n_pred_rooms - n_gt_rooms),
        "room_tp": tp, "room_fp": fp, "room_fn": fn,
        "n_gt_doors": gt_door_count,
        "n_pred_doors": n_pred_doors,
        "door_count_err": abs(n_pred_doors - gt_door_count)
    }

def run_comparative_bootstrap(
    plans_base: List[Dict], plans_tta: List[Dict],
    t_base: List[float], t_tta: List[float],
    ckpt_name: str, ckpt_hash: str,
    n_bootstraps: int = 1000, seed: int = 42
):
    print("\n" + "=" * 90)
    print(f"[*] RUNNING {n_bootstraps} PAIRED BOOTSTRAP RESAMPLES OVER {len(plans_base)} PLANS (Seed={seed})...")
    print("=" * 90)
    np.random.seed(seed)
    N = len(plans_base)

    inv_rooms = {v: k for k, v in ROOM_CLASSES.items()}

    metrics_list = [
        "Merged Wall IoU", "Interior Wall IoU", "Exterior Wall IoU",
        "Door IoU", "Window IoU", "Struct 3-Class mIoU", "Struct 4-Class mIoU",
        "Room Pixel Acc", "Room mIoU (10c)",
        "Room Instance Prec", "Room Instance Rec", "Room Instance F1",
        "Room Count MAE", "Door Count MAE", "Shell Validity Rate"
    ]
    # Add per-class room metrics
    for c in range(1, 11):
        metrics_list.append(f"Room: {inv_rooms[c]}")

    boot_base = {m: np.zeros(n_bootstraps) for m in metrics_list}
    boot_tta = {m: np.zeros(n_bootstraps) for m in metrics_list}
    boot_delta = {m: np.zeros(n_bootstraps) for m in metrics_list}

    def aggregate(records: List[Dict], idxs: np.ndarray) -> Dict[str, float]:
        mw_i = sum(records[i]["mw_i"] for i in idxs)
        mw_u = sum(records[i]["mw_u"] for i in idxs)
        s_i = sum(records[i]["s_i"] for i in idxs)
        s_u = sum(records[i]["s_u"] for i in idxs)
        r_corr = sum(records[i]["r_corr"] for i in idxs)
        r_tot = sum(records[i]["r_tot"] for i in idxs)
        r_i = sum(records[i]["r_i"] for i in idxs)
        r_u = sum(records[i]["r_u"] for i in idxs)

        shells_valid = sum(records[i]["shell_valid"] for i in idxs)
        tot_rc_err = sum(records[i]["room_count_err"] for i in idxs)
        tot_dc_err = sum(records[i]["door_count_err"] for i in idxs)

        tot_tp = sum(records[i]["room_tp"] for i in idxs)
        tot_fp = sum(records[i]["room_fp"] for i in idxs)
        tot_fn = sum(records[i]["room_fn"] for i in idxs)

        mw = (mw_i / max(mw_u, 1)) * 100
        w_int = (s_i[1] / max(s_u[1], 1)) * 100
        w_ext = (s_i[2] / max(s_u[2], 1)) * 100
        door = (s_i[3] / max(s_u[3], 1)) * 100
        win = (s_i[4] / max(s_u[4], 1)) * 100

        s3 = (mw + door + win) / 3.0
        s4 = (w_int + w_ext + door + win) / 4.0

        r_acc = (r_corr / max(r_tot, 1)) * 100
        per_class_r = {c: ((r_i[c] / max(r_u[c], 1)) * 100) for c in range(1, 11)}
        r_miou = np.mean(list(per_class_r.values()))

        prec = (tot_tp / max(tot_tp + tot_fp, 1)) * 100
        rec = (tot_tp / max(tot_tp + tot_fn, 1)) * 100
        f1 = (2 * prec * rec) / max(prec + rec, 1e-6)

        res = {
            "Merged Wall IoU": mw,
            "Interior Wall IoU": w_int,
            "Exterior Wall IoU": w_ext,
            "Door IoU": door,
            "Window IoU": win,
            "Struct 3-Class mIoU": s3,
            "Struct 4-Class mIoU": s4,
            "Room Pixel Acc": r_acc,
            "Room mIoU (10c)": r_miou,
            "Room Instance Prec": prec,
            "Room Instance Rec": rec,
            "Room Instance F1": f1,
            "Room Count MAE": tot_rc_err / len(idxs),
            "Door Count MAE": tot_dc_err / len(idxs),
            "Shell Validity Rate": (shells_valid / len(idxs)) * 100
        }
        for c in range(1, 11):
            res[f"Room: {inv_rooms[c]}"] = per_class_r[c]

        return res

    for b in range(n_bootstraps):
        idxs = np.random.choice(N, size=N, replace=True)
        agg_b = aggregate(plans_base, idxs)
        agg_t = aggregate(plans_tta, idxs)

        for m in metrics_list:
            boot_base[m][b] = agg_b[m]
            boot_tta[m][b] = agg_t[m]
            boot_delta[m][b] = agg_t[m] - agg_b[m]

    # Full aggregation over original 400 plans
    all_idxs = np.arange(N)
    full_base = aggregate(plans_base, all_idxs)
    full_tta = aggregate(plans_tta, all_idxs)

    mean_t_base = np.mean(t_base) * 1000
    mean_t_tta = np.mean(t_tta) * 1000

    print("\n" + "=" * 115)
    print(f"OFFICIAL VALIDATION BENCHMARK RESULTS (N=400 Plans)")
    print(f"Model Checkpoint: {ckpt_name} (SHA256: {ckpt_hash[:16]}...)")
    print(f"Inference Latency: Baseline = {mean_t_base:.1f} ms/plan | 4-Fold TTA = {mean_t_tta:.1f} ms/plan (+{mean_t_tta-mean_t_base:.1f} ms)")
    print("=" * 115)
    print(f"{'Metric':<25} | {'Baseline (95% CI)':<22} | {'4-Fold TTA (95% CI)':<22} | {'Delta (TTA - Base)':<20} | {'p-value':<8}")
    print("-" * 115)

    summary_out = {
        "checkpoint": ckpt_name,
        "sha256": ckpt_hash,
        "latency_base_ms": float(mean_t_base),
        "latency_tta_ms": float(mean_t_tta),
        "metrics": {}
    }

    for m in metrics_list:
        val_b = full_base[m]
        ci_b = (np.percentile(boot_base[m], 2.5), np.percentile(boot_base[m], 97.5))

        val_t = full_tta[m]
        ci_t = (np.percentile(boot_tta[m], 2.5), np.percentile(boot_tta[m], 97.5))

        delta = val_t - val_b
        ci_d = (np.percentile(boot_delta[m], 2.5), np.percentile(boot_delta[m], 97.5))

        # p-value: fraction of bootstraps where delta <= 0 (or delta >= 0 for error metrics)
        if "MAE" in m:
            p_val = np.mean(boot_delta[m] >= 0)  # for errors, reduction is good
        else:
            p_val = np.mean(boot_delta[m] <= 0)

        sig_star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))

        unit = "" if ("MAE" in m) else "%"
        print(f"{m:<25} | {val_b:6.2f}{unit} [{ci_b[0]:.2f}, {ci_b[1]:.2f}] | {val_t:6.2f}{unit} [{ci_t[0]:.2f}, {ci_t[1]:.2f}] | {delta:+6.2f}{unit} [{ci_d[0]:+.2f}, {ci_d[1]:+.2f}] | {p_val:.3f} {sig_star}")

        summary_out["metrics"][m] = {
            "baseline": float(val_b),
            "baseline_ci95": [float(ci_b[0]), float(ci_b[1])],
            "tta": float(val_t),
            "tta_ci95": [float(ci_t[0]), float(ci_t[1])],
            "delta": float(delta),
            "delta_ci95": [float(ci_d[0]), float(ci_d[1])],
            "p_value": float(p_val)
        }

    print("=" * 115)
    print("Significance levels: *** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant (p >= 0.05)")

    out_json = PROJECT_ROOT / "outputs" / "val_400_tta_benchmark.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\n[+] Saved full benchmark results to: {out_json}")

if __name__ == "__main__":
    ckpt = "models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth"
    run_evaluation(ckpt_path=ckpt, split="val")
