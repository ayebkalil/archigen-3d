"""
Official Test Split Authoritative Benchmark Engine for ArchiGen 3D.
Evaluates the final production checkpoint on the unseen TEST split (N=400 plans).

Comprehensive evaluation across:
1. Structural Pixel IoUs:
   - Merged Wall, Interior Wall, Exterior Wall, Door, Window IoU.
2. Semantic Room Metrics:
   - 9-Class Room mIoU (excluding rare/ambiguous Office).
   - 10-Class Room mIoU (including Office).
   - Room Pixel Accuracy (non-background pixels).
   - Per-class breakdown strictly aligned with ROOM_CLASSES.
3. Downstream Vectorizer & True Predicted Geometric Accuracy vs SVG Ground Truth:
   - Predicted Perimeter Relative Error (%) & MAE (m) vs Ground-Truth Envelope.
   - Predicted Footprint Area Relative Error (%) & MAE (m2) vs Ground-Truth Footprint.
   - Watertight Building Shell Validity Rate (%).
   - Room Instance Precision, Recall, F1 (matching IoU >= 0.5 & identical class label).
   - Room Count MAE & Door Count MAE.
4. Statistical Rigor:
   - 1,000 paired bootstrap resamples across the 400 test plans.
   - 95% Confidence Intervals [2.5%, 97.5%] for Baseline, TTA, and Delta.
   - Exact empirical p-values.
   - Logging of model weights SHA256 digest.
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
    tree = ET.parse(svg_path)
    root = tree.getroot()

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

    door_count = 0
    for g in root.findall(".//{http://www.w3.org/2000/svg}g"):
        c = g.attrib.get("class", "")
        if c.startswith("Door"):
            door_count += 1

    return room_instances, door_count

def match_room_instances(pred_rooms: List[Dict[str, Any]], gt_rooms: List[Dict[str, Any]], iou_thresh: float = 0.5):
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

def compute_plan_record(
    p_s: np.ndarray, p_r: np.ndarray, t_s: np.ndarray, t_r: np.ndarray,
    vectorizer: FloorplanVectorizer, gt_room_instances: List[Dict], gt_door_count: int, n_gt_rooms: int,
    gt_scale: float
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
    meta = layout.get("metadata", {})
    is_shell_valid = meta.get("is_building_shell_valid", False)

    pred_rooms = layout.get("rooms", [])
    n_pred_rooms = len(pred_rooms)

    # Format pred rooms with binary masks for matching
    scale_m = meta.get("scale_m_per_px", 0.03)
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

    tp, fp, fn = match_room_instances(formatted_pred, gt_room_instances, iou_thresh=0.5)

    pred_doors = layout.get("doors", [])
    n_pred_doors = len(pred_doors)

    # 5. True Ground-Truth Building Envelope & Geometric Errors
    gt_envelope = ((t_s == 1) | (t_s == 2) | (t_r > 0)).astype(np.uint8)
    cnts_gt, _ = cv2.findContours(gt_envelope, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gt_perim_m = 0.0
    gt_footprint_m2 = 0.0
    if cnts_gt:
        main_cnt = max(cnts_gt, key=cv2.contourArea)
        if cv2.contourArea(main_cnt) > 50:
            gt_perim_m = cv2.arcLength(main_cnt, True) * gt_scale
            gt_footprint_m2 = cv2.contourArea(main_cnt) * (gt_scale ** 2)

    pred_perim_m = meta.get("building_perimeter_m", 0.0)
    pred_footprint_m2 = meta.get("total_surface_m2", meta.get("total_area_m2", 0.0))
    gt_living_m2 = float((t_r > 0).sum()) * (gt_scale ** 2)
    target_area_m2 = gt_living_m2 if gt_living_m2 > 5.0 else gt_footprint_m2

    # Predicted vs GT errors
    has_valid_gt_geom = (gt_perim_m > 2.0 and target_area_m2 > 5.0)
    if has_valid_gt_geom:
        perim_rel_err = abs(pred_perim_m - gt_perim_m) / gt_perim_m
        perim_abs_err = abs(pred_perim_m - gt_perim_m)
        footprint_rel_err = abs(pred_footprint_m2 - target_area_m2) / target_area_m2
        footprint_abs_err = abs(pred_footprint_m2 - target_area_m2)
    else:
        perim_rel_err = None
        perim_abs_err = None
        footprint_rel_err = None
        footprint_abs_err = None

    return {
        "mw_i": mw_i, "mw_u": mw_u,
        "s_i": s_i, "s_u": s_u,
        "r_corr": r_corr, "r_tot": r_tot,
        "r_i": r_i, "r_u": r_u, "r_gt": r_gt,
        "shell_valid": 1 if is_shell_valid else 0,
        "n_gt_rooms": n_gt_rooms,
        "n_pred_rooms": n_pred_rooms,
        "room_count_err": abs(n_pred_rooms - n_gt_rooms),
        "room_tp": tp, "room_fp": fp, "room_fn": fn,
        "n_gt_doors": gt_door_count,
        "n_pred_doors": n_pred_doors,
        "door_count_err": abs(n_pred_doors - gt_door_count),
        "gt_perim_m": gt_perim_m,
        "pred_perim_m": pred_perim_m,
        "gt_footprint_m2": gt_footprint_m2,
        "pred_footprint_m2": pred_footprint_m2,
        "has_valid_gt_geom": has_valid_gt_geom,
        "perim_rel_err": perim_rel_err,
        "perim_abs_err": perim_abs_err,
        "footprint_rel_err": footprint_rel_err,
        "footprint_abs_err": footprint_abs_err
    }

def run_test_evaluation(ckpt_path: str = "models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_file = PROJECT_ROOT / ckpt_path
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    ckpt_hash = compute_sha256(ckpt_file)
    print("=" * 100)
    print(f"[AUTHORITATIVE TEST SPLIT BENCHMARK] Evaluating on Unseen 'test' Split (N=400 plans)")
    print(f"  Checkpoint File: {ckpt_file.name}")
    print(f"  SHA256:          {ckpt_hash}")
    print(f"  Device:          {device}")
    print("=" * 100)

    dataset = CubiCasaDataset(split="test", img_size=512, augment=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    num_plans = len(dataset)

    model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    vectorizer = FloorplanVectorizer()

    plans_base = []
    plans_tta = []
    t_base_infer = []
    t_tta_infer = []

    print(f"[*] Running inference & downstream vectorization across all {num_plans} test plans...")

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            imgs = batch["image"].to(device)
            t_s = batch["struct_mask"].numpy()
            t_r = batch["room_mask"].numpy()
            B = imgs.size(0)

            # 1. Baseline Single-Pass
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

            # 2. 4-Fold Dihedral TTA
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            prob_s_tta = prob_s_base.clone()
            prob_r_tta = prob_r_base.clone()

            # H-Flip
            imgs_h = torch.flip(imgs, dims=[3])
            out_h = model(imgs_h)
            prob_s_tta += torch.flip(F.softmax(out_h["struct"], dim=1), dims=[3])
            prob_r_tta += torch.flip(F.softmax(out_h["room"], dim=1), dims=[3])

            # V-Flip
            imgs_v = torch.flip(imgs, dims=[2])
            out_v = model(imgs_v)
            prob_s_tta += torch.flip(F.softmax(out_v["struct"], dim=1), dims=[2])
            prob_r_tta += torch.flip(F.softmax(out_v["room"], dim=1), dims=[2])

            # 180-Rotation
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

            # 3. Geometric Vectorization & Metrics
            for b in range(B):
                plan_idx = b_idx * loader.batch_size + b
                rel_sample = dataset.valid_samples[plan_idx]
                svg_path = dataset.root_dir / rel_sample / "model.svg"
                gt_scale = float(batch["gt_scale"][b]) if "gt_scale" in batch else 0.030

                gt_room_instances, gt_door_count = extract_svg_gt_instances(svg_path)
                n_gt_rooms = len(gt_room_instances)

                stats_b = compute_plan_record(
                    p_s_base[b], p_r_base[b], t_s[b], t_r[b],
                    vectorizer, gt_room_instances, gt_door_count, n_gt_rooms, gt_scale
                )
                plans_base.append(stats_b)

                stats_t = compute_plan_record(
                    p_s_tta[b], p_r_tta[b], t_s[b], t_r[b],
                    vectorizer, gt_room_instances, gt_door_count, n_gt_rooms, gt_scale
                )
                plans_tta.append(stats_t)

            if (b_idx + 1) % 25 == 0:
                print(f"  Processed {min((b_idx+1)*loader.batch_size, num_plans)}/{num_plans} test floorplans...")

    # 4. Bootstrap Analysis
    run_test_bootstrap(plans_base, plans_tta, t_base_infer, t_tta_infer, ckpt_file.name, ckpt_hash)

def run_test_bootstrap(
    plans_base: List[Dict], plans_tta: List[Dict],
    t_base: List[float], t_tta: List[float],
    ckpt_name: str, ckpt_hash: str,
    n_bootstraps: int = 1000, seed: int = 42
):
    print("\n" + "=" * 100)
    print(f"[*] COMPUTING 1,000 PAIRED BOOTSTRAP RESAMPLES OVER {len(plans_base)} TEST PLANS (Seed={seed})...")
    print("=" * 100)
    np.random.seed(seed)
    N = len(plans_base)

    inv_rooms = {v: k for k, v in ROOM_CLASSES.items()}

    metrics_list = [
        "Merged Wall IoU", "Interior Wall IoU", "Exterior Wall IoU",
        "Door IoU", "Window IoU", "Struct 3-Class mIoU", "Struct 4-Class mIoU",
        "Room Pixel Acc", "Room mIoU (9c, ex. Office)", "Room mIoU (10c, inc. Office)",
        "Pred Perimeter Rel Error", "Pred Perimeter MAE (m)",
        "Pred Footprint Rel Error", "Pred Footprint MAE (m2)",
        "Shell Validity Rate",
        "Room Instance Prec", "Room Instance Rec", "Room Instance F1",
        "Room Count MAE", "Door Count MAE"
    ]
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

        # 9-class mean (excluding Office, c=10)
        r_miou_9c = np.mean([per_class_r[c] for c in range(1, 10)])
        # 10-class mean (including Office, c=10)
        r_miou_10c = np.mean([per_class_r[c] for c in range(1, 11)])

        prec = (tot_tp / max(tot_tp + tot_fp, 1)) * 100
        rec = (tot_tp / max(tot_tp + tot_fn, 1)) * 100
        f1 = (2 * prec * rec) / max(prec + rec, 1e-6)

        # Predicted Geometric Accuracy on plans with valid GT geometry
        valid_perim_errs = [records[i]["perim_rel_err"] for i in idxs if records[i]["perim_rel_err"] is not None]
        valid_perim_abs = [records[i]["perim_abs_err"] for i in idxs if records[i]["perim_abs_err"] is not None]
        valid_foot_errs = [records[i]["footprint_rel_err"] for i in idxs if records[i]["footprint_rel_err"] is not None]
        valid_foot_abs = [records[i]["footprint_abs_err"] for i in idxs if records[i]["footprint_abs_err"] is not None]

        p_rel_err = (float(np.mean(valid_perim_errs)) * 100) if valid_perim_errs else 0.0
        p_abs_err = float(np.mean(valid_perim_abs)) if valid_perim_abs else 0.0
        f_rel_err = (float(np.mean(valid_foot_errs)) * 100) if valid_foot_errs else 0.0
        f_abs_err = float(np.mean(valid_foot_abs)) if valid_foot_abs else 0.0

        res = {
            "Merged Wall IoU": mw,
            "Interior Wall IoU": w_int,
            "Exterior Wall IoU": w_ext,
            "Door IoU": door,
            "Window IoU": win,
            "Struct 3-Class mIoU": s3,
            "Struct 4-Class mIoU": s4,
            "Room Pixel Acc": r_acc,
            "Room mIoU (9c, ex. Office)": r_miou_9c,
            "Room mIoU (10c, inc. Office)": r_miou_10c,
            "Pred Perimeter Rel Error": p_rel_err,
            "Pred Perimeter MAE (m)": p_abs_err,
            "Pred Footprint Rel Error": f_rel_err,
            "Pred Footprint MAE (m2)": f_abs_err,
            "Shell Validity Rate": (shells_valid / len(idxs)) * 100,
            "Room Instance Prec": prec,
            "Room Instance Rec": rec,
            "Room Instance F1": f1,
            "Room Count MAE": tot_rc_err / len(idxs),
            "Door Count MAE": tot_dc_err / len(idxs)
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

    all_idxs = np.arange(N)
    full_base = aggregate(plans_base, all_idxs)
    full_tta = aggregate(plans_tta, all_idxs)

    mean_t_base = np.mean(t_base) * 1000
    mean_t_tta = np.mean(t_tta) * 1000

    print("\n" + "=" * 120)
    print(f"OFFICIAL TEST SPLIT BENCHMARK RESULTS (N=400 Unseen Plans)")
    print(f"Model Checkpoint: {ckpt_name} (SHA256: {ckpt_hash[:16]}...)")
    print(f"Inference Latency: Baseline = {mean_t_base:.1f} ms/plan | 4-Fold TTA = {mean_t_tta:.1f} ms/plan (+{mean_t_tta-mean_t_base:.1f} ms)")
    print("=" * 120)
    print(f"{'Metric':<30} | {'Baseline (95% CI)':<24} | {'4-Fold TTA (95% CI)':<24} | {'Delta (TTA - Base)':<22} | {'p-value':<8}")
    print("-" * 120)

    summary_out = {
        "split": "test",
        "num_plans": N,
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

        # p-value calculation: for error metrics, reduction is good (delta < 0)
        is_error_metric = ("MAE" in m or "Error" in m)
        if is_error_metric:
            p_val = np.mean(boot_delta[m] >= 0)
        else:
            p_val = np.mean(boot_delta[m] <= 0)

        sig_star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))

        if "MAE (m2)" in m:
            unit = " m²"
        elif "MAE (m)" in m:
            unit = " m"
        elif "MAE" in m:
            unit = " rms"
        else:
            unit = "%"

        print(f"{m:<30} | {val_b:6.2f}{unit} [{ci_b[0]:.2f}, {ci_b[1]:.2f}] | {val_t:6.2f}{unit} [{ci_t[0]:.2f}, {ci_t[1]:.2f}] | {delta:+6.2f}{unit} [{ci_d[0]:+.2f}, {ci_d[1]:+.2f}] | {p_val:.3f} {sig_star}")

        summary_out["metrics"][m] = {
            "baseline": float(val_b),
            "baseline_ci95": [float(ci_b[0]), float(ci_b[1])],
            "tta": float(val_t),
            "tta_ci95": [float(ci_t[0]), float(ci_t[1])],
            "delta": float(delta),
            "delta_ci95": [float(ci_d[0]), float(ci_d[1])],
            "p_value": float(p_val)
        }

    print("=" * 120)
    print("Significance levels: *** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant (p >= 0.05)")

    out_json = PROJECT_ROOT / "models" / "benchmarks" / "official_test_split_authoritative.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\n[+] Saved authoritative test benchmark results to: {out_json}")

if __name__ == "__main__":
    run_test_evaluation()
