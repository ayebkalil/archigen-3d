"""
Quantitative Benchmark and Evaluation Suite for Floorplan Models on CubiCasa5K.
Evaluates:
- Structural elements: Merged Wall IoU, Internal/External Wall IoU, Door IoU, Window IoU, mIoU
- Semantic room types: Room mIoU (11 classes), Room Pixel Accuracy
- 3D Geometry: Building Shell Validity Rate, Polygon Validity Rate
Supports models: 'baseline', 'multitask', 'hybrid', or 'all' for comparative table.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import segmentation_models_pytorch as smp
from safetensors.torch import load_file
from shapely.geometry import Polygon

from src.floorplan.dataset import CubiCasaDataset, ROOM_CLASSES
from src.floorplan.models import MultiTaskFloorplanUNet
from src.floorplan.vectorizer import FloorplanVectorizer


def evaluate_model_pipeline(
    model_type: str = "hybrid",
    split: str = "test",
    max_samples: Optional[int] = None,
    batch_size: int = 8,
    baseline_weights: str = "models/saved/floorplan_baseline/best.safetensors",
    multitask_weights: str = "models/saved/floorplan_unetpp/multitask_resnet34_best.pth"
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[EVALUATION] Starting benchmark for model: '{model_type}' on split: '{split}' (Device: {device})")

    # 1. Load Dataset
    dataset = CubiCasaDataset(split=split, img_size=512, augment=False)
    if max_samples and max_samples < len(dataset):
        dataset = Subset(dataset, list(range(max_samples)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_samples = len(dataset)

    # 2. Load Models
    baseline_model = None
    multitask_model = None

    if model_type in ["baseline", "hybrid", "all"]:
        if Path(baseline_weights).exists():
            baseline_model = smp.Unet(encoder_name="resnet34", classes=4).to(device)
            baseline_model.load_state_dict(load_file(baseline_weights))
            baseline_model.eval()
        else:
            print(f"[WARNING] Baseline weights not found at {baseline_weights}")

    if model_type in ["multitask", "hybrid", "all"]:
        if Path(multitask_weights).exists():
            multitask_model = MultiTaskFloorplanUNet(
                encoder_name="resnet34",
                num_struct_classes=5,
                num_room_classes=12
            ).to(device)
            ckpt = torch.load(multitask_weights, map_location=device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            multitask_model.load_state_dict(state_dict)
            multitask_model.eval()
        else:
            print(f"[WARNING] MultiTask weights not found at {multitask_weights}")

    vectorizer = FloorplanVectorizer()

    # Metric accumulators
    m_wall_inter, m_wall_union = 0, 0
    door_inter, door_union = 0, 0
    win_inter, win_union = 0, 0

    wall_int_inter, wall_int_union = 0, 0
    wall_ext_inter, wall_ext_union = 0, 0

    r_inter = {c: 0 for c in range(1, 12)}
    r_union = {c: 0 for c in range(1, 12)}
    r_correct_px, r_total_px = 0, 0

    # True Ground-Truth Accuracy Metrics
    living_area_inter = 0
    living_area_union = 0
    room_count_errors = []
    perimeter_rel_errors = []
    scale_errors = []
    autoscale_count = 0
    ext_wall_correct = 0
    ext_wall_total = 0
    ext_wall_tp = 0
    ext_wall_fp = 0
    ext_wall_fn = 0

    # Object-level detection accumulators (within 20px radius)
    door_gt_total = 0
    door_gt_detected = 0
    win_gt_total = 0
    win_gt_detected = 0

    valid_shell_count = 0
    valid_polygon_count = 0
    total_polygon_count = 0
    total_rooms_count = 0

    processed_samples = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            gt_struct = batch["struct_mask"].numpy()  # 0: bg, 1: wall_int, 2: wall_ext, 3: door, 4: win
            gt_room = batch["room_mask"].numpy()      # 0: bg, 1..11: room classes
            gt_mask4 = batch["mask"].numpy()          # 0: bg, 1: wall, 2: door, 3: win
            b_size = imgs.size(0)

            # Inferences based on model_type
            if model_type == "baseline":
                logits = baseline_model(imgs)
                pred_mask4 = torch.argmax(logits, dim=1).cpu().numpy()
                pred_room = None
                pred_struct5 = None
            elif model_type == "multitask":
                outputs = multitask_model(imgs)
                pred_struct5 = torch.argmax(outputs["struct"], dim=1).cpu().numpy()
                pred_room = torch.argmax(outputs["room"], dim=1).cpu().numpy()
                pred_mask4 = np.zeros_like(pred_struct5)
                pred_mask4[(pred_struct5 == 1) | (pred_struct5 == 2)] = 1
                pred_mask4[pred_struct5 == 3] = 2
                pred_mask4[pred_struct5 == 4] = 3
            elif model_type == "hybrid":
                logits = baseline_model(imgs)
                pred_mask4 = torch.argmax(logits, dim=1).cpu().numpy()
                if multitask_model is not None:
                    outputs = multitask_model(imgs)
                    pred_room = torch.argmax(outputs["room"], dim=1).cpu().numpy()
                    pred_struct5 = torch.argmax(outputs["struct"], dim=1).cpu().numpy()
                else:
                    pred_room = None
                    pred_struct5 = None

            # Calculate metrics per plan
            for b in range(b_size):
                p_m4 = pred_mask4[b]
                t_m4 = gt_mask4[b]
                t_s5 = gt_struct[b]
                t_r = gt_room[b]

                # Merged Wall IoU
                p_wall = (p_m4 == 1)
                t_wall = (t_m4 == 1)
                m_wall_inter += np.logical_and(p_wall, t_wall).sum()
                m_wall_union += np.logical_or(p_wall, t_wall).sum()

                # Door IoU
                p_door = (p_m4 == 2)
                t_door = (t_m4 == 2)
                door_inter += np.logical_and(p_door, t_door).sum()
                door_union += np.logical_or(p_door, t_door).sum()

                # Window IoU
                p_win = (p_m4 == 3)
                t_win = (t_m4 == 3)
                win_inter += np.logical_and(p_win, t_win).sum()
                win_union += np.logical_or(p_win, t_win).sum()

                # Object-level door & window localization
                import cv2
                # GT doors
                _, _, _, gt_door_cents = cv2.connectedComponentsWithStats((t_door).astype(np.uint8))
                _, _, _, pred_door_cents = cv2.connectedComponentsWithStats((p_door).astype(np.uint8))
                gt_door_pts = gt_door_cents[1:] if len(gt_door_cents) > 1 else []
                pred_door_pts = pred_door_cents[1:] if len(pred_door_cents) > 1 else []
                door_gt_total += len(gt_door_pts)
                for g_pt in gt_door_pts:
                    if any(np.hypot(p_pt[0] - g_pt[0], p_pt[1] - g_pt[1]) < 20.0 for p_pt in pred_door_pts):
                        door_gt_detected += 1

                # GT windows
                _, _, _, gt_win_cents = cv2.connectedComponentsWithStats((t_win).astype(np.uint8))
                _, _, _, pred_win_cents = cv2.connectedComponentsWithStats((p_win).astype(np.uint8))
                gt_win_pts = gt_win_cents[1:] if len(gt_win_cents) > 1 else []
                pred_win_pts = pred_win_cents[1:] if len(pred_win_cents) > 1 else []
                win_gt_total += len(gt_win_pts)
                for g_pt in gt_win_pts:
                    if any(np.hypot(p_pt[0] - g_pt[0], p_pt[1] - g_pt[1]) < 20.0 for p_pt in pred_win_pts):
                        win_gt_detected += 1

                # Internal & External Wall IoU if 5-class available
                if pred_struct5 is not None:
                    p_s5 = pred_struct5[b]
                    wall_int_inter += np.logical_and(p_s5 == 1, t_s5 == 1).sum()
                    wall_int_union += np.logical_or(p_s5 == 1, t_s5 == 1).sum()

                    wall_ext_inter += np.logical_and(p_s5 == 2, t_s5 == 2).sum()
                    wall_ext_union += np.logical_or(p_s5 == 2, t_s5 == 2).sum()

                # Room metrics if room model available
                if pred_room is not None:
                    p_r = pred_room[b]
                    for c in range(1, 12):
                        p_c = (p_r == c)
                        t_c = (t_r == c)
                        r_inter[c] += np.logical_and(p_c, t_c).sum()
                        r_union[c] += np.logical_or(p_c, t_c).sum()

                    non_bg = (t_r > 0)
                    r_correct_px += np.logical_and(p_r == t_r, non_bg).sum()
                    r_total_px += non_bg.sum()

                    # Living Area Footprint IoU (all non-background rooms combined)
                    p_living = (p_r > 0)
                    t_living = (t_r > 0)
                    living_area_inter += np.logical_and(p_living, t_living).sum()
                    living_area_union += np.logical_or(p_living, t_living).sum()

                    # Ground truth room instances (connected components >= 100px)
                    gt_num_rooms = 0
                    for c in range(1, 12):
                        c_mask = (t_r == c).astype(np.uint8)
                        cnts, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        gt_num_rooms += sum(1 for cnt in cnts if cv2.contourArea(cnt) >= 100)

                # 3D Vectorizer and Geometric Quality
                p_s_input = pred_struct5[b] if pred_struct5 is not None else None
                p_r_input = pred_room[b] if pred_room is not None else None

                layout = vectorizer.vectorize(
                    mask=p_m4,
                    struct_mask=p_s_input,
                    room_mask=p_r_input
                )

                meta = layout.get("metadata", {})
                pred_scale = meta.get("scale_meters_per_pixel", 0.030)

                if pred_room is not None:
                    pred_num_rooms = len(layout.get("rooms", []))
                    room_count_errors.append(abs(pred_num_rooms - gt_num_rooms))

                if meta.get("is_building_shell_valid", False):
                    valid_shell_count += 1

                for w in layout.get("walls", []):
                    pts = w.get("polygon_meters", [])
                    if len(pts) >= 3:
                        total_polygon_count += 1
                        poly = Polygon(pts)
                        if poly.is_valid:
                            valid_polygon_count += 1

                        # Vectorizer exterior wall vs GT evaluation
                        poly_px = (np.array(pts) / max(pred_scale, 1e-4)).astype(np.int32)
                        patch = np.zeros((512, 512), dtype=np.uint8)
                        cv2.fillPoly(patch, [poly_px], 1)
                        ext_px = int(np.logical_and(patch == 1, t_s5 == 2).sum())
                        int_px = int(np.logical_and(patch == 1, t_s5 == 1).sum())
                        if ext_px + int_px >= 20:
                            gt_is_ext = (ext_px > int_px)
                            pred_is_ext = w.get("is_exterior", False)
                            if pred_is_ext == gt_is_ext:
                                ext_wall_correct += 1
                            ext_wall_total += 1
                            if pred_is_ext and gt_is_ext:
                                ext_wall_tp += 1
                            elif pred_is_ext and not gt_is_ext:
                                ext_wall_fp += 1
                            elif not pred_is_ext and gt_is_ext:
                                ext_wall_fn += 1

                total_rooms_count += len(layout.get("rooms", []))

                # Scale Relative Error & Auto-scale Trigger
                gt_scale_for_plan = float(batch["gt_scale"][b]) if "gt_scale" in batch else pred_scale
                if meta.get("scale_source") == "door_width_estimation":
                    autoscale_count += 1
                if gt_scale_for_plan > 0:
                    err_s = abs(pred_scale - gt_scale_for_plan) / gt_scale_for_plan
                    scale_errors.append(min(err_s, 2.0))

                # Exterior Perimeter relative error vs GT
                gt_ext_mask = (t_s5 == 2).astype(np.uint8) if pred_struct5 is not None else (t_wall).astype(np.uint8)
                gt_cnts, _ = cv2.findContours(gt_ext_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if gt_cnts:
                    gt_perimeter_px = sum(cv2.arcLength(c, True) for c in gt_cnts)
                    gt_perimeter_m = gt_perimeter_px * gt_scale_for_plan
                    pred_perimeter_m = meta.get("building_perimeter_m", 0)
                    if gt_perimeter_m > 1.0:
                        rel_err = abs(pred_perimeter_m - gt_perimeter_m) / gt_perimeter_m
                        perimeter_rel_errors.append(min(rel_err, 2.0))

                processed_samples += 1

            if processed_samples % 50 == 0 or processed_samples == total_samples:
                print(f"  Processed {processed_samples}/{total_samples} floorplans...")

    # Final calculations
    wall_merged_iou = m_wall_inter / max(m_wall_union, 1)
    door_iou = door_inter / max(door_union, 1)
    win_iou = win_inter / max(win_union, 1)
    struct_miou = (wall_merged_iou + door_iou + win_iou) / 3.0

    wall_int_iou = (wall_int_inter / max(wall_int_union, 1)) if wall_int_union > 0 else 0.0
    wall_ext_iou = (wall_ext_inter / max(wall_ext_union, 1)) if wall_ext_union > 0 else 0.0

    room_ious = [r_inter[c] / max(r_union[c], 1) for c in range(1, 12) if r_union[c] > 0]
    room_miou = (sum(room_ious) / len(room_ious)) if room_ious else 0.0
    room_acc = (r_correct_px / max(r_total_px, 1)) if r_total_px > 0 else 0.0

    living_area_iou = (living_area_inter / max(living_area_union, 1)) if living_area_union > 0 else 0.0
    room_count_mae = float(np.mean(room_count_errors)) if room_count_errors else 0.0
    perimeter_rel_err = float(np.mean(perimeter_rel_errors)) * 100.0 if perimeter_rel_errors else 0.0
    scale_rel_err = float(np.mean(scale_errors)) * 100.0 if scale_errors else 0.0
    autoscale_rate = (autoscale_count / max(processed_samples, 1)) * 100.0

    ext_wall_acc = (ext_wall_correct / max(ext_wall_total, 1)) * 100.0 if ext_wall_total > 0 else 0.0
    ext_wall_prec = (ext_wall_tp / max(ext_wall_tp + ext_wall_fp, 1)) * 100.0 if (ext_wall_tp + ext_wall_fp) > 0 else 0.0
    ext_wall_rec = (ext_wall_tp / max(ext_wall_tp + ext_wall_fn, 1)) * 100.0 if (ext_wall_tp + ext_wall_fn) > 0 else 0.0

    door_obj_recall = (door_gt_detected / max(door_gt_total, 1)) * 100.0
    win_obj_recall = (win_gt_detected / max(win_gt_total, 1)) * 100.0

    shell_valid_rate = (valid_shell_count / max(processed_samples, 1)) * 100.0
    polygon_valid_rate = (valid_polygon_count / max(total_polygon_count, 1)) * 100.0

    results = {
        "model_type": model_type,
        "plans_evaluated": processed_samples,
        "wall_merged_iou": round(wall_merged_iou * 100, 2),
        "door_iou": round(door_iou * 100, 2),
        "window_iou": round(win_iou * 100, 2),
        "struct_miou": round(struct_miou * 100, 2),
        "wall_internal_iou": round(wall_int_iou * 100, 2) if wall_int_union > 0 else None,
        "wall_external_iou": round(wall_ext_iou * 100, 2) if wall_ext_union > 0 else None,
        "room_miou": round(room_miou * 100, 2) if r_total_px > 0 else None,
        "room_pixel_accuracy": round(room_acc * 100, 2) if r_total_px > 0 else None,
        "living_area_iou": round(living_area_iou * 100, 2) if living_area_union > 0 else None,
        "room_count_mae": round(room_count_mae, 2) if room_count_errors else None,
        "perimeter_relative_error_pct": round(perimeter_rel_err, 2) if perimeter_rel_errors else None,
        "scale_relative_error_pct": round(scale_rel_err, 2) if scale_errors else None,
        "autoscale_trigger_rate_pct": round(autoscale_rate, 2),
        "ext_wall_accuracy_pct": round(ext_wall_acc, 2) if ext_wall_total > 0 else None,
        "ext_wall_precision_pct": round(ext_wall_prec, 2) if ext_wall_total > 0 else None,
        "ext_wall_recall_pct": round(ext_wall_rec, 2) if ext_wall_total > 0 else None,
        "door_object_recall_pct": round(door_obj_recall, 2) if door_gt_total > 0 else None,
        "win_object_recall_pct": round(win_obj_recall, 2) if win_gt_total > 0 else None,
        "watertight_shell_rate_pct": round(shell_valid_rate, 2),
        "polygon_validity_rate_pct": round(polygon_valid_rate, 2),
        "avg_rooms_per_plan": round(total_rooms_count / max(processed_samples, 1), 1)
    }

    print("\n" + "=" * 65)
    print(f"       BENCHMARK RESULTS: {model_type.upper()} ({processed_samples} PLANS)")
    print("=" * 65)
    print(f"  Wall Merged IoU:          {results['wall_merged_iou']}%")
    if results['wall_internal_iou'] is not None:
        print(f"  Wall Internal IoU:        {results['wall_internal_iou']}%")
    if results['wall_external_iou'] is not None:
        print(f"  Wall External IoU:        {results['wall_external_iou']}%")
    print(f"  Door IoU:                 {results['door_iou']}%")
    print(f"  Window IoU:               {results['window_iou']}%")
    print(f"  Structure Foreground mIoU:{results['struct_miou']}%")
    if results['room_miou'] is not None:
        print(f"  Room Semantic mIoU:       {results['room_miou']}%")
        print(f"  Room Pixel Accuracy:      {results['room_pixel_accuracy']}%")
    print(f"  Watertight 3D Shell Rate: {results['watertight_shell_rate_pct']}%")
    print(f"  3D Polygon Validity Rate: {results['polygon_validity_rate_pct']}%")
    print(f"  Avg Rooms Extracted:      {results['avg_rooms_per_plan']}")
    if results.get('living_area_iou') is not None:
        print(f"  Living Area Footprint IoU:{results['living_area_iou']}%")
    if results.get('room_count_mae') is not None:
        print(f"  Room Count MAE:           {results['room_count_mae']}")
    if results.get('perimeter_relative_error_pct') is not None:
        print(f"  Perimeter Rel Error:      {results['perimeter_relative_error_pct']}%")
    if results.get('scale_relative_error_pct') is not None:
        print(f"  Scale Relative Error:     {results['scale_relative_error_pct']}%")
    print(f"  Auto-Scale Trigger Rate:  {results['autoscale_trigger_rate_pct']}%")
    if results.get('ext_wall_accuracy_pct') is not None:
        print(f"  Exterior Wall Accuracy:   {results['ext_wall_accuracy_pct']}% (Prec: {results['ext_wall_precision_pct']}%, Rec: {results['ext_wall_recall_pct']}%)")
    if results.get('door_object_recall_pct') is not None:
        print(f"  Door Object Recall (20px):{results['door_object_recall_pct']}%")
    if results.get('win_object_recall_pct') is not None:
        print(f"  Window Object Recall (20px):{results['win_object_recall_pct']}%")
    print("=" * 65 + "\n")

    return results


def run_full_comparative_benchmark(split: str = "test", max_samples: Optional[int] = None):
    print("\n" + "#" * 70)
    print("    ARCHIGEN 3D: CUBICASA5K FULL COMPARATIVE BENCHMARK SUITE")
    print("#" * 70)

    res_baseline = evaluate_model_pipeline("baseline", split=split, max_samples=max_samples)
    res_multitask = evaluate_model_pipeline("multitask", split=split, max_samples=max_samples)
    res_hybrid = evaluate_model_pipeline("hybrid", split=split, max_samples=max_samples)

    print("\n" + "=" * 80)
    print(f"                   OVERALL MODEL COMPARISON TABLE ({split.upper()} SET)")
    print("=" * 80)
    header = f"{'Metric':<28} | {'Baseline (ResNet34)':<20} | {'MultiTask':<18} | {'Hybrid (Engine)':<15}"
    print(header)
    print("-" * 80)

    rows = [
        ("Merged Wall IoU", f"{res_baseline['wall_merged_iou']}%", f"{res_multitask['wall_merged_iou']}%", f"{res_hybrid['wall_merged_iou']}%"),
        ("Wall Internal IoU", "N/A", f"{res_multitask.get('wall_internal_iou', 'N/A')}%", f"{res_hybrid.get('wall_internal_iou', 'N/A')}%"),
        ("Wall External IoU", "N/A", f"{res_multitask.get('wall_external_iou', 'N/A')}%", f"{res_hybrid.get('wall_external_iou', 'N/A')}%"),
        ("Door IoU", f"{res_baseline['door_iou']}%", f"{res_multitask['door_iou']}%", f"{res_hybrid['door_iou']}%"),
        ("Window IoU", f"{res_baseline['window_iou']}%", f"{res_multitask['window_iou']}%", f"{res_hybrid['window_iou']}%"),
        ("Door Object Recall (20px)", f"{res_baseline.get('door_object_recall_pct', 'N/A')}%", f"{res_multitask.get('door_object_recall_pct', 'N/A')}%", f"{res_hybrid.get('door_object_recall_pct', 'N/A')}%"),
        ("Window Object Recall (20px)", f"{res_baseline.get('win_object_recall_pct', 'N/A')}%", f"{res_multitask.get('win_object_recall_pct', 'N/A')}%", f"{res_hybrid.get('win_object_recall_pct', 'N/A')}%"),
        ("Structure Foreground mIoU", f"{res_baseline['struct_miou']}%", f"{res_multitask['struct_miou']}%", f"{res_hybrid['struct_miou']}%"),
        ("Room Semantic mIoU", "N/A", f"{res_multitask['room_miou']}%", f"{res_hybrid['room_miou']}%"),
        ("Room Pixel Accuracy", "N/A", f"{res_multitask['room_pixel_accuracy']}%", f"{res_hybrid['room_pixel_accuracy']}%"),
        ("Living Area Footprint IoU", "N/A", f"{res_multitask.get('living_area_iou', 'N/A')}%", f"{res_hybrid.get('living_area_iou', 'N/A')}%"),
        ("Room Count MAE", "N/A", f"{res_multitask.get('room_count_mae', 'N/A')}", f"{res_hybrid.get('room_count_mae', 'N/A')}"),
        ("Exterior Wall Accuracy", "N/A", f"{res_multitask.get('ext_wall_accuracy_pct', 'N/A')}%", f"{res_hybrid.get('ext_wall_accuracy_pct', 'N/A')}%"),
        ("Perimeter Relative Error", f"{res_baseline.get('perimeter_relative_error_pct', 'N/A')}%", f"{res_multitask.get('perimeter_relative_error_pct', 'N/A')}%", f"{res_hybrid.get('perimeter_relative_error_pct', 'N/A')}%"),
        ("Scale Relative Error", f"{res_baseline.get('scale_relative_error_pct', 'N/A')}%", f"{res_multitask.get('scale_relative_error_pct', 'N/A')}%", f"{res_hybrid.get('scale_relative_error_pct', 'N/A')}%"),
        ("Auto-Scale Trigger Rate", f"{res_baseline['autoscale_trigger_rate_pct']}%", f"{res_multitask['autoscale_trigger_rate_pct']}%", f"{res_hybrid['autoscale_trigger_rate_pct']}%"),
        ("Watertight 3D Shell Rate", f"{res_baseline['watertight_shell_rate_pct']}%", f"{res_multitask['watertight_shell_rate_pct']}%", f"{res_hybrid['watertight_shell_rate_pct']}%"),
        ("Avg Rooms / Plan", f"{res_baseline['avg_rooms_per_plan']}", f"{res_multitask['avg_rooms_per_plan']}", f"{res_hybrid['avg_rooms_per_plan']}")
    ]

    for label, b_val, m_val, h_val in rows:
        print(f"{label:<28} | {b_val:<20} | {m_val:<18} | {h_val:<15}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate floorplan models on CubiCasa5K")
    parser.add_argument("--model", type=str, default="multitask", choices=["baseline", "multitask", "hybrid", "all"])
    parser.add_argument("--split", type=str, default="val", choices=["test", "val", "train"])
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for rapid run")
    parser.add_argument("--weights", type=str, default="models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth", help="Path to multitask model weights")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    args = parser.parse_args()

    if args.model == "all":
        run_full_comparative_benchmark(split=args.split, max_samples=args.max_samples)
    else:
        evaluate_model_pipeline(
            model_type=args.model,
            split=args.split,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            multitask_weights=args.weights
        )
