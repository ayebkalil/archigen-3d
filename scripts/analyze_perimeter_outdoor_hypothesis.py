"""
Script: analyze_perimeter_outdoor_hypothesis.py
Tests the user's hypothesis on the 400 validation plans:
1. Compares Envelope WITH Outdoor (class 6) vs Envelope WITHOUT Outdoor (Indoor heated footprint only).
2. Computes Signed Perimeter Error: (P_pred - P_gt) / P_gt in pixels and in meters.
3. Computes Footprint IoU with and without Outdoor.
4. Identifies the 10 worst plans, prints their characteristics, and saves visual overlays to outputs/perimeter_worst_10/.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import binary_fill_holes
from shapely.geometry import Polygon
from shapely.validation import make_valid
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset, ROOM_CLASSES
from src.floorplan.models import MultiTaskFloorplanUNet
from src.floorplan.vectorizer import FloorplanVectorizer

def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    largest_lbl = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_lbl).astype(np.uint8)

def extract_envelope_polygon(wall_mask: np.ndarray, room_mask: np.ndarray, include_outdoor: bool, scale_m: float) -> Tuple[float, float, int, Polygon]:
    """
    Constructs the filled envelope polygon, simplified with approxPolyDP(eps=2.0).
    Returns: (perimeter_px, perimeter_m, num_vertices, polygon)
    """
    if include_outdoor:
        raw = ((wall_mask > 0) | (room_mask > 0)).astype(np.uint8)
    else:
        # Exclude class 6 (Outdoor / Terrace / Balcony)
        raw = ((wall_mask > 0) | ((room_mask > 0) & (room_mask != 6))).astype(np.uint8)

    filled = binary_fill_holes(raw).astype(np.uint8)
    largest = keep_largest_component(filled)
    
    cnts, _ = cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 0.0, 0, Polygon()

    largest_cnt = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest_cnt, 2.0, True).reshape(-1, 2)
    if len(approx) < 3:
        return 0.0, 0.0, 0, Polygon()

    poly = Polygon(approx)
    if not poly.is_valid:
        repaired = make_valid(poly)
        if repaired.geom_type == "Polygon":
            poly = repaired
        elif repaired.geom_type in ["MultiPolygon", "GeometryCollection"]:
            polys = [g for g in repaired.geoms if g.geom_type == "Polygon" and not g.is_empty]
            poly = max(polys, key=lambda p: p.area) if polys else Polygon()

    perim_px = float(poly.length) if poly.is_valid else 0.0
    perim_m = float(perim_px * scale_m)
    num_pts = len(poly.exterior.coords) - 1 if (poly.is_valid and not poly.is_empty) else len(approx)
    return perim_px, perim_m, num_pts, poly

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 90)
    print(f"ANALYZING PERIMETER ERROR & OUTDOOR CLASS HYPOTHESIS (Device: {device})")
    print("Dataset: CubiCasa5K Validation Split (N = 400 plans)")
    print("=" * 90)

    dataset = CubiCasaDataset(split="val", img_size=512, augment=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    num_plans = len(dataset)

    ckpt_path = PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth"
    model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    plans_results = []
    plan_idx = 0

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            imgs = batch["image"].to(device)
            gt_s = batch["struct_mask"].numpy()
            gt_r = batch["room_mask"].numpy()
            B = imgs.size(0)

            out = model(imgs)
            p_s = torch.argmax(out["struct"], dim=1).cpu().numpy().astype(np.uint8)
            p_r = torch.argmax(out["room"], dim=1).cpu().numpy().astype(np.uint8)

            for i in range(B):
                # Scale: in CubiCasa, ~0.03 m/px
                scale_m = 0.03

                # 1. Config A: WITH Outdoor
                p_px_gt_a, p_m_gt_a, pts_gt_a, poly_gt_a = extract_envelope_polygon(
                    ((gt_s[i] == 1) | (gt_s[i] == 2)), gt_r[i], include_outdoor=True, scale_m=scale_m
                )
                p_px_pr_a, p_m_pr_a, pts_pr_a, poly_pr_a = extract_envelope_polygon(
                    ((p_s[i] == 1) | (p_s[i] == 2)), p_r[i], include_outdoor=True, scale_m=scale_m
                )

                # 2. Config B: WITHOUT Outdoor (Indoor only)
                p_px_gt_b, p_m_gt_b, pts_gt_b, poly_gt_b = extract_envelope_polygon(
                    ((gt_s[i] == 1) | (gt_s[i] == 2)), gt_r[i], include_outdoor=False, scale_m=scale_m
                )
                p_px_pr_b, p_m_pr_b, pts_pr_b, poly_pr_b = extract_envelope_polygon(
                    ((p_s[i] == 1) | (p_s[i] == 2)), p_r[i], include_outdoor=False, scale_m=scale_m
                )

                # Footprint IoU A
                inter_a = (poly_gt_a.intersection(poly_pr_a).area) if (poly_gt_a.is_valid and poly_pr_a.is_valid) else 0.0
                union_a = (poly_gt_a.union(poly_pr_a).area) if (poly_gt_a.is_valid and poly_pr_a.is_valid) else 1.0
                iou_a = (inter_a / union_a) * 100.0 if union_a > 0 else 0.0

                # Footprint IoU B
                inter_b = (poly_gt_b.intersection(poly_pr_b).area) if (poly_gt_b.is_valid and poly_pr_b.is_valid) else 0.0
                union_b = (poly_gt_b.union(poly_pr_b).area) if (poly_gt_b.is_valid and poly_pr_b.is_valid) else 1.0
                iou_b = (inter_b / union_b) * 100.0 if union_b > 0 else 0.0

                # Signed & Abs relative errors
                signed_err_a = (p_px_pr_a - p_px_gt_a) / max(p_px_gt_a, 1e-5) * 100.0
                abs_err_a = abs(p_px_pr_a - p_px_gt_a) / max(p_px_gt_a, 1e-5) * 100.0

                signed_err_b = (p_px_pr_b - p_px_gt_b) / max(p_px_gt_b, 1e-5) * 100.0
                abs_err_b = abs(p_px_pr_b - p_px_gt_b) / max(p_px_gt_b, 1e-5) * 100.0

                # Meter version signed & abs
                abs_err_m_a = abs(p_m_pr_a - p_m_gt_a) / max(p_m_gt_a, 1e-5) * 100.0
                abs_err_m_b = abs(p_m_pr_b - p_m_gt_b) / max(p_m_gt_b, 1e-5) * 100.0

                has_outdoor_gt = bool(np.any(gt_r[i] == 6))
                has_outdoor_pr = bool(np.any(p_r[i] == 6))
                outdoor_px_gt = int(np.sum(gt_r[i] == 6))

                plans_results.append({
                    "plan_idx": plan_idx,
                    "has_outdoor_gt": has_outdoor_gt,
                    "has_outdoor_pr": has_outdoor_pr,
                    "outdoor_px_gt": outdoor_px_gt,
                    # Config A (With Outdoor)
                    "iou_a": iou_a,
                    "p_gt_a": p_px_gt_a,
                    "p_pr_a": p_px_pr_a,
                    "p_m_gt_a": p_m_gt_a,
                    "p_m_pr_a": p_m_pr_a,
                    "signed_err_a": signed_err_a,
                    "abs_err_a": abs_err_a,
                    "pts_gt_a": pts_gt_a,
                    "pts_pr_a": pts_pr_a,
                    # Config B (Without Outdoor)
                    "iou_b": iou_b,
                    "p_gt_b": p_px_gt_b,
                    "p_pr_b": p_px_pr_b,
                    "p_m_gt_b": p_m_gt_b,
                    "p_m_pr_b": p_m_pr_b,
                    "signed_err_b": signed_err_b,
                    "abs_err_b": abs_err_b,
                    "pts_gt_b": pts_gt_b,
                    "pts_pr_b": pts_pr_b,
                    # Data tensors for visual overlay
                    "gt_s": gt_s[i],
                    "gt_r": gt_r[i],
                    "p_s": p_s[i],
                    "p_r": p_r[i]
                })

                plan_idx += 1

            if (b_idx + 1) % 25 == 0:
                print(f"  Evaluated {plan_idx}/{num_plans} floorplans...")

    # Filter plans where GT has valid envelope
    valid_a = [r for r in plans_results if r["p_gt_a"] > 50]
    valid_b = [r for r in plans_results if r["p_gt_b"] > 50]

    print("\n" + "=" * 90)
    print("RESULTS: WITH OUTDOOR (Config A) vs WITHOUT OUTDOOR (Config B)")
    print("=" * 90)

    mean_iou_a = np.mean([r["iou_a"] for r in valid_a])
    mean_abs_err_a = np.mean([r["abs_err_a"] for r in valid_a])
    mean_signed_err_a = np.mean([r["signed_err_a"] for r in valid_a])
    mean_pts_gt_a = np.mean([r["pts_gt_a"] for r in valid_a])
    mean_pts_pr_a = np.mean([r["pts_pr_a"] for r in valid_a])

    mean_iou_b = np.mean([r["iou_b"] for r in valid_b])
    mean_abs_err_b = np.mean([r["abs_err_b"] for r in valid_b])
    mean_signed_err_b = np.mean([r["signed_err_b"] for r in valid_b])
    mean_pts_gt_b = np.mean([r["pts_gt_b"] for r in valid_b])
    mean_pts_pr_b = np.mean([r["pts_pr_b"] for r in valid_b])

    print(f"1. Configuration A (WITH Outdoor Class 6):")
    print(f"   • Footprint IoU:              {mean_iou_a:.2f}%")
    print(f"   • Abs Relative Perimeter Error: {mean_abs_err_a:.2f}%")
    print(f"   • Signed Perimeter Error:       {mean_signed_err_a:+.2f}%")
    print(f"   • Polygon Vertices:             GT={mean_pts_gt_a:.1f} vs Pred={mean_pts_pr_a:.1f} (Ratio: {mean_pts_pr_a/max(mean_pts_gt_a,1):.2f}x)")

    print(f"\n2. Configuration B (WITHOUT Outdoor Class 6 - Indoor Heated Envelope Only):")
    print(f"   • Footprint IoU:              {mean_iou_b:.2f}%")
    print(f"   • Abs Relative Perimeter Error: {mean_abs_err_b:.2f}%")
    print(f"   • Signed Perimeter Error:       {mean_signed_err_b:+.2f}%")
    print(f"   • Polygon Vertices:             GT={mean_pts_gt_b:.1f} vs Pred={mean_pts_pr_b:.1f} (Ratio: {mean_pts_pr_b/max(mean_pts_gt_b,1):.2f}x)")

    delta_err = mean_abs_err_b - mean_abs_err_a
    delta_iou = mean_iou_b - mean_iou_a
    print(f"\n3. Impact of Excluding Outdoor:")
    print(f"   • Delta Abs Perimeter Error:   {delta_err:+.2f}%")
    print(f"   • Delta Footprint IoU:         {delta_iou:+.2f}%")

    # Analyze Plans with Heavy Outdoor Presence (GT outdoor > 5% of pixels)
    heavy_outdoor = [r for r in valid_a if r["outdoor_px_gt"] > (512 * 512 * 0.05)]
    print(f"\n4. Sub-Cohort with Significant Outdoor Spaces (N = {len(heavy_outdoor)} plans):")
    if heavy_outdoor:
        ho_err_a = np.mean([r["abs_err_a"] for r in heavy_outdoor])
        ho_err_b = np.mean([r["abs_err_b"] for r in heavy_outdoor])
        ho_iou_a = np.mean([r["iou_a"] for r in heavy_outdoor])
        ho_iou_b = np.mean([r["iou_b"] for r in heavy_outdoor])
        print(f"   • With Outdoor:    Abs Err = {ho_err_a:.2f}%, IoU = {ho_iou_a:.2f}%")
        print(f"   • Without Outdoor: Abs Err = {ho_err_b:.2f}%, IoU = {ho_iou_b:.2f}%")

    # ----------------------------------------------------
    # TOP-10 WORST PLANS ANALYSIS & VISUALIZATION
    # ----------------------------------------------------
    worst_10 = sorted(valid_a, key=lambda r: r["abs_err_a"], reverse=True)[:10]
    print("\n" + "=" * 90)
    print("TOP-10 WORST PLANS BY PERIMETER ERROR (Config A):")
    print("=" * 90)
    print(f"{'Rank':<5} | {'Plan Idx':<8} | {'Abs Err %':<10} | {'Signed %':<10} | {'IoU %':<8} | {'GT P(m)':<8} | {'Pred P(m)':<10} | {'Outdoor GT px':<12}")
    print("-" * 90)

    out_vis_dir = PROJECT_ROOT / "outputs/perimeter_worst_10"
    out_vis_dir.mkdir(parents=True, exist_ok=True)

    for rank, w in enumerate(worst_10):
        print(f"#{rank+1:<4} | {w['plan_idx']:<8} | {w['abs_err_a']:<10.2f} | {w['signed_err_a']:<+10.2f} | {w['iou_a']:<8.2f} | {w['p_m_gt_a']:<8.1f} | {w['p_m_pr_a']:<10.1f} | {w['outdoor_px_gt']:<12}")

        # Render 3-panel comparison image:
        # Panel 1: Ground Truth (Walls grey, Rooms colored, Outdoor highlighted in Gold)
        # Panel 2: Prediction (Walls grey, Rooms colored, Outdoor in Gold)
        # Panel 3: Difference Overlay (GT Envelope contour green, Pred Envelope contour red)
        canvas = Image.new("RGB", (512 * 3, 512), (255, 255, 255))
        
        # Color palettes
        gt_img = np.zeros((512, 512, 3), dtype=np.uint8) + 245
        pr_img = np.zeros((512, 512, 3), dtype=np.uint8) + 245
        diff_img = np.zeros((512, 512, 3), dtype=np.uint8) + 250

        # Draw GT rooms
        gt_r = w["gt_r"]
        gt_s = w["gt_s"]
        gt_img[gt_r > 0] = [180, 210, 240]  # Light blue for indoor rooms
        gt_img[gt_r == 6] = [255, 215, 0]   # Gold for Outdoor
        gt_img[(gt_s == 1) | (gt_s == 2)] = [40, 40, 40]  # Dark grey for walls

        # Draw Pred rooms
        p_r = w["p_r"]
        p_s = w["p_s"]
        pr_img[p_r > 0] = [180, 210, 240]
        pr_img[p_r == 6] = [255, 215, 0]
        pr_img[(p_s == 1) | (p_s == 2)] = [40, 40, 40]

        # Draw Envelope Contours on diff_img
        # GT contour in Green
        raw_gt = ((gt_s == 1) | (gt_s == 2) | (gt_r > 0)).astype(np.uint8)
        cnts_g, _ = cv2.findContours(binary_fill_holes(raw_gt).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_g:
            cv2.drawContours(diff_img, [max(cnts_g, key=cv2.contourArea)], -1, (34, 197, 94), 3)

        # Pred contour in Red
        raw_pr = ((p_s == 1) | (p_s == 2) | (p_r > 0)).astype(np.uint8)
        cnts_p, _ = cv2.findContours(binary_fill_holes(raw_pr).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_p:
            cv2.drawContours(diff_img, [max(cnts_p, key=cv2.contourArea)], -1, (239, 68, 68), 2)

        # Highlight Outdoor mismatch in Yellow on diff
        mismatch_out = ((gt_r == 6) ^ (p_r == 6))
        diff_img[mismatch_out] = [234, 179, 8]

        canvas.paste(Image.fromarray(gt_img), (0, 0))
        canvas.paste(Image.fromarray(pr_img), (512, 0))
        canvas.paste(Image.fromarray(diff_img), (1024, 0))

        # Add Titles
        draw = ImageDraw.Draw(canvas)
        draw.text((15, 10), f"Ground Truth (Plan #{w['plan_idx']}) - Gold: Outdoor", fill=(20, 20, 20))
        draw.text((512 + 15, 10), f"Prediction - Abs Err: {w['abs_err_a']:.1f}% (Signed: {w['signed_err_a']:+.1f}%)", fill=(20, 20, 20))
        draw.text((1024 + 15, 10), f"Envelopes: GT=Green, Pred=Red | Yellow=Outdoor Mismatch", fill=(20, 20, 20))

        out_path = out_vis_dir / f"worst_{rank+1:02d}_plan_{w['plan_idx']:03d}.png"
        canvas.save(out_path)

    print(f"\n[+] Saved top-10 worst plan visualization overlays to: {out_vis_dir}")

    # Save summary json
    out_json = PROJECT_ROOT / "outputs/perimeter_outdoor_hypothesis_analysis.json"
    clean_summary = {
        "num_valid_plans": len(valid_a),
        "config_a_with_outdoor": {
            "mean_footprint_iou_pct": float(mean_iou_a),
            "mean_abs_perimeter_error_pct": float(mean_abs_err_a),
            "mean_signed_perimeter_error_pct": float(mean_signed_err_a),
            "mean_gt_vertices": float(mean_pts_gt_a),
            "mean_pred_vertices": float(mean_pts_pr_a)
        },
        "config_b_without_outdoor": {
            "mean_footprint_iou_pct": float(mean_iou_b),
            "mean_abs_perimeter_error_pct": float(mean_abs_err_b),
            "mean_signed_perimeter_error_pct": float(mean_signed_err_b),
            "mean_gt_vertices": float(mean_pts_gt_b),
            "mean_pred_vertices": float(mean_pts_pr_b)
        },
        "top_10_worst_plans": [
            {
                "rank": r + 1,
                "plan_idx": int(w["plan_idx"]),
                "abs_err_pct": float(w["abs_err_a"]),
                "signed_err_pct": float(w["signed_err_a"]),
                "footprint_iou_pct": float(w["iou_a"]),
                "p_m_gt": float(w["p_m_gt_a"]),
                "p_m_pred": float(w["p_m_pr_a"]),
                "outdoor_px_gt": int(w["outdoor_px_gt"]),
                "overlay_file": f"worst_{r+1:02d}_plan_{w['plan_idx']:03d}.png"
            }
            for r, w in enumerate(worst_10)
        ]
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(clean_summary, f, indent=2)
    print(f"[+] Summary json saved to: {out_json}")

if __name__ == "__main__":
    run_experiment()
