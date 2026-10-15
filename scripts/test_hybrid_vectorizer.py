"""
Hybrid Room Extraction & Robust Evaluation Engine for FloorplanVectorizer.
Implements:
1. Primary Partition: Room head predictions masked by walls (walls split same-class blobs).
2. Same-Class Adjacency Merging: Touching components of the same room class merge into one.
3. Sub-Threshold Fragment Absorption: Regions below physical area threshold (1.5 m² for living, 0.8 m² for closet/bath)
   are absorbed into the neighbor sharing the longest boundary, or discarded if isolated noise.
4. Instance-Level Ground Truth: Parses SVG <g class="Space ..."> instances directly from model.svg.
5. Instance Metrics: Computes true Precision, Recall, and F1 at IoU >= 0.5 threshold.
6. Unified Footprint Metrics: Footprint IoU (predicted envelope vs GT envelope) and unified perimeter error.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset, ROOM_CLASSES, parse_svg_points

def parse_svg_room_instances(svg_path: Path, scale: float, pad_x: int, pad_y: int, canvas_size: int = 512):
    """Parses ground-truth room instances directly from SVG Space elements."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    instances = []

    for g in root.findall(".//{http://www.w3.org/2000/svg}g"):
        c = g.attrib.get("class", "")
        if c.startswith("Space"):
            # Identify class
            matched_class = None
            for r_name, r_id in ROOM_CLASSES.items():
                if r_name in c and r_id > 0:
                    matched_class = r_id
                    break
            if matched_class is None:
                continue

            # Each polygon or group of polygons in this Space element is one room instance
            poly_pts_list = []
            for poly in g.findall(".//{http://www.w3.org/2000/svg}polygon"):
                if "points" in poly.attrib:
                    pts = parse_svg_points(poly.attrib["points"], scale, pad_x, pad_y)
                    if len(pts) >= 3:
                        poly_pts_list.append(pts)

            if poly_pts_list:
                mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
                cv2.fillPoly(mask, poly_pts_list, 1)
                area_px = mask.sum()
                if area_px >= 100:  # Valid room instance
                    instances.append({
                        "class_id": matched_class,
                        "class_name": [k for k, v in ROOM_CLASSES.items() if v == matched_class][0],
                        "mask": mask,
                        "area_px": area_px
                    })
    return instances

def extract_hybrid_rooms(
    pred_room: np.ndarray,
    wall_binary: np.ndarray,
    scale_m_per_px: float = 0.030,
    min_area_habitable_m2: float = 1.5,
    min_area_service_m2: float = 0.8
):
    """
    Hybrid room partitioner:
    1. Masks room predictions by walls so walls split same-class rooms.
    2. Merges touching same-class fragments.
    3. Absorbs sub-threshold fragments into adjacent room with longest shared border.
    """
    h, w = pred_room.shape
    px_area_habitable = int(round(min_area_habitable_m2 / (scale_m_per_px ** 2)))
    px_area_service = int(round(min_area_service_m2 / (scale_m_per_px ** 2)))

    # Non-wall room mask
    clean_room = pred_room.copy()
    clean_room[wall_binary > 0] = 0

    # 1. Label connected components per class
    labeled_map = np.zeros((h, w), dtype=np.int32)
    region_info = {}
    current_label = 1

    for c_id in range(1, 12):
        c_bin = (clean_room == c_id).astype(np.uint8)
        num_c, labels_c = cv2.connectedComponents(c_bin)
        for lbl in range(1, num_c):
            r_mask = (labels_c == lbl)
            area = r_mask.sum()
            if area < 10:  # Skip single-pixel noise immediately
                continue
            labeled_map[r_mask] = current_label
            is_service = c_id in [4, 7, 8]  # Bath, Storage, Closet
            min_thresh = px_area_service if is_service else px_area_habitable
            region_info[current_label] = {
                "class_id": c_id,
                "area_px": area,
                "min_thresh": min_thresh
            }
            current_label += 1

    # 2. Absorb small sub-threshold fragments into longest-boundary neighbor
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    small_labels = [lbl for lbl, info in region_info.items() if info["area_px"] < info["min_thresh"]]

    for s_lbl in small_labels:
        if s_lbl not in region_info:
            continue
        s_mask = (labeled_map == s_lbl).astype(np.uint8)
        dilated = cv2.dilate(s_mask, kernel, iterations=1)
        border = dilated - s_mask

        # Find neighboring labels in the border
        neighbor_pixels = labeled_map[border > 0]
        neighbor_candidates = [n for n in neighbor_pixels if n > 0 and n != s_lbl]

        if neighbor_candidates:
            # Pick neighbor with highest boundary contact count
            from collections import Counter
            counts = Counter(neighbor_candidates)
            best_neighbor = counts.most_common(1)[0][0]

            # Absorb s_lbl into best_neighbor
            labeled_map[s_mask > 0] = best_neighbor
            region_info[best_neighbor]["area_px"] += region_info[s_lbl]["area_px"]
            del region_info[s_lbl]
        else:
            # Isolated small noise with no room neighbor: clear to 0
            labeled_map[s_mask > 0] = 0
            del region_info[s_lbl]

    # 3. Build final room instances
    final_rooms = []
    unique_labels = np.unique(labeled_map)
    for lbl in unique_labels:
        if lbl == 0:
            continue
        r_mask = (labeled_map == lbl).astype(np.uint8)
        c_id = region_info[lbl]["class_id"]
        area_px = r_mask.sum()
        area_m2 = area_px * (scale_m_per_px ** 2)

        cnts, _ = cv2.findContours(r_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest_c = max(cnts, key=cv2.contourArea)
            approx = cv2.approxPolyDP(largest_c, 2.5, True).reshape(-1, 2)
            if len(approx) >= 3:
                final_rooms.append({
                    "class_id": c_id,
                    "class_name": [k for k, v in ROOM_CLASSES.items() if v == c_id][0],
                    "mask": r_mask,
                    "area_m2": round(area_m2, 2),
                    "polygon_px": approx.tolist()
                })

    return final_rooms

def evaluate_instance_matching(pred_rooms, gt_instances, iou_thresh: float = 0.50):
    """Computes Instance Precision, Recall, and F1 at IoU >= 0.5 matching threshold."""
    if not gt_instances:
        return 0, len(pred_rooms), 0

    matched_gt = set()
    matched_pred = set()

    for p_idx, p in enumerate(pred_rooms):
        p_mask = p["mask"]
        p_cls = p["class_id"]
        best_iou = 0.0
        best_g_idx = None

        for g_idx, g in enumerate(gt_instances):
            if g_idx in matched_gt or g["class_id"] != p_cls:
                continue
            g_mask = g["mask"]
            inter = np.logical_and(p_mask, g_mask).sum()
            union = np.logical_or(p_mask, g_mask).sum()
            iou = inter / max(union, 1)
            if iou > best_iou:
                best_iou = iou
                best_g_idx = g_idx

        if best_iou >= iou_thresh and best_g_idx is not None:
            matched_gt.add(best_g_idx)
            matched_pred.add(p_idx)

    tp = len(matched_pred)
    fp = len(pred_rooms) - tp
    fn = len(gt_instances) - len(matched_gt)
    return tp, fp, fn

def benchmark_hybrid_vectorizer(num_plans: int = 50):
    dataset = CubiCasaDataset(split="val", img_size=512, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"\n[HYBRID ROOM BENCHMARK] Evaluating on {num_plans} validation floorplans...")

    # Load cached prediction or run model
    from src.floorplan.models import MultiTaskFloorplanUNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
    ckpt = torch.load("models/saved/floorplan_unetpp/multitask_resnet34_best_backup_25ep.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Metrics: Old (Baseline Vectorizer) vs New (Hybrid Vectorizer)
    old_errors = []
    new_errors = []
    old_signed = []
    new_signed = []

    old_tp, old_fp, old_fn = 0, 0, 0
    new_tp, new_fp, new_fn = 0, 0, 0

    footprint_ious = []
    perimeter_rel_errors = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= num_plans:
                break

            imgs = batch["image"].to(device)
            out = model(imgs)
            pred_s = torch.argmax(out["struct"], dim=1)[0].cpu().numpy()
            pred_r = torch.argmax(out["room"], dim=1)[0].cpu().numpy()

            gt_s = batch["struct_mask"][0].numpy()
            gt_r = batch["room_mask"][0].numpy()
            sample_id = batch["sample_id"][0]
            gt_scale = float(batch["gt_scale"][0])

            # 1. Parse true Ground-Truth instances from SVG
            svg_path = dataset.root_dir / sample_id / "model.svg"
            img_path = dataset.root_dir / sample_id / "F1_scaled.png"
            from PIL import Image
            with Image.open(img_path) as im:
                iw, ih = im.size
            scale = min(512 / iw, 512 / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            pad_x = (512 - nw) // 2
            pad_y = (512 - nh) // 2

            gt_instances = parse_svg_room_instances(svg_path, scale, pad_x, pad_y, 512)
            n_gt = len(gt_instances)

            # 2. Old Method (Raw class contours >= 100px)
            old_rooms = []
            for c in range(1, 12):
                c_mask = (pred_r == c).astype(np.uint8)
                cnts, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    if cv2.contourArea(cnt) >= 100:
                        m = np.zeros((512, 512), dtype=np.uint8)
                        cv2.drawContours(m, [cnt], -1, 1, -1)
                        old_rooms.append({"class_id": c, "mask": m})
            n_old = len(old_rooms)

            # 3. New Hybrid Method
            wall_bin = ((pred_s == 1) | (pred_s == 2)).astype(np.uint8)
            new_rooms = extract_hybrid_rooms(pred_r, wall_bin, scale_m_per_px=gt_scale)
            n_new = len(new_rooms)

            old_errors.append(abs(n_old - n_gt))
            new_errors.append(abs(n_new - n_gt))
            old_signed.append(n_old - n_gt)
            new_signed.append(n_new - n_gt)

            # Match instances
            o_tp, o_fp, o_fn = evaluate_instance_matching(old_rooms, gt_instances)
            n_tp, n_fp, n_fn = evaluate_instance_matching(new_rooms, gt_instances)
            old_tp += o_tp; old_fp += o_fp; old_fn += o_fn
            new_tp += n_tp; new_fp += n_fp; new_fn += n_fn

            # 4. Footprint IoU & Unified Perimeter
            pred_envelope = ((pred_s == 1) | (pred_s == 2) | (pred_r > 0)).astype(np.uint8)
            gt_envelope = ((gt_s == 1) | (gt_s == 2) | (gt_r > 0)).astype(np.uint8)
            
            # Fill holes in envelopes
            def fill_envelope(bin_img):
                cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filled = np.zeros_like(bin_img)
                if cnts:
                    cv2.drawContours(filled, cnts, -1, 1, -1)
                return filled, cnts

            pred_filled, p_cnts = fill_envelope(pred_envelope)
            gt_filled, g_cnts = fill_envelope(gt_envelope)

            ft_inter = np.logical_and(pred_filled, gt_filled).sum()
            ft_union = np.logical_or(pred_filled, gt_filled).sum()
            footprint_ious.append(ft_inter / max(ft_union, 1))

            if p_cnts and g_cnts:
                p_len = cv2.arcLength(max(p_cnts, key=cv2.contourArea), True)
                g_len = cv2.arcLength(max(g_cnts, key=cv2.contourArea), True)
                if g_len > 10:
                    perimeter_rel_errors.append(abs(p_len - g_len) / g_len)

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{num_plans} plans... (GT: {n_gt} rooms | Old: {n_old} | New: {n_new})")

    # Metrics Summary
    old_prec = old_tp / max(old_tp + old_fp, 1) * 100
    old_rec = old_tp / max(old_tp + old_fn, 1) * 100
    old_f1 = 2 * old_prec * old_rec / max(old_prec + old_rec, 1e-4)

    new_prec = new_tp / max(new_tp + new_fp, 1) * 100
    new_rec = new_tp / max(new_tp + new_fn, 1) * 100
    new_f1 = 2 * new_prec * new_rec / max(new_prec + new_rec, 1e-4)

    print("\n" + "=" * 70)
    print("      HYBRID ROOM EXTRACTION VS OLD BASELINE COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<35} {'Old Vectorizer':<16} {'New Hybrid Vectorizer':<20}")
    print("-" * 70)
    print(f"{'Room Count MAE':<35} {np.mean(old_errors):<16.2f} {np.mean(new_errors):<20.2f}")
    print(f"{'Mean Signed Error (Over-split)':<35} {np.mean(old_signed):<+16.2f} {np.mean(new_signed):<+20.2f}")
    print(f"{'Instance Precision (IoU >= 0.5)':<35} {old_prec:<16.2f}% {new_prec:<20.2f}%")
    print(f"{'Instance Recall    (IoU >= 0.5)':<35} {old_rec:<16.2f}% {new_rec:<20.2f}%")
    print(f"{'Instance F1-Score  (IoU >= 0.5)':<35} {old_f1:<16.2f}% {new_f1:<20.2f}%")
    print("-" * 70)
    print(f"\n--- UNIFIED ENVELOPE FOOTPRINT METRICS ---")
    print(f"Mean Building Footprint IoU:           {np.mean(footprint_ious)*100:.2f}%")
    print(f"Mean Unified Perimeter Relative Error: {np.mean(perimeter_rel_errors)*100:.2f}% (Median: {np.median(perimeter_rel_errors)*100:.2f}%)")
    print("=" * 70)

if __name__ == "__main__":
    benchmark_hybrid_vectorizer(num_plans=50)
