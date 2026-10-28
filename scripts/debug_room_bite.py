import sys
sys.path.insert(0, 'archigen-3d/backend')
sys.path.insert(0, 'archigen-3d')
import cv2
import numpy as np
from PIL import Image
from scipy.special import softmax
from app.floorplan_engine import FloorplanPipeline

pipeline = FloorplanPipeline.get_instance()
raw_img = Image.open('archigen-3d/frontend/public/sample_floorplan.png').convert('RGB')
iw, ih = raw_img.size
size = 512
scale = min(size / iw, size / ih)
nw, nh = int(iw * scale), int(ih * scale)
pad_x = (size - nw) // 2
pad_y = (size - nh) // 2
resized = raw_img.resize((nw, nh), Image.Resampling.BILINEAR)
canvas = Image.new('RGB', (size, size), (255, 255, 255))
canvas.paste(resized, (pad_x, pad_y))

arr = np.array(canvas, dtype=np.float32) / 255.0
arr = arr.transpose(2, 0, 1)
arr = (arr - pipeline.mean_np) / pipeline.std_np
tensor_np = np.expand_dims(arr, 0)

outs = pipeline.onnx_multitask.run(None, {'input': tensor_np})
prob_s = softmax(outs[0], axis=1)
prob_r = softmax(outs[1], axis=1)
struct_pred = np.argmax(prob_s, axis=1).squeeze(0).astype(np.uint8)
room_pred = np.argmax(prob_r, axis=1).squeeze(0).astype(np.uint8)

mask_4class = np.zeros_like(struct_pred)
mask_4class[(struct_pred == 1) | (struct_pred == 2)] = 1
mask_4class[struct_pred == 3] = 2
mask_4class[struct_pred == 4] = 3

# Let's inspect the bounding box of room_4:
scale_m_per_px = 0.053  # or whatever pipeline calculates
# Points: x in [1.38, 4.78], y in [13.39, 18.27]
px_min_x = int(1.38 / 0.053)
px_max_x = int(4.78 / 0.053)
px_min_y = int(13.39 / 0.053)
px_max_y = int(18.27 / 0.053)

# Let's test the difference between current envelope and watertight structural envelope
wall_binary = (mask_4class == 1).astype(np.uint8)
all_structure = ((mask_4class >= 1) & (mask_4class <= 3)).astype(np.uint8)

# Morphological close with larger kernel (e.g. 5x5 or 7x7) to seal exterior wall gaps
k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
sealed_structure = cv2.morphologyEx(all_structure, cv2.MORPH_CLOSE, k_close)

# Let's test vectorizer with sealed envelope
# First, update vectorizer._vectorize_rooms logic in Python to see room_4 points
# In step 3:
all_structure = ((mask_4class >= 1) & (mask_4class <= 3)).astype(np.uint8)
k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
sealed_structure = cv2.morphologyEx(all_structure, cv2.MORPH_CLOSE, k_close)
envelope_sealed = ((sealed_structure > 0) | (room_pred > 0)).astype(np.uint8)
cnts_sealed, _ = cv2.findContours(envelope_sealed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filled_sealed = np.zeros_like(envelope_sealed)
if cnts_sealed:
    cv2.drawContours(filled_sealed, cnts_sealed, -1, 1, -1)

# Now test unassigned absorption with filled_sealed
wall_clean = cv2.morphologyEx((mask_4class == 1).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
mask_interior = (filled_sealed > 0) & (wall_clean == 0)

# Also, in unassigned absorption: use iterative dilation until all unassigned interior pixels are absorbed!
labeled_map = np.zeros_like(room_pred, dtype=np.int32)
region_info = {}
current_label = 1
for c_id in range(1, 12):
    c_bin = ((room_pred == c_id) & (wall_clean == 0)).astype(np.uint8)
    num_c, labels_c = cv2.connectedComponents(c_bin)
    for lbl in range(1, num_c):
        r_mask = (labels_c == lbl)
        if r_mask.sum() < 10:
            continue
        labeled_map[r_mask] = current_label
        region_info[current_label] = {"class_id": c_id, "area_px": int(r_mask.sum())}
        current_label += 1

unassigned = mask_interior & (labeled_map == 0)
print("Initial unassigned interior pixels:", unassigned.sum())

# Iteratively absorb unassigned interior into touching neighbors (up to 8 iterations or until stable)
k_abs = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
for iter_idx in range(12):
    unassigned = mask_interior & (labeled_map == 0)
    if not unassigned.any():
        break
    num_u, labels_u = cv2.connectedComponents(unassigned.astype(np.uint8))
    for u_lbl in range(1, num_u):
        comp = (labels_u == u_lbl).astype(np.uint8)
        dilated = cv2.dilate(comp, k_abs, iterations=1)
        touching = labeled_map[(dilated > 0) & (labeled_map > 0)]
        if len(touching) > 0:
            from collections import Counter
            best = Counter(touching).most_common(1)[0][0]
            labeled_map[comp > 0] = best
            if best in region_info:
                region_info[best]["area_px"] += int(comp.sum())

unassigned_after = mask_interior & (labeled_map == 0)
print("Unassigned interior pixels after iterative absorption:", unassigned_after.sum())

# Check room_4 area and contour
# Find which label corresponds to bedroom in room_4 region
crop_lbls = labeled_map[px_min_y:px_max_y, px_min_x:px_max_x]
from collections import Counter
common_lbl = Counter(crop_lbls[crop_lbls > 0]).most_common(1)[0][0]
r4_mask = (labeled_map == common_lbl).astype(np.uint8)
# Fill holes in r4_mask
cnts_ext, _ = cv2.findContours(r4_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filled_r4 = np.zeros_like(r4_mask)
cv2.drawContours(filled_r4, cnts_ext, -1, 1, -1)

# Morphological close to smooth out jagged text artifacts
k_smooth = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed_r4 = cv2.morphologyEx(filled_r4, cv2.MORPH_CLOSE, k_smooth)

cnts_final, _ = cv2.findContours(closed_r4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(cnts_final, key=cv2.contourArea)
approx = cv2.approxPolyDP(largest, 2.5, True).reshape(-1, 2)
print("Smoothed filled room_4 points count:", len(approx))
print("Smoothed filled room_4 contour points:")
for pt in approx:
    print(f"  [{pt[0]*0.053:.2f}, {pt[1]*0.053:.2f}]")




