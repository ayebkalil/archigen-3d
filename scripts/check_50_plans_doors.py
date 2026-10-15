from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np

root = Path("data/CubiCasa5k/cubicasa5k/cubicasa5k")
with open(root / "train.txt", "r") as f:
    samples = [line.strip().lstrip("/") for line in f if line.strip()][:50]

print(f"Inspecting 50 floorplans for Door & Window annotations:")
zero_door_count = 0
zero_window_count = 0
total_plans = 0

door_classes_seen = set()
door_ids_seen = set()

results = []

for idx, s_rel in enumerate(samples):
    folder = root / s_rel
    svg_p = folder / "model.svg"
    img_p = folder / "F1_scaled.png"
    if not svg_p.exists() or not img_p.exists():
        continue
    
    total_plans += 1
    tree = ET.parse(svg_p)
    r = tree.getroot()
    
    # Check all g elements
    num_door_groups = 0
    num_door_polys = 0
    num_window_groups = 0
    num_window_polys = 0
    
    for g in r.findall(".//{http://www.w3.org/2000/svg}g"):
        cls = g.attrib.get("class", "")
        gid = g.attrib.get("id", "")
        
        is_door = (gid == "Door") or ("Door" in cls)
        is_win = (gid == "Window") or ("Window" in cls)
        
        if is_door:
            num_door_groups += 1
            door_classes_seen.add(cls)
            door_ids_seen.add(gid)
            polys = g.findall("{http://www.w3.org/2000/svg}polygon")
            num_door_polys += len(polys)
            
        if is_win:
            num_window_groups += 1
            polys = g.findall("{http://www.w3.org/2000/svg}polygon")
            num_window_polys += len(polys)
            
    results.append({
        "sample": s_rel,
        "door_groups": num_door_groups,
        "door_polys": num_door_polys,
        "win_groups": num_window_groups,
        "win_polys": num_window_polys
    })
    if num_door_groups == 0 or num_door_polys == 0:
        zero_door_count += 1
    if num_window_groups == 0 or num_window_polys == 0:
        zero_window_count += 1

print(f"Checked {total_plans} floorplans:")
print(f"  Plans with ZERO door polygons: {zero_door_count} / {total_plans}")
print(f"  Plans with ZERO window polygons: {zero_window_count} / {total_plans}")
print(f"  Door classes seen: {door_classes_seen}")
print(f"  Door IDs seen: {door_ids_seen}")

# Print first 20 in detail
for r in results[:20]:
    print(f"Sample {r['sample'][-10:]} | Door Groups: {r['door_groups']:2d}, Door Polys: {r['door_polys']:2d} | Win Groups: {r['win_groups']:2d}, Win Polys: {r['win_polys']:2d}")
