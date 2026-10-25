#!/usr/bin/env python3
"""
scripts/download_assets.py
--------------------------
Downloads curated CC0 3D furniture models from the official Kenney Furniture Kit
mirror (ETdoFresh/kenney.nl) into data/furniture_raw/.

License: CC0 1.0 Universal (Public Domain).
"""

import os
import sys
import time
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_RAW_URL = "https://raw.githubusercontent.com/ETdoFresh/kenney.nl/master/furniturekit_updated/Models/GLTF%20format/"

# 23 curated models covering Bedroom, Living, Dining, Office, Kitchen, Bathroom
CURATED_ASSETS = {
    # Bedroom
    "bed_double": "bedDouble.glb",
    "bed_single": "bedSingle.glb",
    "nightstand": "cabinetBedDrawerTable.glb",
    "wardrobe": "bookcaseClosedDoors.glb",
    # Living Room
    "sofa_3seater": "loungeSofaLong.glb",
    "armchair": "loungeDesignChair.glb",
    "coffee_table": "tableCoffee.glb",
    "tv_unit": "cabinetTelevision.glb",
    "rug_living": "rugRectangle.glb",
    "potted_plant": "pottedPlant.glb",
    # Dining Room
    "dining_table": "table.glb",
    "dining_chair": "chairModernCushion.glb",
    "sideboard": "bookcaseClosedWide.glb",
    # Office
    "desk": "desk.glb",
    "office_chair": "chairDesk.glb",
    "bookcase": "bookcaseOpen.glb",
    "floor_lamp": "lampRoundFloor.glb",
    # Kitchen
    "kitchen_counter": "kitchenCabinet.glb",
    "kitchen_sink_unit": "kitchenSink.glb",
    "refrigerator": "kitchenFridgeLarge.glb",
    # Bathroom
    "toilet": "toilet.glb",
    "bathroom_vanity": "bathroomSink.glb",
    "bathtub": "bathtub.glb",
}

def download_assets(target_dir: str = "data/furniture_raw") -> dict:
    os.makedirs(target_dir, exist_ok=True)
    results = {}
    
    logger.info(f"Downloading {len(CURATED_ASSETS)} curated assets to {target_dir}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ArchiGen3D-AssetFetcher/1.0"}
    
    for item_id, remote_file in CURATED_ASSETS.items():
        out_path = os.path.join(target_dir, remote_file)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            logger.info(f"  [CACHED] {item_id} -> {remote_file} ({os.path.getsize(out_path)} bytes)")
            results[item_id] = {"status": "cached", "path": out_path, "bytes": os.path.getsize(out_path)}
            continue
            
        url = REPO_RAW_URL + remote_file
        success = False
        last_err = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    content = resp.read()
                    if len(content) < 500:
                        raise ValueError(f"File too small ({len(content)} bytes)")
                    with open(out_path, "wb") as f:
                        f.write(content)
                    results[item_id] = {"status": "downloaded", "path": out_path, "bytes": len(content)}
                    logger.info(f"  [OK] {item_id} ({remote_file}, {len(content)} bytes)")
                    success = True
                    break
            except Exception as e:
                last_err = e
                time.sleep(1.0)
                
        if not success:
            logger.error(f"  [FAILED] {item_id} ({remote_file}): {last_err}")
            results[item_id] = {"status": "failed", "error": str(last_err)}
            
    return results

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "data", "furniture_raw")
    results = download_assets(raw_dir)
    failed = [k for k, v in results.items() if v.get("status") == "failed"]
    if failed:
        logger.error(f"Failed to download {len(failed)} assets: {failed}")
        sys.exit(1)
    else:
        logger.info(f"All {len(results)} assets ready in {raw_dir}")
