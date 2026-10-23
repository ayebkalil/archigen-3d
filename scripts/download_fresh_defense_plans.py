"""
download_fresh_defense_plans.py
Downloads 12-14 pristine, untouched floor plans from open Wikimedia Commons / HABS
archives for the final defense evaluation.
Includes metadata, licenses, and sources.
"""

import urllib.request
import urllib.parse
import json
import time
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_ROOT / "data/real_external_plans/fresh_defense_eval"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "ArchiGen3D-Academic-Research/1.0 (khalil.ayeb@tekup.tn; contact@archigen3d.org)"

# Queries targeting residential floor plans
queries = [
    "floor plan apartment residential",
    "house floor plan architectural drawing",
    "residential floor plan ground floor",
    "villa floor plan architecture"
]

candidates = []

for q in queries:
    url = f"https://commons.wikimedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(q)}&srnamespace=6&format=json&srlimit=20"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        search_res = data.get("query", {}).get("search", [])
        for item in search_res:
            title = item["title"]
            # Exclude tif, pdf, video, non-plans
            if any(ext in title.lower() for ext in [".jpg", ".png", ".jpeg"]):
                if not any(bad in title.lower() for bad in ["elevation", "section", "view of", "portrait", "statue", "site plan"]):
                    if title not in [c["title"] for c in candidates]:
                        candidates.append({"title": title, "pageid": item["pageid"]})
    except Exception as e:
        print(f"Error querying {q}: {e}")
    time.sleep(0.5)

print(f"Discovered {len(candidates)} potential candidates on Wikimedia Commons.")

# Fetch imageinfo for selected diverse candidates
selected = []
target_count = 12

for c in candidates:
    if len(selected) >= target_count:
        break
    title = c["title"]
    info_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.parse.quote(title)}&prop=imageinfo&iiprop=url|size|extmetadata&format=json"
    req = urllib.request.Request(info_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        pages = data.get("query", {}).get("pages", {})
        for pid, page in pages.items():
            if "imageinfo" in page and len(page["imageinfo"]) > 0:
                info = page["imageinfo"][0]
                img_url = info.get("url")
                w = info.get("width", 0)
                h = info.get("height", 0)
                meta = info.get("extmetadata", {})
                license_name = meta.get("LicenseShortName", {}).get("value", "Public Domain / CC")
                
                # Check resolution criteria (at least 600px width/height, max 4000px)
                if 600 <= w <= 4000 and 600 <= h <= 4000:
                    clean_name = f"fresh_{len(selected)+1:02d}_" + title.replace("File:", "").replace(" ", "_")[:40] + Path(img_url).suffix.lower()
                    save_path = TARGET_DIR / clean_name
                    
                    # Download image
                    dl_req = urllib.request.Request(img_url, headers={"User-Agent": USER_AGENT})
                    with urllib.request.urlopen(dl_req) as dl_resp:
                        img_bytes = dl_resp.read()
                    
                    # Validate image with PIL
                    with open(save_path, "wb") as f:
                        f.write(img_bytes)
                    try:
                        with Image.open(save_path) as im:
                            im.verify()
                        
                        selected.append({
                            "id": f"fresh_{len(selected)+1:02d}",
                            "filename": clean_name,
                            "original_title": title,
                            "url": img_url,
                            "dimensions": [w, h],
                            "license": license_name,
                            "file_size_bytes": len(img_bytes),
                            "status": "LOCKED_FOR_DEFENSE_EVAL"
                        })
                        print(f"[{len(selected):02d}/{target_count}] Downloaded: {clean_name} ({w}x{h}, {license_name})")
                    except Exception:
                        save_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"Error fetching info for {title}: {e}")
    time.sleep(0.5)

manifest_file = TARGET_DIR / "fresh_defense_manifest.json"
with open(manifest_file, "w", encoding="utf-8") as f:
    json.dump(selected, f, indent=2)

print(f"\n[+] Successfully collected {len(selected)} pristine floor plans.")
print(f"[+] Manifest saved to: {manifest_file}")
