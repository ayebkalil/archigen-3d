"""
download_heldout_modern_plans.py
Downloads and verifies 8 untouched modern residential listing floor plans
from Wikimedia Commons for the held-out evaluation set.
"""

import urllib.request
import urllib.parse
import json
import time
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data/real_external_plans/heldout_modern"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODERN_PLANS = [
    {
        "id": "modern_01",
        "title": "File:2 bhk Bungalow floor plan.jpg",
        "filename": "modern_01_2bhk_bungalow.jpg",
        "description": "Modern 2-BHK residential bungalow architectural floor plan with kitchen, living room, bedrooms, bathrooms.",
        "license": "CC BY-SA 4.0",
        "style": "Modern CAD Bungalow"
    },
    {
        "id": "modern_02",
        "title": "File:CAD of House Plan from Design to Build.jpg",
        "filename": "modern_02_cad_house_plan.jpg",
        "description": "High-resolution architectural CAD layout of modern residential single-family home.",
        "license": "CC BY-SA 4.0",
        "style": "Contemporary CAD Blueprint"
    },
    {
        "id": "modern_03",
        "title": "File:Floor plan 1395 Sq.ft.jpg",
        "filename": "modern_03_floor_plan_1395sqft.jpg",
        "description": "1395 Sq Ft contemporary residential listing floor plan with dining, lounge, bedrooms.",
        "license": "CC BY-SA 4.0",
        "style": "Modern Real Estate Listing"
    },
    {
        "id": "modern_04",
        "title": "File:30BY35 home plan by ER. AYUSH TOMAR 9997233546.jpg",
        "filename": "modern_04_30by35_residential.jpg",
        "description": "30x35 modern compact urban residential home layout.",
        "license": "CC0",
        "style": "Compact Urban Residence"
    },
    {
        "id": "modern_05",
        "title": "File:House Plan of savar DOHS.jpg",
        "filename": "modern_05_savar_dohs_residence.jpg",
        "description": "Modern suburban duplex residence floor plan layout.",
        "license": "CC BY-SA 4.0",
        "style": "Suburban Duplex"
    },
    {
        "id": "modern_06",
        "title": "File:Caroline Court Apartments Vancouver floor plan.png",
        "filename": "modern_06_caroline_court_apartment.png",
        "description": "Multi-room urban apartment listing floor plan with hallway, kitchen, living room.",
        "license": "Public domain",
        "style": "Urban Apartment Listing"
    },
    {
        "id": "modern_07",
        "title": "File:Swedish-apartment-design-with-open-floor-plan-1-554x368.jpg",
        "filename": "modern_07_swedish_open_plan_apartment.jpg",
        "description": "Contemporary Scandinavian open floor plan apartment layout.",
        "license": "CC BY 2.0",
        "style": "Scandinavian Open Concept"
    },
    {
        "id": "modern_08",
        "title": "File:Simon House Plan..JPG",
        "filename": "modern_08_simon_house_plan.jpg",
        "description": "Modern residential family house architectural floor plan.",
        "license": "Public domain",
        "style": "Modern Single Family"
    }
]

def main():
    print("=" * 80)
    print("DOWNLOADING HELD-OUT MODERN LISTING FLOORPLANS")
    print("=" * 80)

    # 1. Fetch direct URLs from Commons API
    titles = [p["title"] for p in MODERN_PLANS]
    t_param = "|".join(titles)
    api_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.parse.quote(t_param)}&prop=imageinfo&iiprop=url|size&format=json"

    req = urllib.request.Request(api_url, headers={"User-Agent": "ArchiGen3D-Heldout/1.0"})
    with urllib.request.urlopen(req) as resp:
        api_data = json.loads(resp.read().decode())

    title_to_url = {}
    title_to_size = {}
    for pid, page in api_data.get("query", {}).get("pages", {}).items():
        t = page.get("title", "")
        if "imageinfo" in page and page["imageinfo"]:
            ii = page["imageinfo"][0]
            title_to_url[t] = ii.get("url")
            title_to_size[t] = [ii.get("width"), ii.get("height")]

    manifest = []
    for item in MODERN_PLANS:
        t = item["title"]
        url = title_to_url.get(t)
        if not url:
            print(f"[!] Could not find URL for {t}")
            continue

        dest_path = OUT_DIR / item["filename"]
        if dest_path.exists() and dest_path.stat().st_size > 1000:
            print(f"Already downloaded {item['id']}: {item['filename']}")
        else:
            print(f"Downloading {item['id']}: {item['filename']} ...")
            downloaded = False
            for attempt in range(4):
                try:
                    time.sleep(1.5 + attempt * 1.5)
                    dl_req = urllib.request.Request(url, headers={"User-Agent": "ArchiGen3D-Heldout/1.0 (academic floorplan benchmark; contact ayebk@users.noreply.github.com)"})
                    with urllib.request.urlopen(dl_req) as r, open(dest_path, "wb") as f:
                        f.write(r.read())
                    downloaded = True
                    break
                except Exception as e:
                    print(f"  Attempt {attempt + 1} failed: {e}. Retrying...")
            if not downloaded:
                print(f"[!] Failed to download {item['id']}")
                continue

        # Verify image
        with Image.open(dest_path) as im:
            w, h = im.size
            fmt = im.format
            print(f"  Verified {dest_path.name}: {w}x{h} ({fmt})")

        manifest_entry = {
            "id": item["id"],
            "filename": item["filename"],
            "title": item["title"],
            "style": item["style"],
            "description": item["description"],
            "license": item["license"],
            "source_url": url,
            "resolution": [w, h]
        }
        manifest.append(manifest_entry)

    manifest_path = OUT_DIR / "heldout_modern_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[+] Successfully saved {len(manifest)} modern plans to: {OUT_DIR}")
    print(f"[+] Manifest saved to: {manifest_path}")

if __name__ == "__main__":
    main()
