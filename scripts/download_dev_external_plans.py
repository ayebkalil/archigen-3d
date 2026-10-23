"""
Download Script for 20 Diverse External Floorplans (Development Set).
Builds a fresh, independent dev set (CAD blueprints, architectural drawings,
historic blueprints, residential layouts) outside CubiCasa5K and strictly
disjoint from the 12 locked defense evaluation plans.
"""

import urllib.request
import urllib.parse
import json
import hashlib
from pathlib import Path
from PIL import Image
import io

OUT_DIR = Path("data/dev_external_plans")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "ArchiGen3D-Research/1.0 (academic floorplan generalization research; contact@archigen.org)"
}

SEARCH_TERMS = [
    "architectural blueprint house floor plan",
    "CAD floor plan house",
    "architectural drawing residential floor plan",
    "floor plan drawing residence blueprint",
    "measured drawing floor plan HABS"
]

def download_dev_set(target_count: int = 20):
    all_titles = []
    seen = set()

    # Load 12 locked plans filenames/hashes to ensure ZERO overlap
    locked_manifest = Path("data/real_external_plans/fresh_defense_eval/fresh_defense_manifest.json")
    locked_hashes = set()
    if locked_manifest.exists():
        with open(locked_manifest, "r", encoding="utf-8") as f:
            for item in json.load(f):
                locked_hashes.add(item.get("sha256"))

    print(f"Searching Wikimedia Commons for diverse floorplans across {len(SEARCH_TERMS)} categories...")
    for term in SEARCH_TERMS:
        url = f"https://commons.wikimedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(term)}&srnamespace=6&srlimit=25&format=json"
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for item in data.get("query", {}).get("search", []):
                    t = item.get("title", "")
                    tl = t.lower()
                    if (tl.endswith(".jpg") or tl.endswith(".png") or tl.endswith(".jpeg")) and not tl.endswith(".svg"):
                        if t not in seen:
                            seen.add(t)
                            all_titles.append(t)
        except Exception as e:
            print(f"Search error for {term}: {e}")

    print(f"Found {len(all_titles)} candidate images. Downloading up to {target_count} diverse plans...")

    saved_metadata = []
    for idx, title in enumerate(all_titles):
        if len(saved_metadata) >= target_count:
            break

        info_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.parse.quote(title)}&prop=imageinfo&iiprop=url|size|extmetadata&format=json"
        req = urllib.request.Request(info_url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                info_data = json.loads(resp.read().decode("utf-8"))
                pages = info_data.get("query", {}).get("pages", {})
                for pid, page in pages.items():
                    ii = page.get("imageinfo", [{}])[0]
                    img_url = ii.get("url")
                    if not img_url:
                        continue

                    w, h = ii.get("width", 0), ii.get("height", 0)
                    if w < 500 or h < 500:
                        continue

                    meta = ii.get("extmetadata", {})
                    license_name = meta.get("LicenseShortName", {}).get("value", "Public Domain")

                    # Download image bytes
                    img_req = urllib.request.Request(img_url, headers=HEADERS)
                    with urllib.request.urlopen(img_req, timeout=20) as img_resp:
                        raw_bytes = img_resp.read()

                    sha256 = hashlib.sha256(raw_bytes).hexdigest()
                    if sha256 in locked_hashes:
                        print(f"Skipping {title} (already in locked defense set)")
                        continue

                    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    # Check aspect ratio isn't extremely elongated (e.g. panoramic strip)
                    aspect = max(pil_img.width / pil_img.height, pil_img.height / pil_img.width)
                    if aspect > 3.0:
                        continue

                    plan_idx = len(saved_metadata) + 1
                    clean_title = "".join(c if c.isalnum() or c in "._-" else "_" for c in title.replace("File:", ""))[:30]
                    ext = ".jpg" if ".jpg" in img_url.lower() else ".png"
                    out_fn = f"dev_{plan_idx:02d}_{clean_title}{ext}"
                    out_path = OUT_DIR / out_fn

                    # Save preserving high resolution (max 1500)
                    pil_img.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
                    pil_img.save(out_path)

                    record = {
                        "id": f"dev_{plan_idx:02d}",
                        "filename": out_fn,
                        "title": title,
                        "license": license_name,
                        "dimensions": [pil_img.width, pil_img.height],
                        "sha256": sha256,
                        "source_url": img_url
                    }
                    saved_metadata.append(record)
                    print(f"  [+] [{plan_idx}/{target_count}] Saved {out_fn} ({pil_img.width}x{pil_img.height})")

        except Exception as e:
            print(f"  [!] Failed downloading {title}: {e}")

    manifest_p = OUT_DIR / "dev_manifest.json"
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(saved_metadata, f, indent=2)

    print(f"\nSuccessfully collected {len(saved_metadata)} plans in {OUT_DIR}")
    print(f"Dev manifest saved to {manifest_p}")

if __name__ == "__main__":
    download_dev_set(20)
