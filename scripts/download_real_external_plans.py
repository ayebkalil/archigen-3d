import urllib.request
import urllib.parse
import json
from pathlib import Path
from PIL import Image
import io

OUT_DIR = Path("data/real_external_plans")
OUT_DIR.mkdir(parents=True, exist_ok=True)

headers = {
    "User-Agent": "ArchiGen3D-Research/1.0 (academic architectural evaluation; contact@archigen.org)"
}

search_terms = [
    "house floor plan",
    "villa floor plan",
    "residence floor plan",
    "HABS floor plan house"
]

all_titles = []
seen_titles = set()

for term in search_terms:
    url = f"https://commons.wikimedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(term)}&srnamespace=6&srlimit=15&format=json"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for item in data.get("query", {}).get("search", []):
                t = item.get("title", "")
                # Only bitmaps (jpg/png), not svg or pdf
                t_lower = t.lower()
                if (t_lower.endswith(".jpg") or t_lower.endswith(".png")) and not t_lower.endswith(".svg"):
                    if t not in seen_titles:
                        seen_titles.add(t)
                        all_titles.append(t)
    except Exception as e:
        print(f"Search failed for {term}: {e}")

print(f"Found {len(all_titles)} candidate files.")

saved_metadata = []
for idx, title in enumerate(all_titles, 1):
    if len(saved_metadata) >= 12:
        break

    info_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.parse.quote(title)}&prop=imageinfo&iiprop=url|size|extmetadata&format=json"
    req = urllib.request.Request(info_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            info_data = json.loads(resp.read().decode("utf-8"))
            pages = info_data.get("query", {}).get("pages", {})
            for pid, page in pages.items():
                ii = page.get("imageinfo", [{}])[0]
                img_url = ii.get("url")
                if not img_url:
                    continue

                w = ii.get("width", 0)
                h = ii.get("height", 0)
                if w < 400 or h < 400:
                    continue

                meta = ii.get("extmetadata", {})
                desc = meta.get("ImageDescription", {}).get("value", title)
                license_name = meta.get("LicenseShortName", {}).get("value", "Public Domain")
                credit = meta.get("Credit", {}).get("value", "Wikimedia Commons")

                # Clean title for filename
                clean_title = "".join(c if c.isalnum() or c in "._-" else "_" for c in title.replace("File:", ""))
                ext = ".jpg" if ".jpg" in img_url.lower() else ".png"
                out_filename = f"real_{len(saved_metadata)+1:02d}_{clean_title[:32]}{ext}"
                out_path = OUT_DIR / out_filename

                print(f"Downloading [{len(saved_metadata)+1}/12]: {title} ({w}x{h})...")
                img_req = urllib.request.Request(img_url, headers=headers)
                with urllib.request.urlopen(img_req, timeout=20) as img_resp:
                    img_bytes = img_resp.read()
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    # Resize max dimension to 1024 for reasonable storage and clean processing
                    pil_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    pil_img.save(out_path)

                saved_metadata.append({
                    "id": f"real_{len(saved_metadata)+1:02d}",
                    "filename": out_filename,
                    "title": title,
                    "description": desc[:150],
                    "license": license_name,
                    "credit": credit[:150],
                    "source_url": img_url,
                    "downloaded_size": [pil_img.width, pil_img.height],
                    "original_size": [w, h]
                })
                print(f"  [+] Saved {out_filename} ({pil_img.width}x{pil_img.height})")
    except Exception as e:
        print(f"  [!] Failed {title}: {e}")

manifest_file = OUT_DIR / "real_external_manifest.json"
with open(manifest_file, "w") as f:
    json.dump(saved_metadata, f, indent=2)

print(f"\n[+] Successfully saved {len(saved_metadata)} real external plans to {OUT_DIR}")
