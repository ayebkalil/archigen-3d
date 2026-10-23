import json
import re
from pathlib import Path
from PIL import Image

p_dir = Path("data/real_external_plans/fresh_defense_eval")
manifest_p = p_dir / "fresh_defense_manifest.json"
with open(manifest_p, "r", encoding="utf-8") as f:
    manifest = json.load(f)

for item in manifest:
    old_p = p_dir / item["filename"]
    if old_p.exists():
        im = Image.open(old_p)
        ext = ".jpg" if im.format == "JPEG" else ".png"
        clean_title = re.sub(r'[^a-zA-Z0-9_\-]', '', item["original_title"].replace("File:", "").replace(" ", "_"))[:35]
        new_name = f"{item['id']}_{clean_title}{ext}"
        new_p = p_dir / new_name
        im.close()
        old_p.rename(new_p)
        item["filename"] = new_name

with open(manifest_p, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("Cleaned filenames and updated manifest:")
for item in manifest:
    print(f" - [{item['id']}] {item['filename']} ({item['dimensions']}, {item['license']})")
