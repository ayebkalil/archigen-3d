import urllib.request
from pathlib import Path

base = "https://raw.githubusercontent.com/CubiCasa/CubiCasa5k/master/floortrans/loaders/"
for fname in ["svg_loader.py", "svg_utils.py", "house.py"]:
    url = base + fname
    print(f"Downloading {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode('utf-8')
            out_p = Path("scripts") / f"cubicasa_{fname}"
            out_p.write_text(content, encoding='utf-8')
            print(f"  Saved to {out_p} ({len(content)} chars)")
    except Exception as e:
        print(f"  Error: {e}")
