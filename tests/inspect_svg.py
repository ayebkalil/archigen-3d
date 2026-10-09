import xml.etree.ElementTree as ET
from pathlib import Path

svg_paths = list(Path("data/CubiCasa5k").rglob("model.svg"))[:3]
for p in svg_paths:
    print(f"=== File: {p} ===")
    tree = ET.parse(p)
    root = tree.getroot()
    classes = set()
    for el in root.iter():
        c = el.attrib.get("class")
        tag = el.tag.split("}")[-1]
        if c:
            classes.add(f"{tag}: {c}")
    for c in sorted(list(classes)):
        print(" ", c)
