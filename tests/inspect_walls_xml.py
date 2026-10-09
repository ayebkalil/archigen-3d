import xml.etree.ElementTree as ET
from pathlib import Path

p = Path("data/CubiCasa5k/cubicasa5k/cubicasa5k/colorful/10052/model.svg")
tree = ET.parse(p)
root = tree.getroot()

for elem in root.iter():
    c = elem.attrib.get("class", "")
    if "Wall" in c:
        print(f"Element {elem.tag} class='{c}'")
        for child in elem:
            c_c = child.attrib.get("class", "")
            print(f"   Child {child.tag} class='{c_c}'")
