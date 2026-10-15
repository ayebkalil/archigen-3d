import xml.etree.ElementTree as ET
from pathlib import Path

root = Path('data/CubiCasa5k/cubicasa5k/cubicasa5k')
sample_p = root / 'high_quality_architectural/6044/model.svg'
tree = ET.parse(sample_p)
r = tree.getroot()

for g in r.findall('.//{http://www.w3.org/2000/svg}g'):
    if g.attrib.get('id') == 'Door' or g.attrib.get('class', '').startswith('Door'):
        print(f"Group id='{g.attrib.get('id')}', class='{g.attrib.get('class')}'")
        # Direct polygon children:
        direct = g.findall('{http://www.w3.org/2000/svg}polygon')
        print(f"  Direct polygon children count: {len(direct)}")
        for i, p in enumerate(direct):
            print(f"    Direct poly {i}: points='{p.attrib.get('points')}'")
        
        # Nested polygon children:
        all_polys = g.findall('.//{http://www.w3.org/2000/svg}polygon')
        print(f"  All (including nested) polygon children count: {len(all_polys)}")
        for i, p in enumerate(all_polys):
            if p not in direct:
                print(f"    Nested poly {i}: points='{p.attrib.get('points')}'")
        break
