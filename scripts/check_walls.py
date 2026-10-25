import os, sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np, glob
from src.floorplan.vectorizer import FloorplanVectorizer
f = glob.glob('data/CubiCasa5k/cached_512/val/*.npz')[0]
d = np.load(f)
v = FloorplanVectorizer()
layout = v.vectorize(d['mask'], room_mask=d['room_mask'], struct_mask=d['struct_mask'])
print('Walls in layout for Plan 0:')
for w in layout['walls']:
    pts = w['polygon_meters']
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    print(f"  {w['id']}: is_ext={w['is_exterior']} area={w['area_m2']} num_pts={len(pts)} x:[{min(xs):.2f}, {max(xs):.2f}], y:[{min(ys):.2f}, {max(ys):.2f}]")
