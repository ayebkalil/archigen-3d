import requests
import json
from pathlib import Path

url = 'http://localhost:8000/predict/floorplan'
img_path = r'C:/Users/ayebk/.gemini/antigravity/brain/387c7b87-863d-4c00-831b-93abdc64d4d4/.user_uploaded/media_1793110631118_19fdf9f9.png'

with open(img_path, 'rb') as f:
    r = requests.post(url, files={'file': ('plan.png', f, 'image/png')})

print('Status:', r.status_code)
if r.status_code == 200:
    data = r.json()
    rooms = data.get('rooms', [])
    print(f'Total rooms detected: {len(rooms)}')
    for rm in rooms:
        area = rm.get('area_sqm') or rm.get('area') or 0.0
        print(f"  Room {rm.get('id')}: type={rm.get('room_type'):12s} area={float(area):.2f} m2 is_fragment={rm.get('is_fragment')}")
else:
    print('Error:', r.text[:500])
