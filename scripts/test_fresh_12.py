import requests

with open('archigen-3d/frontend/public/sample_floorplan.png', 'rb') as f:
    b = f.read()

r = requests.post('http://localhost:8000/predict/floorplan', files={'file': ('plan.jpg', b, 'image/jpeg')})
print('Status:', r.status_code)
if r.status_code == 200:
    data = r.json()
    print('Rooms count:', data.get('num_rooms'))
    for rm in data.get('rooms', []):
        print(f"  {rm['id']}: type={rm['room_type']:20s} area={rm['surface_m2']:.2f} m2 num_pts={len(rm['polygon_points'])}")
else:
    print('Error:', r.text[:500])
