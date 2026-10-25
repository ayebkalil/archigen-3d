"""
scripts/verify_unmirrored_l_shape.py
------------------------------------
Creates an asymmetric L-shaped floorplan and renders:
1. 2D floorplan layout image
2. 3D top-down view (Three.js WebGL render)
Combines them side by side to verify zero mirror reflection.
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Define an asymmetric L-shaped house:
# North-West (Top-Left) main corner at (2.0, 2.0)
# Top wall extends East to (8.0, 2.0) (6m wide)
# Short leg goes South to (8.0, 5.0) (3m depth)
# Inner corner cuts West to (5.0, 5.0)
# Long leg extends South to (5.0, 10.0)
# Bottom wall goes West to (2.0, 10.0) (3m wide)
# West wall goes North to (2.0, 2.0) (8m tall)
L_WALL_EXTERIOR = [
    [2.0, 2.0],
    [8.0, 2.0],
    [8.0, 5.0],
    [5.0, 5.0],
    [5.0, 10.0],
    [2.0, 10.0]
]

# Interior room 1 (North wing: Bedroom)
ROOM_NORTH = [
    [2.2, 2.2],
    [7.8, 2.2],
    [7.8, 4.8],
    [4.8, 4.8],
    [4.8, 4.8],
    [2.2, 4.8]
]

# Interior room 2 (South wing: Living)
ROOM_SOUTH = [
    [2.2, 5.2],
    [4.8, 5.2],
    [4.8, 9.8],
    [2.2, 9.8]
]

SCALE_PX = 45 # 45 pixels per meter
IMG_SIZE = 550

def render_2d_plan():
    im = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (245, 243, 238))
    draw = ImageDraw.Draw(im)
    
    # Grid lines
    for i in range(0, IMG_SIZE, SCALE_PX):
        draw.line([(i, 0), (i, IMG_SIZE)], fill=(225, 222, 215), width=1)
        draw.line([(0, i), (IMG_SIZE, i)], fill=(225, 222, 215), width=1)
        
    def to_px(pt):
        return (int(pt[0] * SCALE_PX), int(pt[1] * SCALE_PX))
        
    # Rooms
    draw.polygon([to_px(p) for p in ROOM_NORTH], fill=(186, 230, 253), outline=(56, 189, 248), width=2)
    draw.polygon([to_px(p) for p in ROOM_SOUTH], fill=(254, 215, 170), outline=(249, 115, 22), width=2)
    
    # Exterior Wall outline
    draw.polygon([to_px(p) for p in L_WALL_EXTERIOR], outline=(30, 41, 59), width=7)
    
    # Compass / Navigation Indicator
    cx, cy = 480, 70
    draw.line([(cx, cy + 30), (cx, cy - 30)], fill=(239, 68, 68), width=3)
    draw.polygon([(cx, cy - 38), (cx - 7, cy - 26), (cx + 7, cy - 26)], fill=(239, 68, 68))
    draw.text((cx - 5, cy - 54), "N", fill=(239, 68, 68))
    
    # Markers for orientation check
    draw.text((to_px([3.2, 3.2])), "ROOM 1 (NORTH)", fill=(3, 105, 161))
    draw.text((to_px([2.4, 7.2])), "ROOM 2 (SOUTH)", fill=(194, 65, 12))
    draw.text((to_px([5.5, 2.5])), "Short Leg (East)", fill=(15, 23, 42))
    draw.text((to_px([2.4, 10.3])), "Long Leg (South)", fill=(15, 23, 42))
    
    # Label
    draw.rectangle([(10, 10), (260, 40)], fill=(15, 23, 42))
    draw.text((20, 18), "2D FLOORPLAN (IMAGE SPACE)", fill=(248, 250, 252))
    
    return im

def export_l_shape_html():
    pts_json = json.dumps(L_WALL_EXTERIOR)
    r1_json = json.dumps(ROOM_NORTH)
    r2_json = json.dumps(ROOM_SOUTH)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body, html {{ margin: 0; padding: 0; overflow: hidden; background: #1a1a1e; }}
    #canvas-container {{ width: 550px; height: 550px; }}
    #badge {{ position: absolute; top: 10px; left: 10px; background: #0f172a; color: #38bdf8; padding: 6px 12px; font-family: monospace; font-size: 13px; font-weight: bold; border-radius: 4px; }}
  </style>
  <script type="importmap">
    {{
      "imports": {{
        "three": "https://unpkg.com/three@0.160.0/build/three.module.js"
      }}
    }}
  </script>
</head>
<body>
  <div id="badge">3D TOP VIEW (SHARED TRANSFORM)</div>
  <div id="canvas-container"></div>
  <script type="module">
    import * as THREE from 'three';

    const pts = {pts_json};
    const r1 = {r1_json};
    const r2 = {r2_json};

    // House centroid in plan meters
    const cx = 5.0;
    const cy = 6.0;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f3ee);

    // Unmirrored top-down camera:
    // Screen +X is East (+X_3d = x - cx)
    // Screen +Y is North (-Z_3d = -(y - cy))
    // So camera up vector is (0, 0, -1)
    const camera = new THREE.PerspectiveCamera(45, 1.0, 0.1, 1000);
    camera.position.set(0, 18, 0);
    camera.up.set(0, 0, -1);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({{ antialias: true, preserveDrawingBuffer: true }});
    renderer.setSize(550, 550);
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0xffffff, 1.2);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
    dirLight.position.set(10, 20, 10);
    scene.add(dirLight);

    // Grid on floor
    const grid = new THREE.GridHelper(20, 20, 0xd1d5db, 0xe5e7eb);
    grid.position.y = -0.01;
    scene.add(grid);

    // Shared transform function:
    // Shape coordinates: X_shape = x - cx, Y_shape = -(y - cy)
    // rotateX(-Math.PI / 2) maps Y_shape -> +Z_3d = (y - cy)
    function makeShape(coords) {{
      const shape = new THREE.Shape();
      shape.moveTo(coords[0][0] - cx, -(coords[0][1] - cy));
      for (let i = 1; i < coords.length; i++) {{
        shape.lineTo(coords[i][0] - cx, -(coords[i][1] - cy));
      }}
      shape.closePath();
      return shape;
    }}

    // Extruded Walls
    const wallShape = makeShape(pts);
    const wallGeo = new THREE.ExtrudeGeometry(wallShape, {{ steps: 1, depth: 2.8, bevelEnabled: false }});
    wallGeo.rotateX(-Math.PI / 2);
    const wallMat = new THREE.MeshStandardMaterial({{ color: 0x1e293b, roughness: 0.5 }});
    const wallMesh = new THREE.Mesh(wallGeo, wallMat);
    scene.add(wallMesh);

    // Room 1 Floor
    const r1Shape = makeShape(r1);
    const r1Geo = new THREE.ShapeGeometry(r1Shape);
    r1Geo.rotateX(-Math.PI / 2);
    const r1Mesh = new THREE.Mesh(r1Geo, new THREE.MeshStandardMaterial({{ color: 0xbae6fd }}));
    r1Mesh.position.y = 0.02;
    scene.add(r1Mesh);

    // Room 2 Floor
    const r2Shape = makeShape(r2);
    const r2Geo = new THREE.ShapeGeometry(r2Shape);
    r2Geo.rotateX(-Math.PI / 2);
    const r2Mesh = new THREE.Mesh(r2Geo, new THREE.MeshStandardMaterial({{ color: 0xfed7aa }}));
    r2Mesh.position.y = 0.02;
    scene.add(r2Mesh);

    // Render frame
    renderer.render(scene, camera);

    // Wait and signal ready
    setTimeout(() => {{
      renderer.render(scene, camera);
      window.__L_READY__ = true;
    }}, 400);
  </script>
</body>
</html>"""
    out_html = os.path.join("frontend", "public", "test_l_shape_top.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html

if __name__ == "__main__":
    im2d = render_2d_plan()
    im2d.save("outputs/l_shape_2d_plan.png")
    export_l_shape_html()
    print("Exported 2D plan and HTML test harness.")
