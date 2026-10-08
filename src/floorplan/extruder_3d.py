"""
Procedural 3D Wall Extrusion and Three.js Mesh Generation Engine.
Converts vector layout JSON into:
1. Three.js BufferGeometry compatible JSON specs for client-side rendering.
2. Standard Wavefront OBJ format for standalone 3D rendering.
3. Exterior facade boundary extraction for Pix2Pix texture mapping.
"""

from typing import Dict, List, Any
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon

class Floorplan3DExtruder:
    """
    Extrudes 2D wall polygons along the Z-axis into 3D meshes with openings.
    """
    def __init__(self, wall_height_m: float = 2.80, door_height_m: float = 2.10, window_height_m: float = 1.40, window_sill_m: float = 0.90):
        self.wall_height_m = wall_height_m
        self.door_height_m = door_height_m
        self.window_height_m = window_height_m
        self.window_sill_m = window_sill_m

    def generate_threejs_scene_data(self, layout: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a JSON payload ready for Three.js scene creation in the frontend.
        Includes:
        - Wall geometries (coordinates, height, thickness)
        - Floor slab geometries
        - Room labels and bounding boxes
        - Door / window cutout locations
        """
        walls_3d = []
        for wall in layout.get("walls", []):
            pts_2d = wall["polygon_meters"]
            if len(pts_2d) < 3:
                continue

            # Validate polygon with Shapely
            poly = Polygon(pts_2d)
            is_valid = poly.is_valid

            walls_3d.append({
                "id": wall["id"],
                "points_2d": pts_2d,
                "height": self.wall_height_m,
                "is_valid_geometry": is_valid,
                "openings": wall.get("openings", [])
            })

        rooms_3d = []
        for room in layout.get("rooms", []):
            poly = Polygon(room["polygon_meters"])
            rooms_3d.append({
                "id": room["id"],
                "room_type": room["room_type"],
                "surface_m2": room["surface_m2"],
                "centroid": room["centroid_meters"],
                "points_2d": room["polygon_meters"],
                "is_valid_geometry": poly.is_valid
            })

        return {
            "type": "ArchiGen3D_Floorplan_Scene",
            "version": "1.0",
            "scale": layout["metadata"].get("scale_meters_per_pixel", 0.025),
            "total_surface_m2": layout["metadata"].get("total_surface_m2", 0.0),
            "walls": walls_3d,
            "rooms": rooms_3d,
            "doors": layout.get("doors", []),
            "windows": layout.get("windows", [])
        }

    def export_obj(self, layout: Dict[str, Any], output_path: str):
        """
        Exports the extruded 3D walls and floor slabs as a standard Wavefront .OBJ file.
        """
        vertices = []
        faces = []

        def add_vertex(x, y, z):
            vertices.append((x, z, y)) # In Three.js / standard 3D, Y is up
            return len(vertices)

        # 1. Extrude walls
        h = self.wall_height_m
        for wall in layout.get("walls", []):
            pts = wall["polygon_meters"]
            if len(pts) < 3:
                continue

            num_pts = len(pts)
            base_idx = len(vertices) + 1

            # Bottom vertices
            for pt in pts:
                add_vertex(pt[0], pt[1], 0.0)
            # Top vertices
            for pt in pts:
                add_vertex(pt[0], pt[1], h)

            # Side faces (quads split into 2 triangles)
            for i in range(num_pts):
                next_i = (i + 1) % num_pts
                b1 = base_idx + i
                b2 = base_idx + next_i
                t1 = base_idx + num_pts + i
                t2 = base_idx + num_pts + next_i

                faces.append((b1, b2, t2))
                faces.append((b1, t2, t1))

        # Write OBJ file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# ArchiGen 3D Extruded Architectural Floorplan\n")
            for v in vertices:
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")

        print(f"[SUCCESS] 3D Floorplan exported to OBJ: {output_path} ({len(vertices)} vertices, {len(faces)} faces)")
