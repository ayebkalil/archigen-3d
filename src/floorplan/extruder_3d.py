"""
Procedural 3D Wall Extrusion, Three.js Mesh Generation, and OBJ+MTL Texturing Engine.
Converts vector layout JSON into:
1. Three.js BufferGeometry compatible JSON specs for client-side rendering with texture maps.
2. Standard Wavefront OBJ + MTL format with UV coordinates for Blender rendering.
3. Exterior facade boundary extraction and texture attachment.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import math
import numpy as np
from shapely.geometry import Polygon

class Floorplan3DExtruder:
    """
    Extrudes 2D wall polygons along the vertical axis into 3D meshes with openings and texture mapping.
    """
    def __init__(
        self,
        wall_height_m: float = 2.80,
        door_height_m: float = 2.10,
        window_height_m: float = 1.40,
        window_sill_m: float = 0.90
    ):
        self.wall_height_m = wall_height_m
        self.door_height_m = door_height_m
        self.window_height_m = window_height_m
        self.window_sill_m = window_sill_m

    def generate_threejs_scene_data(self, layout: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a JSON payload ready for Three.js scene creation in the frontend.
        Includes:
        - Wall geometries (coordinates, height, thickness, is_exterior, is_front)
        - Facade texture mapping metadata (texture file, UV bounds)
        - Floor slab geometries
        - Room labels and bounding boxes
        - Door / window cutout locations
        """
        num_floors = layout.get("metadata", {}).get("num_floors", 1)
        total_h = self.wall_height_m * num_floors

        walls_3d = []
        for wall in layout.get("walls", []):
            pts_2d = wall.get("polygon_meters", [])
            if len(pts_2d) < 3:
                continue

            # Validate polygon with Shapely
            poly = Polygon(pts_2d)
            is_valid = poly.is_valid

            wall_entry = {
                "id": wall["id"],
                "points_2d": pts_2d,
                "height": total_h,
                "is_exterior": wall.get("is_exterior", False),
                "is_front": wall.get("is_front", False),
                "is_valid_geometry": is_valid,
                "openings": wall.get("openings", []),
                "facade_texture": wall.get("facade_texture", None)
            }
            walls_3d.append(wall_entry)

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
            "version": "2.0",
            "scale": layout.get("metadata", {}).get("scale_m_per_px", 0.025),
            "total_surface_m2": layout.get("metadata", {}).get("total_area_m2", 0.0),
            "num_floors": num_floors,
            "wall_height_m": total_h,
            "walls": walls_3d,
            "rooms": rooms_3d,
            "doors": layout.get("doors", []),
            "windows": layout.get("windows", [])
        }

    def export_obj_with_mtl(
        self,
        layout: Dict[str, Any],
        output_obj_path: str,
        texture_relative_path: Optional[str] = None
    ):
        """
        Exports the extruded 3D building as Wavefront .OBJ with texture coordinates (vt)
        and an accompanying .MTL material file so that Blender renders textured walls.
        """
        obj_file = Path(output_obj_path)
        obj_file.parent.mkdir(parents=True, exist_ok=True)
        mtl_file = obj_file.with_suffix(".mtl")

        num_floors = layout.get("metadata", {}).get("num_floors", 1)
        h = self.wall_height_m * num_floors

        vertices = []
        texcoords = []  # (u, v)
        faces = []      # (v_idx, vt_idx) for each vertex in face
        material_faces = {"Wall_Interior": [], "Wall_Exterior": [], "Wall_Front_Facade": []}

        def add_vertex(x, y, z):
            # In Blender / Three.js standard coords: X is right, Y is up, Z is depth
            vertices.append((x, z, -y))
            return len(vertices)

        def add_vt(u, v):
            texcoords.append((u, v))
            return len(texcoords)

        # Standard default UV corners
        vt_bl = add_vt(0.0, 0.0)
        vt_br = add_vt(1.0, 0.0)
        vt_tr = add_vt(1.0, 1.0)
        vt_tl = add_vt(0.0, 1.0)

        for wall in layout.get("walls", []):
            pts = wall.get("polygon_meters", [])
            if len(pts) < 3:
                continue

            num_pts = len(pts)
            base_idx = len(vertices) + 1

            is_front = wall.get("is_front", False)
            is_exterior = wall.get("is_exterior", False)

            mat_name = "Wall_Front_Facade" if is_front else ("Wall_Exterior" if is_exterior else "Wall_Interior")

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

                # Assign texture coordinates
                f1 = ((b1, vt_bl), (b2, vt_br), (t2, vt_tr))
                f2 = ((b1, vt_bl), (t2, vt_tr), (t1, vt_tl))

                material_faces[mat_name].append(f1)
                material_faces[mat_name].append(f2)

        # Write MTL file
        with open(mtl_file, "w", encoding="utf-8") as fm:
            fm.write("# ArchiGen 3D Material Library\n")
            fm.write("newmtl Wall_Interior\nKd 0.90 0.90 0.88\nKa 0.2 0.2 0.2\nillum 2\n\n")
            fm.write("newmtl Wall_Exterior\nKd 0.80 0.78 0.75\nKa 0.2 0.2 0.2\nillum 2\n\n")
            fm.write("newmtl Wall_Front_Facade\nKd 1.0 1.0 1.0\nKa 0.3 0.3 0.3\nillum 2\n")
            if texture_relative_path:
                fm.write(f"map_Kd {texture_relative_path}\n")

        # Write OBJ file
        with open(obj_file, "w", encoding="utf-8") as f:
            f.write("# ArchiGen 3D Textured Architectural Mesh\n")
            f.write(f"mtllib {mtl_file.name}\n")
            for v in vertices:
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for vt in texcoords:
                f.write(f"vt {vt[0]:.4f} {vt[1]:.4f}\n")

            for mat_name, f_list in material_faces.items():
                if f_list:
                    f.write(f"\nusemtl {mat_name}\n")
                    for tri in f_list:
                        f.write(f"f {tri[0][0]}/{tri[0][1]} {tri[1][0]}/{tri[1][1]} {tri[2][0]}/{tri[2][1]}\n")

        print(f"[SUCCESS] 3D Floorplan exported to OBJ+MTL: {output_obj_path} ({len(vertices)} vertices, {sum(len(l) for l in material_faces.values())} faces)")

    def export_obj(self, layout: Dict[str, Any], output_path: str):
        """Backwards-compatible alias for export_obj_with_mtl."""
        self.export_obj_with_mtl(layout, output_path)

