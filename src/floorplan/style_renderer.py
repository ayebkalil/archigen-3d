"""
Procedural Architectural Style Re-Renderer for CubiCasa5K Floorplans.
Generates synthetic domain-shifted floorplans (CAD blueprints, B&W drawings,
hatched walls, door swing arcs, dimension annotations) directly from vector
SVG ground-truth, while preserving 100% exact segmentation masks.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import math
import random

from src.floorplan.dataset import parse_svg_points, ROOM_CLASSES, STRUCTURE_CLASSES

CAD_BLUE_BG = (14, 36, 61)       # Classic AutoCAD Dark Blueprint Blue
CAD_CYAN_LINE = (165, 243, 252)   # Cyan CAD stroke
CAD_WHITE_LINE = (245, 245, 250)  # White CAD stroke
CAD_DIM_COLOR = (252, 211, 77)    # Yellow dimension marks

def create_hatch_pattern(size: int = 32, angle: int = 45, spacing: int = 8, color=(40, 40, 40)) -> np.ndarray:
    """Generates a seamless 2D diagonal hatching texture."""
    pat = np.full((size, size, 3), 255, dtype=np.uint8)
    for x in range(-size, 2 * size, spacing):
        cv2.line(pat, (x, 0), (x + size, size), color, 1, cv2.LINE_AA)
    return pat

def render_floorplan_style(
    svg_path: Path,
    img_size: int = 512,
    style: str = "cad_blueprint"
) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    """
    Parses SVG ground-truth and procedurally renders a floorplan in the requested architectural style.
    Returns: (restyled_image, struct_mask, room_mask)
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Determine SVG viewBox / bounds
    viewbox = root.attrib.get("viewBox", "").split()
    if len(viewbox) == 4:
        vw, vh = float(viewbox[2]), float(viewbox[3])
    else:
        # Fallback to width / height attributes
        vw = float(root.attrib.get("width", "1000").replace("px", ""))
        vh = float(root.attrib.get("height", "1000").replace("px", ""))

    scale = min(img_size / vw, img_size / vh)
    nw, nh = int(round(vw * scale)), int(round(vh * scale))
    pad_x = (img_size - nw) // 2
    pad_y = (img_size - nh) // 2

    # 1. Initialize Ground Truth Masks
    struct_mask = np.zeros((img_size, img_size), dtype=np.uint8)
    room_mask = np.zeros((img_size, img_size), dtype=np.uint8)

    # Collect parsed geometric elements
    room_polys = []      # list of (pts, room_id)
    ext_wall_polys = []  # list of pts
    int_wall_polys = []  # list of pts
    door_polys = []      # list of pts
    window_polys = []    # list of pts
    door_elements = []   # raw XML elements for swing arcs

    # Parse Rooms
    for g in root.findall(".//{http://www.w3.org/2000/svg}g"):
        c = g.attrib.get("class", "")
        if c.startswith("Space"):
            for r_name, r_id in ROOM_CLASSES.items():
                if r_name in c:
                    for poly in g.findall(".//{http://www.w3.org/2000/svg}polygon"):
                        if "points" in poly.attrib:
                            pts = parse_svg_points(poly.attrib["points"], scale, pad_x, pad_y)
                            if len(pts) >= 3:
                                room_polys.append((pts, r_id))
                                cv2.fillPoly(room_mask, [pts], r_id)
                    break

    # Parse Walls
    for g in root.findall(".//{http://www.w3.org/2000/svg}g"):
        c = g.attrib.get("class", "")
        if c == "Wall External":
            for poly in g.findall("{http://www.w3.org/2000/svg}polygon"):
                if "points" in poly.attrib:
                    pts = parse_svg_points(poly.attrib["points"], scale, pad_x, pad_y)
                    if len(pts) >= 3:
                        ext_wall_polys.append(pts)
                        cv2.fillPoly(struct_mask, [pts], 2) # Wall_External
        elif c == "Wall" or (c.startswith("Wall ") and "External" not in c):
            for poly in g.findall("{http://www.w3.org/2000/svg}polygon"):
                if "points" in poly.attrib:
                    pts = parse_svg_points(poly.attrib["points"], scale, pad_x, pad_y)
                    if len(pts) >= 3:
                        int_wall_polys.append(pts)
                        cv2.fillPoly(struct_mask, [pts], 1) # Wall_Internal

    # Parse Doors & Windows
    for g in root.findall(".//{http://www.w3.org/2000/svg}g"):
        c = g.attrib.get("class", "")
        if c.startswith("Door"):
            door_elements.append(g)
            for poly in g.findall(".//{http://www.w3.org/2000/svg}polygon"):
                if "points" in poly.attrib:
                    pts = parse_svg_points(poly.attrib["points"], scale, pad_x, pad_y)
                    if len(pts) >= 3:
                        door_polys.append(pts)
                        cv2.fillPoly(struct_mask, [pts], 3) # Door
        elif c.startswith("Window"):
            for poly in g.findall(".//{http://www.w3.org/2000/svg}polygon"):
                if "points" in poly.attrib:
                    pts = parse_svg_points(poly.attrib["points"], scale, pad_x, pad_y)
                    if len(pts) >= 3:
                        window_polys.append(pts)
                        cv2.fillPoly(struct_mask, [pts], 4) # Window

    # If style is "random", pick one with balanced weights
    if style == "random":
        style = random.choice([
            "cad_blueprint",
            "black_and_white",
            "hatched_walls",
            "cad_white_blue",
            "faint_pencil_scan",
            "annotated_dimensions"
        ])

    # 2. Render Procedural Canvas based on Selected Style
    if style in ("cad_blueprint", "cad_white_blue"):
        bg_color = CAD_BLUE_BG if style == "cad_blueprint" else (22, 45, 80)
        canvas = np.full((img_size, img_size, 3), bg_color, dtype=np.uint8)

        # Draw Rooms as subtle dark tint
        for pts, _ in room_polys:
            cv2.fillPoly(canvas, [pts], (18, 42, 70))

        # Draw Exterior Walls (Solid or crisp white line)
        for pts in ext_wall_polys:
            cv2.fillPoly(canvas, [pts], (220, 235, 255))
            cv2.polylines(canvas, [pts], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # Draw Interior Walls
        for pts in int_wall_polys:
            cv2.fillPoly(canvas, [pts], (180, 210, 240))
            cv2.polylines(canvas, [pts], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # Draw Door Openings & Swing Arcs
        for pts in door_polys:
            # Cut opening in wall
            cv2.fillPoly(canvas, [pts], bg_color)
            # Add classic CAD door swing arc
            cx, cy = np.mean(pts, axis=0).astype(int)
            r = int(np.linalg.norm(pts[0] - pts[1])) if len(pts) > 1 else 12
            r = max(8, min(r, 24))
            cv2.ellipse(canvas, (cx, cy), (r, r), 0, 0, 90, (120, 200, 255), 1, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (cx + r, cy), (120, 200, 255), 1, cv2.LINE_AA)

        # Draw Windows (Double Parallel Lines)
        for pts in window_polys:
            cv2.fillPoly(canvas, [pts], bg_color)
            if len(pts) >= 4:
                cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.line(canvas, tuple(pts[2]), tuple(pts[3]), (255, 255, 255), 1, cv2.LINE_AA)

        # Optional CAD Dimension Ticks
        if style == "cad_blueprint" or random.random() > 0.5:
            for pts in ext_wall_polys[:4]:
                if len(pts) >= 2:
                    p1, p2 = tuple(pts[0]), tuple(pts[1])
                    # Offset line
                    cv2.line(canvas, p1, p2, CAD_DIM_COLOR, 1, cv2.LINE_AA)
                    cv2.drawMarker(canvas, p1, CAD_DIM_COLOR, cv2.MARKER_TILTED_CROSS, 4, 1)
                    cv2.drawMarker(canvas, p2, CAD_DIM_COLOR, cv2.MARKER_TILTED_CROSS, 4, 1)

    elif style == "black_and_white":
        canvas = np.full((img_size, img_size, 3), 255, dtype=np.uint8)

        # Solid black exterior walls
        for pts in ext_wall_polys:
            cv2.fillPoly(canvas, [pts], (20, 20, 20))

        # Solid dark gray / black interior walls
        for pts in int_wall_polys:
            cv2.fillPoly(canvas, [pts], (40, 40, 40))

        # Clear door openings and draw swing arc
        for pts in door_polys:
            cv2.fillPoly(canvas, [pts], (255, 255, 255))
            cx, cy = np.mean(pts, axis=0).astype(int)
            r = int(np.linalg.norm(pts[0] - pts[1])) if len(pts) > 1 else 12
            r = max(8, min(r, 22))
            cv2.ellipse(canvas, (cx, cy), (r, r), 0, 0, 90, (50, 50, 50), 1, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (cx + r, cy), (50, 50, 50), 1, cv2.LINE_AA)

        # Windows
        for pts in window_polys:
            cv2.fillPoly(canvas, [pts], (255, 255, 255))
            cv2.polylines(canvas, [pts], isClosed=True, color=(100, 100, 100), thickness=1)

    elif style == "hatched_walls":
        canvas = np.full((img_size, img_size, 3), 255, dtype=np.uint8)
        hatch_tex = create_hatch_pattern(32, 45, 6, (60, 60, 60))

        # Tile texture over entire image
        tiled_hatch = np.tile(hatch_tex, (img_size // 32 + 1, img_size // 32 + 1, 1))[:img_size, :img_size]

        wall_mask = np.zeros((img_size, img_size), dtype=np.uint8)
        for pts in ext_wall_polys + int_wall_polys:
            cv2.fillPoly(wall_mask, [pts], 255)

        # Doors cutout
        for pts in door_polys:
            cv2.fillPoly(wall_mask, [pts], 0)

        # Apply hatching inside walls
        canvas[wall_mask == 255] = tiled_hatch[wall_mask == 255]

        # Outlines
        for pts in ext_wall_polys:
            cv2.polylines(canvas, [pts], isClosed=True, color=(20, 20, 20), thickness=2, lineType=cv2.LINE_AA)
        for pts in int_wall_polys:
            cv2.polylines(canvas, [pts], isClosed=True, color=(40, 40, 40), thickness=1, lineType=cv2.LINE_AA)

        # Door swings
        for pts in door_polys:
            cx, cy = np.mean(pts, axis=0).astype(int)
            r = 12
            cv2.ellipse(canvas, (cx, cy), (r, r), 0, 0, 90, (40, 40, 40), 1, cv2.LINE_AA)

    elif style == "faint_pencil_scan":
        # Parchment / sepia paper background
        canvas = np.full((img_size, img_size, 3), (242, 238, 228), dtype=np.uint8)

        # Add slight paper grain noise
        noise = np.random.normal(0, 4, canvas.shape).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Faint pencil strokes for walls
        for pts in ext_wall_polys + int_wall_polys:
            cv2.polylines(canvas, [pts], isClosed=True, color=(80, 75, 70), thickness=1, lineType=cv2.LINE_AA)

        for pts in door_polys:
            cx, cy = np.mean(pts, axis=0).astype(int)
            cv2.ellipse(canvas, (cx, cy), (12, 12), 0, 0, 90, (90, 85, 80), 1, cv2.LINE_AA)

        # Soft blur to simulate scanned paper
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0.5)

    elif style == "annotated_dimensions":
        canvas = np.full((img_size, img_size, 3), 255, dtype=np.uint8)
        # Black walls
        for pts in ext_wall_polys:
            cv2.fillPoly(canvas, [pts], (30, 30, 30))
        for pts in int_wall_polys:
            cv2.fillPoly(canvas, [pts], (60, 60, 60))

        # Add dimension lines, arrows and numbers around perimeter
        for pts in ext_wall_polys[:6]:
            if len(pts) >= 2:
                p1 = tuple(pts[0])
                p2 = tuple(pts[1])
                dist_m = np.linalg.norm(np.array(p1) - np.array(p2)) * 0.03
                if dist_m > 1.0:
                    cv2.line(canvas, p1, p2, (200, 30, 30), 1, cv2.LINE_AA)
                    mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                    cv2.putText(canvas, f"{dist_m:.2f}m", (mx + 2, my - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 20, 20), 1, cv2.LINE_AA)

    else:
        # Fallback to white background with clean black walls
        canvas = np.full((img_size, img_size, 3), 255, dtype=np.uint8)
        for pts in ext_wall_polys + int_wall_polys:
            cv2.fillPoly(canvas, [pts], (30, 30, 30))

    pil_img = Image.fromarray(canvas)
    return pil_img, struct_mask, room_mask
