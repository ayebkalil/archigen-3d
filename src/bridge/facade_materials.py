"""Experimental geometry-locked assembly of neural material and opening images.

This is compositing, not an end-to-end neural facade prediction or a PBR estimator.
The material swatch must be reviewed for unwanted objects before use.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.models.facade_diffusion import uv_box


def compose_material_facade(fixture, material, opening_source, tile_meters=2.0):
    size = opening_source.width
    if opening_source.height != size or size < 64:
        raise ValueError('Opening source must be a square facade texture')
    length, height = fixture['wall_length_m'], fixture['wall_height_m']
    if not all(math.isfinite(v) and v > 0 for v in (length, height, tile_meters)):
        raise ValueError('Physical wall and tile dimensions must be finite and positive')
    bounds = fixture['uv_bounds']
    wall = uv_box([bounds['u_min'], bounds['v_min'], bounds['u_max'], bounds['v_max']], size)
    x0, y0, x1, y1 = wall
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Wall is smaller than a texture pixel')
    width, pixels_high = x1 - x0 + 1, y1 - y0 + 1
    tile_size = (max(1, round((x1-x0)*tile_meters/length)),
                 max(1, round((y1-y0)*tile_meters/height)))
    tile = material.convert('RGB').resize(tile_size, Image.Resampling.LANCZOS)
    tiled = Image.new('RGB', (width, pixels_high))
    for row, y in enumerate(range(0, pixels_high, tile.height)):
        for col, x in enumerate(range(0, width, tile.width)):
            patch = ImageOps.mirror(tile) if col % 2 else tile
            patch = ImageOps.flip(patch) if row % 2 else patch
            tiled.paste(patch, (x, y))
    result = Image.new('RGB', (size, size), (220, 220, 220))
    result.paste(tiled, (x0, y0))
    mask = Image.new('L', (size, size))
    draw = ImageDraw.Draw(mask)
    opening_source = opening_source.convert('RGB')
    for opening in fixture['openings']:
        if opening['type'] not in ('window', 'door'):
            raise ValueError('Only explicit window/door regions are supported')
        a, b, c, d = uv_box(opening['uv_rect'], size)
        if not (x0 <= a < c <= x1 and y0 <= b < d <= y1):
            raise ValueError('Opening must be inside the wall UV bounds')
        if np.any(np.asarray(mask)[b:d+1, a:c+1]):
            raise ValueError('Overlapping opening masks are ambiguous')
        # No source-facade pixels outside these known regions are copied.
        result.paste(opening_source.crop((a, b, c+1, d+1)), (a, b))
        draw.rectangle((a, b, c, d), fill=255)
    return result, mask
