"""One physical material for a house, with neural edits limited by actual openings."""
import hashlib
import json

import cv2
import numpy as np
from PIL import Image


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def opening_mask(spec, size, trim_m=.045):
    """Pixel centers in physical facade coordinates; no edits outside this support."""
    bounds = spec['uv_bounds']
    u = (np.arange(size) + .5) / size
    v = 1 - (np.arange(size) + .5) / size
    along = (u - bounds['u_min']) / (bounds['u_max'] - bounds['u_min']) * spec['wall_length_m']
    height = (v - bounds['v_min']) / (bounds['v_max'] - bounds['v_min']) * spec['wall_height_m']
    valid = ((along >= 0) & (along <= spec['wall_length_m']))[None] & ((height >= 0) & (height <= spec['wall_height_m']))[:, None]
    mask = np.zeros((size, size), bool)
    for op in spec['openings']:
        mask |= ((along >= op['position_along_wall_m'] - op['width_m'] / 2 - trim_m)
                 & (along <= op['position_along_wall_m'] + op['width_m'] / 2 + trim_m))[None] & (
                    (height >= op['sill_m'] - trim_m) & (height <= op['sill_m'] + op['height_m'] + trim_m))[:, None]
    return mask & valid


def lock_surface(base, proposal, spec, trim_m=.045):
    """Hard compositor, not a learned correction or a semantic-image detector."""
    if base.width != base.height:
        raise ValueError('Expected a square letterboxed surface')
    proposed = proposal.convert('RGB').resize(base.size, Image.Resampling.BICUBIC)
    original, neural = np.asarray(base.convert('RGB')), np.asarray(proposed)
    mask = opening_mask(spec, base.width, trim_m)
    # A feather is confined to the allowed support; it never leaks onto the wall.
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(distance / 2, 0, 1)[:, :, None]
    result = np.rint(original * (1 - alpha) + neural * alpha).astype(np.uint8)
    result[~mask] = original[~mask]
    return Image.fromarray(result), mask


def facade_chains(house, panels, requested_period_m=2):
    """Walk physical wall axes, then use face-line intersections for corner phase."""
    by_id = {wall['id']: wall for wall in house['walls']}
    result = {}
    for level in house['storeys']:
        pending = {p['wall_id']: p for p in panels if p['level_id'] == level['id']}
        while pending:
            first = next(iter(pending)); chain = [pending.pop(first)]
            while pending:
                end = np.array(by_id[chain[-1]['wall_id']]['axis_meters'][1])
                following = next((key for key in pending if np.linalg.norm(
                    np.array(by_id[key]['axis_meters'][0]) - end) < 1e-4), None)
                if following is None:
                    break
                chain.append(pending.pop(following))
            starts, ends = [], []
            def intersection(a, b):
                origin_a, origin_b = np.asarray(a['facade_frame']['origin']), np.asarray(b['facade_frame']['origin'])
                da, db = np.asarray(a['facade_frame']['direction']), np.asarray(b['facade_frame']['direction'])
                matrix = np.column_stack((da, -db))
                if abs(np.linalg.det(matrix)) < 1e-8:
                    return None
                return origin_a + da * np.linalg.solve(matrix, origin_b - origin_a)[0]
            closed = np.linalg.norm(np.asarray(by_id[chain[-1]['wall_id']]['axis_meters'][1]) -
                                    np.asarray(by_id[chain[0]['wall_id']]['axis_meters'][0])) < 1e-4
            for i, spec in enumerate(chain):
                frame = spec['facade_frame']
                fallback_start = np.asarray(frame['origin']) + np.asarray(frame['direction']) * frame['length_m']
                start = intersection(chain[i - 1], spec) if i or closed else None
                end = intersection(spec, chain[(i + 1) % len(chain)]) if i + 1 < len(chain) or closed else None
                starts.append(fallback_start if start is None else start)
                ends.append(np.asarray(frame['origin']) if end is None else end)
            lengths = [float(np.linalg.norm(b - a)) for a, b in zip(starts, ends)]
            period = sum(lengths) / max(1, round(sum(lengths) / requested_period_m))
            phase = 0.
            for spec, start, end, length in zip(chain, starts, ends, lengths):
                if length < 1e-4:
                    raise ValueError('Degenerate exterior material run')
                result[spec['wall_id']] = {'start': start.tolist(), 'direction': ((end - start) / length).tolist(),
                    'phase_m': phase, 'period_m': period, 'elevation_m': level['elevation'], 'closed_chain': bool(closed),
                    'face_length_m': length, 'chain_length_m': sum(lengths)}
                phase += length
    return result


def material_surface(spec, tile, phase, size=1024, neutral=(127, 127, 127)):
    """Sample the same tile in metres, preserving physical scale on all faces."""
    bounds, frame = spec['uv_bounds'], spec['facade_frame']
    u, v = (np.arange(size) + .5) / size, 1 - (np.arange(size) + .5) / size
    s = (u - bounds['u_min']) / (bounds['u_max'] - bounds['u_min']) * spec['wall_length_m']
    y = (v - bounds['v_min']) / (bounds['v_max'] - bounds['v_min']) * spec['wall_height_m']
    positions = np.asarray(frame['origin'])[None] + s[:, None] * np.asarray(frame['direction'])[None]
    distance = phase['phase_m'] + (positions - np.asarray(phase['start'])) @ np.asarray(phase['direction'])
    xmap = np.mod(distance / phase['period_m'], 1) * tile.width
    ymap = np.mod(-(phase['elevation_m'] + y) / phase['period_m'], 1) * tile.height
    pixels = np.asarray(tile.convert('RGB'))
    xgrid, ygrid = np.meshgrid(xmap.astype(np.float32), ymap.astype(np.float32))
    sampled = cv2.remap(pixels, xgrid, ygrid, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    valid = ((s >= 0) & (s <= spec['wall_length_m']))[None] & ((y >= 0) & (y <= spec['wall_height_m']))[:, None]
    sampled[~valid] = neutral
    return Image.fromarray(sampled)


