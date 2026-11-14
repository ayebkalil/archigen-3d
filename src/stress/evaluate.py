"""Structural output checks: these do not establish perceptual reconstruction accuracy."""
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def mask_metrics(expected_path, actual_path, directory):
    from scipy.ndimage import binary_erosion, binary_dilation
    expected = np.asarray(Image.open(expected_path))>127
    actual = np.asarray(Image.open(actual_path))>127
    if expected.shape!=actual.shape:
        raise ValueError('Mask reference dimensions differ; no implicit resizing allowed')
    union = np.logical_or(expected,actual).sum()
    iou = float(np.logical_and(expected,actual).sum()/union) if union else 1.
    a,b = expected ^ binary_erosion(expected), actual ^ binary_erosion(actual)
    # Two pixels at source resolution; not scale invariant across source sizes.
    recall = (a & binary_dilation(b,iterations=2)).sum()/max(a.sum(),1)
    precision = (b & binary_dilation(a,iterations=2)).sum()/max(b.sum(),1)
    f1 = float(2*recall*precision/max(recall+precision,1e-12))
    # Downsample only the artifact, never the metric arrays.
    rgb = np.zeros((*expected.shape,3),dtype=np.uint8)
    rgb[expected & actual] = (180,180,180)
    rgb[expected & ~actual] = (255,60,60)
    rgb[actual & ~expected] = (60,140,255)
    image = Image.fromarray(rgb); image.thumbnail((768,768))
    image.save(Path(directory)/'mask-errors.png')
    return {'foreground_iou':iou,'mask_boundary_f1_2px':f1,
            'foreground_expected_pixels':int(expected.sum()),'foreground_actual_pixels':int(actual.sum())}


def evaluate(model, response, directory):
    directory = Path(directory)
    metrics, issues = {}, []
    if model == 'furniture':
        import trimesh
        data = base64.b64decode(response['model_glb_base64'].split(',', 1)[-1])
        (directory / 'model.glb').write_bytes(data)
        scene = trimesh.load(io.BytesIO(data), file_type='glb', force='scene', process=False)
        meshes = list(scene.geometry.values())
        metrics.update(glb_bytes=len(data), meshes=len(meshes), vertices=sum(len(m.vertices) for m in meshes),
                       faces=sum(len(m.faces) for m in meshes))
        if not meshes or not metrics['vertices'] or not metrics['faces']: issues.append('empty_geometry')
        if any(not np.isfinite(m.vertices).all() for m in meshes): issues.append('nonfinite_vertices')
        if meshes:
            metrics['degenerate_faces'] = sum(int((~m.nondegenerate_faces()).sum()) for m in meshes)
            if metrics['degenerate_faces']: issues.append('degenerate_faces')
            extents = scene.bounds[1] - scene.bounds[0]
            metrics.update({f'extent_{axis}_m': float(v) for axis,v in zip('xyz', extents)})
            if np.any(extents <= 1e-7): issues.append('degenerate_extent')
        quality = response.get('quality', {})
        for key in ('atlas_coverage','components','watertight','peak_allocated_mib','peak_reserved_mib','peak_device_sampled_mib'):
            if isinstance(quality.get(key), (int,float)): metrics[key] = quality[key]
        for key, name in [('preview_image_base64','preview.png')]:
            if response.get(key): (directory / name).write_bytes(base64.b64decode(response[key].split(',')[-1]))
    elif model == 'floorplan':
        from shapely.geometry import Polygon
        metrics.update(rooms=len(response.get('rooms', [])), walls=len(response.get('walls', [])),
                       doors=len(response.get('doors', [])), windows=len(response.get('windows', [])))
        for room in response.get('rooms', []):
            polygon = Polygon(room.get('polygon_points', []))
            if not polygon.is_valid or polygon.area <= 0: issues.append('invalid_room_polygon')
        if not metrics['rooms'] or not metrics['walls']: issues.append('missing_rooms_or_walls')
        reference = directory/'expected-geometry.json'
        if reference.exists():
            from scipy.optimize import linear_sum_assignment
            geometry = json.loads(reference.read_text())
            factor = geometry['canvas_dimensions'][0]/geometry['source_dimensions'][0]*geometry['scale_m_per_canvas_pixel']
            expected = [Polygon(np.asarray(p)*factor) for p in geometry['room_polygons_source_px']]
            actual = [Polygon(r['polygon_points']) for r in response.get('rooms',[]) if len(r.get('polygon_points',[]))>=3]
            actual = [p for p in actual if p.is_valid and p.area>0]
            overlap = np.zeros((len(expected),len(actual)))
            for i,a in enumerate(expected):
                for j,b in enumerate(actual): overlap[i,j] = a.intersection(b).area/max(a.union(b).area,1e-12)
            matched = linear_sum_assignment(-overlap)
            metrics.update(expected_rooms=len(expected),room_count_error=metrics['rooms']-len(expected),
                           synthetic_room_matched_iou_mean=float(overlap[matched].sum()/max(len(expected),1)))
            if metrics['room_count_error']: issues.append('synthetic_room_count_mismatch')
    else:
        for key, name in [('texture_base64','texture.png'), ('sketch_base64','sketch.png')]:
            data = base64.b64decode(response[key].split(',')[-1])
            (directory / name).write_bytes(data)
            image = np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))
            metrics[name.split('.')[0]+'_std'] = float(image.std())
            if image.std() < 1: issues.append('uniform_'+name)
    return {'quality': 'unusable' if any(i.startswith(('empty','nonfinite','degenerate_extent')) for i in issues)
            else 'degraded' if issues else 'unreviewed', 'issues': issues, 'metrics': metrics,
            'visual_review': 'not_reviewed', 'ground_truth': 'synthetic_room_geometry' if model=='floorplan' and (directory/'expected-geometry.json').exists() else 'unavailable',
            'limitation': 'Structural checks do not measure fidelity to real geometry or perceptual quality.'}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['mask','output'],required=True)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--model',default='furniture')
    args = parser.parse_args()
    directory = Path(args.directory)
    if args.mode=='mask':
        result = mask_metrics(directory/'expected-alpha.png',directory/'raw-mask.png',directory)
    else:
        result = evaluate(args.model,json.loads((directory/'validation-input.json').read_text(encoding='utf-8')),directory)
    (directory/('mask-metrics.json' if args.mode=='mask' else 'evaluation.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__': main()
