"""Coordinate-safe extraction and neural atlas baking for the furniture experiment."""
import threading
import numpy as np
from PIL import Image, ImageOps
import trimesh

_REMBG_LOCK = threading.Lock()
_REMBG_SESSION = None


def _get_rembg_session():
    """Reuse the u2netp ONNX session across opaque photo requests."""
    global _REMBG_SESSION
    with _REMBG_LOCK:
        if _REMBG_SESSION is None:
            import rembg
            _REMBG_SESSION = rembg.new_session('u2netp', providers=['CPUExecutionProvider'])
        return _REMBG_SESSION


def prepare_input(image, remove_bg=True):
    image = ImageOps.exif_transpose(image).convert('RGBA')
    alpha = np.asarray(image)[..., 3]
    source = 'provided alpha' if np.any(alpha < 255) else 'opaque image'
    if remove_bg and source == 'opaque image':
        import rembg
        session = _get_rembg_session()
        image = rembg.remove(image, session=session, post_process_mask=False)
        source = 'u2netp CPU mask (review thin parts)'
    rgba = np.array(image)
    yy, xx = np.where(rgba[..., 3] > 16)
    if len(xx) == 0:
        raise ValueError('No visible foreground found')
    cropped = image.crop((int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1))
    side = int(np.ceil(max(cropped.size) / .85))
    square = Image.new('RGBA', (side, side))
    square.paste(cropped, ((side-cropped.width)//2, (side-cropped.height)//2))
    square = square.resize((512, 512), Image.Resampling.LANCZOS)
    pixels = np.asarray(square).astype(np.float32)/255
    rgb = pixels[..., :3]*pixels[..., 3:] + .5*(1-pixels[..., 3:])
    return Image.fromarray(np.rint(rgb*255).astype(np.uint8)), square, source


def extract_surface(density, radius, level=25.0):
    """Volume axes are X,Y,Z. skimage returns these axes unchanged (no torchmcubes swap)."""
    from skimage.measure import marching_cubes
    density = np.asarray(density, dtype=np.float32)
    if density.ndim != 3 or not np.isfinite(density).all():
        raise ValueError('Density must be a finite 3D volume')
    if not density.min() < level < density.max():
        raise ValueError('No surface crosses the requested density level')
    spacing = tuple(2*radius/(n-1) for n in density.shape)
    vertices, faces, _, _ = marching_cubes(density, level=level, spacing=spacing, allow_degenerate=False)
    mesh = trimesh.Trimesh(vertices=vertices-radius, faces=faces, process=False)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals(multibody=True)
    return mesh


def to_room_coordinates(vertices, height_m):
    """TripoSR X camera-facing, Y right, Z up -> room X right, Y up, Z camera-facing.

    Cyclic permutation has determinant +1: no reflection or winding reversal.
    One uniform scale preserves the inferred proportions. No floor flattening.
    """
    if not np.isfinite(height_m) or not .05 <= height_m <= 10:
        raise ValueError('Furniture height must be between 0.05 and 10 metres')
    result = np.asarray(vertices, dtype=np.float64)[:, [1, 2, 0]].copy()
    extent = np.ptp(result, axis=0)
    if extent[1] <= 1e-8:
        raise ValueError('Degenerate furniture height')
    result[:, [0, 2]] -= (result[:, [0, 2]].min(0)+result[:, [0, 2]].max(0))/2
    result[:, 1] -= result[:, 1].min()
    return result * (height_m/extent[1])


def _rasterize_triangle_chunk(vertices, f_valid, a, b, c, det, low, high, nx, counts, positions, covered):
    """Vectorized barycentric rasterization for a bounded batch of triangles."""
    t_rep = np.repeat(np.arange(len(f_valid), dtype=np.int32), counts)
    offsets = np.arange(len(t_rep), dtype=np.int32)
    starts = np.zeros(len(f_valid), dtype=np.int32)
    if len(f_valid) > 1:
        np.cumsum(counts[:-1], out=starts[1:])
    local = offsets - np.repeat(starts, counts)
    nx_rep = np.repeat(nx, counts)
    px = np.repeat(low[:, 0], counts) + (local % nx_rep)
    py = np.repeat(low[:, 1], counts) + (local // nx_rep)

    ax, ay = a[t_rep, 0], a[t_rep, 1]
    bx, by = b[t_rep, 0], b[t_rep, 1]
    cx, cy = c[t_rep, 0], c[t_rep, 1]
    dt = det[t_rep]

    wa = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / dt
    wb = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / dt
    wc = 1.0 - wa - wb

    inside = (wa >= -1e-6) & (wb >= -1e-6) & (wc >= -1e-6)
    if not inside.any():
        return
    ti = t_rep[inside]
    fv = f_valid[ti]
    xyz = (
        wa[inside, None] * vertices[fv[:, 0]]
        + wb[inside, None] * vertices[fv[:, 1]]
        + wc[inside, None] * vertices[fv[:, 2]]
    )
    py_i = py[inside]
    px_i = px[inside]
    positions[py_i, px_i] = xyz
    covered[py_i, px_i] = True


def rasterize_atlas(vertices, faces, uvs, resolution):
    """Vectorized CPU barycentric rasterization. V points up; PIL row zero is top."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    uvs = np.asarray(uvs, dtype=np.float32)
    positions = np.zeros((resolution, resolution, 3), np.float32)
    covered = np.zeros((resolution, resolution), bool)
    if len(faces) == 0:
        raise ValueError('UV atlas contains no rasterized surface')

    coords = np.column_stack((uvs[:, 0] * resolution - .5, (1.0 - uvs[:, 1]) * resolution - .5))
    p = coords[faces]
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    det = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    low = np.maximum(np.ceil(p.min(axis=1)).astype(np.int32), 0)
    high = np.minimum(np.floor(p.max(axis=1)).astype(np.int32), resolution - 1)
    valid = (np.abs(det) >= 1e-10) & (high[:, 0] >= low[:, 0]) & (high[:, 1] >= low[:, 1])
    tri_idx = np.nonzero(valid)[0]
    if len(tri_idx) == 0:
        raise ValueError('UV atlas contains no rasterized surface')

    low, high, det = low[tri_idx], high[tri_idx], det[tri_idx]
    a, b, c, f_valid = a[tri_idx], b[tri_idx], c[tri_idx], faces[tri_idx]
    nx = high[:, 0] - low[:, 0] + 1
    counts = nx * (high[:, 1] - low[:, 1] + 1)

    max_samples = 1_000_000
    n_valid = len(tri_idx)
    start = 0
    while start < n_valid:
        if counts[start] > max_samples:
            # Handle an unusually large triangle row-by-row so memory stays bounded.
            x = np.arange(low[start, 0], high[start, 0] + 1, dtype=np.float32)
            row_step = max(1, max_samples // max(1, int(nx[start])))
            for y0 in range(int(low[start, 1]), int(high[start, 1]) + 1, row_step):
                y1 = min(int(high[start, 1]), y0 + row_step - 1)
                gx, gy = np.meshgrid(x, np.arange(y0, y1 + 1, dtype=np.float32))
                wa = ((b[start, 1] - c[start, 1]) * (gx - c[start, 0]) + (c[start, 0] - b[start, 0]) * (gy - c[start, 1])) / det[start]
                wb = ((c[start, 1] - a[start, 1]) * (gx - c[start, 0]) + (a[start, 0] - c[start, 0]) * (gy - c[start, 1])) / det[start]
                wc = 1.0 - wa - wb
                inside = (wa >= -1e-6) & (wb >= -1e-6) & (wc >= -1e-6)
                if inside.any():
                    fv = f_valid[start]
                    positions[gy[inside].astype(np.int32), gx[inside].astype(np.int32)] = (
                        wa[inside, None] * vertices[fv[0]]
                        + wb[inside, None] * vertices[fv[1]]
                        + wc[inside, None] * vertices[fv[2]]
                    )
                    covered[gy[inside].astype(np.int32), gx[inside].astype(np.int32)] = True
            start += 1
            continue

        # Accumulate as many triangles as fit inside the sample budget.
        end = start
        acc = 0
        while end < n_valid and end - start < 32768 and acc + int(counts[end]) <= max_samples:
            acc += int(counts[end])
            end += 1
        _rasterize_triangle_chunk(
            vertices, f_valid[start:end], a[start:end], b[start:end], c[start:end],
            det[start:end], low[start:end], high[start:end], nx[start:end], counts[start:end],
            positions, covered,
        )
        start = end

    if not covered.any():
        raise ValueError('UV atlas contains no rasterized surface')
    return positions, covered


def _connected_normal_charts(vertices, faces):
    """Partition faces into edge-connected charts sharing a dominant normal direction."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    dom = np.argmax(np.abs(fn), axis=1)
    sgn = (fn[np.arange(len(faces)), dom] < 0).astype(np.int32)
    labels = dom * 2 + sgn

    edges = np.vstack([
        np.sort(faces[:, [0, 1]], axis=1),
        np.sort(faces[:, [1, 2]], axis=1),
        np.sort(faces[:, [2, 0]], axis=1),
    ])
    edge_faces = np.tile(np.arange(len(faces), dtype=np.int32), 3)
    stride = np.int64(len(vertices)) + 1
    keys = edges[:, 0].astype(np.int64) * stride + edges[:, 1].astype(np.int64)
    order = np.argsort(keys)
    keys_s, ef_s = keys[order], edge_faces[order]
    same = keys_s[:-1] == keys_s[1:]
    f_a, f_b = ef_s[:-1][same], ef_s[1:][same]
    keep = labels[f_a] == labels[f_b]
    adj = csr_matrix((np.ones(int(keep.sum()), np.int8), (f_a[keep], f_b[keep])), shape=(len(faces), len(faces)))
    n_comp, comp_id = connected_components(adj, directed=False)
    return n_comp, comp_id, labels


def _pack_and_project_charts(vertices, faces, comp_id, labels, resolution, padding):
    """Project each connected chart orthographically and shelf-pack into [0, 1]^2."""
    n_comp = int(comp_id.max()) + 1
    comp_label = np.zeros(n_comp, np.int32)
    comp_label[comp_id] = labels

    tri_xyz = vertices[faces]
    u_axis = np.array([1, 1, 2, 2, 0, 0])[labels]
    v_axis = np.array([2, 2, 0, 0, 1, 1])[labels]
    idx_f = np.arange(len(faces))[:, None]
    u_coords = tri_xyz[idx_f, np.arange(3)[None, :], u_axis[:, None]].copy()
    v_coords = tri_xyz[idx_f, np.arange(3)[None, :], v_axis[:, None]]
    u_coords[(labels % 2) == 1] *= -1.0

    c_umin = np.full(n_comp, np.inf, np.float32)
    c_umax = np.full(n_comp, -np.inf, np.float32)
    c_vmin = np.full(n_comp, np.inf, np.float32)
    c_vmax = np.full(n_comp, -np.inf, np.float32)
    np.minimum.at(c_umin, comp_id, u_coords.min(axis=1))
    np.maximum.at(c_umax, comp_id, u_coords.max(axis=1))
    np.minimum.at(c_vmin, comp_id, v_coords.min(axis=1))
    np.maximum.at(c_vmax, comp_id, v_coords.max(axis=1))

    w = np.maximum(c_umax - c_umin, 1e-6)
    h = np.maximum(c_vmax - c_vmin, 1e-6)
    rot = h > w
    cw = np.where(rot, h, w)
    ch = np.where(rot, w, h)

    pad_uv = padding / float(resolution)
    order_c = np.argsort(-ch)

    def try_pack(scale):
        x, y, shelf_h = pad_uv, pad_uv, 0.0
        ox = np.empty(n_comp, np.float32)
        oy = np.empty(n_comp, np.float32)
        for ci in order_c:
            bw, bh = float(cw[ci]) * scale, float(ch[ci]) * scale
            if bw + 2 * pad_uv > 1.0:
                return None
            if x + bw + pad_uv > 1.0:
                y += shelf_h + pad_uv
                x, shelf_h = pad_uv, 0.0
            if y + bh + pad_uv > 1.0:
                return None
            ox[ci], oy[ci] = x, y
            x += bw + pad_uv
            if bh > shelf_h:
                shelf_h = bh
        return ox, oy

    total_area = float(np.sum((cw + 2 * pad_uv) * (ch + 2 * pad_uv)))
    lo, hi = 0.01, min(0.95 / float(cw.max()), 1.0 / np.sqrt(max(1e-9, total_area))) * 1.5
    best = None
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        res = try_pack(mid)
        if res is not None:
            best = (mid, res[0], res[1])
            lo = mid
        else:
            hi = mid
    if best is None:
        while (res := try_pack(lo)) is None:
            lo *= 0.5
        best = (lo, res[0], res[1])
    scale, ox, oy = best

    pair_keys = np.repeat(comp_id, 3).astype(np.int64) * len(vertices) + faces.ravel().astype(np.int64)
    uniq_keys, inv_idx = np.unique(pair_keys, return_inverse=True)
    mapping = (uniq_keys % len(vertices)).astype(np.int32)
    u_comp = (uniq_keys // len(vertices)).astype(np.int32)
    faces_out = inv_idx.reshape(-1, 3).astype(np.int32)

    canonical = vertices[mapping]
    cl = comp_label[u_comp]
    ua = np.array([1, 1, 2, 2, 0, 0])[cl]
    va = np.array([2, 2, 0, 0, 1, 1])[cl]
    u_val = canonical[np.arange(len(canonical)), ua]
    v_val = canonical[np.arange(len(canonical)), va]
    u_val = np.where((cl % 2) == 1, -u_val, u_val)

    lu = u_val - c_umin[u_comp]
    lv = v_val - c_vmin[u_comp]
    r_v = rot[u_comp]
    pu = np.where(r_v, cw[u_comp] - lv, lu) * scale + ox[u_comp]
    pv = np.where(r_v, lu, lv) * scale + oy[u_comp]
    uvs = np.column_stack([pu, pv]).astype(np.float32)
    return canonical, faces_out, uvs


def _unwrap_fast(vertices, faces, resolution=1024):
    """Fast overlap-checked orthogonal-chart unwrapping (~0.25s on 190k triangles)."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    padding = max(4, resolution // 128)
    _, comp_id, labels = _connected_normal_charts(vertices, faces)

    span = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    tol = max(1e-4, span * 0.02)
    check_res = min(512, resolution)

    for _ in range(4):
        canonical, faces_out, uvs = _pack_and_project_charts(
            vertices, faces, comp_id, labels, resolution, padding
        )
        pos_fwd, cov = rasterize_atlas(canonical, faces_out, uvs, check_res)
        pos_rev, _ = rasterize_atlas(canonical, faces_out[::-1], uvs, check_res)
        conflict = np.zeros((check_res, check_res), bool)
        conflict[cov] = np.linalg.norm(pos_fwd[cov] - pos_rev[cov], axis=1) > tol
        if not conflict.any():
            return canonical, faces_out, uvs

        # Split only charts whose pixels collide at different 3D depths.
        coords = np.column_stack((uvs[:, 0] * check_res - .5, (1.0 - uvs[:, 1]) * check_res - .5))
        tri_pts = np.concatenate([coords[faces_out], coords[faces_out].mean(axis=1, keepdims=True)], axis=1)
        px = np.clip(np.rint(tri_pts[..., 0]).astype(np.int32), 0, check_res - 1)
        py = np.clip(np.rint(tri_pts[..., 1]).astype(np.int32), 0, check_res - 1)
        bad_charts = np.unique(comp_id[conflict[py, px].any(axis=1)])
        if len(bad_charts) == 0:
            return canonical, faces_out, uvs
        centroids = vertices[faces].mean(axis=1)
        next_id = int(comp_id.max()) + 1
        for bc in bad_charts:
            mask = comp_id == bc
            if int(mask.sum()) < 2:
                continue
            d_axis = int(labels[mask][0]) // 2
            depths = centroids[mask, d_axis]
            upper = depths > np.median(depths)
            if upper.any() and not upper.all():
                idx = np.nonzero(mask)[0][upper]
                comp_id[idx] = next_id
                next_id += 1

    return canonical, faces_out, uvs


def bake_neural_atlas(mesh, query_color, resolution=1024):
    from scipy.ndimage import distance_transform_edt
    canonical, faces, uvs = _unwrap_fast(mesh.vertices, mesh.faces, resolution=resolution)
    print('Furniture atlas: unwrap complete, rasterizing surface positions', flush=True)
    positions, covered = rasterize_atlas(canonical, faces, uvs, resolution)
    colors = np.asarray(query_color(positions[covered]), dtype=np.float32)
    if not np.isfinite(colors).all():
        raise ValueError('Nonfinite neural surface colors')
    pixels = np.zeros((resolution, resolution, 3), np.uint8)
    pixels[covered] = np.rint(np.clip(colors, 0, 1)*255).astype(np.uint8)
    # Extend every empty texel to its nearest surface color, including tiny charts
    # and coarse mip levels whose footprint extends beyond a fixed-width gutter.
    nearest = distance_transform_edt(~covered, return_distances=False, return_indices=True)
    pixels[~covered] = pixels[nearest[0][~covered], nearest[1][~covered]]
    return canonical, faces, uvs, Image.fromarray(pixels), float(covered.mean())
