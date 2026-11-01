"""Coordinate-safe extraction and neural atlas baking for the furniture experiment."""
import numpy as np
from PIL import Image, ImageOps
import trimesh


def prepare_input(image, remove_bg=True):
    image = ImageOps.exif_transpose(image).convert('RGBA')
    alpha = np.asarray(image)[..., 3]
    source = 'provided alpha' if np.any(alpha < 255) else 'opaque image'
    if remove_bg and source == 'opaque image':
        import rembg
        session = rembg.new_session('u2netp', providers=['CPUExecutionProvider'])
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


def rasterize_atlas(vertices, faces, uvs, resolution):
    """CPU barycentric rasterization. xatlas V points up; PIL row zero is top."""
    positions = np.zeros((resolution, resolution, 3), np.float32)
    covered = np.zeros((resolution, resolution), bool)
    coords = np.column_stack((uvs[:, 0]*resolution-.5, (1-uvs[:, 1])*resolution-.5))
    for triangle in faces:
        p = coords[triangle]
        low = np.maximum(np.ceil(p.min(0)).astype(int), 0)
        high = np.minimum(np.floor(p.max(0)).astype(int), resolution-1)
        if np.any(high < low):
            continue
        a, b, c = p
        det = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(det) < 1e-10:
            continue
        x, y = np.meshgrid(np.arange(low[0], high[0]+1), np.arange(low[1], high[1]+1))
        wa = ((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/det
        wb = ((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/det
        wc = 1-wa-wb
        inside = (wa >= -1e-6) & (wb >= -1e-6) & (wc >= -1e-6)
        xyz = wa[..., None]*vertices[triangle[0]] + wb[..., None]*vertices[triangle[1]] + wc[..., None]*vertices[triangle[2]]
        positions[y[inside], x[inside]] = xyz[inside]
        covered[y[inside], x[inside]] = True
    if not covered.any():
        raise ValueError('UV atlas contains no rasterized surface')
    return positions, covered


def bake_neural_atlas(mesh, query_color, resolution=1024):
    import xatlas
    from scipy.ndimage import distance_transform_edt
    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    options = xatlas.PackOptions()
    options.resolution = resolution
    options.padding = max(4, resolution//128)
    options.bilinear = True
    atlas.generate(pack_options=options)
    print('Furniture atlas: unwrap complete, rasterizing surface positions', flush=True)
    mapping, faces, uvs = atlas[0]
    canonical = mesh.vertices[mapping]
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
