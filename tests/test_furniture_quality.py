"""Analytic regression tests for the experimental image-to-3D geometry path."""
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image

from src.furniture.quality_geometry import prepare_input, extract_surface, to_room_coordinates, rasterize_atlas, bake_neural_atlas


def test_marching_cubes_preserves_native_xyz_on_asymmetric_field():
    axis = np.linspace(-1, 1, 50)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1)
    center = np.array([.2, -.1, .05])
    radii = np.array([.24, .4, .65])
    density = 26 - np.sum(((points-center)/radii)**2, axis=-1)
    mesh = extract_surface(density, radius=1)
    np.testing.assert_allclose(mesh.bounds.mean(0), center, atol=.015)
    np.testing.assert_allclose(mesh.extents, radii*2, atol=.025)
    assert mesh.is_watertight and mesh.volume > 0


def test_room_transform_is_not_a_reflection_and_preserves_thin_foot_geometry():
    matrix = np.eye(3)[[1, 2, 0]]
    assert np.linalg.det(matrix) == 1
    vertices = np.array([[-.3, -.2, 0], [.3, .2, 1.], [.1, .1, .01]])
    out = to_room_coordinates(vertices, .8)
    assert out[:, 1].min() == 0 and out[:, 1].max() == .8
    assert out[2, 1] == pytest.approx(.008)  # No blanket 2.5 cm flattening.
    np.testing.assert_allclose(np.linalg.norm(out[1]-out[0]), .8*np.linalg.norm(vertices[1]-vertices[0]))
    np.testing.assert_allclose((out.min(0)+out.max(0))[[0, 2]], 0)


def test_uv_rasterization_samples_canonical_positions_with_top_origin_image_rows():
    vertices = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], np.float32)
    faces = np.array([[0,1,2], [0,2,3]])
    positions, mask = rasterize_atlas(vertices, faces, vertices[:, :2], 16)
    assert mask.all()
    np.testing.assert_allclose(positions[0, 0], [1/32, 31/32, 0], atol=1e-6)
    np.testing.assert_allclose(positions[-1, -1], [31/32, 1/32, 0], atol=1e-6)


def test_preprocessing_keeps_thin_and_soft_alpha_foreground():
    rgba = np.zeros((64, 64, 4), np.uint8)
    rgba[5:55, 30:33] = [200, 100, 50, 200]
    prepared, preview, source = prepare_input(Image.fromarray(rgba), remove_bg=True)
    assert prepared.size == preview.size == (512, 512)
    assert source == 'provided alpha'
    assert 0 < np.asarray(preview)[..., 3].max() < 255
    with pytest.raises(ValueError, match='foreground'):
        prepare_input(Image.new('RGBA', (32, 32)), remove_bg=False)


def test_quality_mode_delegates_without_loading_legacy_or_falling_back(monkeypatch):
    from src.furniture.image_to_3d import ImageTo3DGenerator
    def fail(*args, **kwargs):
        raise RuntimeError('quality reconstruction failed')
    dummy = SimpleNamespace(device='cpu', device_name='quality test', generate_from_image=fail)
    monkeypatch.setenv('ARCHIGEN_FURNITURE_QUALITY', '1')
    monkeypatch.setattr('src.furniture.quality_engine.QualityFurnitureEngine', lambda **kw: dummy)
    monkeypatch.setattr(ImageTo3DGenerator, '_init_triposr', lambda self: pytest.fail('Legacy loader called'))
    model = ImageTo3DGenerator()
    with pytest.raises(RuntimeError, match='quality reconstruction failed'):
        model.generate_from_image(Image.new('RGB', (8, 8)))


@pytest.mark.parametrize('height', [0, -1, float('nan'), 11])
def test_invalid_metric_scale_is_rejected(height):
    with pytest.raises(ValueError):
        to_room_coordinates(np.array([[0,0,0], [1,1,1]]), height)


def test_warmup_and_requests_share_one_furniture_model(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from src.furniture.image_to_3d import ImageTo3DGenerator
    constructions = []
    monkeypatch.setattr(ImageTo3DGenerator, '_instance', None)
    monkeypatch.setattr(ImageTo3DGenerator, '__init__', lambda self: constructions.append(self))
    with ThreadPoolExecutor(max_workers=8) as pool:
        instances = list(pool.map(lambda _: ImageTo3DGenerator.get_instance(), range(16)))
    assert len(constructions) == 1
    assert all(instance is constructions[0] for instance in instances)


def test_fast_atlas_unwrapping_keeps_layered_surfaces_non_overlapping():
    # Two parallel front-facing squares in the same mesh must never collide in UV space.
    import trimesh
    v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, .5], [1, 0, .5], [1, 1, .5], [0, 1, .5],
    ], dtype=np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    canonical, faces_out, uvs, texture, coverage = bake_neural_atlas(
        mesh, lambda pts: np.clip(pts, 0, 1), resolution=128
    )
    assert texture.size == (128, 128)
    assert coverage > 0.2
    pos_fwd, cov = rasterize_atlas(canonical, faces_out, uvs, 128)
    pos_rev, _ = rasterize_atlas(canonical, faces_out[::-1], uvs, 128)
    np.testing.assert_allclose(pos_fwd[cov], pos_rev[cov], atol=1e-5)
