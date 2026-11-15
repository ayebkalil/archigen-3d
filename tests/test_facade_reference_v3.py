"""Physical appearance constraints and real reference-guided semantic inference."""
import numpy as np
import json
from PIL import Image
import pytest

from experiments.facade_reference_v3.materials import (
    opening_mask, lock_surface, facade_chains, material_surface, fingerprint)


def panel():
    return {'wall_id': 'a', 'level_id': 'ground', 'wall_length_m': 8., 'wall_height_m': 2.,
            'uv_bounds': {'u_min': 0, 'u_max': 1, 'v_min': .375, 'v_max': .625},
            'facade_frame': {'origin': [8., 0], 'direction': [-1., 0], 'length_m': 8.},
            'openings': [{'position_along_wall_m': 2., 'width_m': 1., 'height_m': 1., 'sill_m': .5}]}


def test_edits_are_bounded_in_metres_including_letterbox_and_trim():
    spec = panel()
    mask = opening_mask(spec, 800, trim_m=0)
    y, x = np.where(mask)
    assert (x.min(), x.max(), y.min(), y.max()) == (150, 249, 350, 449)
    expanded = opening_mask(spec, 800, trim_m=.05)
    assert expanded.sum() == 110 * 110
    # Increasing physical facade height preserves the opening's metric dimensions.
    spec['wall_height_m'] = 4.
    assert opening_mask(spec, 800, trim_m=0).sum() == 100 * 50


def test_arbitrary_neural_predictions_cannot_change_protected_wall_pixels():
    rng = np.random.default_rng(42)
    base = Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
    proposal = Image.new('RGB', (512, 512), 'black')
    result, mask = lock_surface(base, proposal, panel())
    assert np.array_equal(np.asarray(result)[~mask], np.asarray(base)[~mask])
    assert np.any(np.asarray(result)[mask] != np.asarray(base)[mask])
    spec = panel(); spec['openings'] = []
    blank, mask = lock_surface(base, proposal, spec)
    assert not mask.any() and np.array_equal(blank, base)


def test_corner_phase_closes_and_preserves_physical_scale_across_detached_buildings():
    panels, walls = [], []
    for prefix, offset in [('house', 0), ('annex', 20)]:
        points = np.array([[0, 0], [8, 0], [8, 4], [0, 4], [0, 0]], float) + [offset, 0]
        for i, (a, b) in enumerate(zip(points, points[1:])):
            spec = panel(); spec['wall_id'] = f'{prefix}-{i}'
            length = float(np.linalg.norm(b - a))
            spec['facade_frame'] = {'origin': b.tolist(), 'direction': ((a - b) / length).tolist(), 'length_m': length}
            spec['wall_length_m'] = length
            panels.append(spec); walls.append({'id': spec['wall_id'], 'axis_meters': [a.tolist(), b.tolist()]})
    phases = facade_chains({'walls': walls, 'storeys': [{'id': 'ground', 'elevation': 0}]}, panels)
    assert len(phases) == 8
    json.dumps(phases)  # Runtime phase metadata must contain portable Python scalars.
    for prefix in ('house', 'annex'):
        chain = [phases[f'{prefix}-{i}'] for i in range(4)]
        assert all(p['closed_chain'] and p['period_m'] == 2 for p in chain)
        assert [p['phase_m'] for p in chain] == [0, 8, 12, 20]
        assert (chain[-1]['phase_m'] + chain[-1]['face_length_m']) % 2 == 0


def test_shared_material_uses_identical_physical_scale_and_neutral_padding():
    spec = panel(); tile = Image.new('RGB', (16, 16), (210, 200, 190))
    phase = {'start': [0, 0], 'direction': [1, 0], 'phase_m': 0., 'period_m': 2., 'elevation_m': 0.}
    pixels = np.asarray(material_surface(spec, tile, phase, size=256))
    assert np.all(pixels[96:160] == [210, 200, 190])
    assert np.all(pixels[:96] == 127) and np.all(pixels[160:] == 127)
    assert fingerprint({'a': 1, 'b': 2}) == fingerprint({'b': 2, 'a': 1})
    assert fingerprint({'strength': .35}) != fingerprint({'strength': .5})


def test_real_img2img_pipeline_reuses_components_and_accepts_semantic7():
    torch = pytest.importorskip('torch')
    diffusers = pytest.importorskip('diffusers')
    torch.set_num_threads(2)
    from experiments.facade_semantic_v2.architecture import upgrade_conditioning
    unet = diffusers.UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
        down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'), up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'),
        block_out_channels=(16, 32), layers_per_block=1, cross_attention_dim=16, attention_head_dim=4, norm_num_groups=8)
    control = upgrade_conditioning(diffusers.ControlNetModel.from_unet(unet, conditioning_embedding_out_channels=(4, 8, 8, 8)))
    vae = diffusers.AutoencoderKL(block_out_channels=(16, 32, 32, 32), norm_num_groups=8,
        down_block_types=('DownEncoderBlock2D',) * 4, up_block_types=('UpDecoderBlock2D',) * 4, latent_channels=4)
    old = diffusers.StableDiffusionControlNetPipeline(vae=vae, unet=unet, controlnet=control,
        text_encoder=None, tokenizer=None, scheduler=diffusers.DDIMScheduler(), safety_checker=None,
        feature_extractor=None, requires_safety_checker=False)
    pipeline = diffusers.StableDiffusionControlNetImg2ImgPipeline(**old.components, requires_safety_checker=False)
    assert pipeline.unet is old.unet and pipeline.controlnet is old.controlnet and pipeline.vae is old.vae
    pipeline.set_progress_bar_config(disable=True)
    with torch.no_grad():
        result = pipeline(prompt_embeds=torch.randn(1, 8, 16), negative_prompt_embeds=torch.zeros(1, 8, 16),
            image=Image.new('RGB', (64, 64), 'white'), control_image=torch.rand(1, 7, 64, 64),
            strength=.5, num_inference_steps=4, generator=torch.Generator().manual_seed(42), output_type='latent').images
    assert result.shape == (1, 4, 8, 8) and torch.isfinite(result).all()
