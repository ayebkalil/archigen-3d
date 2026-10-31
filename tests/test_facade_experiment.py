"""Regression contracts for opt-in high-resolution facade synthesis."""
import io
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.architecture import FacadeGenerator512, MultiScaleFacadeDiscriminator, NEUTRAL_STYLE
from src.models.losses import OpeningSharpnessLoss, VGG19PerceptualLoss
from src.bridge.facade_bridge import FacadeBridge
from src.bridge.facade_experimental import ExperimentalFacadeInference


class TinyPrior(torch.nn.Module):
    def forward(self, x):
        return x * .5


def small_generator():
    with patch('src.models.architecture.UNetGenerator', TinyPrior):
        return FacadeGenerator512(width=4, blocks=1)


def test_512_prior_preservation_and_style_gradient():
    torch.set_num_threads(2)
    model = small_generator().train()
    assert not model.coarse.training
    x = torch.zeros(2, 3, 512, 512)
    s = torch.tensor([0, 3])
    initial = model(x, s)
    assert initial.shape == x.shape
    assert torch.equal(initial, x)
    # After a nonzero residual head, loss must reach both style embedding and encoder.
    torch.nn.init.normal_(model.output.weight, std=.02)
    out = model(x, s)
    out.square().mean().backward()
    assert model.style_embedding.weight.grad[[0, 3]].abs().sum() > 0
    assert model.stem.weight.grad is not None
    assert not torch.allclose(out[0], out[1])


def test_discriminators_have_distinct_scales_and_input_gradients():
    d = MultiScaleFacadeDiscriminator(width=4)
    photo = torch.randn(1, 3, 512, 512, requires_grad=True)
    result = d(torch.zeros_like(photo), photo, torch.tensor([NEUTRAL_STYLE]))
    assert [item[0].shape[-1] for item in result] == [64, 32]
    sum(item[0].mean() for item in result).backward()
    assert photo.grad.abs().sum() > 0


def test_sharpness_is_masked_and_differentiable():
    loss = OpeningSharpnessLoss()
    target = torch.zeros(1, 3, 32, 32)
    pred = torch.randn_like(target).requires_grad_()
    sketch = torch.ones_like(target)  # no opening
    assert loss(pred, target, sketch) == 0
    sketch[:, :, 8:24, 8:24] = torch.tensor([0, 128, 255]).view(1, 3, 1, 1) / 127.5 - 1
    value = loss(pred, target, sketch)
    assert value > 0
    value.backward()
    assert pred.grad.abs().sum() > 0
    assert loss(target, target, sketch) == 0


def test_vgg_exact_layers_and_prediction_gradients():
    loss = VGG19PerceptualLoss(pretrained=False)
    assert loss.indices == (3, 8, 13, 22, 31)
    pred = torch.randn(1, 3, 32, 32, requires_grad=True)
    target = torch.randn_like(pred, requires_grad=True)
    loss(pred, target).backward()
    assert pred.grad.abs().sum() > 0
    assert target.grad is None
    assert all(p.grad is None for p in loss.parameters())


def test_untrained_styles_are_reported_as_neutral():
    engine = ExperimentalFacadeInference.__new__(ExperimentalFacadeInference)
    engine.supported_styles = []
    info = engine.describe('Traditional Brick')
    assert engine.style_id('Traditional Brick') == NEUTRAL_STYLE
    assert info['style_conditioned'] is False
    assert info['supported_styles'] == []
    with pytest.raises(ValueError): engine.style_id('unknown')


def test_512_bridge_uvs_and_no_heuristic_tint():
    class Engine:
        resolution, neural_style = 512, True
        def generate(self, data, style):
            assert Image.open(io.BytesIO(data)).size == (512, 512)
            buf = io.BytesIO(); Image.new('RGB', (512, 512), (128, 128, 128)).save(buf, format='PNG')
            return buf.getvalue()
    bridge = FacadeBridge.__new__(FacadeBridge)
    bridge.inference_engine, bridge.enhance_textures = Engine(), False
    wall = {'polygon_meters': [[0, 0], [8, 0], [8, .2], [0, .2]], 'openings': []}
    with patch('src.bridge.facade_bridge.apply_facade_style', side_effect=AssertionError('No tint allowed')):
        result = bridge.render_wall_facade(wall, num_floors=2, is_front=True, style='Traditional Brick')
    assert result['photo_image'].size == (512, 512)
    assert np.asarray(result['photo_image']).min() == 128
    bounds = result['uv_bounds']
    for op in result['openings']:
        u = bounds['u_min'] + op['position_along_wall_m'] / 8 * (bounds['u_max'] - bounds['u_min'])
        assert u == pytest.approx((op['uv_rect'][0]+op['uv_rect'][2])/2, abs=1e-4)


def test_missing_experiment_never_silently_loads_production(tmp_path, monkeypatch):
    monkeypatch.setenv('ARCHIGEN_FACADE_EXPERIMENT', str(tmp_path/'missing.pth'))
    with pytest.raises(FileNotFoundError):
        FacadeBridge(prefer_cuda=False)


def test_experimental_api_metadata_does_not_claim_legacy_styles(monkeypatch):
    import asyncio
    import importlib
    from types import SimpleNamespace
    from backend.app import main
    production = importlib.import_module('app.floorplan_engine')
    engine = SimpleNamespace(describe=lambda style: {'resolution': 512, 'supported_styles': []})
    pipeline = SimpleNamespace(fallback_model=None, onnx_fallback=None,
        checkpoint_hashes={'facade_experimental': True}, bridge=SimpleNamespace(inference_engine=engine))
    monkeypatch.setattr(production.FloorplanPipeline, 'get_instance', lambda: pipeline)
    response = asyncio.run(main.get_model_metadata())
    assert response.target_resolution == [512, 512]
    assert response.supported_styles == []
