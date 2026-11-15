"""V2 semantic geometry, leakage protection, real adapter backprop and portability."""
import copy
import json

import numpy as np
from PIL import Image
import pytest

from experiments.facade_semantic_v2.conditioning import (
    cmp_labels, irf_labels, IRF_RGB, IRF_TO_CANONICAL, control_channels, letterbox, facade_crop)
from experiments.facade_semantic_v2.data import building_key, split_group


def test_distinct_semantics_and_exact_original_annotations():
    labels = np.array([[0, 1, 2, 3, 4, 5]], np.uint8)
    channels = control_channels(labels)
    assert channels.shape == (7, 1, 6)
    assert np.array_equal(channels[1:].sum(0), np.ones((1, 6)))
    assert channels[3, 0, 2] == 1 and channels[4, 0, 3] == 1
    assert np.array_equal(irf_labels(Image.fromarray(IRF_RGB[None])), IRF_TO_CANONICAL[None])
    assert cmp_labels(np.array([[2, 3, 4, 7]], np.uint8)).tolist() == [[1, 2, 3, 4]]
    with pytest.raises(ValueError, match='colours'):
        irf_labels(Image.new('RGB', (4, 4), (1, 2, 3)))


def test_letterbox_preserves_opening_aspect_and_padding_mask():
    labels = np.ones((100, 400), np.uint8); labels[20:80, 100:140] = 2
    photo = Image.fromarray(np.repeat((labels * 60)[:, :, None], 3, axis=2))
    image, mask, valid = letterbox(photo, labels)
    y, x = np.where(mask == 2)
    assert abs((x.max() - x.min() + 1) / (y.max() - y.min() + 1) - 2 / 3) < .03
    assert valid.sum() == 512 * 128
    assert not valid[:192].any() and not mask[:192].any()
    assert tuple(np.asarray(image)[0, 0]) == (127, 127, 127)


def test_storey_crop_keeps_intersecting_door_body_and_aligned_pair():
    labels = np.ones((300, 500), np.uint8)
    labels[25:65, 50:100] = 2; labels[175:215, 50:100] = 2
    labels[180:300, 300:345] = 3
    photo = Image.fromarray(np.repeat(labels[:, :, None], 3, axis=2))
    seen_ground = False
    for seed in range(10):
        image, mask = facade_crop(photo, labels, np.random.default_rng(seed), 1)
        assert np.array_equal(np.asarray(image)[:, :, 0], mask)
        if (mask == 3).any():
            assert (mask == 3).sum() == 120 * 45
            seen_ground = True
    assert seen_ground


def test_building_url_grouping_ignores_tracking_query():
    a = building_key('https://www.archdaily.com/123/a-house?ad_source=foo', 'A_1')
    b = building_key('https://www.archdaily.com/123/a-house/?ad_source=bar', 'B_7')
    assert a == b and split_group(a) == split_group(b)
    assert {split_group('building' + str(i)) for i in range(100)} == {'train', 'val', 'test'}


def tiny_models():
    diffusers = pytest.importorskip('diffusers')
    import torch
    torch.set_num_threads(2)
    unet = diffusers.UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
        down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'), up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'),
        block_out_channels=(16, 32), layers_per_block=1, cross_attention_dim=16, attention_head_dim=4, norm_num_groups=8)
    control = diffusers.ControlNetModel.from_unet(unet, conditioning_embedding_out_channels=(4, 8, 8, 8))
    return unet, control


def test_real_conditioning_upgrade_preserves_parent_edge_response():
    import torch
    from experiments.facade_semantic_v2.architecture import upgrade_conditioning
    _, control = tiny_models()
    edge = torch.rand(1, 1, 64, 64)
    old = control.controlnet_cond_embedding.conv_in(edge.repeat(1, 3, 1, 1)).detach()
    upgrade_conditioning(control)
    actual = control.controlnet_cond_embedding.conv_in(torch.cat([edge, torch.rand(1, 6, 64, 64)], 1))
    assert torch.allclose(old, actual, atol=1e-6)
    assert control.config.conditioning_channels == 7


def test_joint_backprop_adapter_roundtrip_and_frozen_base(tmp_path):
    pytest.importorskip('peft')
    import torch
    from experiments.facade_semantic_v2.architecture import upgrade_conditioning, add_facade_lora, save_lora, restore_lora
    from experiments.facade_sd15_controlnet.train import predict_noise
    unet, control = tiny_models()
    base = copy.deepcopy(unet.state_dict())
    upgrade_conditioning(control); params = add_facade_lora(unet, 2)
    # Zero-init ControlNet initially gates semantic gradients; several steps must open the gate.
    optimizer = torch.optim.AdamW(list(control.parameters()) + params, lr=.01)
    inputs = (torch.randn(1, 4, 8, 8), torch.tensor([10]), torch.randn(1, 8, 16), torch.rand(1, 7, 64, 64))
    initial = control.controlnet_cond_embedding.conv_in.weight.detach().clone()
    target = torch.randn(1, 4, 8, 8)
    for _ in range(3):
        loss = (predict_noise(control, unet, *inputs) - target).square().mean()
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    assert not torch.equal(initial[:, 1:], control.controlnet_cond_embedding.conv_in.weight[:, 1:])
    assert any('lora_B' in name and p.abs().sum() > 0 for name, p in unet.named_parameters())
    for name, p in unet.named_parameters():
        if not p.requires_grad:
            old_name = name.replace('.base_layer.', '.')
            assert torch.equal(p, base[old_name])
    save_lora(unet, tmp_path)
    other, _ = tiny_models(); other.load_state_dict(base); add_facade_lora(other, 2)
    restore_lora(other, tmp_path)
    unet.eval(); other.eval()
    assert torch.allclose(unet(inputs[0], inputs[1], inputs[2]).sample,
                          other(inputs[0], inputs[1], inputs[2]).sample, atol=1e-6)


def test_masked_loss_ignores_padding_and_has_finite_gradients():
    import torch
    from types import SimpleNamespace
    from experiments.facade_semantic_v2.losses import facade_loss
    scheduler = SimpleNamespace(alphas_cumprod=torch.tensor([.99, .5]), config=SimpleNamespace(prediction_type='epsilon'))
    config = {'background_loss_weight': .15, 'opening_loss_weight': 2, 'min_snr_gamma': 5}
    labels = torch.ones(1, 16, 16).long(); labels[:, 4:12, 4:12] = 2
    valid = torch.ones(1, 16, 16); valid[:, :8] = 0
    prediction = torch.ones(1, 4, 4, 4, requires_grad=True)
    a = facade_loss(prediction, torch.zeros_like(prediction), labels, valid, torch.tensor([1]), scheduler, config)
    altered = prediction.detach().clone(); altered[:, :, :2] = 10000
    b = facade_loss(altered, torch.zeros_like(altered), labels, valid, torch.tensor([1]), scheduler, config)
    assert torch.allclose(a, b)
    a.backward(); assert torch.isfinite(prediction.grad).all() and not prediction.grad[:, :, :2].any()


def test_kaggle_notebook_pins_private_inputs_and_compiles(tmp_path):
    from experiments.facade_semantic_v2.manage import stage_kernel
    metadata = stage_kernel(tmp_path, 'a' * 40, 'smoke')
    assert metadata['is_private'] and len(metadata['dataset_sources']) == 2 and metadata['kernel_sources']
    notebook = json.loads((tmp_path / 'train_facade_semantic.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<notebook>', 'exec')
    assert 'a' * 40 in (tmp_path / 'train_facade_semantic.ipynb').read_text()


def test_validator_rejects_building_leakage_and_corrupt_masks(tmp_path):
    from experiments.facade_semantic_v2.data import validate_dataset
    from experiments.facade_sd15_controlnet.data import digest, write_json
    rows = []
    for i, split in enumerate(('train', 'val')):
        photo = tmp_path / f'{i}.jpg'; labels = tmp_path / f'{i}.png'
        Image.new('RGB', (20, 10), (i * 90, 100, 40)).save(photo)
        Image.fromarray(np.ones((10, 20), np.uint8)).save(labels)
        rows.append({'id': str(i), 'source': 'IRF', 'split': split, 'group': 'same-building',
            'photo': photo.name, 'labels': labels.name, 'photo_sha256': digest(photo),
            'labels_sha256': digest(labels), 'native_size': [20, 10]})
    manifest = {'schema_version': 2, 'conditioning': 'facade_semantic7_v2', 'samples': rows,
                'counts': {'IRF': {'train': 1, 'val': 1, 'test': 0}}}
    write_json(tmp_path / 'sources.json', {'IRF': str(tmp_path)})
    write_json(tmp_path / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='crosses'):
        validate_dataset(tmp_path)
    rows[1]['group'] = 'different-building'; write_json(tmp_path / 'manifest.json', manifest)
    validate_dataset(tmp_path)
    Image.fromarray(np.ones((8, 20), np.uint8)).save(tmp_path / '1.png')
    with pytest.raises(ValueError, match='dimensions'):
        validate_dataset(tmp_path, verify_hashes=False)


def test_metrics_do_not_reward_or_penalize_letterbox_padding():
    pytest.importorskip('skimage')
    from experiments.facade_semantic_v2.evaluate import pair_metrics
    labels = np.ones((64, 64), np.uint8); valid = np.zeros_like(labels); valid[16:48] = 1
    photo = Image.new('RGB', (64, 64), (90, 90, 90))
    generated = np.asarray(photo).copy(); generated[:16] = 255; generated[48:] = 0
    scores = pair_metrics(Image.fromarray(generated), photo, labels, valid, perceptual=False)
    assert scores['ssim'] == 1 and scores['psnr'] >= 99


def test_actual_pipeline_accepts_semantic_tensor_and_reloads_lora(tmp_path):
    pytest.importorskip('peft')
    import torch
    from diffusers import StableDiffusionControlNetPipeline, AutoencoderKL, DDIMScheduler
    from experiments.facade_semantic_v2.architecture import upgrade_conditioning, add_facade_lora, save_lora
    unet, control = tiny_models()
    add_facade_lora(unet, 2); save_lora(unet, tmp_path)
    unet.delete_adapters('default')
    vae = AutoencoderKL(block_out_channels=(16, 32, 32, 32), norm_num_groups=8,
        down_block_types=('DownEncoderBlock2D',) * 4, up_block_types=('UpDecoderBlock2D',) * 4, latent_channels=4)
    pipeline = StableDiffusionControlNetPipeline(vae=vae, unet=unet, controlnet=upgrade_conditioning(control),
        text_encoder=None, tokenizer=None, scheduler=DDIMScheduler(), safety_checker=None,
        feature_extractor=None, requires_safety_checker=False)
    pipeline.load_lora_weights(str(tmp_path), weight_name='facade_lora.safetensors')
    pipeline.set_progress_bar_config(disable=True)
    with torch.no_grad():
        result = pipeline(prompt_embeds=torch.randn(1, 8, 16), negative_prompt_embeds=torch.zeros(1, 8, 16),
            image=torch.rand(1, 7, 64, 64), height=64, width=64, num_inference_steps=2, output_type='latent').images
    assert result.shape == (1, 4, 8, 8) and torch.isfinite(result).all()


def test_full_dataset_audit_keeps_locked_splits_and_skips_sparse_training(tmp_path):
    pytest.importorskip('openpyxl')
    from openpyxl import Workbook
    from experiments.facade_sd15_controlnet.data import package_dataset, write_json, digest
    from experiments.facade_semantic_v2.data import build_dataset, validate_dataset, FacadeDataset
    cmp = tmp_path / 'cmp'; cmp.mkdir(); irf = tmp_path / 'irf'; irf.mkdir()
    rows = []
    for i, split in enumerate(('train', 'val', 'test', 'train')):
        photo = cmp / f'{i}.jpg'; Image.new('RGB', (32, 24), (i * 40, 100, 80)).save(photo)
        labels = np.full((24, 32), 2 if i != 3 else 1, np.uint8); labels[3, 3] = 3
        Image.fromarray(labels).save(photo.with_suffix('.png'))
        rows.append({'id': str(i), 'split': split, 'photo': photo.name, 'photo_sha256': digest(photo)})
    write_json(cmp / 'source.json', {'samples': rows, 'source': 'synthetic CMP fixture', 'license': 'fixture'})
    package_dataset(cmp / 'source.json', cmp, tmp_path / 'cmp-pack')
    book = Workbook(); sheet = book.active; sheet.title = 'case information'
    sheet.append(['Sample ID', 'Case URL', 'Architects', 'Photographs'])
    counts = dict.fromkeys(('train', 'val', 'test'), 0)
    for i in range(2000):
        url = 'https://www.archdaily.com/fixture-' + str(i)
        split = split_group(building_key(url, str(i)))
        if counts[split] >= 10:
            continue
        counts[split] += 1
        stem = f'A_{i:04d}'
        rng = np.random.default_rng(i)
        Image.fromarray(rng.integers(0, 255, (256, 512, 3), dtype=np.uint8)).save(irf / (stem + '.jpg'))
        labels = np.full((256, 512), 2, np.uint8); labels[40:100, 60:120] = 3
        Image.fromarray(IRF_RGB[labels]).save(irf / (stem + '.png'))
        write_json(irf / (stem + '.json'), {'imageWidth': 512, 'imageHeight': 256, 'shapes': [{'label': 'wall'}]})
        sheet.append([stem + '.png', url, 'Fixture architect', 'Fixture photographer'])
        if min(counts.values()) == 10:
            break
    book.save(irf / 'sample_list.xlsx')
    root = tmp_path / 'prepared'
    manifest = build_dataset(tmp_path / 'cmp-pack', irf, root)
    validate_dataset(root)
    assert manifest['counts']['CMP'] == {'train': 2, 'val': 1, 'test': 1}
    assert manifest['counts']['IRF'] == {'train': 10, 'val': 10, 'test': 10}
    train = FacadeDataset(root, 'train', source='CMP')
    assert len(train) == 1 and train.rows[0]['id'] == 'CMP/0'
    assert len(json.loads((root / 'irf-credits.json').read_text())) == 30
    val = FacadeDataset(root, 'val', augment=True)
    assert not val.augment
