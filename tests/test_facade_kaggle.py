"""Portable experiment contracts. No pretrained weights or CUDA are required."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from experiments.facade_sd15_controlnet.data import (
    condition_from_labels, labels_from_bridge, package_dataset, validate_dataset, FacadePairs, write_json,
)
from experiments.facade_sd15_controlnet.manage import stage_kernel, install_bundle
from src.data.facade_palette import CMP_TRAINING_LUT


def tiny_dataset(tmp_path):
    rows = []
    for index, split in enumerate(['train', 'val', 'test']):
        photo = tmp_path / f'source{index}.jpg'
        Image.new('RGB', (32, 24), (40 + index * 20, 90, 130)).save(photo)
        labels = np.full((24, 32), 2, dtype=np.uint8); labels[4:14, 3:12] = 3; labels[12:24, 19:26] = 4
        Image.fromarray(labels).save(photo.with_suffix('.png'))
        rows.append({'id': f'{split}/1.jpg', 'split': split, 'photo': photo.name,
            'photo_sha256': hashlib.sha256(photo.read_bytes()).hexdigest()})
    write_json(tmp_path / 'source.json', {'samples': rows, 'source': 'CMP test fixture', 'license': 'CC BY-SA'})
    package_dataset(tmp_path / 'source.json', tmp_path, tmp_path / 'dataset')
    return tmp_path / 'dataset'


def test_original_pairs_retain_splits_hashes_and_no_style_labels(tmp_path):
    root = tiny_dataset(tmp_path)
    manifest = validate_dataset(root)
    assert manifest['splits'] == {'train': 1, 'val': 1, 'test': 1}
    assert manifest['style_labels'] is None
    assert len(FacadePairs(root, 'train')) == 1
    with (root / manifest['samples'][0]['photo']).open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='checksum'): validate_dataset(root)


def test_runtime_and_training_conditioning_match_canonical_bridge_palette():
    labels = np.full((32, 32), 2, np.uint8)
    labels[3:12, 4:10] = 3; labels[15:32, 21:29] = 4; labels[:2] = 1
    expected, mask = condition_from_labels(labels, 64)
    recovered = labels_from_bridge(Image.fromarray(CMP_TRAINING_LUT[labels]))
    actual, recovered_mask = condition_from_labels(recovered, 64)
    assert np.array_equal(np.asarray(expected), np.asarray(actual))
    assert np.array_equal(np.asarray(mask), np.asarray(recovered_mask))
    assert set(np.unique(expected)) == {0, 255}
    assert np.asarray(mask)[14, 12] == 255


def test_kaggle_source_is_commit_pinned_and_notebook_cells_compile(tmp_path):
    import nbformat
    stage_kernel(tmp_path, owner='khalilayeb', commit='a' * 40)
    notebook = nbformat.read(tmp_path / 'train_facade_controlnet.ipynb', as_version=4)
    nbformat.validate(notebook)
    assert 'a' * 40 in ''.join(cell.source for cell in notebook.cells)
    for cell in notebook.cells:
        if cell.cell_type == 'code': compile(cell.source, '<notebook>', 'exec')
    metadata = json.loads((tmp_path / 'kernel-metadata.json').read_text())
    assert metadata['is_private'] is True and metadata['enable_gpu'] is True


def test_portable_mlflow_replay_is_idempotent_and_preserves_metric_history(tmp_path):
    pytest.importorskip('mlflow')
    from mlflow.tracking import MlflowClient
    from experiments.facade_sd15_controlnet.tracking import RunTracker, import_tracking
    run = RunTracker(tmp_path / 'kaggle', 'Facade portability fixture', {'seed': 42}, {'git_commit': 'abc'})
    run.metrics({'loss': .9}, 1); run.metrics({'loss': .7}, 2)
    report = tmp_path / 'report.json'; report.write_text('{"fixture": true}')
    run.artifact(report); run.finish()
    uri = 'sqlite:///' + (tmp_path / 'local.db').as_posix()
    imported = import_tracking(tmp_path / 'kaggle', uri)
    assert import_tracking(tmp_path / 'kaggle', uri) == imported
    client = MlflowClient(tracking_uri=uri)
    assert [(m.step, m.value) for m in client.get_metric_history(imported, 'loss')] == [(1, .9), (2, .7)]
    assert client.get_run(imported).data.tags['kaggle_source_run_id'] == run.run_id
    assert client.list_artifacts(imported, 'reports')[0].path == 'reports/report.json'
    resumed = RunTracker(tmp_path / 'kaggle', 'Facade portability fixture', {'seed': 42}, {'git_commit': 'abc'})
    resumed.metrics({'loss': .5}, 3); resumed.finish()
    assert resumed.run_id != run.run_id
    assert (tmp_path / 'kaggle/tracking-history' / run.run_id / 'tracking-events.jsonl').exists()
    assert import_tracking(tmp_path / 'kaggle', uri) != imported


def test_installer_rejects_archive_traversal_and_modified_weights(tmp_path):
    unsafe = tmp_path / 'bad.zip'
    with zipfile.ZipFile(unsafe, 'w') as zipped: zipped.writestr('../outside.txt', 'unsafe')
    with pytest.raises(ValueError, match='Unsafe'): install_bundle(unsafe, tmp_path / 'install')
    invalid = tmp_path / 'changed.zip'
    with zipfile.ZipFile(invalid, 'w') as zipped:
        zipped.writestr('export/model.json', json.dumps({'weights_sha256': 'not-the-weight-hash'}))
        zipped.writestr('export/controlnet/diffusion_pytorch_model.safetensors', b'not weights')
    with pytest.raises(ValueError, match='checksum'): install_bundle(invalid, tmp_path / 'install')


def test_opening_weighted_denoising_loss_prioritizes_openings():
    import torch
    from experiments.facade_sd15_controlnet.train import opening_weighted_loss
    prediction, target = torch.zeros(1, 4, 16, 16), torch.zeros(1, 4, 16, 16)
    prediction[:, :, 2:4, 2:4] = 1
    mask = torch.zeros(1, 1, 16, 16); mask[:, :, 2:4, 2:4] = 1
    assert opening_weighted_loss(prediction, target, mask, 2) > opening_weighted_loss(prediction, target, mask, 0)
    assert opening_weighted_loss(prediction, target, mask, 0).item() == pytest.approx(torch.nn.functional.mse_loss(prediction, target).item())


def test_tiny_controlnet_backpropagates_through_frozen_unet():
    diffusers = pytest.importorskip('diffusers')
    import torch
    from experiments.facade_sd15_controlnet.train import predict_noise, opening_weighted_loss
    torch.set_num_threads(2)
    unet = diffusers.UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
        down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
        up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'), block_out_channels=(16, 32),
        layers_per_block=1, cross_attention_dim=16, attention_head_dim=4, norm_num_groups=8)
    controlnet = diffusers.ControlNetModel.from_unet(unet, conditioning_embedding_out_channels=(4, 8, 8, 8))
    unet.requires_grad_(False)
    noisy, hidden, condition = torch.randn(1, 4, 8, 8), torch.randn(1, 8, 16), torch.randn(1, 3, 64, 64)
    prediction = predict_noise(controlnet, unet, noisy, torch.tensor([10]), hidden, condition)
    loss = opening_weighted_loss(prediction, torch.randn_like(prediction), torch.zeros(1, 1, 64, 64))
    loss.backward()
    assert all(parameter.grad is None for parameter in unet.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in controlnet.parameters())


def test_deferred_diffusion_does_not_run_during_floorplan_upload():
    from src.bridge.facade_bridge import FacadeBridge
    from types import SimpleNamespace
    bridge = FacadeBridge.__new__(FacadeBridge)
    bridge.inference_engine = SimpleNamespace(defer_automatic_generation=True)
    bridge.device_name = 'test worker'
    layout = {'metadata': {}, 'walls': [{'id': 'exterior', 'is_exterior': True, 'polygon_meters': [[0, 0], [4, 0], [4, .2], [0, .2]]}]}
    result = bridge.process_layout_facades(layout)
    assert result['front_wall_id'] == 'exterior'
    assert result['facade_results'] == {}
    assert layout['metadata']['facade_mode'] == 'on_demand'
