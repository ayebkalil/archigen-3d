"""Offline geometry/resource contracts; these do not certify model image quality."""
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from src.models.facade_diffusion import (
    ROOT, cached_models, cmp_control, fixture_control, prompt_for, resource_report, uv_box,
)


def test_bottom_origin_uvs_preserve_physical_door_and_upper_window_positions():
    fixture = {'uv_bounds': {'u_min': 0.1, 'v_min': 0.1, 'u_max': 0.9, 'v_max': 0.9},
               'openings': [{'uv_rect': [.2, .1, .3, .4]}, {'uv_rect': [.6, .6, .8, .8]}]}
    edge, mask = fixture_control(fixture, 100)
    mask = np.asarray(mask)
    assert mask[80, 25] == 255  # Ground-floor door is low in top-origin image.
    assert mask[30, 70] == 255  # Upper-floor window is high.
    assert mask[20, 25] == 0  # No flipped duplicate opening.
    assert np.asarray(edge)[60, 20, 0] == 255
    assert uv_box([.2, .1, .3, .4], 100) == (20, 60, 30, 90)


@pytest.mark.parametrize('rect', [[0, 0, 0, 1], [-.1, 0, 1, 1], [0, 0, float('nan'), 1]])
def test_invalid_uvs_cannot_silently_move_openings(rect):
    with pytest.raises(ValueError):
        uv_box(rect, 1024)


def test_cmp_edges_come_from_label_boundaries_and_opening_ids_only():
    labels = np.full((32, 32), 2, np.uint8)
    labels[8:16, 4:12] = 3
    labels[20:32, 20:28] = 4
    labels[2:4, :] = 5  # Cornice must not become an opening mask.
    edges, mask = cmp_control(labels, 64)
    assert edges.size == (64, 64)
    assert set(np.unique(edges)) == {0, 255}
    assert np.asarray(mask)[20, 12] == 255
    assert np.asarray(mask)[45, 45] == 255
    assert np.asarray(mask)[5, 10] == 0
    assert np.asarray(edges)[20, 7, 0] == 255
    for invalid in (np.zeros((32, 32, 3), np.uint8), np.full((2, 2), 13), np.full((2, 2), 1.5)):
        with pytest.raises(ValueError):
            cmp_control(invalid)


def test_insufficient_resources_block_loading_and_download_checks_no_gpu(tmp_path):
    with patch('shutil.disk_usage', return_value=SimpleNamespace(free=3 * 1024**3)), \
         patch('psutil.virtual_memory', return_value=SimpleNamespace(available=1024**3)), \
         patch('src.models.facade_diffusion.gpu_memory', return_value={'total_mib': 4096, 'used_mib': 2800}) as gpu:
        report = resource_report(tmp_path)
        assert not report['ready'] and len(report['blockers']) == 3
        gpu.reset_mock()
        download = resource_report(tmp_path, download=True)
        assert not download['ready'] and len(download['blockers']) == 1
        gpu.assert_not_called()


def test_cached_weights_reduce_disk_requirement_but_not_runtime_memory_guard(tmp_path):
    with patch('shutil.disk_usage', return_value=SimpleNamespace(free=3 * 1024**3)), \
         patch('psutil.virtual_memory', return_value=SimpleNamespace(available=9 * 1024**3)), \
         patch('src.models.facade_diffusion.gpu_memory', return_value={'total_mib': 4096, 'used_mib': 800}):
        assert resource_report(tmp_path, cached=True)['ready']
        assert not resource_report(tmp_path, cached=False)['ready']


def test_disk_offload_accepts_lower_ram_only_with_room_for_memory_mapped_weights(tmp_path):
    with patch('shutil.disk_usage', return_value=SimpleNamespace(free=20 * 1024**3)) as disk, \
         patch('psutil.virtual_memory', return_value=SimpleNamespace(available=4 * 1024**3)), \
         patch('src.models.facade_diffusion.gpu_memory', return_value={'total_mib': 4096, 'used_mib': 300}):
        assert not resource_report(tmp_path, cached=True)['ready']
        assert resource_report(tmp_path, cached=True, disk_dir=tmp_path)['ready']
        disk.return_value.free = 5 * 1024**3
        assert not resource_report(tmp_path, cached=True, disk_dir=tmp_path)['ready']


def test_partial_cache_is_not_treated_as_ready(tmp_path):
    config = {'base': {'repo': 'owner/model', 'revision': 'abc', 'files': ['config.json', 'model.safetensors']}}
    path = tmp_path / 'models--owner--model/snapshots/abc'
    path.mkdir(parents=True)
    (path / 'config.json').write_text('{}')
    assert cached_models(config, tmp_path) is None
    (path / 'model.safetensors').write_bytes(b'fixture')
    assert cached_models(config, tmp_path) == {'base': path}


def test_model_sources_are_pinned_and_styles_are_text_guidance():
    config = json.loads((ROOT / 'configs/facade_diffusion.json').read_text())
    assert set(config) == {'base', 'controlnet', 'vae'}
    for model in config.values():
        assert len(model['revision']) == 40
        assert not any('*' in name for name in model['files'])
    assert 'brick mortar' in prompt_for('Traditional Brick')
    with pytest.raises(KeyError):
        prompt_for('invented style')


def test_weight_download_resumes_partial_chunk_and_verifies_content(tmp_path):
    from src.models.facade_download import download_weight
    contents = b'abcdefghijklm'
    dest = tmp_path / 'model.safetensors'
    dest.with_suffix('.safetensors.archigen.part').write_bytes(b'abcde')
    spec = {'size': len(contents), 'sha256': hashlib.sha256(contents).hexdigest()}
    def fetch(url, start, end, size):
        assert size == len(contents)
        return contents[start:end+1]
    with patch('src.models.facade_download.CHUNK', 4), \
         patch('src.models.facade_download.fetch_range', side_effect=fetch):
        download_weight('https://example.invalid/model', dest, spec)
    assert dest.read_bytes() == contents
    # A cached corrupt file must not be accepted or silently replaced.
    dest.write_bytes(b'x' * len(contents))
    with pytest.raises(ValueError, match='checksum'):
        download_weight('https://example.invalid/model', dest, spec)


@pytest.mark.parametrize('status,content_range,body', [
    (200, 'bytes 0-3/4', b'abcd'), (206, 'bytes 4-7/8', b'abcd'), (206, 'bytes 0-3/4', b'ab')])
def test_range_download_rejects_full_wrong_or_truncated_responses(status, content_range, body):
    from src.models.facade_download import fetch_range
    with patch('requests.get') as get, patch('src.models.facade_download.time.sleep'):
        response = get.return_value.__enter__.return_value
        response.status_code = status
        response.headers = {'Content-Range': content_range}
        response.iter_content.return_value = [body]
        with pytest.raises(ValueError):
            fetch_range('https://example.invalid/model', 0, 3, 4)
