"""
Unit tests for model registry cryptographic integrity and export lineage.
Ensures that deployed ONNX runtime artifacts strictly match their recorded
hashes and originate from verified PyTorch champion checkpoints.
"""

import json
import hashlib
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def test_onnx_champion_hash_matches_exported_from_registry():
    """
    Asserts that:
    1. The ONNX champion on disk matches the SHA256 recorded in model_registry.json.
    2. The checkpoint it was exported from exists and its SHA256 matches exported_from_sha256.
    3. The registry's champion_floorplan section confirms this exact lineage.
    Fails immediately if models have been altered, desynchronized, or corrupted.
    """
    registry_path = PROJECT_ROOT / "models" / "model_registry.json"
    assert registry_path.exists(), f"Registry file not found at {registry_path}"

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    # 1. Locate ONNX champion entry
    models_dict = registry.get("models", {})
    onnx_entry = None
    for key, val in models_dict.items():
        if "floorplan_multitask.onnx" in key:
            onnx_entry = val
            break

    assert onnx_entry is not None, "floorplan_multitask.onnx entry not found in model_registry.json"
    assert "exported_from_checkpoint" in onnx_entry, "Missing exported_from_checkpoint in registry"
    assert "exported_from_sha256" in onnx_entry, "Missing exported_from_sha256 in registry"

    # 2. Check disk ONNX file
    onnx_path = PROJECT_ROOT / "models" / "saved" / "floorplan_unetpp" / "floorplan_multitask.onnx"
    assert onnx_path.exists(), f"ONNX champion file not found at {onnx_path}"
    disk_onnx_sha = compute_sha256(onnx_path)
    assert disk_onnx_sha == onnx_entry["sha256"], (
        f"ONNX champion hash mismatch! Disk: {disk_onnx_sha}, Registry: {onnx_entry['sha256']}"
    )

    # 3. Check source PyTorch checkpoint
    source_rel_path = onnx_entry["exported_from_checkpoint"]
    source_path = PROJECT_ROOT / source_rel_path
    assert source_path.exists(), f"Exported-from source checkpoint not found at {source_path}"
    disk_source_sha = compute_sha256(source_path)
    assert disk_source_sha == onnx_entry["exported_from_sha256"], (
        f"Source checkpoint hash mismatch! Disk: {disk_source_sha}, Registry: {onnx_entry['exported_from_sha256']}"
    )

    # 4. Check authoritative champion_floorplan section
    champ_floorplan = registry.get("champion_floorplan", {})
    assert champ_floorplan.get("sha256_onnx") == disk_onnx_sha, (
        f"champion_floorplan.sha256_onnx ({champ_floorplan.get('sha256_onnx')}) != disk ({disk_onnx_sha})"
    )
    assert champ_floorplan.get("sha256_pytorch") == disk_source_sha, (
        f"champion_floorplan.sha256_pytorch ({champ_floorplan.get('sha256_pytorch')}) != disk ({disk_source_sha})"
    )

def test_onnx_fallback_hash_matches_exported_from_registry():
    """
    Asserts that the fallback ONNX router matches the registry hash and
    originates from the blend alpha=0.3 checkpoint.
    """
    registry_path = PROJECT_ROOT / "models" / "model_registry.json"
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    champ_floorplan = registry.get("champion_floorplan", {})
    fb_onnx_path = PROJECT_ROOT / "models" / "saved" / "floorplan_unetpp" / "floorplan_fallback.onnx"
    fb_pth_path = PROJECT_ROOT / "models" / "saved" / "floorplan_unetpp" / "multitask_resnet34_blend_alpha_0_3.pth"

    assert fb_onnx_path.exists(), f"Fallback ONNX not found at {fb_onnx_path}"
    assert fb_pth_path.exists(), f"Fallback PyTorch not found at {fb_pth_path}"

    disk_fb_onnx_sha = compute_sha256(fb_onnx_path)
    disk_fb_pth_sha = compute_sha256(fb_pth_path)

    assert disk_fb_onnx_sha == champ_floorplan["sha256_fallback_onnx"], (
        f"Fallback ONNX sha mismatch: {disk_fb_onnx_sha} != {champ_floorplan['sha256_fallback_onnx']}"
    )
    assert disk_fb_pth_sha == champ_floorplan["sha256_fallback_pytorch"], (
        f"Fallback PyTorch sha mismatch: {disk_fb_pth_sha} != {champ_floorplan['sha256_fallback_pytorch']}"
    )
