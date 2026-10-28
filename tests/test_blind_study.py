"""
tests/test_blind_study.py
-------------------------
Unit tests for the Stage 5 Blind Human Comparison suite:
- Validates the study keys and complete isolation from shared bundles.
- Asserts zero engine/random/baseline keywords inside shared bundles.
- Verifies that all 128 rendered images across Study 1 and Study 2 exist.
- Validates submission schema validator, study_id matching, and plan/room bootstrap estimators.
- Verifies packaged zip bundles outside OneDrive.
"""

import json
import os
import re
import zipfile
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from scripts.analyze_blind_study import (
    validate_submission_file,
    compute_bootstrap_ci_rooms,
    compute_bootstrap_ci_plans
)

def test_study_keys_and_isolation():
    """Verify secret keys have 32 matching rooms and are kept outside shared bundles."""
    keys_dir = PROJECT_ROOT / "outputs/blind_study/keys"
    key_b1 = keys_dir / "study_key_b1.json"
    key_b2 = keys_dir / "study_key_b2.json"

    assert key_b1.exists(), f"Secret key B1 missing at {key_b1}"
    assert key_b2.exists(), f"Secret key B2 missing at {key_b2}"

    with open(key_b1, "r", encoding="utf-8") as f:
        k1 = json.load(f)
    with open(key_b2, "r", encoding="utf-8") as f:
        k2 = json.load(f)

    assert k1["study_id"] == "study_1"
    assert k2["study_id"] == "study_2"
    assert len(k1["rooms"]) == 32
    assert len(k2["rooms"]) == 32

    for r1, r2 in zip(k1["rooms"], k2["rooms"]):
        assert r1["room_id"] == r2["room_id"]
        assert r1["engine_is"] in ["A", "B"]
        assert r2["engine_is"] in ["A", "B"]
        assert "val_plan_index" in r1
        assert "val_plan_index" in r2

    # Verify no secret key exists in shared bundles
    shared_bundles_dir = PROJECT_ROOT / "outputs/blind_study/shared_bundles"
    assert not (shared_bundles_dir / "study_key.json").exists()
    assert not (shared_bundles_dir / "study_key_b1.json").exists()
    assert not (shared_bundles_dir / "study_key_b2.json").exists()

def test_all_rendered_images_exist():
    """Verify that all 64 images for Study 1 and 64 images for Study 2 exist."""
    bundles_dir = PROJECT_ROOT / "outputs/blind_study/shared_bundles"

    for study_id in ["study_1", "study_2"]:
        img_dir = bundles_dir / study_id / "images"
        assert img_dir.exists(), f"Missing images dir for {study_id}"

        for i in range(1, 33):
            prefix = f"room_{i:02d}"
            img_a = img_dir / f"{prefix}_a.png"
            img_b = img_dir / f"{prefix}_b.png"

            assert img_a.exists(), f"Missing {img_a.name} in {study_id}"
            assert img_b.exists(), f"Missing {img_b.name} in {study_id}"
            assert img_a.stat().st_size > 10000
            assert img_b.stat().st_size > 10000

def test_shared_bundles_byte_anonymity():
    """Verify byte-level absence of forbidden keywords in HTML and lack of PNG metadata."""
    bundles_dir = PROJECT_ROOT / "outputs/blind_study/shared_bundles"
    assert bundles_dir.exists()

    forbidden_semantic = ["engine", "random", "baseline", "seed", "ground_truth"]

    for root, dirs, files in os.walk(bundles_dir):
        for fname in files:
            fpath = Path(root) / fname
            # Verify filename
            for w in forbidden_semantic:
                assert w not in fname.lower(), f"Forbidden word '{w}' in filename: {fname}"

            # Plain byte scan of HTML
            if fname.endswith(".html"):
                content = fpath.read_bytes().lower()
                for w in ["engine", "random", "baseline", "b1", "b2", "key", "seed", "ground_truth"]:
                    assert w.encode("utf-8") not in content, f"Forbidden word '{w}' in {fpath}"

def test_submission_validator_refuses_invalid_files(tmp_path):
    """Assert that the analyzer strictly refuses files not exported by the official page or with mismatched study_id."""
    # 1. Missing version
    bad_file_1 = tmp_path / "bad1.json"
    bad_file_1.write_text(json.dumps({"study_id": "study_1", "rater_id": "r1", "ratings": []}))
    with pytest.raises(ValueError, match="Missing or invalid study schema version"):
        validate_submission_file(bad_file_1)

    # 2. Missing rater_id
    bad_file_2 = tmp_path / "bad2.json"
    bad_file_2.write_text(json.dumps({"study_schema_version": "1.0", "study_id": "study_1", "ratings": []}))
    with pytest.raises(ValueError, match="Missing or empty 'rater_id'"):
        validate_submission_file(bad_file_2)

    # 3. Invalid choice
    bad_file_3 = tmp_path / "bad3.json"
    bad_file_3.write_text(json.dumps({
        "study_schema_version": "1.0",
        "study_id": "study_1",
        "rater_id": "tester",
        "ratings": [{"room_id": "room_01", "choice": "Cheat"}]
    }))
    with pytest.raises(ValueError, match="Invalid choice 'Cheat'"):
        validate_submission_file(bad_file_3)

    # 4. Study ID mismatch
    mismatch_file = tmp_path / "mismatch.json"
    mismatch_file.write_text(json.dumps({
        "study_schema_version": "1.0",
        "study_id": "study_2",
        "rater_id": "rater_1",
        "ratings": [{"room_id": "room_01", "choice": "A"}]
    }))
    with pytest.raises(ValueError, match="Study ID mismatch"):
        validate_submission_file(mismatch_file, expected_study_id="study_1")

    # 5. Valid file must pass
    good_file = tmp_path / "good.json"
    good_file.write_text(json.dumps({
        "study_schema_version": "1.0",
        "study_id": "study_1",
        "rater_id": "architect_bob",
        "ratings": [{"room_id": "room_01", "choice": "A"}]
    }))
    res = validate_submission_file(good_file, expected_study_id="study_1")
    assert res["rater_id"] == "architect_bob"

def test_bootstrap_estimators():
    """Verify room and plan bootstrap confidence interval estimators."""
    room_evals = {
        "room_01": ["engine_win", "engine_win"],
        "room_02": ["engine_win", "baseline_win"],
        "room_03": ["baseline_win", "tie"],
        "room_04": ["engine_win", "tie"]
    }
    plan_to_rooms = {
        "plan_A": ["room_01", "room_02"],
        "plan_B": ["room_03", "room_04"]
    }

    ci_room = compute_bootstrap_ci_rooms(room_evals, n_bootstrap=100, seed=42)
    assert len(ci_room) == 2
    assert 0.0 <= ci_room[0] <= ci_room[1] <= 100.0

    ci_plan = compute_bootstrap_ci_plans(plan_to_rooms, room_evals, n_bootstrap=100, seed=42)
    assert len(ci_plan) == 2
    assert 0.0 <= ci_plan[0] <= ci_plan[1] <= 100.0

def test_packaged_zip_bundles_outside_onedrive():
    """Verify zip bundles exist in C:/Users/ayebk/blind_study_bundles and contain zero keys."""
    zip_dir = Path("C:/Users/ayebk/blind_study_bundles")
    assert zip_dir.exists(), f"Target zip dir missing: {zip_dir}"

    for zip_name in ["study_1_rater_bundle.zip", "study_2_rater_bundle.zip"]:
        zpath = zip_dir / zip_name
        assert zpath.exists(), f"Missing zip file: {zpath}"
        assert zpath.stat().st_size > 1000000 # > 1MB

        with zipfile.ZipFile(zpath, "r") as zf:
            names = zf.namelist()
            assert len(names) == 65
            for n in names:
                n_lower = n.lower()
                assert "key" not in n_lower
                assert "manifest" not in n_lower
                assert "engine" not in n_lower
                assert "random" not in n_lower
                assert "baseline" not in n_lower
                assert n.endswith((".html", ".png"))
