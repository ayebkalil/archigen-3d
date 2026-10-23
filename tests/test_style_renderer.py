"""
Unit tests for Procedural StyleRenderer and Restyled Dataset.
Ensures zero mask leakage, identical vector ground-truth, and valid image tensors.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from src.floorplan.style_renderer import render_floorplan_style
from src.floorplan.restyled_dataset import RestyledCubiCasaDataset

SUPPORTED_STYLES = [
    "cad_blueprint",
    "cad_white_blue",
    "black_and_white",
    "hatched_walls",
    "faint_pencil_scan",
    "annotated_dimensions"
]

def test_style_renderer_styles():
    """Verify that render_floorplan_style produces valid RGB 512x512 images for all 6 drafting styles."""
    cubicasa_dir = Path("data/CubiCasa5k")
    svg_files = list(cubicasa_dir.glob("**/model.svg"))
    if not svg_files:
        pytest.skip("CubiCasa5k dataset not present locally for full SVG render test.")

    svg_path = svg_files[0]

    for style in SUPPORTED_STYLES:
        img, struct_mask, room_mask = render_floorplan_style(svg_path, img_size=512, style=style)
        assert isinstance(img, Image.Image)
        assert img.size == (512, 512)
        assert img.mode == "RGB"
        assert struct_mask.shape == (512, 512)
        assert room_mask.shape == (512, 512)

def test_style_renderer_mask_preservation():
    """Verify that procedural restyling leaves ground-truth segmentation masks bit-for-bit identical."""
    cubicasa_dir = Path("data/CubiCasa5k")
    svg_files = list(cubicasa_dir.glob("**/model.svg"))
    if not svg_files:
        pytest.skip("CubiCasa5k dataset not present locally for full SVG render test.")

    svg_path = svg_files[0]

    _, struct_cad, room_cad = render_floorplan_style(svg_path, img_size=512, style="cad_blueprint")
    _, struct_pencil, room_pencil = render_floorplan_style(svg_path, img_size=512, style="faint_pencil_scan")

    # Masks rendered from vector SVG must be 100% identical regardless of visual style
    np.testing.assert_array_equal(struct_cad, struct_pencil)
    np.testing.assert_array_equal(room_cad, room_pencil)

def test_restyled_dataset_item_shape():
    """Verify that RestyledCubiCasaDataset outputs tensors with expected dimensions."""
    root_dir = Path("data/CubiCasa5k/cubicasa5k/cubicasa5k")
    if not root_dir.exists() or not (root_dir / "val.txt").exists():
        pytest.skip("CubiCasa5k validation split not present.")

    dataset = RestyledCubiCasaDataset(
        p_restyle=0.5,
        root_dir=str(root_dir),
        split="val",
        img_size=512,
        cache_dir=None
    )

    if len(dataset) == 0:
        pytest.skip("Dataset is empty.")

    item = dataset[0]
    assert "image" in item
    assert "structure_mask" in item or "struct_mask" in item
    assert item["image"].shape == (3, 512, 512)
    assert item["image"].dtype == torch.float32
