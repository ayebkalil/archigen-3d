"""Precision recovery must change predictions, retain gates and preserve gradients."""
import json
from pathlib import Path

import pytest

from experiments.facade_material_v3.teacher import (
    correct_weighted_ce_bias, precision_teacher_loss, teacher_accepted, teacher_selection_key)


def config():
    return json.loads((Path(__file__).parents[1]/'experiments/facade_material_v3/config.json').read_text())


def test_inverse_cost_correction_recovers_known_probabilities_and_input_gradients():
    import torch
    from experiments.facade_material_v3.architecture import make_teacher
    torch.set_num_threads(2)
    model=make_teacher(8).eval().requires_grad_(False)
    image=torch.randn(1,3,32,32,requires_grad=True)
    before=model(image).detach()
    weights=config()['teacher_legacy_class_weights']
    correct_weighted_ce_bias(model,weights)
    after=model(image)
    expected=before-torch.tensor(weights).log()[None,:,None,None]
    assert torch.allclose(after,expected,atol=1e-6)
    after.square().mean().backward()
    assert image.grad.abs().sum()>0 and all(p.grad is None for p in model.parameters())
    with pytest.raises(ValueError):correct_weighted_ce_bias(model,[0,1,1,1,1,1])


def test_gate_aware_selection_cannot_replace_a_passing_teacher_with_high_recall():
    c=config()
    good=dict(miou=.4,wall_iou=.8,window_iou=.5,door_iou=.25,opening_iou=.5,wall_false_opening_rate=.04)
    bad={**good,'window_iou':.95,'door_iou':.8,'opening_iou':.9,'wall_false_opening_rate':.051}
    assert teacher_accepted(good,c) and not teacher_accepted(bad,c)
    assert teacher_selection_key(good,c)>teacher_selection_key(bad,c)
    assert c['teacher_max_wall_fp']==.05
    assert not teacher_accepted({**good,'opening_iou':.1},c)


def test_precision_loss_penalizes_painted_openings_and_excludes_invalid_pixels():
    import torch
    torch.set_num_threads(2)
    target=torch.ones(1,8,8,dtype=torch.long); valid=torch.ones(1,8,8)
    logits=torch.zeros(1,6,8,8,requires_grad=True)
    loss=precision_teacher_loss(logits,target,valid,dice_weight=0,false_opening_weight=1)
    loss.backward()
    # Gradient descent increases wall logits and decreases opening logits.
    assert logits.grad[:,1].mean()<0 and logits.grad[:,2:4].mean()>0
    good=torch.zeros_like(logits);good[:,1]=8
    bad=torch.zeros_like(logits);bad[:,2]=8
    assert precision_teacher_loss(bad,target,valid)>precision_teacher_loss(good,target,valid)+1
    assert precision_teacher_loss(bad,target,torch.zeros_like(valid)).item()==0


def test_recovery_dataset_is_attached_and_notebook_pins_immutable_source(tmp_path):
    from experiments.facade_material_v3.manage import stage_kernel
    metadata=stage_kernel(tmp_path,commit='a'*40,mode='train')
    assert metadata['is_private']
    assert 'khalilayeb/archigen-v3-teacher-recovery' in metadata['dataset_sources']
    notebook=json.loads((tmp_path/'train_facade_material.ipynb').read_text())
    code=''.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    assert 'a'*40 in code and 'MODE = "staged"' in code


def test_installer_cannot_label_a_retained_initialization_as_new_training(tmp_path,monkeypatch):
    import zipfile
    from experiments.facade_material_v3 import manage
    archive=tmp_path/'parent-only.zip'
    with zipfile.ZipFile(archive,'w') as zipped:zipped.writestr('export/model.json','{}')
    monkeypatch.setattr(manage,'validate_export',lambda root:{'selected_step':0})
    with pytest.raises(ValueError,match='step-zero initialization'):
        manage.install(archive,tmp_path/'installed')
    assert not (tmp_path/'installed').exists()
