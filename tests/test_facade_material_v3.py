"""Learned V3 conditioning, actual gradient flow, data honesty and portable exports."""
import json
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest

from experiments.facade_material_v3.data import wall_patch,material_reference,MaterialDataset
from experiments.facade_material_v3.losses import predicted_clean,semantic_feedback,material_statistics
from experiments.facade_material_v3.teacher import segmentation_scores


def test_references_and_blank_crops_never_erase_an_annotated_opening():
    labels=np.ones((256,256),np.uint8);labels[60:200,100:140]=2
    pixels=np.full((256,256,3),[180,120,80],np.uint8);pixels[labels==2]=[0,80,240]
    photo=Image.fromarray(pixels)
    for seed in range(10):
        chosen=wall_patch(photo,labels,np.random.default_rng(seed),minimum=64)
        assert chosen is not None
        crop,mask,box=chosen
        assert (mask==1).all() and np.array_equal(crop,pixels[box[1]:box[3],box[0]:box[2]])
        reference,fallback=material_reference(photo,labels,np.random.default_rng(seed))
        assert not fallback and np.all(np.asarray(reference)==[180,120,80])
    assert wall_patch(photo,np.full_like(labels,2),np.random.default_rng(0)) is None


def dataset_fixture(root):
    (root/'photos').mkdir();(root/'labels').mkdir()
    rows=[]
    for i,split in enumerate(('train','train','val','test')):
        photo=np.full((256,256,3),[190,130+i*10,80],np.uint8)
        labels=np.ones((256,256),np.uint8);labels[60:180,110:145]=2
        photo[labels==2]=[20,60,130]
        Image.fromarray(photo).save(root/'photos'/f'{i}.png');Image.fromarray(labels).save(root/'labels'/f'{i}.png')
        rows.append({'id':f'IRF/{i}','source':'IRF','split':split,'group':f'building-{i}','photo':f'{i}.png','labels':f'labels/{i}.png'})
    (root/'manifest.json').write_text(json.dumps({'samples':rows}))
    (root/'sources.json').write_text(json.dumps({'IRF':str(root/'photos')}))


def test_material_dataset_respects_splits_and_aligns_ten_channels(tmp_path):
    dataset_fixture(tmp_path)
    train=MaterialDataset(tmp_path,'train',augment=True,blank_probability=1)
    val=MaterialDataset(tmp_path,'val',augment=True,blank_probability=1)
    assert len(train)==2 and len(val)==1 and val.blank_probability==0
    sample=train[0]
    assert sample['blank_crop'] and sample['condition'].shape==(10,512,512)
    assert np.all(sample['labels'].numpy()==1)
    assert np.array_equal(sample['condition'][7:],sample['reference'])
    assert np.all(sample['condition'][1:7].sum(0).numpy()==1)
    assert (val[0]['labels']==2).any() and not val[0]['blank_crop']


def test_blank_perfect_prediction_and_false_opening_diagnostics():
    correct=np.zeros((6,6),int);correct[1,1]=100
    assert segmentation_scores(correct)['opening_iou']==1
    wrong=np.zeros((6,6),int);wrong[1,2]=100
    scores=segmentation_scores(wrong)
    assert scores['opening_iou']==0 and scores['wall_false_opening_rate']==1


def test_clean_prediction_equations_and_feedback_gradients():
    import torch
    torch.set_num_threads(2)
    from experiments.facade_material_v3.architecture import make_teacher
    alpha=torch.tensor([.81]);clean=torch.randn(1,4,8,8);noise=torch.randn_like(clean)
    noisy=.9*clean+np.sqrt(.19)*noise
    scheduler=SimpleNamespace(alphas_cumprod=alpha,config=SimpleNamespace(prediction_type='epsilon'))
    assert torch.allclose(predicted_clean(noisy,noise,torch.tensor([0]),scheduler),clean,atol=1e-6)
    scheduler.config.prediction_type='v_prediction'
    velocity=.9*noise-np.sqrt(.19)*clean
    assert torch.allclose(predicted_clean(noisy,velocity,torch.tensor([0]),scheduler),clean,atol=1e-6)
    teacher=make_teacher(8).eval().requires_grad_(False)
    image=torch.randn(1,3,32,32,requires_grad=True)
    labels=torch.ones(1,32,32).long();labels[:,10:20,12:18]=2
    valid=torch.ones(1,32,32);valid[:,:4]=0
    ce,fp=semantic_feedback(teacher(image),labels,valid)
    loss=ce+fp+material_statistics(image,torch.ones_like(image)*.7,labels,valid)
    loss.backward()
    assert torch.isfinite(image.grad).all() and image.grad.abs().sum()>0
    assert all(p.grad is None for p in teacher.parameters())
    logits=torch.zeros(1,6,32,32);logits[:,2]=10
    _,bad=semantic_feedback(logits,labels,valid)
    logits[:,2]=0;logits[:,1]=10
    _,good=semantic_feedback(logits,labels,valid)
    assert bad>good+.9


def test_upgrade_preserves_v2_response_and_learns_reference_kernels():
    import torch
    from experiments.facade_material_v3.architecture import upgrade_material_conditioning
    from test_facade_semantic_v2 import tiny_models
    _,control=tiny_models()
    from experiments.facade_semantic_v2.architecture import upgrade_conditioning
    control=upgrade_conditioning(control)
    semantic=torch.rand(1,7,64,64)
    old=control.controlnet_cond_embedding.conv_in(semantic).detach()
    upgrade_material_conditioning(control)
    actual=control.controlnet_cond_embedding.conv_in(torch.cat([semantic,torch.rand(1,3,64,64)],1))
    assert torch.allclose(old,actual,atol=1e-6)
    actual.square().mean().backward()
    assert control.controlnet_cond_embedding.conv_in.weight.grad[:,7:].abs().sum()>0
    assert control.config.conditioning_channels==10


def test_feedback_reaches_latents_through_a_real_frozen_vae_decoder():
    import torch
    diffusers=pytest.importorskip('diffusers')
    from torch.utils.checkpoint import checkpoint
    from experiments.facade_material_v3.architecture import make_teacher
    torch.set_num_threads(2)
    vae=diffusers.AutoencoderKL(block_out_channels=(16,32,32,32),norm_num_groups=8,
        down_block_types=('DownEncoderBlock2D',)*4,up_block_types=('UpDecoderBlock2D',)*4,latent_channels=4).eval().requires_grad_(False)
    teacher=make_teacher(8).eval().requires_grad_(False)
    latent=torch.randn(1,4,8,8,requires_grad=True)
    image=checkpoint(lambda z:vae.decode(z).sample,latent,use_reentrant=False)
    labels=torch.ones(1,64,64).long();labels[:,15:40,22:34]=2;valid=torch.ones(1,64,64)
    ce,fp=semantic_feedback(teacher(image),labels,valid)
    loss=ce+fp+material_statistics(image,torch.ones_like(image)*.6,labels,valid)
    loss.backward()
    assert torch.isfinite(latent.grad).all() and latent.grad.abs().sum()>0
    assert all(p.grad is None for p in vae.parameters())
    assert all(p.grad is None for p in teacher.parameters())


def test_real_pipeline_exports_reloads_and_accepts_material10(tmp_path):
    import torch
    diffusers=pytest.importorskip('diffusers');pytest.importorskip('peft')
    from test_facade_semantic_v2 import tiny_models
    from experiments.facade_semantic_v2.architecture import upgrade_conditioning,add_facade_lora,save_lora
    from experiments.facade_material_v3.architecture import upgrade_material_conditioning
    from experiments.facade_material_v3.runtime import material_control
    unet,control=tiny_models();upgrade_material_conditioning(upgrade_conditioning(control))
    add_facade_lora(unet,2);save_lora(unet,tmp_path/'lora');unet.delete_adapters('default')
    control.save_pretrained(tmp_path/'controlnet',safe_serialization=True)
    other=diffusers.ControlNetModel.from_pretrained(tmp_path/'controlnet')
    assert other.config.conditioning_channels==10
    vae=diffusers.AutoencoderKL(block_out_channels=(16,32,32,32),norm_num_groups=8,
        down_block_types=('DownEncoderBlock2D',)*4,up_block_types=('UpDecoderBlock2D',)*4,latent_channels=4)
    pipe=diffusers.StableDiffusionControlNetPipeline(vae=vae,unet=unet,controlnet=other,text_encoder=None,tokenizer=None,
        scheduler=diffusers.DDIMScheduler(),safety_checker=None,feature_extractor=None,requires_safety_checker=False)
    pipe.load_lora_weights(str(tmp_path/'lora'),weight_name='facade_lora.safetensors');pipe.set_progress_bar_config(disable=True)
    control_input=material_control(torch.zeros(7,512,512),Image.new('RGB',(512,512),'white'))
    assert control_input.shape==(1,10,512,512) and (control_input[:,7:]==1).all()
    condition=torch.nn.functional.interpolate(control_input,(64,64),mode='nearest')
    with torch.no_grad():
        output=pipe(prompt_embeds=torch.randn(1,8,16),negative_prompt_embeds=torch.zeros(1,8,16),image=condition,
            num_inference_steps=2,height=64,width=64,output_type='latent').images
    assert output.shape==(1,4,8,8) and torch.isfinite(output).all()
    with pytest.raises(ValueError,match='finite'):material_control(torch.full((7,512,512),float('nan')),Image.new('RGB',(512,512)))


def test_private_notebook_pins_source_and_uses_v2_parent(tmp_path):
    from experiments.facade_material_v3.manage import stage_kernel
    metadata=stage_kernel(tmp_path,'b'*40,'smoke')
    assert metadata['is_private'] and metadata['kernel_sources']==['khalilayeb/archigen-facade-semantic7-lora-v2-train']
    notebook=json.loads((tmp_path/'train_facade_material.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<notebook>','exec')
    text=(tmp_path/'train_facade_material.ipynb').read_text()
    assert 'b'*40 in text and 'experiments.facade_material_v3.kaggle_run' in text
    stage_kernel(tmp_path/'full','c'*40,'train')
    full=json.loads((tmp_path/'full/train_facade_material.ipynb').read_text())
    assert 'MODE = "staged"' in ''.join(line for cell in full['cells'] for line in cell['source'])


def test_installer_rejects_traversal_before_writing_destination(tmp_path):
    from experiments.facade_material_v3.manage import install
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as zipped:zipped.writestr('../escape.txt','bad')
    with pytest.raises(ValueError,match='Unsafe'):install(archive,tmp_path/'model')
    assert not (tmp_path/'model').exists() and not (tmp_path/'escape.txt').exists()


def test_staged_job_never_starts_full_training_after_failed_preflight(tmp_path,monkeypatch):
    from experiments.facade_material_v3 import kaggle_run
    import subprocess
    calls=[]
    def failed(command,**kwargs):
        calls.append(command[-1]);raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(kaggle_run.subprocess,'run',failed)
    with pytest.raises(subprocess.CalledProcessError):kaggle_run.run_staged(tmp_path)
    assert calls==['smoke']
    def succeeded(command,**kwargs):calls.append(command[-1])
    monkeypatch.setattr(kaggle_run.subprocess,'run',succeeded)
    smoke=tmp_path/'facade-material-v3-smoke';smoke.mkdir()
    (smoke/'reload-check.json').write_text(json.dumps({'ten_channel_inference':False,'smoke_only':True}))
    monkeypatch.setattr(kaggle_run,'validate_export',lambda *args,**kwargs:{'config':{'max_train_steps':2}})
    with pytest.raises(RuntimeError,match='Smoke completion'):kaggle_run.run_staged(tmp_path)
    assert calls==['smoke','smoke']
