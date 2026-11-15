"""Portable material10 inference with the same local offload strategy as V2."""
import gc
import json
from pathlib import Path
import threading
import time

import numpy as np

from experiments.facade_sd15_controlnet.data import digest
from experiments.facade_semantic_v2.runtime import SemanticFacadeRuntime


def validate_export(root,allow_smoke=False):
    root=Path(root);metadata=json.loads((root/'model.json').read_text())
    if metadata.get('architecture')!='sd15_semantic10_material_lora_v3':raise ValueError('Wrong V3 architecture')
    config=metadata['config']
    if config['conditioning']!='facade_semantic7_material_rgb_v3' or config['resolution']!=512:
        raise ValueError('Unsupported V3 conditioning or resolution')
    if metadata['quality_status'].startswith('SMOKE') and not allow_smoke:raise ValueError('Smoke models cannot be installed as quality checkpoints')
    for name,key in [('controlnet/diffusion_pytorch_model.safetensors','weights_sha256'),('lora/facade_lora.safetensors','lora_sha256')]:
        if digest(root/name)!=metadata[key]:raise ValueError('V3 weight checksum mismatch: '+name)
    if json.loads((root/'controlnet/config.json').read_text())['conditioning_channels']!=10:
        raise ValueError('Expected semantic7 + material RGB')
    return metadata


def material_control(semantic,reference):
    import torch
    semantic=torch.as_tensor(semantic,dtype=torch.float32)
    if semantic.shape==(7,512,512):semantic=semantic[None]
    if semantic.shape!=(1,7,512,512) or not torch.isfinite(semantic).all() or semantic.min()<0 or semantic.max()>1:
        raise ValueError('Expected finite semantic7 in [0,1] at 512')
    if reference.size!=(512,512):raise ValueError('Material reference must align with the 512 facade UV bounds')
    ref=torch.from_numpy(np.asarray(reference.convert('RGB'),np.float32).transpose(2,0,1).copy())[None]/255
    return torch.cat([semantic,ref],1)


class MaterialFacadeRuntime(SemanticFacadeRuntime):
    def __init__(self,model,allow_download=False,allow_smoke=False):
        self.root=Path(model).resolve();self.metadata=validate_export(self.root,allow_smoke)
        self.config=dict(self.metadata['config']);self.allow_download=allow_download;self.pipeline=None
        self.lock=threading.Lock();self.last_profile=None

    def generate_control(self,control,style=None,seed=None):
        raise ValueError('V3 requires an explicit material reference; use generate_material')

    def generate_material(self,semantic,reference,seed=None):
        import torch
        from src.models.facade_diffusion import gpu_memory
        condition=material_control(semantic,reference)
        with self.lock:
            self._load();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            try:
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    image=self.pipeline(self.config['prompt'],image=condition,negative_prompt=self.config['negative_prompt'],
                        height=512,width=512,num_inference_steps=self.config['inference_steps'],guidance_scale=self.config['guidance_scale'],
                        controlnet_conditioning_scale=self.config['controlnet_conditioning_scale'],
                        generator=torch.Generator(device='cpu').manual_seed(self.config['seed'] if seed is None else seed)).images[0]
                self.pipeline.maybe_free_model_hooks()
                self.last_profile={'seconds':time.perf_counter()-started,'peak_torch_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                    'peak_torch_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'teacher_required_for_inference':False}
                if gpu_memory()['used_mib']>self.config['local_vram_budget_mib']:raise RuntimeError('Local device memory budget exceeded')
                return image
            finally:gc.collect();torch.cuda.empty_cache()
