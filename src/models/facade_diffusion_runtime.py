"""Lazy heavyweight runtime, confined to the isolated diffusion environment."""
import gc
from pathlib import Path


def load_pipeline(paths, *, disk_dir=None):
    import torch
    from accelerate import disk_offload
    from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection

    options = dict(torch_dtype=torch.float16, use_safetensors=True,
                   local_files_only=True, low_cpu_mem_usage=True)
    components = {}

    def load(name, cls, path, variant=None):
        print(f'Loading {name} ({"disk" if disk_dir else "CPU"} offload)', flush=True)
        model = cls.from_pretrained(path, **options, **({'variant': variant} if variant else {}))
        model.eval()
        if disk_dir:
            # Each component is offloaded immediately, before loading the next one.
            # safetensors + low_cpu_mem_usage avoid materializing a duplicate state dict.
            disk_offload(model, Path(disk_dir) / name, execution_device=torch.device('cuda:0'), offload_buffers=True)
            gc.collect()
        components[name] = model

    load('controlnet', ControlNetModel, paths['controlnet'], 'fp16')
    load('vae', AutoencoderKL, paths['vae'])
    load('text_encoder', CLIPTextModel, paths['base'] / 'text_encoder', 'fp16')
    load('text_encoder_2', CLIPTextModelWithProjection, paths['base'] / 'text_encoder_2', 'fp16')
    load('unet', UNet2DConditionModel, paths['base'] / 'unet', 'fp16')
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        paths['base'], **components, **options, variant='fp16', add_watermarker=False)
    if not disk_dir:
        pipe.enable_sequential_cpu_offload(gpu_id=0)
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    if pipe._execution_device.type != 'cuda':
        raise RuntimeError('Offload hooks did not select CUDA execution')
    return pipe
