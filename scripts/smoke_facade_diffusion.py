"""Exercise SDXL APIs/offload with tiny RANDOM weights. Not an image quality test.

Run with .venv-facade-diffusion/Scripts/python.exe. No network/model downloads.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from diffusers import (
    AutoencoderKL, ControlNetModel, DDIMScheduler,
    StableDiffusionXLControlNetPipeline, UNet2DConditionModel,
)
from PIL import Image
from src.models.facade_diffusion import gpu_memory


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--offload', choices=['cpu', 'disk'], default='cpu')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This smoke test checks local CUDA offload')
    if gpu_memory()['used_mib'] > 2000:
        raise RuntimeError('Free GPU memory before running the tiny smoke test')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(.4)
    torch.cuda.reset_peak_memory_stats()
    unet = UNet2DConditionModel(
        sample_size=32, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(32, 64), down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
        up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'), cross_attention_dim=64,
        attention_head_dim=(2, 4), use_linear_projection=True,
        addition_embed_type='text_time', addition_time_embed_dim=8,
        projection_class_embeddings_input_dim=80,
    ).half()
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4,
        block_out_channels=(32, 64), down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D'),
        up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D'), layers_per_block=1,
        sample_size=32, force_upcast=False,
    ).half()
    # Match ControlNet's image-to-latent downsampling (one stride-two stage).
    controlnet = ControlNetModel.from_unet(unet, conditioning_embedding_out_channels=(8, 16)).half()
    pipe = StableDiffusionXLControlNetPipeline(
        unet=unet, controlnet=controlnet, vae=vae, scheduler=DDIMScheduler(),
        text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None,
        add_watermarker=False,
    )
    temporary = tempfile.TemporaryDirectory(prefix='archigen-diffusion-smoke-')
    if args.offload == 'disk':
        from accelerate import disk_offload
        for name, model in [('unet', unet), ('vae', vae), ('controlnet', controlnet)]:
            disk_offload(model, Path(temporary.name) / name, execution_device=torch.device('cuda:0'), offload_buffers=True)
    else:
        pipe.enable_sequential_cpu_offload(gpu_id=0)
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    steps = []
    def callback(pipe, step, timestep, kwargs):
        steps.append(step)
        return kwargs
    embeddings = torch.zeros(1, 8, 64, dtype=torch.float16)
    pooled = torch.zeros(1, 32, dtype=torch.float16)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        result = pipe(
            prompt_embeds=embeddings, pooled_prompt_embeds=pooled,
            negative_prompt_embeds=embeddings, negative_pooled_prompt_embeds=pooled,
            image=Image.new('RGB', (64, 64)), width=64, height=64, guidance_scale=5.,
            num_inference_steps=2, controlnet_conditioning_scale=1.,
            generator=torch.Generator('cpu').manual_seed(42), callback_on_step_end=callback,
        ).images[0]
    assert result.size == (64, 64) and steps == [0, 1]
    print(json.dumps({'status': 'tiny random-weight CUDA smoke passed', 'image_quality_test': False,
                      'offload': args.offload,
                      'steps': len(steps), 'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
                      'peak_reserved_mib': torch.cuda.max_memory_reserved()/1024**2, 'gpu': gpu_memory()}))


if __name__ == '__main__':
    main()
