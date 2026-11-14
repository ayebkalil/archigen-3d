"""Lazy local SD1.5 runtime. All heavyweight dependencies remain in this worker."""
import gc
import io
import json
from pathlib import Path
import threading
import time

from PIL import Image

from .data import condition_from_labels, digest, labels_from_bridge

STYLE_TEXT = {
    'Modern Minimalist': 'modern white stucco, slim black metal frames',
    'Mediterranean Stone': 'pale limestone blocks, warm natural stone',
    'Traditional Brick': 'red brick masonry, crisp mortar joints',
    'Lush Vegetation': 'climbing ivy between the windows, living green facade',
}


class FacadeControlNetRuntime:
    def __init__(self, model, allow_download=False):
        self.root = Path(model).resolve()
        self.metadata = json.loads((self.root / 'model.json').read_text(encoding='utf-8'))
        if self.metadata.get('architecture') != 'sd15_controlnet_cmp_edges_v1': raise ValueError('Wrong facade architecture')
        self.config = self.metadata['config']
        if self.config['resolution'] != 512 or self.config['conditioning'] != 'cmp_semantic_edges_v1': raise ValueError('Unsupported conditioning/resolution')
        if digest(self.root / 'controlnet/diffusion_pytorch_model.safetensors') != self.metadata['weights_sha256']:
            raise ValueError('ControlNet weight checksum mismatch')
        self.allow_download = allow_download
        self.pipeline = None
        self.lock = threading.Lock()
        self.last_profile = None

    def _load(self):
        import torch
        import psutil
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler
        if self.pipeline is not None: return
        if not torch.cuda.is_available(): raise RuntimeError('This local profile requires CUDA; CPU inference has not been validated.')
        if psutil.virtual_memory().available < 4 * 1024**3: raise RuntimeError('Free at least 4 GiB of system RAM for CPU-offloaded inference.')
        from src.models.facade_diffusion import gpu_memory
        device = gpu_memory()
        budget = min(self.config['local_vram_budget_mib'], device['total_mib'])
        headroom = budget - device['used_mib'] - 256
        if headroom < 768: raise RuntimeError('Insufficient GPU headroom below the configured budget; close other GPU workloads.')
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(headroom / device['total_mib'], 0)
        options = {'torch_dtype': torch.float16, 'use_safetensors': True, 'low_cpu_mem_usage': True}
        controlnet = ControlNetModel.from_pretrained(self.root / 'controlnet', **options)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(self.config['base_model'],
            revision=self.config['base_revision'], variant='fp16', controlnet=controlnet,
            safety_checker=None, requires_safety_checker=False, feature_extractor=None,
            local_files_only=not self.allow_download, **options)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_sequential_cpu_offload(gpu_id=0)
        pipe.enable_vae_slicing(); pipe.enable_vae_tiling()
        pipe.set_progress_bar_config(disable=True)
        self.pipeline = pipe

    def generate_control(self, control, style=None, seed=None):
        import torch
        from src.models.facade_diffusion import gpu_memory
        if style is not None and style not in STYLE_TEXT: raise ValueError('Unknown facade style')
        prompt = self.config['prompt'] + (', ' + STYLE_TEXT[style] if style else '')
        with self.lock:
            self._load()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                    image = self.pipeline(prompt, image=control.convert('RGB').resize((512, 512), Image.Resampling.NEAREST),
                        negative_prompt=self.config['negative_prompt'], height=512, width=512,
                        num_inference_steps=self.config['inference_steps'], guidance_scale=self.config['guidance_scale'],
                        controlnet_conditioning_scale=self.config['controlnet_conditioning_scale'],
                        generator=torch.Generator(device='cpu').manual_seed(self.config['seed'] if seed is None else seed)).images[0]
                device = gpu_memory()
                self.last_profile = {'seconds': time.perf_counter() - started,
                    'peak_torch_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
                    'peak_torch_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
                    'device_used_after_mib': device['used_mib']}
                if device['used_mib'] > self.config['local_vram_budget_mib']:
                    raise RuntimeError('Device memory exceeded the configured budget; result not accepted.')
                return image
            finally:
                gc.collect(); torch.cuda.empty_cache()

    def generate_sketch(self, image_bytes, style=None):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((512, 512), Image.Resampling.NEAREST)
        condition, _ = condition_from_labels(labels_from_bridge(image), 512)
        return self.generate_control(condition, style)

    def describe(self, style=None):
        return {'model': 'sd15_controlnet_cmp_edges_v1', 'resolution': 512,
                'weights_sha256': self.metadata['weights_sha256'],
                'training_run_id': self.metadata['run_id'], 'git_commit': self.metadata['git_commit'],
                'requested_style': style, 'style_conditioned': style in STYLE_TEXT,
                'style_conditioning': 'frozen pretrained text prior; no supervised CMP style labels',
                'effective_style': style or 'CMP neutral', 'supported_styles': list(STYLE_TEXT),
                'trained_style_labels': [], 'quality_status': self.metadata['quality_status'],
                'geometry_guaranteed': False, 'loaded': self.pipeline is not None, 'last_profile': self.last_profile}
