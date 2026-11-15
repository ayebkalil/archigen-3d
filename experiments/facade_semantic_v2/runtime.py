"""V2 export loader: semantic7 + UNet LoRA, sequential CPU offload for local CUDA."""
import gc
import io
import json
from pathlib import Path
import threading
import time

from PIL import Image

from experiments.facade_sd15_controlnet.data import digest, labels_from_bridge
from experiments.facade_sd15_controlnet.runtime import STYLE_TEXT
from .conditioning import cmp_labels, control_channels


def validate_export(root):
    root = Path(root)
    metadata = json.loads((root / 'model.json').read_text(encoding='utf-8'))
    if metadata.get('architecture') != 'sd15_semantic7_lora_facade_v2':
        raise ValueError('Wrong facade architecture')
    if metadata['config']['conditioning'] != 'facade_semantic7_v2' or metadata['config']['resolution'] != 512:
        raise ValueError('Unsupported conditioning/resolution')
    for filename, key in [('controlnet/diffusion_pytorch_model.safetensors', 'weights_sha256'),
                          ('lora/facade_lora.safetensors', 'lora_sha256')]:
        if digest(root / filename) != metadata[key]:
            raise ValueError('Model checksum mismatch: ' + filename)
    config = json.loads((root / 'controlnet/config.json').read_text())
    if config['conditioning_channels'] != 7:
        raise ValueError('Export must have seven conditioning channels')
    return metadata


class SemanticFacadeRuntime:
    def __init__(self, model, allow_download=False):
        self.root = Path(model).resolve()
        self.metadata = validate_export(self.root)
        self.config = dict(self.metadata['config'])
        self.allow_download, self.pipeline = allow_download, None
        self.lock, self.last_profile = threading.Lock(), None

    def _load(self):
        import psutil
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler
        from src.models.facade_diffusion import gpu_memory
        if self.pipeline is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; CPU-only inference has not been profiled')
        if psutil.virtual_memory().available < 4 * 1024**3:
            raise RuntimeError('Free at least 4 GiB system RAM for offloaded inference')
        device = gpu_memory()
        headroom = min(self.config['local_vram_budget_mib'], device['total_mib']) - device['used_mib'] - 256
        if headroom < 768:
            raise RuntimeError('Insufficient GPU headroom; close other GPU workloads')
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(headroom / device['total_mib'], 0)
        options = dict(torch_dtype=torch.float16, use_safetensors=True, low_cpu_mem_usage=True)
        controlnet = ControlNetModel.from_pretrained(self.root / 'controlnet', **options)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(self.config['base_model'],
            revision=self.config['base_revision'], variant='fp16', controlnet=controlnet,
            safety_checker=None, requires_safety_checker=False, feature_extractor=None,
            local_files_only=not self.allow_download, **options)
        pipe.load_lora_weights(str(self.root / 'lora'), weight_name='facade_lora.safetensors')
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_sequential_cpu_offload(gpu_id=0)
        pipe.enable_vae_slicing(); pipe.enable_vae_tiling(); pipe.set_progress_bar_config(disable=True)
        self.pipeline = pipe

    def generate_control(self, control, style=None, seed=None):
        import torch
        from src.models.facade_diffusion import gpu_memory
        if style is not None and style not in STYLE_TEXT:
            raise ValueError('Unknown facade style')
        control = torch.as_tensor(control, dtype=torch.float32)
        if control.shape != (1, 7, 512, 512) or not torch.isfinite(control).all() or control.min() < 0 or control.max() > 1:
            raise ValueError('Expected finite semantic control B7x512x512 in [0,1]')
        prompt = self.config['prompt'] + (', ' + STYLE_TEXT[style] if style else '')
        with self.lock:
            self._load(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
            try:
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                    image = self.pipeline(prompt, image=control, negative_prompt=self.config['negative_prompt'],
                        height=512, width=512, num_inference_steps=self.config['inference_steps'],
                        guidance_scale=self.config['guidance_scale'], controlnet_conditioning_scale=self.config['controlnet_conditioning_scale'],
                        generator=torch.Generator(device='cpu').manual_seed(self.config['seed'] if seed is None else seed)).images[0]
                self.pipeline.maybe_free_model_hooks()
                self.last_profile = {'seconds': time.perf_counter() - started,
                    'peak_torch_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
                    'peak_torch_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2}
                if gpu_memory()['used_mib'] > self.config['local_vram_budget_mib']:
                    raise RuntimeError('Device exceeds local VRAM budget')
                return image
            finally:
                gc.collect(); torch.cuda.empty_cache()

    def generate_sketch(self, image_bytes, style=None):
        import torch
        sketch = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if sketch.size != (512, 512):
            raise ValueError('Bridge sketch must already be letterboxed to 512')
        channels = control_channels(cmp_labels(labels_from_bridge(sketch)))
        return self.generate_control(torch.from_numpy(channels)[None], style)
