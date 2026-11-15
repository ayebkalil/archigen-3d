"""Reuse V2 weights with a layout/material image as the denoising starting point."""
import gc
import time

from experiments.facade_semantic_v2.runtime import SemanticFacadeRuntime


class ReferenceFacadeRuntime(SemanticFacadeRuntime):
    def _load(self):
        from diffusers import StableDiffusionControlNetImg2ImgPipeline
        if isinstance(self.pipeline, StableDiffusionControlNetImg2ImgPipeline):
            return
        super()._load()
        old = self.pipeline
        old.remove_all_hooks()
        self.pipeline = StableDiffusionControlNetImg2ImgPipeline(**old.components, requires_safety_checker=False)
        self.pipeline.enable_sequential_cpu_offload(gpu_id=0)
        self.pipeline.enable_vae_slicing(); self.pipeline.enable_vae_tiling()
        self.pipeline.set_progress_bar_config(disable=True)

    def generate_reference(self, control, reference, settings):
        import torch
        from src.models.facade_diffusion import gpu_memory
        control = torch.as_tensor(control, dtype=torch.float32)
        if (control.shape != (1, 7, 512, 512) or reference.size != (512, 512)
                or not torch.isfinite(control).all() or control.min() < 0 or control.max() > 1):
            raise ValueError('Expected aligned 512 reference and semantic7 tensor')
        if not 0 < settings['strength'] < 1 or int(settings['steps'] * settings['strength']) < 1:
            raise ValueError('Reference strength must retain a meaningful starting image and denoising step')
        with self.lock:
            self._load(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
            try:
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                    image = self.pipeline(settings['prompt'], image=reference, control_image=control,
                        strength=settings['strength'], negative_prompt=settings['negative_prompt'],
                        height=512, width=512, num_inference_steps=settings['steps'],
                        guidance_scale=settings['guidance_scale'], controlnet_conditioning_scale=settings['control_scale'],
                        generator=torch.Generator(device='cpu').manual_seed(settings['seed'])).images[0]
                self.pipeline.maybe_free_model_hooks()
                self.last_profile = {'seconds': time.perf_counter() - started,
                    'peak_torch_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
                    'peak_torch_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2}
                if gpu_memory()['used_mib'] > self.config['local_vram_budget_mib']:
                    raise RuntimeError('GPU budget exceeded')
                return image
            finally:
                gc.collect(); torch.cuda.empty_cache()
