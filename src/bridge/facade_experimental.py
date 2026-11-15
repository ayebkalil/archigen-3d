"""Opt-in 512 inference. Missing/invalid experimental weights fail explicitly."""
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.facade_palette import FACADE_STYLES, NEUTRAL_STYLE
from src.models.artifacts import verify_model, ModelArtifactError


class ExperimentalFacadeInference:
    resolution = 512
    neural_style = True  # Bypasses legacy post-hoc tinting, even for unlabelled/neutral weights.

    def __init__(self, path, device='cpu'):
        self.path, self.device = Path(path), device
        try:
            verify_model(self.path)
        except ModelArtifactError as error:
            if not self.path.is_file():
                raise FileNotFoundError(str(error)) from error
            raise
        if self.path.suffix == '.onnx':
            sidecar = self.path.with_suffix('.json')
            verify_model(sidecar)
            self.metadata = json.loads(sidecar.read_text(encoding='utf-8'))
            if self.metadata.get('architecture') != 'facade_residual_film_512_v1' or self.metadata.get('resolution') != 512:
                raise ValueError('Experimental ONNX sidecar must describe facade_residual_film_512_v1 at 512 pixels')
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(self.path), providers=['CPUExecutionProvider'])
            self.generator = None
        else:
            from src.models.facade_experiment import load_generator, MemoryBudget
            self.budget = MemoryBudget() if str(device).startswith('cuda') else None
            self.generator, self.metadata = load_generator(self.path, device)
            if self.budget:
                self.budget.check()
            self.session = None
        self.supported_styles = self.metadata.get('supported_style_ids', [])

    def style_id(self, style):
        if style is None:
            return NEUTRAL_STYLE
        if style not in FACADE_STYLES:
            raise ValueError(f'Unknown facade style: {style}')
        requested = FACADE_STYLES.index(style)
        return requested if requested in self.supported_styles else NEUTRAL_STYLE

    def generate(self, image_bytes, style='Modern Minimalist'):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((512, 512), Image.Resampling.NEAREST)
        array = (np.asarray(image, dtype=np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
        style_ids = np.array([self.style_id(style)], dtype=np.int64)
        if self.session is not None:
            output = self.session.run(['photo'], {'sketch': array, 'style_id': style_ids})[0]
        else:
            import torch
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=str(self.device).startswith('cuda')):
                output = self.generator(torch.from_numpy(array).to(self.device), torch.from_numpy(style_ids).to(self.device))
            output = output.float().cpu().numpy()
            if getattr(self, 'budget', None):
                self.budget.check()
        result = Image.fromarray(np.clip((output[0].transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8))
        buffer = io.BytesIO(); result.save(buffer, format='PNG')
        return buffer.getvalue()

    def generate_batch(self, images, style='Modern Minimalist'):
        # Bound memory independently of the number of building walls.
        result = []
        for image in images:
            buffer = io.BytesIO(); image.save(buffer, format='PNG')
            result.append(Image.open(io.BytesIO(self.generate(buffer.getvalue(), style))).convert('RGB'))
        return result

    def describe(self, style):
        conditioned = self.style_id(style) != NEUTRAL_STYLE
        return {'model': 'facade_residual_film_512_v1', 'resolution': 512,
                'requested_style': style, 'style_conditioned': conditioned,
                'effective_style': style if conditioned else 'CMP neutral (unlabelled)',
                'supported_styles': [FACADE_STYLES[i] for i in self.supported_styles]}
