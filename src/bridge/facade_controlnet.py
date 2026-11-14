"""Opt-in localhost worker adapter; production environment needs no diffusers."""
import base64
import io
import json
from urllib.parse import urlsplit

import httpx
from PIL import Image


class RemoteControlNetFacadeInference:
    resolution = 512
    neural_style = True
    model_family = 'sd15_controlnet'
    defer_automatic_generation = True

    def __init__(self, url):
        parsed = urlsplit(url)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'} or parsed.username:
            raise ValueError('The experimental facade worker must use a localhost HTTP URL.')
        self.url = url.rstrip('/')
        with httpx.Client(timeout=10, trust_env=False) as client:
            response = client.get(self.url + '/health'); response.raise_for_status()
        self.metadata = response.json()
        if self.metadata.get('model') != 'sd15_controlnet_cmp_edges_v1' or self.metadata.get('resolution') != 512:
            raise ValueError('Unexpected facade worker architecture/resolution')

    def generate(self, image_bytes, style=None):
        with httpx.Client(timeout=600, trust_env=False) as client:
            response = client.post(self.url + '/generate', json={
                'image_base64': base64.b64encode(image_bytes).decode(), 'style': style})
            response.raise_for_status()
        self.metadata['loaded'] = True
        if response.headers.get('X-ArchiGen-Profile'):
            self.metadata['last_profile'] = json.loads(response.headers['X-ArchiGen-Profile'])
        return response.content

    def generate_batch(self, images, style=None):
        results = []
        for image in images:
            buffer = io.BytesIO(); image.save(buffer, format='PNG')
            results.append(Image.open(io.BytesIO(self.generate(buffer.getvalue(), style))).convert('RGB'))
        return results

    def describe(self, style):
        return {**self.metadata, 'requested_style': style, 'effective_style': style or 'CMP neutral',
                'style_conditioned': style in self.metadata.get('supported_styles', [])}
