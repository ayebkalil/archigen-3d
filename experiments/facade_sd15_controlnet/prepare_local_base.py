"""Download only pinned FP16 SD1.5 components, with bounded resumable ranges."""
import json
from pathlib import Path
import requests

from src.models.facade_download import download_weight


def main():
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    repo, revision = config['base_model'], config['base_revision']
    root = Path.home() / '.cache/huggingface/hub' / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
    response = requests.get(f'https://huggingface.co/api/models/{repo}/revision/{revision}',
                            params={'blobs': 'true'}, timeout=45)
    response.raise_for_status()
    records = {item['rfilename']: item for item in response.json()['siblings']}
    small = ['model_index.json', 'unet/config.json', 'vae/config.json', 'text_encoder/config.json',
             'scheduler/scheduler_config.json', 'tokenizer/merges.txt', 'tokenizer/vocab.json',
             'tokenizer/tokenizer_config.json', 'tokenizer/special_tokens_map.json']
    large = ['text_encoder/model.fp16.safetensors', 'vae/diffusion_pytorch_model.fp16.safetensors',
             'unet/diffusion_pytorch_model.fp16.safetensors']
    for name in small:
        destination = root / name
        if destination.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(f'https://huggingface.co/{repo}/resolve/{revision}/{name}', timeout=45)
        response.raise_for_status()
        if len(response.content) != records[name]['size']:
            raise ValueError('Incomplete configuration download: ' + name)
        destination.write_bytes(response.content)
        print('Cached ' + name, flush=True)
    for name in large:
        spec = records[name]['lfs']
        download_weight(f'https://huggingface.co/{repo}/resolve/{revision}/{name}', root / name, spec)
        print('Verified ' + name, flush=True)
    print('Base ready at ' + str(root), flush=True)


if __name__ == '__main__':
    main()
