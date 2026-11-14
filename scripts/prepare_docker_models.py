"""Restore the production ONNX profile atomically, without DVC or ML packages."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def prepare(root=ROOT, fetch=urllib.request.urlopen):
    root = Path(root)
    manifest = json.loads((root / 'models/artifacts.json').read_text())
    if manifest['schema_version'] != 1:
        raise ValueError('Unsupported model inventory version')
    entries = {item['path']: item for item in manifest['artifacts']}
    for name in manifest['profiles']['runtime']:
        item = entries[name]
        target = (root / name).resolve()
        if not target.is_relative_to(root.resolve() / 'models/saved'):
            raise ValueError('Model path escapes models/saved')
        if target.is_file() and target.stat().st_size == item['size']:
            with target.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256']:
                    print(f'OK {name}', flush=True)
                    continue
        if not item.get('url', '').startswith('https://'):
            raise ValueError(f'Missing HTTPS release URL: {name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            request = urllib.request.Request(item['url'], headers={'User-Agent': 'ArchiGen-Docker'})
            digest, size = hashlib.sha256(), 0
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
                with fetch(request, timeout=120) as response:
                    while block := response.read(4 * 1024 * 1024):
                        size += len(block)
                        if size > item['size']:
                            raise ValueError(f'Download exceeds expected size: {name}')
                        digest.update(block)
                        output.write(block)
            if size != item['size'] or digest.hexdigest() != item['sha256']:
                raise ValueError(f'SHA-256/size mismatch: {name}; existing file preserved')
            os.chmod(temporary, 0o644)
            temporary.replace(target)
            print(f'RESTORED {name} SHA256={digest.hexdigest()}', flush=True)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        prepare()
    except Exception as exc:
        print(f'Model preparation failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
