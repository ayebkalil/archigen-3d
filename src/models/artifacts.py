"""Portable model inventory and integrity checks; no ML dependencies."""
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESTORE_COMMAND = 'python scripts/models.py pull --profile runtime'


class ModelArtifactError(RuntimeError):
    """A deployed model is absent, corrupt, or from another version."""


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def inventory(root=PROJECT_ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / 'models/artifacts.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ModelArtifactError('Unsupported model inventory version')
    for item in manifest['artifacts']:
        if not (root / item['path']).resolve().is_relative_to(root):
            raise ModelArtifactError('Model path escapes project directory')
    return manifest


def inspect_artifacts(profile='runtime', root=PROJECT_ROOT):
    root = Path(root).resolve()
    manifest = inventory(root)
    selected = set(manifest['profiles'][profile])
    results = []
    for item in manifest['artifacts']:
        if item['path'] not in selected:
            continue
        path = root / item['path']
        status = 'missing' if not path.is_file() else 'ok'
        actual = None
        if status == 'ok':
            if path.stat().st_size != item['size']:
                status = 'corrupt'
            else:
                actual = sha256(path)
                if actual != item['sha256']:
                    status = 'corrupt'
        results.append({'path': item['path'], 'status': status, 'expected_sha256': item['sha256'], 'actual_sha256': actual})
    if {row['path'] for row in results} != selected:
        raise ModelArtifactError('Profile references an unknown model')
    return results


def require_artifacts(profile='runtime', root=PROJECT_ROOT):
    results = inspect_artifacts(profile, root)
    errors = [f"{row['status']}: {row['path']}" for row in results if row['status'] != 'ok']
    if errors:
        raise ModelArtifactError('; '.join(errors) + f'. Restore verified weights: python scripts/models.py pull --profile {profile}')
    return results


def verify_model(path, root=PROJECT_ROOT):
    """Verify a pinned deployed artifact; explicit external experiments remain opt-in."""
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_file():
        raise ModelArtifactError(f'Model is missing: {path.name}. Run `{RESTORE_COMMAND}`.')
    entry = next((item for item in inventory(root)['artifacts'] if root / item['path'] == path), None)
    if entry and (path.stat().st_size != entry['size'] or sha256(path) != entry['sha256']):
        raise ModelArtifactError(f'Model checksum mismatch: {entry["path"]}. Run `{RESTORE_COMMAND}` to repair it.')
    return entry
