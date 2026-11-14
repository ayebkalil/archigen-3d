"""Recover only missing model files from verified, configured local DVC objects."""
import configparser
import hashlib
from pathlib import Path
import shutil

import yaml


def recover_local_weights(root):
    root = Path(root).resolve()
    model_root = root/'models/saved'
    caches = [root/'.dvc/cache', root/'.dvc/cache/files/md5']
    config = configparser.ConfigParser()
    config.read(root/'.dvc/config')
    for section in config.sections():
        if section.strip("'").startswith('remote ') and config.has_option(section,'url'):
            url = config.get(section,'url')
            if '://' not in url:
                path = Path(url)
                if not path.is_absolute(): path = root/path
                caches.extend([path,path/'files/md5'])
    caches = [p.resolve() for p in caches if p.is_dir()]
    recovered, warnings = [], []
    for manifest in model_root.rglob('*.dvc'):
        try:
            for output in (yaml.safe_load(manifest.read_text()) or {}).get('outs',[]):
                target = (manifest.parent/output['path']).resolve()
                digest = output.get('md5','')
                if target.exists() or not target.is_relative_to(model_root.resolve()) or target.suffix not in ('.pth','.onnx','.safetensors'):
                    continue
                if len(digest)!=32 or any(c not in '0123456789abcdef' for c in digest): continue
                for cache in caches:
                    candidate = cache/digest[:2]/digest[2:]
                    if not candidate.is_file(): continue
                    if output.get('size') is not None and candidate.stat().st_size!=output['size']:
                        warnings.append(f'{target.name}: local DVC size mismatch'); continue
                    with candidate.open('rb') as stream: actual = hashlib.file_digest(stream,'md5').hexdigest()
                    if actual!=digest:
                        warnings.append(f'{target.name}: local DVC digest mismatch'); continue
                    temporary = target.with_suffix(target.suffix+'.stress-restore.tmp')
                    try:
                        shutil.copyfile(candidate,temporary)
                        temporary.replace(target)
                    finally: temporary.unlink(missing_ok=True)
                    recovered.append({'path':str(target.relative_to(root)), 'dvc_md5':digest,'source':str(candidate)})
                    break
        except (OSError,KeyError,yaml.YAMLError) as exc:
            warnings.append(f'{manifest.name}: {exc}')
    return {'recovered':recovered,'warnings':warnings,'existing_local_caches':[str(p) for p in caches],
            'network_downloads':False}
