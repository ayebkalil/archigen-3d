"""Bounded, resumable HF weight downloads for networks that stall on large GETs."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time

from src.models.facade_diffusion import sha256

CHUNK = 8 * 1024**2


def fetch_range(url, start, end, size):
    import requests
    expected = end - start + 1
    for attempt in range(4):
        try:
            with requests.get(url, headers={'Range': f'bytes={start}-{end}'},
                              params={'download': 'true', 'range_start': str(start)},
                              stream=True, timeout=(15, 45)) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                    raise ValueError('Server ignored or changed requested byte range')
                data = bytearray()
                for block in response.iter_content(1024**2):
                    data.extend(block)
                    if len(data) > expected:
                        raise ValueError('Oversized byte-range response')
                if len(data) != expected:
                    raise ValueError('Truncated byte-range response')
                return data
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def download_weight(url, destination, spec):
    """Only contiguous validated ranges are persisted; final SHA-256 is mandatory."""
    from filelock import FileLock
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + '.archigen.lock', timeout=1):
        if destination.is_file():
            if destination.stat().st_size == spec['size'] and sha256(destination) == spec['sha256']:
                return
            raise ValueError(f'Existing weight failed checksum: {destination}')
        partial = destination.with_suffix(destination.suffix + '.archigen.part')
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > spec['size']:
            raise ValueError(f'Partial weight is too large: {partial}')
        # A killed write may have persisted only part of one range. Re-fetch it.
        if offset != spec['size']:
            offset = offset // CHUNK * CHUNK
        started, resumed = time.monotonic(), offset
        with partial.open('r+b' if partial.exists() else 'w+b') as stream, ThreadPoolExecutor(max_workers=4) as pool:
            stream.truncate(offset)
            stream.seek(offset)
            while offset < spec['size']:
                ranges = [(start, min(start + CHUNK, spec['size']) - 1)
                          for start in range(offset, min(offset + 4*CHUNK, spec['size']), CHUNK)]
                futures = [pool.submit(fetch_range, url, start, end, spec['size']) for start, end in ranges]
                for future in futures:  # Consume in order, even when requests finish out of order.
                    block = future.result()
                    stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                    offset += len(block)
                speed = (offset - resumed) / max(time.monotonic() - started, .001) / 1024**2
                print(f'{destination.name}: {offset / 1024**2:.0f}/{spec["size"] / 1024**2:.0f} MiB ({speed:.2f} MiB/s)', flush=True)
        if sha256(partial) != spec['sha256']:
            raise ValueError(f'Download checksum mismatch: {partial}')
        partial.replace(destination)
