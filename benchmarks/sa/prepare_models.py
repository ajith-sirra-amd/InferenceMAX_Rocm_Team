"""Provision revision-pinned Hub snapshots under the runner user's /data directory."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def verify_file(path, metadata):
    if path.stat().st_size != metadata.size:
        raise ValueError('Incorrect checkpoint file size: ' + str(path))
    expected = metadata.lfs.sha256 if metadata.lfs else metadata.blob_id
    digest = hashlib.sha256() if metadata.lfs else hashlib.sha1()
    if not metadata.lfs:
        digest.update(('blob ' + str(metadata.size) + '\0').encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError('Checkpoint checksum mismatch: ' + str(path))
    return dict(file=metadata.rfilename, bytes=metadata.size, hash=actual,
                mtime_ns=path.stat().st_mtime_ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if os.environ.get('RUNNER_NAME') != config['expected_runner']:
        raise RuntimeError('Selected runner does not match expected_runner')
    root = Path(config['local_model_root'])
    if root != Path('/data') or not root.is_dir():
        raise ValueError('This setup requires the confirmed /data mount')
    owned = root / 'ajith'
    if not owned.exists():
        # Docker is already authorized on this host. Create only our new namespace;
        # never change the ownership of an existing shared/personal directory.
        helper = subprocess.check_output(['docker', 'image', 'inspect', '--format',
            '{{.Id}}', 'ubuntu:22.04'], text=True).strip()
        subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--read-only',
            '--mount', 'type=bind,src=/data,dst=/data', '--entrypoint', 'sh', helper,
            '-c', 'mkdir -m 750 /data/ajith && chown "$1:$2" /data/ajith',
            'sh', str(os.getuid()), str(os.getgid())], check=True)
    if owned.is_symlink() or owned.stat().st_uid != os.getuid() or not os.access(owned, os.W_OK):
        raise RuntimeError('/data/ajith must be a writable directory owned by the runner user')
    report = dict(status='preparing', started_unix=time.time(), models=[])
    target = args.output / 'checkpoint-download.json'
    def save():
        target.write_text(json.dumps(report, indent=2) + '\n')
    os.environ.update(HF_HOME=str(args.output / 'work/hf-download'),
                      HF_XET_CHUNK_CACHE_SIZE_BYTES='0', HF_HUB_DOWNLOAD_TIMEOUT='120')
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    plans = []
    for model, revision in config['model_revisions'].items():
        destination = root / config['local_model_paths'][model]
        if not destination.resolve().is_relative_to(owned.resolve()) or destination.name != revision:
            raise ValueError('Snapshots must be revision-specific directories under /data/ajith')
        info = api.model_info(model, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError('Hub revision did not resolve to the pinned commit')
        size = sum(f.size for f in info.siblings)
        present = sum((destination / f.rfilename).stat().st_size for f in info.siblings
                      if (destination / f.rfilename).is_file())
        plans.append((model, revision, destination, info, max(0, size - present)))
    required = sum(p[-1] for p in plans) + 100 * 1024**3
    free = shutil.disk_usage(root).free
    report.update(available_bytes=free, required_bytes_including_headroom=required)
    save()
    if free < required:
        raise RuntimeError('Insufficient /data capacity for pinned models and 100 GiB headroom')
    try:
        for model, revision, destination, info, needed in plans:
            row = dict(model=model, revision=revision, path=str(destination),
                       bytes=sum(f.size for f in info.siblings), status='downloading')
            report['models'].append(row); save()
            print(json.dumps(row), flush=True)
            destination.mkdir(parents=True, exist_ok=True)
            snapshot_download(model, revision=revision, local_dir=destination, max_workers=8)
            row['status'] = 'verifying'; save()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                verified = list(pool.map(lambda f: verify_file(destination / f.rfilename, f), info.siblings))
            row.update(status='verified', files=verified)
            (destination / '.ipat-checkpoint.json').write_text(json.dumps(row, indent=2) + '\n')
            save()
            print(f'Verified {model}: {len(verified)} files', flush=True)
        report.update(status='completed', finished_unix=time.time()); save()
    except BaseException as error:
        report.update(status='failed', error=str(error), finished_unix=time.time()); save()
        raise


if __name__ == '__main__':
    main()
