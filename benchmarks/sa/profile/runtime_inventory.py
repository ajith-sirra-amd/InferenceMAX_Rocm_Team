"""Record the pinned container's installed Kimi source and package versions."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(importlib.util.find_spec('vllm').origin).parent
    files = set((root / 'models/kimi_k3').rglob('*.py'))
    files.update((root / 'v1/worker/gpu/spec_decode/dspark').rglob('*.py'))
    for relative in (
        'model_executor/models/qwen3_dflash.py', 'model_executor/models/qwen3_dspark.py',
        'model_executor/models/k3_dspark.py', 'config/profiler.py', 'profiler/wrapper.py',
        'v1/attention/backends/mla/rocm_aiter_mla_sparse.py',
        'v1/worker/gpu/model_runner.py', 'v1/worker/gpu_worker.py', 'envs.py',
    ):
        path = root / relative
        if path.is_file():
            files.add(path)
    manifest = []
    for path in sorted(files):
        relative = Path('vllm') / path.relative_to(root)
        data = path.read_bytes()
        target = args.output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        compile(data, str(relative), 'exec')
        manifest.append(dict(file=str(relative), bytes=len(data),
                             sha256=hashlib.sha256(data).hexdigest(), syntax_valid=True))
    packages = {}
    for package in ('vllm', 'torch', 'triton', 'pytorch-triton-rocm', 'aiter',
                    'flydsl', 'amd-kernels', 'transformers', 'fastsafetensors'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    facts = dict(scope='Installed source identity and syntax; not runtime activation or numerical validation',
                 vllm_path=str(root), packages=packages, files=manifest,
                 tools={tool: shutil.which(tool) for tool in ('rocprofv3', 'rocprof-compute', 'amd-smi')})
    (args.output / 'manifest.json').write_text(json.dumps(facts, indent=2) + '\n')
    print(json.dumps(dict(packages=packages, source_files=len(manifest), output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
