"""Bounded read-only inventory of installed profiling-relevant source.

Installed files are evidence for this image, not proof that a particular graph
node used them. Importing AITER records registered schemas but launches no work.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def inventory(output):
    output.mkdir(parents=True, exist_ok=False)
    result = dict(scope='Installed-image source; no measured graph-node binding',
                  python=sys.version, packages={}, imports={}, files=[], schemas={})
    roots = []
    for name in ('aiter', 'vllm', 'torch', 'triton'):
        try:
            dist = importlib.metadata.distribution(name)
            result['packages'][name] = dict(version=dist.version,
                direct_url=dist.read_text('direct_url.json'))
        except importlib.metadata.PackageNotFoundError:
            result['packages'][name] = {'error': 'Distribution metadata absent'}
        if name in ('aiter', 'vllm'):
            try:
                spec = importlib.util.find_spec(name)
                if spec and spec.origin:
                    roots.append(Path(spec.origin).resolve().parent)
                module = importlib.import_module(name)
                path = Path(module.__file__).resolve()
                result['imports'][name] = str(path)
                git = subprocess.run(['git', '-C', str(path.parent), 'rev-parse', 'HEAD'],
                    capture_output=True, text=True, timeout=5)
                result['packages'][name]['git_revision'] = git.stdout.strip() if git.returncode == 0 else None
            except Exception as error:
                result['imports'][name] = {'error': repr(error)}
    # Source trees may sit next to an editable AITER package.
    for root in list(roots):
        if root.name == 'aiter':
            roots += [root.parent / name for name in ('csrc', 'aiter_meta')]
    pattern = re.compile(r'custom.?all.?reduce|allreduce|all_reduce|communicat|'
                         r'gemm.*a16w16|a4w4|fp4|fused.?moe|moe.?fused|moe_flydsl|'
                         r'rocm_aiter|jit/(core|utils)|jit/config', re.I)
    seen, copied, scanned = set(), 0, 0
    for root in roots:
        if not root.is_dir():
            continue
        for directory, dirs, files in os.walk(root):
            dirs[:] = [d for d in sorted(dirs) if d not in ('.git', '__pycache__', 'node_modules')]
            for name in sorted(files):
                scanned += 1
                if scanned > 30000:
                    break
                path = Path(directory, name)
                if path.suffix not in ('.py', '.cpp', '.cc', '.cu', '.cuh', '.h', '.hpp', '.json'):
                    continue
                if not pattern.search(str(path.relative_to(root))):
                    continue
                path = path.resolve()
                if path in seen:
                    continue
                seen.add(path)
                record = dict(path=str(path), bytes=path.stat().st_size)
                if record['bytes'] <= 2 * 1024**2 and copied + record['bytes'] <= 32 * 1024**2:
                    data = path.read_bytes()
                    record['sha256'] = hashlib.sha256(data).hexdigest()
                    saved = output / 'sources' / str(path).lstrip('/')
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    saved.write_bytes(data)
                    record['saved'] = str(saved.relative_to(output))
                    copied += len(data)
                else:
                    record['omitted'] = 'Bounded source inventory size limit'
                result['files'].append(record)
            if scanned > 30000:
                break
    result['scanned_files'] = scanned
    result['copied_source_bytes'] = copied
    saved = [v['path'] for v in result['files'] if 'saved' in v]
    result['source_coverage'] = dict(
        aiter_reduction_cpp=any('custom_all_reduce' in p and Path(p).suffix in ('.cu', '.cpp') for p in saved),
        aiter_reduction_header=any('custom_all_reduce' in p and Path(p).suffix in ('.cuh', '.h', '.hpp') for p in saved),
        reduction_python=any('all_reduce' in p and p.endswith('.py') for p in saved),
        vllm_communicator=any('/vllm/' in p and 'communicator' in p and p.endswith('.py') for p in saved),
        selected_gemm_source=any(re.search(r'gemm.*a16w16', p) for p in saved),
        fp4_gemm_source=any('gemm' in p and re.search(r'a4w4|fp4', p) for p in saved),
        fused_moe_source=any(re.search(r'fused.?moe|moe.?fused|moe_flydsl', p) for p in saved))
    result['source_status'] = 'complete' if all(result['source_coverage'].values()) else 'incomplete'
    result['imported_relevant_python_modules'] = []
    for name, module in sorted(sys.modules.copy().items()):
        path = getattr(module, '__file__', None)
        if name.startswith(('aiter.', 'vllm.')) and path and pattern.search(str(path)):
            result['imported_relevant_python_modules'].append(dict(module=name, path=str(Path(path).resolve())))
    result['invocation_binding'] = 'Package/submodule imports are observed here; no measured reduction graph-node or JIT-kernel binding'
    try:
        import torch
        for name in ('aiter::_gemm_a16w16_asm', 'aten::mm', 'aten::bmm'):
            result['schemas'][name] = [str(s) for s in torch._C._jit_get_schemas_for_operator(name)]
        result['registered_fp4_moe_schemas'] = [str(s) for s in torch._C._jit_get_all_schemas()
            if s.name.startswith('aiter::') and re.search(r'a4w4|fp4|moe', s.name)][:128]
        aiter = sys.modules.get('aiter')
        for name in ('gemm_a16w16', 'gemm_a16w16_asm', 'all_reduce'):
            function = getattr(aiter, name, None)
            if callable(function):
                result['schemas']['python:aiter.' + name] = dict(
                    signature=str(inspect.signature(function)), source=inspect.getsourcefile(function))
    except Exception as error:
        result['schema_error'] = repr(error)
    loaded = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/') and re.search(r'aiter|rccl|nccl|custom.*reduce', fields[5], re.I):
            loaded.add(fields[5])
    result['loaded_libraries'] = []
    for name in sorted(loaded):
        path = Path(name)
        record = dict(path=name)
        if path.is_file():
            record['bytes'] = path.stat().st_size
            if record['bytes'] <= 512 * 1024**2:
                with path.open('rb') as stream:
                    record['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
            else:
                record['omitted'] = 'Bounded binary hashing size limit'
        result['loaded_libraries'].append(record)
    (output / 'inventory.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(source_files=len(result['files']), copied_bytes=copied,
                         source_status=result['source_status'], packages=result['packages'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    inventory(parser.parse_args().output)
