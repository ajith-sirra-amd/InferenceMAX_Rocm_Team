#!/usr/bin/env python3
"""Isolated native AgentX preparation; no serving-runtime or SA source edits."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

TOOLS = Path(__file__).resolve().parent
WORK = Path('/agentic-work')
SOURCE = Path('/sa-client')
PYTHON = WORK / 'runtime/venv/bin/python'
PINS = json.loads((TOOLS / 'pins.json').read_text())


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def source_check():
    paths = [p for p in (SOURCE / 'infx').rglob('*') if p.is_file() and p.suffix in ('.py', '.txt')]
    paths += [SOURCE / 'benchmarks/runtime_settings.sh', SOURCE / 'benchmarks/srt_agentic.sh']
    files = {str(p.relative_to(SOURCE)): digest(p) for p in sorted(paths)}
    require(object_digest(files) == PINS['source_files_sha256'], 'Measured native client source differs from pin')
    require(digest(TOOLS / 'sa-client-constraints.txt') == PINS['constraints_sha256'], 'Constraints differ from SA pin')
    return files


def runtime_check():
    require(platform.python_version() == PINS['python'], 'Wrong client Python version')
    normalize = lambda name: re.sub(r'[-_.]+', '-', name).lower()
    expected = {}
    for line in (TOOLS / 'sa-client-constraints.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            name, version = line.split('==')
            expected[normalize(name)] = version
    actual = {normalize(d.metadata['Name']): d.version for d in importlib.metadata.distributions()}
    require(actual == expected, f'Client package set differs: missing/wrong={sorted(k for k,v in expected.items() if actual.get(k) != v)}, extra={sorted(set(actual)-set(expected))}')
    dist = importlib.metadata.distribution('agentx-harness')
    package = Path(dist.locate_file('aiperf'))
    files = {str(p.relative_to(package.parent)): digest(p) for p in sorted(package.rglob('*')) if p.is_file() and p.suffix in ('.py', '.yaml')}
    require(object_digest(files) == PINS['agentx_files_sha256'], 'AgentX package source differs from the pinned wheel')
    for name in ('python', 'agentx', 'hf'):
        require(os.access(PYTHON.parent / name, os.X_OK), f'Missing client executable: {name}')
    return actual


def cache_environment(offline=True):
    return {
        'AIPERF_RUNTIME_DIR': str(WORK / 'runtime'),
        'HF_HOME': str(WORK / 'cache/huggingface'),
        'HF_HUB_CACHE': str(WORK / 'cache/huggingface/hub'),
        'HF_DATASETS_CACHE': str(WORK / 'cache/huggingface/datasets'),
        'HF_MODULES_CACHE': str(WORK / 'cache/huggingface/modules'),
        'TIKTOKEN_CACHE_DIR': str(WORK / 'cache/tiktoken'),
        'XDG_CACHE_HOME': str(WORK / 'cache'),
        'UV_CACHE_DIR': str(WORK / 'cache/uv'),
        'UV_PYTHON_INSTALL_DIR': str(WORK / 'runtime/python'),
        'TMPDIR': str(WORK / 'tmp'),
        'PYTHONPATH': str(SOURCE),
        'PYTHONSAFEPATH': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
        **({'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'} if offline else {}),
    }


def cache_files():
    """Hash immutable inputs, following HF blob symlinks only within this cache."""
    result = {}
    for relative in ('cache/huggingface/hub', 'cache/huggingface/datasets', 'cache/huggingface/modules', 'cache/tiktoken'):
        for path in sorted((WORK / relative).rglob('*')):
            if not path.is_file() or path.name.endswith('.lock') or '__pycache__' in path.parts:
                continue
            require(path.resolve().is_relative_to(WORK), f'Cache file escapes prepared directory: {path}')
            result[str(path.relative_to(WORK))] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    return result


def cache_refs_check():
    data = PINS['data']
    hub = WORK / 'cache/huggingface/hub'
    for kind in ('dataset', 'tokenizer'):
        prefix = 'datasets' if kind == 'dataset' else 'models'
        repo = hub / (prefix + '--' + data[kind + '_repo'].replace('/', '--'))
        revision = data[kind + '_revision']
        require((repo / 'refs/main').read_text().strip() == revision, f'{kind} main is not bound to pinned revision')
        require(sorted(p.name for p in (repo / 'snapshots').iterdir() if p.is_dir()) == [revision], f'Unexpected {kind} snapshots')


def offline_probe():
    """Use the same default-main APIs that the unmodified native client uses."""
    from datasets import load_dataset
    from aiperf.common.tokenizer import Tokenizer
    data = PINS['data']
    cache_refs_check()
    subprocess.run([str(PYTHON.parent / 'hf'), 'download', '--repo-type', 'dataset', data['dataset_repo']], check=True, timeout=120)
    dataset = load_dataset(data['dataset_repo'], split='train', trust_remote_code=False)
    require(len(dataset) == data['dataset_rows'], 'Native offline loader returned wrong dataset size')
    tokenizer = Tokenizer.from_pretrained(data['tokenizer_repo'], trust_remote_code=True)
    token_ids = tokenizer.encode('Native agentic client cache validation.')
    require(bool(token_ids), 'Native offline tokenizer returned no tokens')
    evidence = {'rows': len(dataset), 'fingerprint': dataset._fingerprint, 'tokenizer_probe_ids': token_ids}
    online = json.loads((WORK / 'online-probe.json').read_text())
    require(evidence == online, 'Offline default-main APIs differ from pinned online inputs')
    write_json(WORK / 'offline-probe.json', evidence)


def prepare_inputs():
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from aiperf.common.tokenizer import Tokenizer
    runtime_check()
    data = PINS['data']
    snapshot_download(data['dataset_repo'], repo_type='dataset', revision=data['dataset_revision'])
    dataset = load_dataset(data['dataset_repo'], split='train', revision=data['dataset_revision'], trust_remote_code=False)
    require(len(dataset) == data['dataset_rows'], 'Pinned dataset row count differs')
    # AutoTokenizer retrieves only tokenizer assets and their required remote code.
    tokenizer = Tokenizer.from_pretrained(data['tokenizer_repo'], trust_remote_code=True, revision=data['tokenizer_revision'], resolve_alias=False)
    for kind in ('dataset', 'tokenizer'):
        prefix = 'datasets' if kind == 'dataset' else 'models'
        repo = WORK / 'cache/huggingface/hub' / (prefix + '--' + data[kind + '_repo'].replace('/', '--'))
        (repo / 'refs').mkdir(exist_ok=True)
        (repo / 'refs/main').write_text(data[kind + '_revision'])
    write_json(WORK / 'online-probe.json', {'rows': len(dataset), 'fingerprint': dataset._fingerprint, 'tokenizer_probe_ids': tokenizer.encode('Native agentic client cache validation.')})
    subprocess.run([str(PYTHON), str(TOOLS / 'prepare.py'), '--offline-probe'], env={**os.environ, **cache_environment()}, check=True, timeout=300)
    write_json(WORK / 'cache-manifest.json', cache_files())
    write_json(WORK / 'packages.json', runtime_check())
    baseline_path = Path('/baseline.json')
    require(digest(baseline_path) == PINS['baseline_recipe_sha256'], 'Baseline recipe changed')
    recipe = json.loads(baseline_path.read_text())
    recipe['name'] = 'kimi-k31-c48-native-sa-agentic-baseline'
    recipe['provenance'].update(sa_source_sha=PINS['sa_source_sha'], client_pins_sha256=digest(TOOLS / 'pins.json'), dataset_revision=data['dataset_revision'], tokenizer_revision=data['tokenizer_revision'])
    recipe['workload'] = {'type': 'native-agentx', 'concurrency': 48, 'benchmark_duration_seconds': 3600, 'dataset': data['dataset_repo'], 'agentx': '1.0.13', 'preset': PINS['agentx_preset']}
    recipe['note'] = 'Native SA AgentX workload and pinned core server recipe; local direct orchestration. Synthetic acceptance 3.57 retained. Power/9405 telemetry unavailable; no natural-quality or energy claim. Profiler off.'
    write_json(WORK / 'recipe.json', recipe)
    # The final marker exists only after all online/offline and identity checks pass.
    write_json(WORK / 'prepared.json', {
        'status': 'ready', 'prepared_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'pins_sha256': digest(TOOLS / 'pins.json'),
        'tools_sha256': {p.name: digest(p) for p in sorted(TOOLS.iterdir()) if p.is_file()},
        'cache_manifest_sha256': digest(WORK / 'cache-manifest.json'),
        'recipe_sha256': digest(WORK / 'recipe.json'),
        'offline_probe': json.loads((WORK / 'offline-probe.json').read_text()),
        'uv_version': (WORK / 'uv-version.txt').read_text().strip(),
        'limits': PINS['limits'],
    })
    print('Native agentic client prepared and offline inputs verified.', flush=True)


def main():
    source_check()
    if sys.argv[1:] == ['--offline-probe']:
        offline_probe()
        return
    if sys.argv[1:] == ['--prepare-inputs']:
        prepare_inputs()
        return
    require(not sys.argv[1:], 'Unknown preparation arguments')
    require(not (WORK / 'prepared.json').exists(), 'Prepared runtime already exists')
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith(('PYTHON', 'AIPERF_', 'AGENTX_')) or key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'VIRTUAL_ENV'):
            environment.pop(key)
    environment.update(cache_environment(offline=False))
    # AgentX considers even HF_HUB_OFFLINE=0 offline; online prep must remove it.
    for directory in ('runtime', 'tmp', 'cache/huggingface/hub', 'cache/huggingface/datasets', 'cache/huggingface/modules', 'cache/tiktoken', 'cache/uv'):
        (WORK / directory).mkdir(parents=True, exist_ok=True)
    os.environ.clear()
    os.environ.update(environment)
    sys.path.insert(0, str(SOURCE))
    from infx.bench import uv
    uv_exe = uv.find(WORK / 'runtime/uv/bin')
    (WORK / 'uv-version.txt').write_text(subprocess.check_output([uv_exe, '--version'], text=True))
    subprocess.run([uv_exe, 'venv', '--python', PINS['python'], str(PYTHON.parent.parent)], check=True, timeout=300)
    wheel = PINS['agentx_wheel_url'] + '#sha256=' + PINS['agentx_wheel_sha256']
    subprocess.run([uv_exe, 'pip', 'install', '--python', str(PYTHON), '--no-config', '-r', str(SOURCE / 'infx/bench/agentic/requirements.txt'), '-c', str(TOOLS / 'sa-client-constraints.txt'), wheel], check=True, timeout=600)
    os.execve(PYTHON, [str(PYTHON), str(TOOLS / 'prepare.py'), '--prepare-inputs'], environment)


if __name__ == '__main__':
    main()
