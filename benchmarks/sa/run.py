#!/usr/bin/env python3
"""Run a pinned SA recipe with its native binder and srtctl orchestrator."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run(argv, **kwargs):
    print('+ ' + ' '.join(map(str, argv)), flush=True)
    return subprocess.run(list(map(str, argv)), check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--allocation', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    work = repo / '.sa-runs' / output.name
    work.mkdir(parents=True, exist_ok=True)
    for name in ('tmp', 'cache', 'hf', 'uv'):
        (output / 'work' / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(output / 'work/tmp'),
                      XDG_CACHE_HOME=str(output / 'work/cache'),
                      UV_CACHE_DIR=str(output / 'work/uv'),
                      UV_PYTHON_INSTALL_DIR=str(output / 'work/uv/python'),
                      UV_LINK_MODE='copy', UV_NO_MODIFY_PATH='1')
    (output / 'request.json').write_text(json.dumps(config, indent=2) + '\n')
    # Reuse only the explicitly requested, currently running, single-node reservation.
    info = subprocess.check_output(['squeue', '-h', '-j', args.allocation, '-o', '%u|%T|%N'], text=True).strip()
    owner, state, nodes = info.split('|')
    local = subprocess.check_output(['hostname', '-s'], text=True).strip()
    if owner != subprocess.check_output(['id', '-un'], text=True).strip() or state != 'RUNNING' or nodes != local:
        raise RuntimeError('Allocation must belong to the runner user and contain only this host: ' + info)
    (output / 'allocation.txt').write_text(info + '\n')
    os.environ.update(SLURM_JOB_ID=args.allocation, SLURM_JOB_NODELIST=nodes,
                      SLURM_NODELIST=nodes, SLURM_JOB_NUM_NODES='1', SLURM_NNODES='1',
                      SLURM_GPUS_PER_NODE='8')
    for folder, remote, sha in (
        ('InferenceX', 'SemiAnalysisAI/InferenceX', config['inferencex_sha']),
        ('srt-slurm', 'NVIDIA/srt-slurm', config['srt_slurm_sha']),
    ):
        dest = work / folder
        if not dest.exists():
            run(['git', 'init', '-q', dest])
            run(['git', '-C', dest, 'remote', 'add', 'origin', f'https://github.com/{remote}.git'])
            run(['git', '-C', dest, 'fetch', '--depth=1', 'origin', sha])
            run(['git', '-C', dest, 'checkout', '--detach', 'FETCH_HEAD'])
        actual = subprocess.check_output(['git', '-C', dest, 'rev-parse', 'HEAD'], text=True).strip()
        if actual != sha:
            raise RuntimeError(f'{folder}: expected {sha}, found {actual}')
    source = work / 'InferenceX/inferencex-e2e'
    if not source.exists():
        source = work / 'InferenceX'
    srt = work / 'srt-slurm'
    for patch in sorted((source / 'runners/srt-slurm/patches').glob('*.patch')):
        run(['git', '-C', srt, 'apply', patch])
    mirror = source / 'benchmarks/multi_node/srt-slurm-recipes'
    shutil.copytree(mirror, srt / 'recipes', dirs_exist_ok=True, symlinks=True)
    shutil.copytree(mirror / 'configs', srt / 'configs', dirs_exist_ok=True, symlinks=True)
    (srt / 'benchmarks/multi_node').mkdir(parents=True, exist_ok=True)
    (srt / 'benchmarks/multi_node/srt-slurm-recipes').symlink_to('../../recipes')
    uv = shutil.which('uv')
    if not uv:
        bootstrap = work / 'bootstrap'
        run([sys.executable, '-m', 'venv', bootstrap])
        run([bootstrap / 'bin/pip', 'install', 'uv'])
        uv = str(bootstrap / 'bin/uv')
    run([uv, 'venv', '--python', '3.12', srt / '.venv'])
    run([uv, 'pip', 'install', '--python', srt / '.venv/bin/python', '-e', srt])
    os.environ['PATH'] = str(srt / '.venv/bin') + ':' + str(Path(uv).parent) + ':' + os.environ['PATH']
    os.environ['PYTHONPATH'] = str(source)
    os.environ['INFERENCEX_REPOSITORY_ROOT'] = str(source)
    os.environ['INFMAX_WORKSPACE'] = str(source)
    runtime = subprocess.check_output(['bash', '-c', 'source "$1"; env -0', 'bash', str(source / 'benchmarks/runtime_settings.sh')])
    os.environ.update(dict(entry.decode().split('=', 1) for entry in runtime.split(b'\0') if entry))
    os.environ.update(config['environment'])
    os.environ['RESULT_FILENAME'] = config['name']
    os.environ['HF_HUB_CACHE'] = '/hf_hub_cache'
    run(['make', 'setup', 'ARCH=x86_64'], cwd=srt)
    # The rest runs in the pinned engine's environment, using its public renderer.
    run([srt / '.venv/bin/python', Path(__file__).with_name('run_bound.py'),
         '--config', args.config.resolve(), '--output', output, '--source', source,
         '--srt', srt], cwd=srt)


if __name__ == '__main__':
    main()
