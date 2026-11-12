#!/usr/bin/env python3
"""Validate then hand off to the unmodified measured SA agentic module."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import (PINS, PYTHON, SOURCE, TOOLS, WORK, cache_environment,
                     cache_refs_check, digest, require, runtime_check, source_check,
                     write_json)


def environment():
    inherited = os.environ
    require(inherited.get('MODEL') == PINS['environment']['MODEL'], 'Wrong served model')
    expected_result_dir = '/logs/agentic' if Path(sys.executable) == PYTHON else '/logs'
    require(inherited.get('RESULT_DIR') == expected_result_dir, 'Expected the per-arm result directory')
    require(inherited.get('CONC') == '48' and inherited.get('DURATION') == '3600', 'Expected native C48 / 3600-second workload')
    port = int(inherited['PORT'])
    require(0 < port < 65536, 'Invalid server port')
    # Drop replay overrides inherited from the serving/runner environment.
    env = {k: v for k, v in inherited.items() if not k.startswith(('AIPERF_', 'AGENTX_', 'AGENTIC_', 'SRT_', 'SRTCTL_', 'SLURM_', 'PYTHON', 'SPEC_', 'KV_', 'TEST')) and k not in ('TOKENIZER', 'SERVED_MODEL_NAME', 'VIRTUAL_ENV')}
    env.update(PINS['environment'])
    env.update(cache_environment())
    env.update(AIPERF_SERVER_URL=f'http://127.0.0.1:{port}', AIPERF_SERVER_METRICS_URLS=f'http://127.0.0.1:{port}/metrics',
               PATH=str(PYTHON.parent) + ':/usr/local/bin:/usr/bin:/bin',
               TMPDIR='/logs/work/agentic-tmp')
    return env


def check_prepared(full):
    source_check()
    runtime_check()
    prepared = json.loads((WORK / 'prepared.json').read_text())
    require(prepared.get('status') == 'ready', 'Prepared client is not ready')
    require(prepared['pins_sha256'] == digest(TOOLS / 'pins.json'), 'Client pins changed after preparation')
    for name, expected in prepared['tools_sha256'].items():
        require(digest(TOOLS / name) == expected, f'Client helper changed: {name}')
    require(prepared['recipe_sha256'] == digest(WORK / 'recipe.json'), 'Prepared recipe changed')
    require(digest('/logs/recipe.json') == prepared['recipe_sha256'], 'Per-arm recipe differs from prepared recipe')
    require(prepared['cache_manifest_sha256'] == digest(WORK / 'cache-manifest.json'), 'Cache manifest changed')
    cache_refs_check()
    if full:
        for relative, expected in json.loads((WORK / 'cache-manifest.json').read_text()).items():
            path = WORK / relative
            require(path.resolve().is_relative_to(WORK), f'Cache file escapes prepared storage: {relative}')
            require(path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'], f'Prepared input changed: {relative}')
    return prepared


def main():
    require(sys.argv[1:] in ([], ['--check']), 'Usage: client.py [--check]')
    env = environment()
    if Path(sys.executable) != PYTHON:
        os.execve(PYTHON, [str(PYTHON), str(TOOLS / 'client.py'), *sys.argv[1:]], env)
    os.environ.clear()
    os.environ.update(env)
    sys.path.insert(0, str(SOURCE))
    from infx.bench.agentic.run import Plan
    from infx.bench.agentic.replay import replay_argv
    from infx.bench.agentic.venv import Runtime
    full = sys.argv[1:] == ['--check']
    prepared = check_prepared(full)
    plan = Plan.from_env(env)
    runtime = Runtime.for_job(env)
    require(runtime.active(), 'Native entry would bootstrap and replace the prepared runtime')
    require(plan.power == 'missing' and not plan.require_power, 'Unexpected power window contract')
    require(plan.replay.concurrency == 48 and plan.replay.duration == 3600 and not plan.replay.unsafe_override, 'Native plan changed')
    Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)
    plan.result_dir.mkdir(parents=True, exist_ok=True)
    server_log = plan.result_dir / 'server.log'
    if server_log.is_symlink():
        require(server_log.readlink() == Path('../server.log'), 'Unexpected agentic server.log target')
    else:
        require(not server_log.exists(), 'Unexpected agentic server.log')
        server_log.symlink_to('../server.log')
    evidence = {'native_plan': asdict(plan), 'native_argv': [str(PYTHON), '-m', 'infx.bench', 'agentic'],
                'agentx_argv': replay_argv(plan.replay, runtime.agentx),
                'client_environment': {k: env[k] for k in sorted(set(PINS['environment']) | set(cache_environment()) | {'PATH'})},
                'preparation': prepared, 'full_cache_verification': full}
    if full:
        write_json('/logs/agentic-client-check.json', evidence)
        print('Native agentic plan, source, runtime and pinned cache validated before GPU startup.', flush=True)
        return
    require(Path('/logs/agentic-client-check.json').is_file(), 'Missing pre-server client check')
    write_json('/logs/agentic-client-start.json', evidence)
    os.chdir('/logs')
    os.execve(PYTHON, [str(PYTHON), '-m', 'infx.bench', 'agentic'], env)


if __name__ == '__main__':
    main()
