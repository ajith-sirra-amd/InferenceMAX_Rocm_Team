#!/usr/bin/env python3
"""Bounded observational diagnostic using the existing natural-screen recipe."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

sys.path.insert(0, '/quality')
import launch as quality_launch
import screen


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def client(endpoint, output):
    from native_spec_decode import parse
    config, rows = screen.assets()
    model = os.environ['MODEL']
    errors = []
    for stage, temperature, selected in (
        ('c1-greedy', 0.0, rows[:1]),
        ('c1-sampled', 1.0, rows[:1]),
        ('c48-sampled', 1.0, rows),
    ):
        stage_dir = output / stage
        stage_dir.mkdir()
        # No other requests can enter this independently launched server.
        with urlopen(endpoint + '/metrics', timeout=10) as response:
            (stage_dir / 'metrics-before.prom').write_bytes(response.read())
        marker = output / 'stage.tmp'
        marker.write_text(stage)
        marker.replace(output / 'stage.txt')
        requests = []
        for row in selected:
            body = screen.prompt(row, model)
            body.update(temperature=temperature, max_tokens=128,
                        return_token_ids=True)
            requests.append((row, body))
        save(stage_dir / 'requests.json', [body for _, body in requests])
        with ThreadPoolExecutor(max_workers=len(selected)) as pool:
            records = list(pool.map(lambda pair: screen.generate(pair[0], pair[1], endpoint), requests))
        with (stage_dir / 'responses.jsonl').open('x') as sink:
            for record in records:
                sink.write(json.dumps(record, ensure_ascii=False) + '\n')
        for record in records:
            if record.get('error'):
                errors.append(dict(stage=stage, id=record['id'], error=record['error']))
        # Async output can complete before the last proposal. Keep the old marker
        # until idle counters settle; saved request IDs bind every captured call.
        deadline, previous, stable = time.monotonic() + 15, None, 0
        while time.monotonic() < deadline:
            with urlopen(endpoint + '/metrics', timeout=5) as response:
                metrics = response.read()
            (stage_dir / 'metrics-after.prom').write_bytes(metrics)
            current = screen.snapshot(metrics.decode(), parse)
            idle = all(value == 0 for key in screen.GAUGES for value in current[key].values())
            stable = stable + 1 if idle and current == previous else 0
            if stable >= 2:
                break
            previous = current
            time.sleep(1)
        else:
            raise RuntimeError('Stage counters did not become idle in15seconds')
        (output / 'stage.txt').unlink()
    proposals = sorted(output.glob('rank-*-*-proposal-*.json'))
    verifies = sorted(output.glob('rank-*-*-verification-*.json'))
    captures = [json.loads(path.read_text()) for path in output.glob('rank-*.json')]
    for record in captures:
        if record.get('diagnostic_error'):
            errors.append(dict(stage=record['stage'], rank=record['rank'], error=record['diagnostic_error']))
    coverage = {}
    for stage in ('c1-greedy', 'c1-sampled', 'c48-sampled'):
        coverage[stage] = {}
        for kind in ('proposal', 'verification', 'sampler', 'ownership', 'draft-graph'):
            coverage[stage][kind] = {str(rank): sorted(r['call'] for r in captures if r['stage'] == stage and r['kind'] == kind and r['rank'] == rank) for rank in range(8)}
        if any(coverage[stage]['proposal'][str(rank)] != [1, 2] for rank in range(8)):
            errors.append(dict(stage=stage, error='Missing first two proposal captures on one or more workers'))
        if not any(coverage[stage]['verification'].values()):
            errors.append(dict(stage=stage, error='No verification capture'))
    if any(coverage['c48-sampled']['verification'][str(rank)] != [1, 2] for rank in range(8)):
        errors.append(dict(stage='c48-sampled', error='Missing first two C48 verification captures on one or more workers'))
    summary = dict(status='failed' if errors else 'collected', request_count=50,
                   max_tokens=128, coverage=coverage, errors=errors,
                   limits='Observational diagnostic; synchronization changes timing. No performance or quality claim. Verification coverage follows native sampling ownership; unused ranks are reported explicitly.')
    save(output / 'summary.json', summary)
    return 1 if errors else 0


def main():
    logs = Path(os.environ['RESULT_DIR'])
    output = logs / 'diagnostic'
    output.mkdir(exist_ok=False)
    validation, _ = screen.assets()
    config = quality_launch.recipe(Path('/fixed-tools/baseline.json'), validation)
    save(logs / 'natural-recipe.json', config)
    extension = 'worker_extension.DiagnosticWorkerExtension'
    config['server_arguments']['worker-extension-cls'] = extension
    os.environ.update(config['server_environment'])
    os.environ.update(KIMI_DIAGNOSTIC_OUTPUT=str(output),
                      PYTHONPATH='/quality-diagnostic:/inferencex',
                      VLLM_COMPUTE_NANS_IN_LOGITS='1')
    argv = quality_launch.command(config, os.environ['MODEL_PATH'], os.environ['PORT'])
    save(logs / 'recipe.json', config)
    (logs / 'vllm_command.txt').write_text(shlex.join(argv) + '\n')
    sys.path.insert(0, '/fixed-tools')
    import audit_server
    audit_server.audit(config, logs, os.environ['MODEL_PATH'], os.environ['PORT'],
                       Path('/opt/runtime-venv/lib/python3.12/site-packages'), argv)
    effective_path = logs / 'effective-server.json'
    effective = json.loads(effective_path.read_text())
    effective.update(sa_config_match=False, sa_throughput_recipe_match=False,
                     expected_natural_config_match=False,
                     expected_natural_core_config_match=True,
                     instrumented_diagnostic=True,
                     instrumentation_arguments={'worker-extension-cls': extension})
    save(effective_path, effective)
    save(output / 'contract.json', dict(
        mode='natural verifier diagnostic, baseline only',
        server_config_sha256=hashlib.sha256((logs / 'recipe.json').read_bytes()).hexdigest(),
        natural_core_config_sha256=hashlib.sha256((logs / 'natural-recipe.json').read_bytes()).hexdigest(),
        helper_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/quality-diagnostic').glob('*.py')},
        hooks='first 2 live proposal and verification calls per scenario',
        hook_initialization='native WorkerWrapperBase.init_device extension hook, before model load',
        server_argument_changes={'worker-extension-cls': extension},
        environment_changes={'PYTHONPATH': os.environ['PYTHONPATH'], 'VLLM_COMPUTE_NANS_IN_LOGITS': '1'},
        scenarios=['C1 temperature0 max_tokens128', 'C1 temperature1 max_tokens128', 'C48 temperature1 max_tokens128'],
        output='raw responses/token IDs; bounded per-worker summaries; no full logits saved'))
    server = None
    result = dict(status='failed', error=None)
    def interrupted(signum, frame):
        raise RuntimeError(f'Diagnostic interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with (output / 'hook-preflight.log').open('xb') as log:
            subprocess.run([sys.executable, '-u', '-c',
                            'import faulthandler; faulthandler.enable(); '
                            'faulthandler.dump_traceback_later(30, repeat=True); '
                            'print("Python startup complete; importing diagnostic hooks", flush=True); '
                            'import diagnostic_hooks; '
                            'print("Installing diagnostic hooks", flush=True); '
                            'diagnostic_hooks.install(); assert diagnostic_hooks.INSTALLED; '
                            'print("Diagnostic hooks installed", flush=True); '
                            'faulthandler.cancel_dump_traceback_later()'],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90)
        with (logs / 'server.log').open('xb') as log:
            server = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            endpoint = 'http://127.0.0.1:' + os.environ['PORT']
            deadline = time.monotonic() + 1800
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited: {server.returncode}')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Readiness exceeded30minutes')
                try:
                    with urlopen(endpoint + '/health', timeout=5) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(2)
            observed = quality_launch.observed_verifier((logs / 'server.log').read_text(errors='replace'), json.loads(config['server_arguments']['speculative-config']))
            observed.update(scope='speculative_config_only',
                            expected_natural_config_match=False,
                            expected_natural_core_config_match=True,
                            speculative_config_match=True,
                            instrumented_diagnostic=True)
            save(logs / 'observed-verifier.json', observed)
            # Only native workers invoke the extension; client startup has no hooks.
            command = ['timeout', '--signal=TERM', '--kill-after=15s', '360s',
                       sys.executable, __file__, 'client', endpoint, str(output)]
            result['client_exit_code'] = subprocess.call(command)
            if result['client_exit_code']:
                raise RuntimeError('Diagnostic capture incomplete; see summary or logs')
            result['status'] = 'collected'
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        (output / 'stage.txt').unlink(missing_ok=True)
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=15)
        save(output / 'launch-result.json', result)
    return 0 if result['status'] == 'collected' else 1


if __name__ == '__main__':
    if sys.argv[1:2] == ['client']:
        raise SystemExit(client(sys.argv[2], Path(sys.argv[3])))
    raise SystemExit(main())
