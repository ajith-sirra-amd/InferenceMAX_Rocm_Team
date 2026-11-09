"""Run the native AgentX client with an explicit diagnostic warmup count."""
from dataclasses import replace
import os
from pathlib import Path
import sys

from infx.bench import proc
from infx.bench.agentic.run import Plan, execute
from infx.bench.agentic.venv import Runtime, bootstrap
from infx.bench.env import BenchError


def diagnostic_plan(environment):
    plan = Plan.from_env(environment)
    warmup = int(environment['SA_PROFILE_WARMUP_REQUESTS_PER_LANE'])
    if warmup < 1 or environment['AIPERF_EXPERIMENTAL_FAST'] != '0':
        raise ValueError('Use a positive warmup count and explicit duration, without fast mode')
    return replace(plan, replay=replace(plan.replay, warmup_requests_per_lane=str(warmup)))


def main():
    os.environ.pop('PYTHONSAFEPATH', None)
    plan = diagnostic_plan(os.environ)
    runtime = Runtime.for_job(os.environ)
    if not runtime.active():
        code = bootstrap(runtime)
        if code:
            return code
        runtime.exec_python([str(Path(__file__).resolve())],
            {**os.environ, 'PYTHONPATH': proc.pythonpath(), 'PYTHONSAFEPATH': '1'})
    return execute(plan, runtime, os.environ)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (BenchError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(1)
