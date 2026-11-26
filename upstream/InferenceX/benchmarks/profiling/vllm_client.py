"""Run native ``vllm bench serve`` with optional measured-request timestamps.

Hooks live only in this client process. The installed vLLM files, request
generation, concurrency, warmup and result calculation are unchanged.
"""

import json
import os
import runpy
import sys
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def measured_requests(module, phase_file):
    original_get_request = module.get_request
    original_functions = dict(module.ASYNC_REQUEST_FUNCS)
    interval = {}

    def save():
        temporary = phase_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(interval) + "\n")
        temporary.replace(phase_file)

    async def get_request(*args, **kwargs):
        # Native benchmark calls this after warmup, at the measured request loop.
        interval["start_unix"] = time.time()
        save()
        async for item in original_get_request(*args, **kwargs):
            yield item

    def track_request(function):
        async def request(*args, **kwargs):
            result = await function(*args, **kwargs)
            if "start_unix" in interval:
                interval["end_unix"] = time.time()
            return result

        return request

    module.get_request = get_request
    module.ASYNC_REQUEST_FUNCS.update(
        {name: track_request(function) for name, function in original_functions.items()}
    )
    try:
        yield
    finally:
        module.get_request = original_get_request
        module.ASYNC_REQUEST_FUNCS.update(original_functions)
        if interval:
            save()


def main():
    sys.argv = ["vllm", "bench", "serve", *sys.argv[1:]]
    phase_file = os.environ.get("INFERENCEX_PROFILE_PHASE_FILE")
    if phase_file:
        from vllm.benchmarks import serve

        with measured_requests(serve, Path(phase_file)):
            runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__")
    else:
        runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__")


if __name__ == "__main__":
    main()
