"""Bounded native Torch export on eight GPUs; synthetic capability only."""
import argparse
import dataclasses
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(device, output):
    print(f'GPU {device}: importing torch and native profiler', flush=True)
    import torch
    import vllm
    from vllm.config import ProfilerConfig
    from vllm.profiler.wrapper import TorchProfilerWrapper

    output.mkdir(parents=True, exist_ok=False)
    print(f'GPU {device}: imports completed; preparing CUDA graph', flush=True)
    torch.set_num_threads(1)
    assert torch.cuda.device_count() == 8
    torch.cuda.set_device(device)
    x = torch.ones((512, 512), device='cuda', dtype=torch.bfloat16)
    y = torch.empty_like(x)
    for _ in range(3):
        torch.mm(x, x, out=y)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.mm(x, x, out=y)
    config = ProfilerConfig(profiler='torch', torch_profiler_dir=str(output),
        torch_profiler_activities=['CPU', 'CUDA'], ignore_frontend=True,
        torch_profiler_with_stack=False, torch_profiler_record_shapes=False,
        torch_profiler_with_memory=False, torch_profiler_use_gzip=True,
        torch_profiler_dump_cuda_time_total=False)
    profile = TorchProfilerWrapper(config, 'capability-' + str(device), device, ['CPU', 'CUDA'])
    print(f'GPU {device}: starting native capture', flush=True)
    profile.start()
    for _ in range(12):
        graph.replay()
        torch.cuda.synchronize()
        profile.step()
    profile.stop()
    print(f'GPU {device}: stopped; validating complete export', flush=True)
    assert float(y[0, 0]) == 512
    traces = list(output.glob('*.pt.trace.json.gz'))
    assert len(traces) == 1, traces
    with gzip.open(traces[0], 'rt') as stream:
        events = json.load(stream)['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('dur', 0) > 0]
    devices = sorted({int(e['args']['device']) for e in kernels})
    assert devices == [device] and len(kernels) >= 12, (device, devices, len(kernels))
    sources = [p for p in (Path(vllm.__file__).parent / 'entrypoints').rglob('*.py')
               if '/start_profile' in p.read_text() and '/stop_profile' in p.read_text()]
    assert sources, 'Profiler API route source not found'
    facts = dict(scope='Synthetic native Torch graph capture/export; not model timing',
        device=device, device_name=torch.cuda.get_device_name(device),
        torch=torch.__version__, hip=torch.version.hip,
        config=dataclasses.asdict(config), kernels=len(kernels), devices=devices,
        trace=traces[0].name, sha256=hashlib.sha256(traces[0].read_bytes()).hexdigest(),
        endpoint_sources=[str(p) for p in sources])
    (output / 'result.json').write_text(json.dumps(facts, indent=2) + '\n')
    print(json.dumps(facts), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if args.device is not None:
        worker(args.device, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    # Separate worker processes avoid sharing profiler state between devices.
    workers = []
    try:
        for device in range(8):
            log = (args.output / f'worker-{device}.log').open('w')
            proc = subprocess.Popen([sys.executable, __file__, '--device', str(device),
                '--output', str(args.output / f'gpu-{device}')], stdout=log, stderr=subprocess.STDOUT)
            log.close()
            workers.append(proc)
        deadline = time.monotonic() + 300
        codes = [p.wait(timeout=max(1, deadline - time.monotonic())) for p in workers]
        assert codes == [0] * 8, codes
        facts = [json.loads((args.output / f'gpu-{d}/result.json').read_text()) for d in range(8)]
        (args.output / 'validation.json').write_text(json.dumps(dict(
            passed=True, workers=facts, limitation='Synthetic capability, no Kimi execution'), indent=2) + '\n')
        print('Eight complete native worker traces validated', flush=True)
    finally:
        for proc in workers:
            if proc.poll() is None:
                proc.terminate()
        for proc in workers:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == '__main__':
    main()
