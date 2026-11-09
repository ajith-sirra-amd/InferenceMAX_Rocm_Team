"""Run the native server and export a bounded rocprof capture before shutdown."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time


def register_library():
    roots = [Path(shutil.which('rocprofv3')).resolve().parent.parent / 'lib', Path('/opt/rocm/lib')]
    roots += [Path(p) / '_rocm_sdk_devel/lib' for p in sys.path if p]
    for root in roots:
        library = root / 'librocprofiler-register.so.0'
        if library.is_file():
            return str(library)
    raise RuntimeError('Cannot locate the installed rocprofiler registration library')


def gpu_children(parent):
    parents = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                parents[int(entry.name)] = int((entry / 'stat').read_text().rsplit(')', 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                pass
    family = {parent}
    while True:
        expanded = family | {pid for pid, ppid in parents.items() if ppid in family}
        if expanded == family:
            break
        family = expanded
    workers = []
    for pid in sorted(family):
        try:
            if any(os.readlink(fd) == '/dev/kfd' for fd in (Path('/proc') / str(pid) / 'fd').iterdir()):
                workers.append(pid)
        except OSError:
            pass
    return workers


def worker_state(parent):
    workers = {}
    for pid in gpu_children(parent):
        try:
            process = Path('/proc', str(pid))
            name = (process / 'comm').read_text().strip()
            if '::Worker' in name:
                workers[pid] = {'name': name, 'attach_ready': any(
                    p.read_text().strip() == 'rocp-bg-attach'
                    for p in (process / 'task').glob('*/comm'))}
        except OSError:
            pass
    return workers


def main(command):
    output = Path(os.environ['SA_PROFILE_OUTPUT'])
    delay = int(os.environ['SA_PROFILE_DELAY_SECONDS'])
    seconds = int(os.environ['SA_PROFILE_SECONDS'])
    # A fresh interpreter initializes profiler registration in each vLLM worker.
    # This is a profiling-only startup override; the replication stays unchanged.
    environment = dict(os.environ, ROCP_TOOL_ATTACH='1', VLLM_WORKER_MULTIPROC_METHOD='spawn')
    environment['LD_PRELOAD'] = ':'.join(filter(None, (register_library(), environment.get('LD_PRELOAD'))))
    result = {'status': 'pending', 'server_argv': command, 'worker_pids': [],
              'worker_start_method': 'spawn'}
    finished = threading.Event()
    launched = time.monotonic()
    server = subprocess.Popen(command, env=environment)

    def forward(signum, _frame):
        if server.poll() is None:
            server.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    def capture():
        jobs, handles = [], []
        try:
            expected = int(os.environ.get('GPU_COUNT', '8'))
            deadline = launched + min(delay, 300)
            while True:
                state = worker_state(server.pid)
                result['preflight_workers'] = state
                if len(state) == expected and all(w['attach_ready'] for w in state.values()):
                    result['preflight_utc_ns'] = time.time_ns()
                    (output / 'attachment-preflight.json').write_text(json.dumps(result, indent=2) + '\n')
                    break
                if time.monotonic() >= deadline or finished.wait(1):
                    raise RuntimeError(f'Expected {expected} attachment-ready vLLM workers before capture; found {state}')
            if finished.wait(max(0, launched + delay - time.monotonic())):
                raise RuntimeError('Server exited before capture')
            state = worker_state(server.pid)
            workers = list(state)
            result['worker_pids'] = workers
            if len(workers) != expected or not all(w['attach_ready'] for w in state.values()):
                raise RuntimeError('Attachment-ready worker coverage changed before capture')
            result['started_utc_ns'] = time.time_ns()
            for pid in workers:
                command = ['rocprofv3', '--attach', str(pid), '--attach-children', 'false',
                           '--attach-duration-msec', str(seconds * 1000), '--attach-sync-output', 'true',
                           '--kernel-trace', '--hip-trace', '--memory-copy-trace', '--marker-trace',
                           '--stats', '-f', 'csv', 'json', '-d', str(output / 'traces')]
                handle = (output / f'attach-{pid}.log').open('w')
                handles.append(handle)
                jobs.append(subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT))
            result['returncodes'] = [job.wait(timeout=seconds + 180) for job in jobs]
            if any(result['returncodes']):
                raise RuntimeError('rocprof attachment or synchronous export failed')
            if server.poll() is not None or not all(Path('/proc', str(pid)).exists() for pid in workers):
                raise RuntimeError('Server or GPU worker exited before trace export completed')
            result['status'] = 'passed'
            result['exported_while_workers_alive'] = True
        except Exception as error:
            result.update(status='failed', error=str(error))
            # Abort this invalid profiling attempt through native server shutdown.
            # Do not wait for a full workload when capture cannot succeed.
            if server.poll() is None:
                try:
                    server.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
        finally:
            for job in jobs:
                if job.poll() is None:
                    job.kill()
                    job.wait()
            for handle in handles:
                handle.close()
            result['finished_utc_ns'] = time.time_ns()
            (output / 'attachment-status.json').write_text(json.dumps(result, indent=2) + '\n')
            print('Profile export: ' + json.dumps(result), flush=True)

    thread = threading.Thread(target=capture, name='profile-export')
    thread.start()
    code = server.wait()
    finished.set()
    thread.join()
    return code or (1 if result['status'] == 'failed' else 0)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
