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


def main(command):
    output = Path(os.environ['SA_PROFILE_OUTPUT'])
    delay = int(os.environ['SA_PROFILE_DELAY_SECONDS'])
    seconds = int(os.environ['SA_PROFILE_SECONDS'])
    environment = dict(os.environ, ROCP_TOOL_ATTACH='1')
    environment['LD_PRELOAD'] = ':'.join(filter(None, (register_library(), environment.get('LD_PRELOAD'))))
    result = {'status': 'pending', 'server_argv': command, 'worker_pids': []}
    finished = threading.Event()
    server = subprocess.Popen(command, env=environment)

    def forward(signum, _frame):
        if server.poll() is None:
            server.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    def capture():
        jobs, handles = [], []
        try:
            if finished.wait(delay):
                raise RuntimeError('Server exited before capture')
            workers = gpu_children(server.pid)
            result['worker_pids'] = workers
            if not workers:
                raise RuntimeError('No GPU processes found in the server process tree')
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
    return code


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
