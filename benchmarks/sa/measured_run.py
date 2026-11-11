"""Supervise the unchanged native workload and a separate profile controller."""
from pathlib import Path
import subprocess
import sys
import time


def run_measured(command, logs, port, profile):
    output = logs / 'profiles/torch'
    output.mkdir(parents=True, exist_ok=False)
    workload = subprocess.Popen(command)
    controller = None
    try:
        with (output / 'controller.log').open('x') as stream:
            controller = subprocess.Popen([sys.executable, str(Path(__file__).parent / 'profile/steady.py'),
                '--output', str(output), '--benchmark-log', str(logs / 'agentic/benchmark.log'),
                '--base-url', f'http://127.0.0.1:{port}', '--workload-pid', str(workload.pid),
                '--settle-seconds', str(profile['delay_seconds']), '--capture-seconds', str(profile['seconds']),
                '--timeout-seconds', str(profile['phase_timeout_seconds']),
                '--poll-seconds', str(profile['poll_seconds']),
                '--export-timeout-seconds', str(profile['export_timeout_seconds'])], stdout=stream, stderr=subprocess.STDOUT)
            while workload.poll() is None:
                code = controller.poll()
                if code is not None and code != 0:
                    raise RuntimeError('Profile controller failed; see profiles/torch/controller.log')
                time.sleep(2)
            if workload.returncode:
                raise subprocess.CalledProcessError(workload.returncode, command)
            if controller.wait(timeout=profile['export_timeout_seconds']) != 0:
                raise RuntimeError('Profile controller failed before completing export')
    finally:
        # Let /stop_profile finish exporting before terminating the workload.
        if controller is not None and controller.poll() is None:
            controller.terminate()
            controller.wait(timeout=profile['export_timeout_seconds'] + 30)
        if workload.poll() is None:
            workload.terminate()
            workload.wait(timeout=1200)
