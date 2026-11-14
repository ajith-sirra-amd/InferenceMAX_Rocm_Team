"""Exact-runtime natural screen with native loaded-draft provenance."""
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
import launch
import screen
sys.path.insert(0, '/pr-tools')
import runtime

def main():
    output = Path('/logs')
    validation, _ = screen.assets()
    config = runtime.candidate_recipe(launch.recipe(Path('/fixed-tools/baseline.json'), validation), 67372, os.environ['IPAT_ARM'])
    os.environ.update(config['server_environment'])
    config['server_arguments']['worker-extension-cls'] = 'model_worker.ModelWorkerExtension'
    argv = launch.command(config, '/model', os.environ['PORT'])
    screen.save(output/'recipe.json', config)
    (output/'command.txt').write_text(shlex.join(argv)+'\n')
    result = dict(status='failed', limits='Natural 48-example screen; not full model quality or performance qualification')
    server = None
    def stop(signum, frame):
        raise RuntimeError(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        with (output/'server.log').open('xb') as log:
            server = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            endpoint = 'http://127.0.0.1:'+os.environ['PORT']
            deadline = time.monotonic()+1800
            while time.monotonic()<deadline:
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited before readiness: {server.returncode}')
                try:
                    with urlopen(endpoint+'/health', timeout=5) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError('Readiness exceeded 1800 seconds')
            records = [json.loads((output/f'rank-{rank}.json').read_text()) for rank in range(8)]
            assert [r['rank'] for r in records] == list(range(8))
            result['loaded_ranks'] = 8
            screen.save(output/'observed-verifier.json', launch.observed_verifier((output/'server.log').read_text(errors='replace'), json.loads(config['server_arguments']['speculative-config'])))
            client = ['timeout', '--signal=TERM', '--kill-after=15s', '375s', sys.executable, '/quality/screen.py', '--endpoint', endpoint, '--model', 'moonshotai/Kimi-K3.1-Pro', '--output', '/logs/quality']
            (output/'client-command.txt').write_text(shlex.join(client)+'\n')
            code = subprocess.call(client)
            if code:
                raise RuntimeError(f'Natural screen incomplete: exit {code}')
            result['status'] = 'collected'
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=15)
            except ProcessLookupError:
                pass
        screen.save(output/'model-result.json', result)
    return 0 if result['status']=='collected' else 1

if __name__ == '__main__':
    raise SystemExit(main())
