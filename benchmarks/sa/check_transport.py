"""Check image and bind-mount execution inside an existing single-node reservation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--allocation', required=True)
a = p.parse_args()
host = subprocess.check_output(['hostname', '-s'], text=True).strip()
owner = subprocess.check_output(['id', '-un'], text=True).strip()
allocation = subprocess.check_output(['squeue', '-h', '-j', a.allocation, '-o', '%u|%T|%N'], text=True).strip()
if allocation != f'{owner}|RUNNING|{host}':
    raise RuntimeError('Reservation must be owned by this user on this host: ' + allocation)
a.output.mkdir(parents=True, exist_ok=True)
marker = a.output / 'transport-marker.txt'
marker.write_text(a.output.name)
image = json.loads(a.config.read_text())['environment']['IMAGE']
task = ['python3', '-c', '''import json, shutil
from pathlib import Path
marker = Path('/logs/transport-marker.txt')
facts = {'vllm': shutil.which('vllm'), 'marker': marker.read_text() if marker.exists() else None}
print(json.dumps(facts), flush=True)
assert facts['vllm'] and facts['marker'], 'Requested image and mount are not active'
''']
base = ['srun', '--jobid', a.allocation, '--overlap', '--nodes', '1', '--ntasks', '1', '--nodelist', host]
commands = {
    'spur_container': [*base, '--container-image', image, '--container-remap-root',
                       '--container-mounts', f'{a.output.resolve()}:/logs:ro', *task],
    'spur_docker': [*base, 'docker', 'run', '--rm', '--read-only',
                    '--name', 'sa-transport-' + a.output.name,
                    '--network', 'host', '--mount', f'type=bind,src={a.output.resolve()},dst=/logs,readonly',
                    '--entrypoint', task[0], image, *task[1:]],
    'native_adapter': [str(Path(__file__).parent / 'spur/srun'), *base[1:],
                       '--container-image', image, '--container-remap-root',
                       '--container-mounts', f'{a.output.resolve()}:/logs:ro', *task],
}
os.environ.update(SA_REAL_SRUN=shutil.which('srun'), SA_OUTPUT=str(a.output.resolve()),
                  SA_CONTAINER_TRANSPORT='docker', SLURM_JOB_ID=a.allocation)
results = {'allocation': allocation, 'image': image, 'checks': {}}
for name, command in commands.items():
    print('Checking ' + name, flush=True)
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        results['checks'][name] = {'returncode': result.returncode, 'output': result.stdout}
    except subprocess.TimeoutExpired as error:
        results['checks'][name] = {'error': str(error)}
    print(json.dumps(results['checks'][name]), flush=True)
(a.output / 'transport-check.json').write_text(json.dumps(results, indent=2) + '\n')
raise SystemExit(0 if results['checks']['native_adapter'].get('returncode') == 0 else 1)
