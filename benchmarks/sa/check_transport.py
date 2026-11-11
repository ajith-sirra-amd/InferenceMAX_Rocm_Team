"""Check image execution directly on the runner or in an explicit reservation."""
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
p.add_argument('--command-json', default=os.environ.get('SA_CHECK_COMMAND', ''))
a = p.parse_args()
host = subprocess.check_output(['hostname', '-s'], text=True).strip()
allocation = None
if a.allocation:
    owner = subprocess.check_output(['id', '-un'], text=True).strip()
    allocation = subprocess.check_output(['squeue', '-h', '-j', a.allocation, '-o', '%u|%T|%N'], text=True).strip()
    if allocation != f'{owner}|RUNNING|{host}':
        raise RuntimeError('Reservation must be owned by this user on this host: ' + allocation)
a.output.mkdir(parents=True, exist_ok=True)
marker = a.output / 'transport-marker.txt'
marker.write_text(a.output.name)
config = json.loads(a.config.read_text())
image = config['environment']['IMAGE']
task = ['python3', '-c', '''import json, shutil
from pathlib import Path
marker = Path('/logs/transport-marker.txt')
facts = {'vllm': shutil.which('vllm'), 'marker': marker.read_text() if marker.exists() else None}
print(json.dumps(facts), flush=True)
assert facts['vllm'] and facts['marker'], 'Requested image and mount are not active'
''']
if a.command_json:
    task = json.loads(a.command_json)
    if not isinstance(task, list) or not task or not all(isinstance(v, str) for v in task):
        raise ValueError('Diagnostic command must be a nonempty JSON string array')
base = ['srun', '--nodes', '1', '--ntasks', '1', '--nodelist', host]
if a.allocation:
    base += ['--jobid', a.allocation, '--overlap']
commands = {
    'spur_container': [*base, '--container-image', image, '--container-remap-root',
                       '--container-mounts', f'{a.output.resolve()}:/logs:ro', *task],
    'spur_docker': [*base, 'docker', 'run', '--rm', '--read-only',
                    '--name', 'sa-transport-' + a.output.name,
                    '--network', 'host', '--mount', f'type=bind,src={a.output.resolve()},dst=/logs,readonly',
                    '--entrypoint', task[0], image, *task[1:]],
    'native_adapter': [str(Path(__file__).parent / 'spur/srun'), *base[1:],
                       '--container-image', image, '--container-remap-root', '--container-writable',
                       '--container-mounts', f'{a.output.resolve()}:/logs' + ('' if a.command_json else ':ro')
                       + f',{Path(__file__).resolve().parent}:/sa-tools:ro', *task],
}
if a.command_json or not a.allocation:
    commands = {'native_adapter': commands['native_adapter']}
os.environ.update(config['environment'])
os.environ.update(SA_REAL_SRUN=shutil.which('srun'), SA_OUTPUT=str(a.output.resolve()),
                  SA_LOCAL_EXECUTION='0' if a.allocation else '1',
                  SLURM_JOB_ID=a.allocation or 'local-' + a.output.name,
                  SA_PASSTHROUGH_ENV=json.dumps(sorted(config['environment'])))
results = {'allocation': allocation, 'image': image, 'checks': {}}
for name, command in commands.items():
    print('Checking ' + name, flush=True)
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
        results['checks'][name] = {'returncode': result.returncode, 'output': result.stdout}
    except subprocess.TimeoutExpired as error:
        results['checks'][name] = {'error': str(error)}
    print(json.dumps(results['checks'][name]), flush=True)
owned = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'label=sa.run=' + a.output.name], text=True).split()
if owned:
    subprocess.run(['docker', 'rm', '-f', *owned], check=True)
(a.output / 'transport-check.json').write_text(json.dumps(results, indent=2) + '\n')
raise SystemExit(0 if results['checks']['native_adapter'].get('returncode') == 0 else 1)
