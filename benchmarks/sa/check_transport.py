"""Check image execution directly on the runner or in an explicit reservation."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import shutil


def saved_evidence_mounts(task, root=Path('/share_nfs/ajith/benchmarks')):
    """Mount only explicitly requested prior-run evidence directories, read-only."""
    mounts = []
    for flag in ('--compiled', '--approved-cache'):
        values = []
        for index, token in enumerate(task):
            if token == flag:
                if index + 1 == len(task):
                    raise ValueError(f'{flag} requires a saved-evidence directory')
                values.append(task[index + 1])
            elif token.startswith(flag + '='):
                values.append(token[len(flag) + 1:])
        if not values:
            continue
        if len(values) != 1:
            raise ValueError(f'{flag} must be specified once')
        target = Path(values[0])
        relative = target.relative_to('/saved-benchmarks')
        if (len(relative.parts) < 2 or '..' in relative.parts
                or not re.fullmatch(r'[0-9]+-[1-9][0-9]*', relative.parts[0])
                or any(character in str(target) for character in ',:')):
            raise ValueError(f'{flag} must name a subdirectory of /saved-benchmarks/<run>-<attempt>/')
        source = root / relative
        resolved = source.resolve()
        if (not source.is_dir() or not resolved.is_relative_to(root.resolve())
                or any(character in str(resolved) for character in ',:')):
            raise ValueError(f'Saved evidence must be an existing directory under {root}: {source}')
        mount = f',{resolved}:{target}:ro'
        if mount not in mounts:
            mounts.append(mount)
    return ''.join(mounts)


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
if config.get('expected_runner') and os.environ.get('RUNNER_NAME') != config['expected_runner']:
    raise RuntimeError('Selected runner does not match expected_runner')
if config.get('require_idle_gpus'):
    idle = subprocess.check_output(['amd-smi', 'process', '--json'], text=True)
    (a.output / 'gpu-idle-check.json').write_text(idle)
    rows = json.loads(idle)
    if (not isinstance(rows, list) or len(rows) != 8
            or {row.get('gpu') for row in rows} != set(range(8))
            or any(row.get('process_list') != [{'process_info': 'No running processes detected'}]
                   for row in rows)):
        raise RuntimeError('GPU activity is present or unknown; see gpu-idle-check.json')
image = config['environment']['IMAGE']
# A cold multi-GB image pull must not consume the diagnostic's execution budget.
inspected = subprocess.run(['docker', 'image', 'inspect', image], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True)
if inspected.returncode:
    with (a.output / 'image-pull.log').open('w') as log:
        subprocess.run(['docker', 'pull', image], stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=1800)
    inspected = subprocess.run(['docker', 'image', 'inspect', image], check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
(a.output / 'image-inspect.json').write_text(inspected.stdout)
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
checkpoint_mount = ''
if '--checkpoint' in task:
    checkpoint = Path(task[task.index('--checkpoint') + 1])
    relative = checkpoint.relative_to('/models')
    model_root = Path(config.get('local_model_root', '/share_nfs/models'))
    source = model_root / relative
    if '..' in relative.parts or not source.is_dir() or not source.resolve().is_relative_to(model_root.resolve()):
        raise ValueError(f'Checkpoint must be an existing directory under {model_root}: {source}')
    checkpoint_mount = f',{source}:{checkpoint}:ro'
evidence_mounts = saved_evidence_mounts(task)
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
                       + f',{Path(__file__).resolve().parent}:/sa-tools:ro' + checkpoint_mount + evidence_mounts, *task],
}
if a.command_json or not a.allocation:
    commands = {'native_adapter': commands['native_adapter']}
os.environ.update(config['environment'])
if a.allocation:
    real_srun = shutil.which('srun')
    if not real_srun:
        raise RuntimeError('Allocation transport requires srun')
    os.environ['SA_REAL_SRUN'] = real_srun
os.environ.update(SA_OUTPUT=str(a.output.resolve()),
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
        partial = error.stdout or ''
        if isinstance(partial, bytes):
            partial = partial.decode(errors='replace')
        results['checks'][name] = {'error': str(error), 'output': partial}
    print(json.dumps(results['checks'][name]), flush=True)
owned = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'label=sa.run=' + a.output.name], text=True).split()
if owned:
    subprocess.run(['docker', 'rm', '-f', *owned], check=True)
(a.output / 'transport-check.json').write_text(json.dumps(results, indent=2) + '\n')
raise SystemExit(0 if results['checks']['native_adapter'].get('returncode') == 0 else 1)
