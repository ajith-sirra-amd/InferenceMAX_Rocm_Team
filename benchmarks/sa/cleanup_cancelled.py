"""Finish one cancelled native run's container cleanup and retain small evidence."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import urllib.request

from cleanup import owned


def locate_run(workspace, run, attempt):
    if any(type(value) is not int or value < 1 for value in (run, attempt)):
        raise ValueError('Run and attempt must be positive integers')
    workspace = workspace.resolve()
    source = workspace / f'sa-benchmark-{run}' / 'run-data' / f'{run}-{attempt}'
    if not source.is_dir() or not source.resolve().is_relative_to(workspace):
        raise ValueError('Cancelled run evidence is missing or outside the workspace')
    return source.resolve()


def select_owned(containers, source, identity):
    selected = []
    for container in containers:
        if owned(container, source.parent) != identity:
            raise ValueError('Container label and saved transport record do not agree')
        selected.append(container['Id'])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    a = parser.parse_args()
    config = json.loads(a.config.read_text())
    if os.environ.get('RUNNER_NAME') != config['expected_runner']:
        raise ValueError('Cleanup must run on the selected model runner')
    if str(a.run) == os.environ['GITHUB_RUN_ID']:
        raise ValueError('Refusing to clean the executing workflow')
    source = locate_run(Path(os.environ['GITHUB_WORKSPACE']), a.run, a.attempt)
    context = json.loads((source / 'execution-context.json').read_text())
    if context['host'] != socket.gethostname().split('.')[0]:
        raise ValueError('Saved model host differs from cleanup host')
    url = (os.environ['GITHUB_API_URL'] + '/repos/' + os.environ['GITHUB_REPOSITORY']
           + f'/actions/runs/{a.run}/attempts/{a.attempt}')
    request = urllib.request.Request(url, headers={
        'Authorization': 'Bearer ' + os.environ['GH_TOKEN']})
    with urllib.request.urlopen(request, timeout=30) as response:
        status = json.load(response)
    if status['status'] != 'completed' or status['conclusion'] != 'cancelled':
        raise ValueError('Target workflow attempt must be completed and cancelled')

    identity = f'{a.run}-{a.attempt}'
    output = a.output / 'native' / ('cancelled-' + identity)
    output.mkdir(parents=True, exist_ok=False)
    ids = subprocess.check_output(['docker', 'ps', '-aq', '--filter',
                                   'label=sa.run=' + identity], text=True).split()
    containers = json.loads(subprocess.check_output(
        ['docker', 'inspect', *ids], text=True)) if ids else []
    targets = select_owned(containers, source, identity)
    report = dict(target_run=a.run, attempt=a.attempt, source=str(source),
                  status='cleaning', containers=[dict(id=c['Id'], name=c['Name'],
                  state=c['State']['Status']) for c in containers])
    report_path = output / 'cleanup-verification.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    # The user cancelled this pre-readiness attempt. The interrupted workflow's
    # 20-minute export grace must not keep these recorded containers alive.
    for container in targets:
        result = subprocess.run(['docker', 'rm', '--force', container], text=True,
                                capture_output=True, timeout=60)
        report.setdefault('removal', []).append(dict(id=container,
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    remaining = subprocess.check_output(['docker', 'ps', '-aq', '--filter',
        'label=sa.run=' + identity], text=True).split()
    report['remaining_target_containers'] = remaining
    report['status'] = 'failed' if remaining else 'stopped'
    report['copied_files'], report['omitted_files'] = [], []
    paths = [*source.iterdir(), *(source / 'native').rglob('*'),
             *(source / 'rendered').rglob('*')]
    for path in paths:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(source):
            continue
        relative = path.relative_to(source)
        if path.stat().st_size > 64 * 1024 * 1024:
            report['omitted_files'].append(str(relative))
            continue
        destination = output / 'original' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        report['copied_files'].append(str(relative))
    for name, command in (
        ('gpu-processes-after.json', ['amd-smi', 'process', '--json']),
        ('containers-after.txt', ['docker', 'ps', '--format', '{{.ID}} {{.Names}} {{.Status}}']),
    ):
        with (output / name).open('w') as stream:
            check = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=30)
        report.setdefault('inventory_returncodes', {})[name] = check.returncode
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('copied_files', 'omitted_files')}, indent=2), flush=True)
    if remaining:
        raise RuntimeError('Cancelled run still has recorded containers')


if __name__ == '__main__':
    main()
