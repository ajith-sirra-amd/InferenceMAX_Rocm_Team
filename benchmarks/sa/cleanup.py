"""Stop only native containers recorded for this run or confirmed completed runs."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request


def owned(container, root):
    identity = container.get('Config', {}).get('Labels', {}).get('sa.run', '')
    if not re.fullmatch(r'[0-9]+-[0-9]+', identity):
        return None
    record = root / identity / 'container-transport.jsonl'
    if not record.is_file() or not record.resolve().is_relative_to(root.resolve()):
        return None
    names = {json.loads(line)['container'] for line in record.read_text().splitlines()}
    return identity if container['Name'].removeprefix('/') in names else None


def completed(identity):
    run, attempt = identity.split('-')
    url = (os.environ['GITHUB_API_URL'] + '/repos/' + os.environ['GITHUB_REPOSITORY']
           + '/actions/runs/' + run + '/attempts/' + attempt)
    request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + os.environ['GH_TOKEN']})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)['status'] == 'completed'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--completed', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    ids = subprocess.check_output(['docker', 'ps', '-q', '--filter', 'label=sa.run'], text=True).split()
    containers = json.loads(subprocess.check_output(['docker', 'inspect', *ids], text=True)) if ids else []
    results = []
    path = a.output / ('completed-container-cleanup.json' if a.completed else 'container-cleanup.json')
    for container in containers:
        identity = owned(container, a.output.parent)
        if not identity:
            continue
        if a.completed:
            if identity == a.output.name or not completed(identity):
                continue
        elif identity != a.output.name:
            continue
        row = dict(run=identity, container=container['Name'], id=container['Id'], status='stopping')
        results.append(row)
        path.write_text(json.dumps(results, indent=2) + '\n')
        print('Stopping recorded native container:', identity, container['Name'], flush=True)
        # Preserve the native profile export allowance before any forced stop.
        subprocess.run(['docker', 'stop', '--time', '1200', container['Id']], check=True, timeout=1230)
        row['status'] = 'stopped'
        path.write_text(json.dumps(results, indent=2) + '\n')
    path.write_text(json.dumps(results, indent=2) + '\n')
    for name, command in (
        ('containers-after-cleanup.txt', ['docker', 'ps', '--format', '{{.ID}} {{.Names}}']),
        ('gpu-memory-after-cleanup.json', ['rocm-smi', '--showmeminfo', 'vram', '--json']),
    ):
        with (a.output / name).open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=30)


if __name__ == '__main__':
    main()
