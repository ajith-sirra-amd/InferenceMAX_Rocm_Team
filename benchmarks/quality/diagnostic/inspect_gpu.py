"""Read GPU memory and process/container ownership without starting a model."""
import datetime
import json
from pathlib import Path
import socket
import subprocess
import sys

out = Path(sys.argv[1]).resolve()
if not out.is_relative_to(Path('/share_nfs/ajith/benchmarks')):
    raise SystemExit('Use shared benchmark storage')
out.mkdir(parents=True, exist_ok=False)

def read(path):
    try:
        return path.read_text().strip()
    except OSError as error:
        return {'error': str(error)}

def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return dict(argv=argv, returncode=result.returncode,
                    stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(argv=argv, error=str(error))

gpus = []
for path in sorted(Path('/sys/class/drm').glob('card*/device/mem_info_vram_total')):
    device = path.parent
    gpus.append(dict(device=str(device), total_bytes=read(path),
                     used_bytes=read(device/'mem_info_vram_used')))
processes = []
for path in sorted(Path('/sys/class/kfd/kfd/proc').glob('[0-9]*')):
    status = read(Path('/proc')/path.name/'status')
    if isinstance(status, str):
        status = [line for line in status.splitlines()
                  if line.startswith(('Name:', 'Pid:', 'PPid:', 'Uid:', 'State:'))]
    processes.append(dict(pid=int(path.name), status=status,
                          cgroup=read(Path('/proc')/path.name/'cgroup'),
                          vram={p.name: read(p) for p in path.glob('vram_*')}))
listing = command(['docker', 'ps', '--no-trunc', '--format', '{{.ID}}'])
containers = []
if listing.get('returncode') == 0:
    for identity in listing['stdout'].splitlines():
        containers.append(command(['docker', 'inspect', '--format',
            '{{json .Id}} {{json .Name}} {{json .State.Pid}} {{json .Config.Image}} {{json .Created}}', identity]))
        containers.append(command(['docker', 'top', identity, '-eo', 'pid,ppid,uid,comm']))
report = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              hostname=socket.gethostname(), mode='read-only GPU ownership inspection',
              gpu_memory=gpus, gpu_processes=processes,
              docker_listing=listing, container_processes=containers,
              limits='No allocations, process signals, container stops or runtime changes. Process command lines and environments are not collected.')
(out/'gpu-ownership.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
