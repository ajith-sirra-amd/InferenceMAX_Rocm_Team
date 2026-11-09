"""Map SRT container steps to Docker inside the user's real Spur reservation."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import uuid


def split_args(args):
    scheduler, container, task = [], {}, []
    output = None
    flags = {'--overlap', '--quiet', '--exclusive', '--oversubscribe', '--pty',
             '--no-container-entrypoint', '--no-container-mount-home',
             '--container-writable', '--container-remap-root'}
    i = 0
    while i < len(args):
        word = args[i]
        if not word.startswith('-'):
            task = args[i:]
            break
        key, sep, value = word.partition('=')
        if not sep and key not in flags:
            i += 1
            value = args[i]
        if 'container-' in key:
            container[key] = value
        elif key in ('--output', '-o'):
            output = value
        elif key == '--open-mode':
            if value != 'append':
                raise ValueError('Only append logs are supported')
        elif key == '--export':
            values = value.split(',')
            if values.pop(0) != 'ALL':
                raise ValueError('Only inherited task environments are supported')
            os.environ.update(item.split('=', 1) for item in values)
        elif key not in ('--quiet', '--oversubscribe'):
            scheduler += [word] if sep or key in flags else [key, value]
        i += 1
    if '--overlap' not in scheduler:
        scheduler.append('--overlap')
    return scheduler, container, task, output


def main(args):
    scheduler, container, task, output = split_args(args)
    if output:
        output = output.replace('%N', os.uname().nodename)
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)
    image = container.get('--container-image')
    if not image:
        os.execv(os.environ['SA_REAL_SRUN'], ['srun', *scheduler, *task])
    image = image.removeprefix('docker://').replace('#', '/', 1).replace(':sha256:', '@sha256:', 1)
    root = Path(os.environ['SA_OUTPUT'])
    name = 'sa-step-' + root.name + '-' + uuid.uuid4().hex[:8]
    work = root / 'work' / 'containers' / name
    for part in ('home', 'cache', 'tmp'):
        (work / part).mkdir(parents=True, exist_ok=True)
    docker = ['docker', 'run', '--rm', '--init', '--read-only', '--user', '0:0', '--name', name,
              '--label', 'sa.run=' + root.name, '--network', 'host', '--ipc', 'host',
              '--device', '/dev/kfd', '--device', '/dev/dri', '--cap-add', 'SYS_PTRACE',
              '--security-opt', 'seccomp=unconfined', '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
              '--mount', f'type=bind,src={work},dst=/work',
              '--mount', f'type=bind,src={work / "home"},dst=/root',
              '--tmpfs', '/work/tmp:rw,exec,size=64g']
    for mount in container.get('--container-mounts', '').split(','):
        if mount:
            source, target, *options = mount.split(':')
            # Device nodes are exposed by Docker's device mechanism above.
            if source in ('/dev/kfd', '/dev/dri'):
                continue
            if target == '/model':
                options = ['ro']
            docker += ['--mount', f'type=bind,src={source},dst={target}' + (',readonly' if 'ro' in options else '')]
    if container.get('--container-workdir'):
        docker += ['--workdir', container['--container-workdir']]
    for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'SLURM_JOB_ID', 'SLURM_NODELIST', 'SLURM_JOB_NODELIST', 'ROCR_VISIBLE_DEVICES'):
        if key in os.environ:
            docker += ['--env', key]
    for key, value in {'TMPDIR': '/work/tmp', 'XDG_CACHE_HOME': '/work/cache',
                       'CCACHE_DIR': '/work/cache/ccache', 'SCCACHE_DIR': '/work/cache/sccache',
                       'PIP_CACHE_DIR': '/work/cache/pip', 'PYTHONDONTWRITEBYTECODE': '1',
                       'UV_PYTHON_INSTALL_DIR': '/work/cache/python', 'UV_LINK_MODE': 'copy'}.items():
        docker += ['--env', key + '=' + value]
    docker += ['--entrypoint', task[0], image, *task[1:]]
    record = {'container': name, 'image': image, 'allocation': os.environ['SLURM_JOB_ID'],
              'mounts': container.get('--container-mounts'), 'transport': 'srun-docker',
              'native_task_sha256': hashlib.sha256(json.dumps(task).encode()).hexdigest()}
    with (root / 'container-transport.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')
    proc = subprocess.Popen([os.environ['SA_REAL_SRUN'], *scheduler, *docker])
    def stop(signum, frame):
        subprocess.run(['docker', 'stop', '--time', '5', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        return proc.wait()
    finally:
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
