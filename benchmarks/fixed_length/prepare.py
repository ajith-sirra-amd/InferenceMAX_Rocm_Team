"""Serialize the frozen SA environment/argv for the existing agentic launcher."""
import hashlib
import json
from pathlib import Path
import shlex
import sys


def prepare(config, output, model, port, source_root):
    argv = ['/opt/runtime-venv/bin/vllm', 'serve', model, '--host', '0.0.0.0', '--port', port]
    for key, value in config['server_arguments'].items():
        if isinstance(value, bool):
            if value:
                argv.append('--' + key)
        else:
            argv.extend(['--' + key, str(value)])
    sources = {}
    for relative in ('vllm/models/kimi_k3/nvidia/dspark_mla.py',
                     'vllm/models/kimi_k3/amd/linear.py',
                     'vllm/models/kimi_k3/amd/ops/attn_res.py'):
        sources[relative] = hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
    (output / 'effective-server.json').write_text(json.dumps(dict(
        argv=argv, environment=config['server_environment'], source_sha256=sources,
        config=config, optimization='none'), indent=2) + '\n')
    shell = [f'export {key}={shlex.quote(value)}'
             for key, value in config['server_environment'].items()]
    shell.append('VLLM_CMD=(' + shlex.join(argv) + ')')
    (output / 'server-config.sh').write_text('\n'.join(shell) + '\n')


if __name__ == '__main__':
    config_path, output, model, port = sys.argv[1:]
    prepare(json.loads(Path(config_path).read_text()), Path(output), model, port,
            Path('/opt/runtime-venv/lib/python3.12/site-packages'))
