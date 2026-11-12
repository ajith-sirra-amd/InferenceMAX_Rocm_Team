"""Audit the explicit agentic-script command against the saved SA reference."""
import hashlib
import json
from pathlib import Path
import os
import sys


def audit(config, output, model, port, source_root, actual_argv):
    argv = ['/opt/runtime-venv/bin/vllm', 'serve', model, '--host', '0.0.0.0', '--port', port]
    for key, value in config['server_arguments'].items():
        if isinstance(value, bool):
            if value:
                argv.append('--' + key)
        else:
            argv.extend(['--' + key, str(value)])
    environment = {key: os.environ.get(key) for key in config['server_environment']}
    if argv != actual_argv or environment != config['server_environment']:
        raise ValueError('Actual server command/environment differs from the saved SA reference')
    sources = {}
    for relative in ('vllm/models/kimi_k3/nvidia/dspark_mla.py',
                     'vllm/models/kimi_k3/amd/linear.py',
                     'vllm/models/kimi_k3/amd/ops/attn_res.py'):
        sources[relative] = hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
    optimization = os.environ['KIMI_FIXED_OPTIMIZATION']
    if optimization == 'attnres':
        manifest = json.loads((output / 'optimization.json').read_text())
        if os.environ.get('KIMI_ATTNRES_PRECOMPUTE_WEIGHTS') != '1':
            raise ValueError('AttnRes candidate enable flag is missing')
        for entry in manifest['files']:
            if sources[entry['path']] != entry['candidate_sha256']:
                raise ValueError('Candidate overlay hash mismatch: ' + entry['path'])
    elif optimization != 'baseline' or os.environ.get('KIMI_ATTNRES_PRECOMPUTE_WEIGHTS', '0') != '0':
        raise ValueError('Unexpected optimization mode or flag')
    (output / 'effective-server.json').write_text(json.dumps(dict(
        argv=actual_argv, environment=environment, source_sha256=sources,
        config=config, optimization=optimization, sa_config_match=True), indent=2) + '\n')


if __name__ == '__main__':
    config_path, output, model, port = sys.argv[1:5]
    audit(json.loads(Path(config_path).read_text()), Path(output), model, port,
          Path('/opt/runtime-venv/lib/python3.12/site-packages'), sys.argv[5:])
