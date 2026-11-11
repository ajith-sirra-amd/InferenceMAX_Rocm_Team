"""Local storage/scheduler bindings; workload and process lifecycle stay native."""
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import yaml

from infx.srt_slurm.single_node import select_recipe, runtime_arguments
from infx.srt_slurm.synthetic_acceptance import plan_commands
from infx.launch.drivers.srt.submit import WORKLOAD_ENV

p = argparse.ArgumentParser()
for name in ('config', 'output', 'source', 'srt'):
    p.add_argument('--' + name, required=True, type=Path)
p.add_argument('--prepare-only', action='store_true')
a = p.parse_args()
c = json.loads(a.config.read_text())
# This SRT revision fixes the public endpoint in ports.py, without a YAML option.
# Bind that infrastructure endpoint locally; server and client use the same native constant.
if 'local_frontend_port' in c:
    port = c['local_frontend_port']
    if type(port) is not int or not (port == 0 or 1024 <= port <= 65535):
        raise ValueError('local_frontend_port must be 0 (automatic) or an unprivileged TCP port')
    with socket.socket() as endpoint:
        endpoint.bind(('0.0.0.0', port))
        port = endpoint.getsockname()[1]
    print(f'Local public endpoint: {port}', flush=True)
    ports = a.srt / 'src/srtctl/ports.py'
    original = ports.read_text()
    needle = 'FRONTEND_PUBLIC_PORT = 8000'
    if original.count(needle) != 1:
        raise ValueError('Unexpected native frontend port definition')
    revised = original.replace(needle, f'FRONTEND_PUBLIC_PORT = {port}')
    ports.write_text(revised)
    (a.output / 'local-endpoint-binding.json').write_text(json.dumps(dict(
        file=str(ports), source_port=8000, local_port=port,
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        revised_sha256=hashlib.sha256(revised.encode()).hexdigest()), indent=2) + '\n')
recipe = str(a.source / c['recipe'])
selected, bound = select_recipe(recipe, os.environ, cpu_power_source='rapl')
(a.output / 'bound-sa-recipe.yaml').write_text(yaml.safe_dump(bound, sort_keys=False))
recipe_file = a.output / 'recipe.yaml'
recipe_file.write_text(yaml.safe_dump(bound, sort_keys=False))
cluster = yaml.safe_load((a.source / 'configs/runners.yaml').read_text())['clusters']['mi355x-amds']['slurm']['srt-slurm']
mounts = {
    str(a.source): '/infmax-workspace',
    str(a.output / 'work/hf'): '/hf_hub_cache',
    '/share_nfs/models': '/models:ro',
    '/dev/kfd': '/dev/kfd', '/dev/dri': '/dev/dri',
}
for host, target in cluster.get('workspace-mounts', {}).items():
    mounts[str(a.source / host)] = target
settings = dict(cluster['extra'])
profile = c.get('profile')
if profile:
    if c['environment']['FRAMEWORK'] != 'vllm' or profile.get('tool') != 'rocprofv3':
        raise ValueError('The native launch profiling hook currently supports vLLM with rocprofv3')
    delay, seconds = profile['delay_seconds'], profile['seconds']
    mode = profile.get('mode', 'launch')
    if mode not in ('launch', 'attach'):
        raise ValueError('Profile mode must be launch or attach')
    if type(delay) is not int or delay < 0 or type(seconds) is not int or not 1 <= seconds <= 180:
        raise ValueError('Use a calibrated nonnegative delay and a 1–180 second capture')
    minimum = profile.get('minimum_seconds')
    if minimum is not None and (type(minimum) is not int or not 1 <= minimum <= seconds):
        raise ValueError('Profile minimum_seconds must be a positive integer no greater than seconds')
    mounts[str(Path(__file__).parent / 'profile')] = '/sa-profile-bin:ro'
    settings['default_bash_preamble'] = 'export PATH="/sa-profile-bin:$PATH"'
settings.update(cluster='mi355x-local', default_partition='amd-spur', default_account='',
                default_time_limit='08:00:00', gpus_per_node=8,
                network_interface='', srtctl_root=str(a.srt), output_dir=str(a.output / 'native'),
                default_mounts=mounts, use_segment_sbatch_directive=False,
                use_exclusive_sbatch_directive=False,
                model_paths={f"hf:{c['environment']['MODEL']}": '/share_nfs/models/' + c['environment']['MODEL']})
if os.environ.get('SA_LOCAL_EXECUTION') == '1':
    settings['release_nodes_before_postprocess'] = False
(a.srt / 'srtslurm.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))
os.environ['SRTSLURM_CONFIG'] = str(a.srt / 'srtslurm.yaml')
extra = ['--set', 'benchmark.stream_output=true', '--set', 'srun_options.container-workdir="/infmax-workspace"',
         '--set', 'srun_options.container-remap-root=""', '--set', 'srun_options.container-writable=""']
if profile:
    for key, value in {'SA_PROFILE_DELAY_SECONDS': str(delay), 'SA_PROFILE_SECONDS': str(seconds), 'SA_PROFILE_MODE': mode,
                       'SA_PROFILE_OUTPUT': '/logs/profiles/rocprof'}.items():
        extra += ['--set', 'roles.agg.env.' + key + '=' + json.dumps(value)]
    if mode == 'launch':
        # Give native shutdown time to export profiler buffers. Workload code is unchanged.
        stage = a.srt / 'src/srtctl/cli/mixins/worker_stage.py'
        original = stage.read_text()
        needle = 'WORKER_TERMINATE_TIMEOUT_SECONDS = 30.0'
        if original.count(needle) != 1:
            raise ValueError('Unexpected native worker shutdown definition')
        revised = original.replace(needle, 'WORKER_TERMINATE_TIMEOUT_SECONDS = 1200.0')
        stage.write_text(revised)
        extra += ['--set', 'roles.agg.env.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS="900"']
        (a.output / 'profile-shutdown-overlay.json').write_text(json.dumps(dict(
            file=str(stage), original_sha256=hashlib.sha256(original.encode()).hexdigest(),
            revised_sha256=hashlib.sha256(revised.encode()).hexdigest(),
            native_worker_timeout_seconds=1200, vllm_worker_timeout_seconds=900), indent=2) + '\n')
    if 'warmup_requests_per_lane' in profile:
        warmup = profile['warmup_requests_per_lane']
        if type(warmup) is not int or warmup < 1 or c['environment']['AIPERF_EXPERIMENTAL_FAST'] != '0':
            raise ValueError('Diagnostic warmup needs a positive count and explicit duration')
        extra += ['--set', 'benchmark.env.SA_PROFILE_WARMUP_REQUESTS_PER_LANE=' + json.dumps(str(warmup)),
                  '--set', 'benchmark.command=' + json.dumps(
                      'env PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 '
                      'PYTHONPATH="/infmax-workspace${PYTHONPATH:+:$PYTHONPATH}" '
                      'python3 /sa-profile-bin/agentic.py')]
plans = plan_commands(str(recipe_file), c['environment']['FRAMEWORK'], runtime_arguments(recipe, os.environ) + extra, os.environ)
if len(plans) != 1:
    raise RuntimeError('Exactly one native recipe is required')
plan = plans[0]
# Resolve native golden overrides before changing only the local draft-model path.
from srtctl.core.overrides import apply_overrides_to_recipe, parse_overrides
import copy
options = argparse.ArgumentParser(add_help=False)
options.add_argument('--set', action='append')
options.add_argument('--unset', action='append')
parsed, _ = options.parse_known_args(plan.command[2:])
effective = copy.deepcopy(bound)
apply_overrides_to_recipe(effective, parse_overrides(parsed.set, parsed.unset))
spec = json.loads(effective['roles']['agg']['args']['speculative-config'])
spec['model'] = '/models/' + spec['model']
plan.command.extend(['--set', 'roles.agg.args.speculative-config=' + json.dumps(json.dumps(spec))])
(a.output / 'native-plan.json').write_text(json.dumps({'selected': selected, 'command': plan.command,
    'acceptance_length': plan.acceptance_length, 'adaptations': [
    ('Execute the native rendered script directly on this runner; no Slurm reservation is used or changed.'
     if os.environ.get('SA_LOCAL_EXECUTION') == '1' else
     'Execute native rendered batch script in the existing single-node allocation; never cancel the reservation.'),
    'Local model/cache/output mounts and auto-detected host interface replace SA host paths.',
    'Docker executes unchanged native task argv, directly in local mode or through Spur in reservation mode.',
    'Container writability follows native SRT options; per-step runtime caches and temporary compiler storage are mounted separately.',
    'Frozen workflow inputs and native workload environment names are explicitly inherited by Docker steps.',
    'Native GPU and RAPL telemetry are requested; unavailable local permissions remain explicit missing evidence.',
    'GitHub release assets use the API route because direct download URLs return HTTP 504.'
]}, indent=2) + '\n')
rendered = a.output / 'rendered'
command = list(plan.command)
command[1] = 'render'
command += ['--to', str(rendered)]
(a.output / 'render-command.json').write_text(json.dumps(command, indent=2) + '\n')
if a.prepare_only:
    raise SystemExit(0)
# Keep native setup and checksum verification; adapt only the download route.
import shutil
setup_env = dict(os.environ)
for tool in ('curl', 'wget'):
    setup_env['SA_REAL_' + tool.upper()] = shutil.which(tool)
setup_env['PATH'] = str(Path(__file__).parent / 'download') + ':' + os.environ['PATH']
subprocess.run(['make', 'setup', 'ARCH=x86_64', 'TACHOMETER_RELEASE=' + c['telemetry_release'], 'CPU_POWER_EXPORTER_RELEASE=v2.66.1'], check=True, env=setup_env)
os.environ.pop('GH_TOKEN', None)
subprocess.run(command, check=True, env={**os.environ, **dict(plan.environment)})
# Keep the scheduler real. Only translate equivalent container defaults on Spur.
if os.environ.get('SA_LOCAL_EXECUTION') != '1':
    os.environ['SA_REAL_SRUN'] = subprocess.check_output(['which', 'srun'], text=True).strip()
    os.environ['SA_REAL_SQUEUE'] = subprocess.check_output(['which', 'squeue'], text=True).strip()
os.environ['PATH'] = str(Path(__file__).parent / 'spur') + ':' + os.environ['PATH']
os.environ['SA_OUTPUT'] = str(a.output)
# Pyxis inherits the workflow inputs; Docker needs them forwarded explicitly.
os.environ['SA_PASSTHROUGH_ENV'] = json.dumps(sorted(
    key for key in os.environ if key in c['environment']
    or any(fnmatch.fnmatchcase(key, pattern) for pattern in WORKLOAD_ENV)))
try:
    subprocess.run(['bash', str(rendered / 'sbatch_script.sh')], check=True)
finally:
    # Clean only containers belonging to this workflow; never cancel the reservation.
    owned = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'label=sa.run=' + a.output.name], text=True).split()
    if owned:
        subprocess.run(['docker', 'rm', '-f', *owned], check=True)

# SA's workflow validates the native AgentX output after the benchmark exits.
logs = a.output / 'native' / os.environ['SLURM_JOB_ID'] / 'logs'
result = logs / (c['name'] + '.json')
if not result.is_file():
    raise RuntimeError('Native benchmark result is missing: ' + str(result))
subprocess.run([sys.executable, '-m', 'infx.results.agentic.validate_agentic_result', str(logs / 'agentic'), '--failed-request-threshold', c['environment']['AIPERF_FAILED_REQUEST_THRESHOLD']], check=True)
if profile:
    print('Server shutdown finished; validating completed profiler files before artifact upload.', flush=True)
    subprocess.run([sys.executable, Path(__file__).with_name('validate_profile.py'),
                    logs, '--seconds', str(seconds), '--require-exports',
                    '--output', a.output / 'profile-capture-validation.json']
                   + (['--require-attachment'] if mode == 'attach' else [])
                   + (['--minimum-seconds', str(minimum)] if minimum is not None else []), check=True)
