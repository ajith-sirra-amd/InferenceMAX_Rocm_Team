"""Local storage/scheduler bindings; workload and process lifecycle stay native."""
import argparse
import fnmatch
import json
import os
from pathlib import Path
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
    if type(delay) is not int or delay < 0 or type(seconds) is not int or not 1 <= seconds <= 180:
        raise ValueError('Use a calibrated nonnegative delay and a 1–180 second capture')
    mounts[str(Path(__file__).parent / 'profile')] = '/sa-profile-bin:ro'
    settings['default_bash_preamble'] = 'export PATH="/sa-profile-bin:$PATH"'
settings.update(cluster='mi355x-local', default_partition='amd-spur', default_account='',
                default_time_limit='08:00:00', gpus_per_node=8,
                network_interface='', srtctl_root=str(a.srt), output_dir=str(a.output / 'native'),
                default_mounts=mounts, use_segment_sbatch_directive=False,
                use_exclusive_sbatch_directive=False,
                model_paths={f"hf:{c['environment']['MODEL']}": '/share_nfs/models/' + c['environment']['MODEL']})
(a.srt / 'srtslurm.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))
os.environ['SRTSLURM_CONFIG'] = str(a.srt / 'srtslurm.yaml')
extra = ['--set', 'benchmark.stream_output=true', '--set', 'srun_options.container-workdir="/infmax-workspace"',
         '--set', 'srun_options.container-remap-root=""', '--set', 'srun_options.container-writable=""']
if profile:
    for key, value in {'SA_PROFILE_DELAY_SECONDS': str(delay), 'SA_PROFILE_SECONDS': str(seconds),
                       'SA_PROFILE_OUTPUT': '/logs/profiles/rocprof'}.items():
        extra += ['--set', 'roles.agg.env.' + key + '=' + json.dumps(value)]
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
    'Execute native rendered batch script in the existing single-node allocation; never cancel the reservation.',
    'Local model/cache/output mounts and auto-detected host interface replace SA host paths.',
    'Spur overlap ignores container options; Docker executes unchanged native task argv inside the reservation.',
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
os.environ['SA_REAL_SRUN'] = subprocess.check_output(['which', 'srun'], text=True).strip()
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
    subprocess.run([sys.executable, Path(__file__).with_name('validate_profile.py'),
                    logs, '--seconds', str(seconds)], check=True)
