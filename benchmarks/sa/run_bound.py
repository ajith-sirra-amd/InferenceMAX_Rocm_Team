"""Local storage/scheduler bindings; workload and process lifecycle stay native."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

from infx.srt_slurm.single_node import select_recipe, runtime_arguments
from infx.srt_slurm.synthetic_acceptance import plan_commands

p = argparse.ArgumentParser()
for name in ('config', 'output', 'source', 'srt'):
    p.add_argument('--' + name, required=True, type=Path)
a = p.parse_args()
c = json.loads(a.config.read_text())
recipe = str(a.source / c['recipe'])
selected, bound = select_recipe(recipe, os.environ)
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
settings.update(cluster='mi355x-local', default_partition='amd-spur', default_account='',
                default_time_limit='08:00:00', gpus_per_node=8,
                network_interface='lo', srtctl_root=str(a.srt), output_dir=str(a.output / 'native'),
                default_mounts=mounts, use_segment_sbatch_directive=False,
                use_exclusive_sbatch_directive=False,
                model_paths={f"hf:{c['environment']['MODEL']}": '/share_nfs/models/' + c['environment']['MODEL']})
(a.srt / 'srtslurm.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))
os.environ['SRTSLURM_CONFIG'] = str(a.srt / 'srtslurm.yaml')
extra = ['--set', 'benchmark.stream_output=true', '--set', 'srun_options.container-workdir="/infmax-workspace"']
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
    'Local model/cache/output mounts and loopback interface replace SA host paths.',
    'Spur transport omits unsupported Pyxis default flags; native workload argv is retained.',
    'CPU power exporter is unavailable locally; GPU telemetry retains the SA exporter configuration.'
]}, indent=2) + '\n')
rendered = a.output / 'rendered'
command = list(plan.command)
command[1] = 'render'
command += ['--to', str(rendered)]
subprocess.run(command, check=True, env={**os.environ, **dict(plan.environment)})
# Keep the scheduler real. Only translate equivalent container defaults on Spur.
os.environ['SA_REAL_SRUN'] = subprocess.check_output(['which', 'srun'], text=True).strip()
os.environ['PATH'] = str(Path(__file__).parent / 'spur') + ':' + os.environ['PATH']
os.environ['SA_OUTPUT'] = str(a.output)
subprocess.run(['bash', str(rendered / 'sbatch_script.sh')], check=True)
