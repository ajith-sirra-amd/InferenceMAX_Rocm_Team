"""Require matched full quality and explicit local AttnRes fixture qualification."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import model_compare
import runtime


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_recipe(recipe, pr, arm):
    result = copy.deepcopy(recipe)
    result.pop('image')
    additional = json.loads(result['server_arguments']['additional-config'])
    if pr == 67372:
        assert additional.pop('dspark_context_projection_sharding', False) == (arm == 'candidate')
    elif pr == 67373:
        assert result['server_environment'].pop('VLLM_ROCM_KIMI_ATTN_RES_NORM_CACHE') == ('1' if arm == 'candidate' else '0')
    else:
        raise ValueError('Unknown frozen PR')
    result['server_arguments']['additional-config'] = additional
    return result


def same_layout_fixture(original):
    replacements = [
        ('reference_prefix = prefix.clone()',
         'reference_prefix = torch.empty((num_tokens, hidden + 7), device=prefix.device, dtype=prefix.dtype)[:, :hidden]\n    reference_prefix.copy_(prefix)'),
        ('reference_blocks = blocks.clone()',
         'reference_blocks = torch.empty((num_tokens, 8, hidden + 13), device=blocks.device, dtype=blocks.dtype)[:, :, :hidden]\n    reference_blocks.copy_(blocks)'),
    ]
    for old, new in replacements:
        assert original.count(old) == 1, 'Frozen fixture input differs'
        original = original.replace(old, new)
    return original


def fixture_gate(component, pair, quality, owner):
    component = Path(component).resolve()
    component.relative_to(Path('/share_nfs/gguasti/benchmarks'))
    validation = json.loads((component / 'fixture-suite/validation.json').read_text())
    manifest = json.loads((component / 'fixture-manifest.json').read_text())
    assert validation['status'] == 'passed' and validation['returncode'] == 0
    assert validation['cases'] == 29 and validation['failures'] == []
    assert validation['fixture'] == manifest
    assert manifest['source_sha'] == pair['candidate']['source_sha']
    cases = ET.parse(component / 'fixture-suite/junit.xml').findall('.//testcase')
    assert len(cases) == 29 and all(not any(c.find(k) is not None for k in ['failure', 'error', 'skipped']) for c in cases)
    image = pair['candidate']['parent_image']
    command = ['docker', 'run', '--rm', '--network', 'none', '--label', 'ipat.pr.output=' + str(owner),
               '--entrypoint', runtime.PYTHON, image, '-c',
               "from pathlib import Path;print(Path('/vllm-workspace/tests/models/kimi_k3/test_amd_attn_res.py').read_text(),end='')"]
    original = subprocess.check_output(command, text=True, timeout=60)
    digest = hashlib.sha256(original.encode()).hexdigest()
    verified = json.loads((quality / 'candidate/runtime-verification.json').read_text())
    assert digest == manifest['original_sha256'] == verified['test_sha256']['tests/models/kimi_k3/test_amd_attn_res.py']
    expected = hashlib.sha256(same_layout_fixture(original).encode()).hexdigest()
    assert expected == manifest['revised_sha256'] == sha(component / 'test_amd_attn_res_same_layout.py')
    return dict(contract='Local full29 same-layout cache exactness; original upstream23pass/sixfail retained',
                input_sha256={str(p): sha(p) for p in [component / 'fixture-manifest.json',
                    component / 'fixture-suite/validation.json', component / 'fixture-suite/junit.xml',
                    component / 'test_amd_attn_res_same_layout.py']})


def main():
    parser = argparse.ArgumentParser()
    for name in ['manifest', 'quality', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--component-output', default='')
    args = parser.parse_args()
    pair = json.loads(args.manifest.read_text())
    runtime.load_pair(args.manifest, pair['pr'])
    comparison = model_compare.compare(args.quality)
    assert comparison['score_denominator'] == 1319, 'Full pinned quality coverage required before timing'
    assert comparison['candidate_correct'] >= comparison['baseline_correct'], 'Full comparison shows score decrease'
    for arm in ['baseline', 'candidate']:
        assert json.loads((args.quality / arm / 'runtime-manifest.json').read_text())['pair'] == pair
    component = None
    if pair['pr'] == 67373:
        assert args.component_output, 'Explicit local AttnRes fixture qualification is required'
        component = fixture_gate(args.component_output, pair, args.quality, args.output)
    destination = args.output / 'pair-start.json'
    assert not destination.exists(), 'Refusing to overwrite timing prerequisite evidence'
    args.output.mkdir(parents=True, exist_ok=True)
    (args.quality / 'model-comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    destination.write_text(json.dumps(dict(pr=pair['pr'], quality_output=str(args.quality),
        quality_comparison_sha256=sha(args.quality / 'model-comparison.json'),
        manifest_sha256=sha(args.manifest), component_qualification=component,
        limits='Matched full1319 zero-shot sampled no-drop prerequisite; broad quality and performance remain separate'), indent=2) + '\n')
    print(json.dumps(dict(status='full_quality_prerequisite_passed', pr=pair['pr'])))


if __name__ == '__main__':
    main()
