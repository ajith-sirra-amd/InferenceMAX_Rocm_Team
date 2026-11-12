"""Compose the existing disjoint MoE-front and DSpark run-local overlays."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(source, output):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Output must be separate from original source')
    if output.exists():
        raise ValueError('Output already exists; use a fresh run directory')
    bundles = Path(__file__).resolve().parent.parent
    names = ('moe_front', 'dspark')
    manifests = {name: json.loads((bundles/name/'manifest.json').read_text()) for name in names}
    if len({m['image'] for m in manifests.values()}) != 1:
        raise ValueError('Component images differ')
    modified = set()
    for name, manifest in manifests.items():
        paths = {entry['path'] for entry in manifest['files']}
        if modified.intersection(paths):
            raise ValueError('Component source overlays overlap')
        modified.update(paths)
        for other, dependency in manifests.items():
            if other != name and paths.intersection(dependency['native_dependency_sha256']):
                raise ValueError('A component changes another component native dependency')
    output.mkdir(parents=True)
    installed, files, verified = {}, [], {}
    for name in names:
        component = output/name
        subprocess.run([sys.executable, str(bundles/name/'install.py'),
                        '--source', str(source), '--output', str(component)], check=True)
        manifest = json.loads((component/'manifest.json').read_text())
        installed[name] = manifest
        for path, digest in manifest['installation']['verified_source_sha256'].items():
            if path in verified and verified[path] != digest:
                raise ValueError('Component baseline hashes disagree: '+path)
            verified[path] = digest
        for entry in manifest['files']:
            files.append(entry)
            for directory, hash_key in [('original', 'baseline_sha256'), ('candidate', 'candidate_sha256')]:
                target = output/directory/entry['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(component/directory/entry['path'], target)
                if sha256(target) != entry[hash_key]:
                    raise ValueError('Copied overlay hash mismatch: '+str(target))
    for path, digest in verified.items():
        if sha256(source/path) != digest:
            raise ValueError('Original source changed during composition: '+path)
    manifest = {
        'name': 'kimi-moe-front-and-dspark',
        'status': 'Combined fixed-length performance qualification pending; model quality unvalidated',
        'image': manifests['moe_front']['image'],
        'files': files,
        'components': installed,
        'enable_environment': {'KIMI_MOE_MERGED_FRONT': '1', 'KIMI_DSPARK_SHARD_CONTEXT_PROJ': '1'},
        'installation': {'source': str(source), 'candidate': str(output/'candidate'),
                         'original': str(output/'original'), 'verified_source_sha256': verified,
                         'status': 'Both original installers passed; three disjoint candidate hashes verified'},
        'isolated_model_runs': {'moe_front': 43192854530, 'dspark': 43181110676},
        'limitations': ['Combined runtime and E2E are not established by isolated improvements',
                        'BF16/router differences remain; synthetic acceptance does not establish model quality'],
    }
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest['installation']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    install(args.source, args.output)
