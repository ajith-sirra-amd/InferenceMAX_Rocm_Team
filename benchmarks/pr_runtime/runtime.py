"""Resolve and verify prepared, digest-pinned full-source PR runtimes.

No image is built or experiment dispatched by this module. Runtime manifests
are build outputs, not substitutes for GPU/model qualification.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess


PAIRS = {
    67372: ("0e2871508b20815f2e5969f1697a0de5b1b90881",
            "c4aea1c4b27bc2a932a177ce8291ea1ba29d6365"),
    67373: ("8a8cad0253ecf942f0c00c7e55a6bc7ddfee1dc1",
            "d5d2c30af2599a164e7a18c7567aa401c5c40a8e"),
}
SUITES = {
    67372: "tests/models/test_dspark_mla.py",
    67373: "tests/models/kimi_k3/test_amd_attn_res.py",
}
PYTHON = "/opt/runtime-venv/bin/python"
SOURCE = "/opt/ipat/vllm"


def load_pair(path, pr):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1 or data.get("pr") != pr or pr not in PAIRS:
        raise ValueError("Unknown manifest schema or PR")
    if data.get("source_repository") != "ajith-sirra-amd/vllm":
        raise ValueError("Unexpected vLLM source repository")
    for arm, sha in zip(("baseline", "candidate"), PAIRS[pr]):
        item = data[arm]
        if item["source_sha"] != sha:
            raise ValueError(f"{arm}: revision differs from the frozen pair")
        if not re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", item["image"]):
            raise ValueError(f"{arm}: use a digest-pinned image")
        if not re.fullmatch(r"[0-9a-f]{64}", item["dependencies_sha256"]):
            raise ValueError(f"{arm}: missing dependency identity")
    if data["baseline"]["image"] == data["candidate"]["image"]:
        raise ValueError("Baseline and candidate cannot use the same artifact")
    if data["baseline"]["dependencies_sha256"] != data["candidate"]["dependencies_sha256"]:
        raise ValueError("Paired dependencies differ")
    return data


def candidate_recipe(recipe, pr, arm):
    """Apply only the declared opt-in; retain every other core setting."""
    if pr not in PAIRS or arm not in ("baseline", "candidate"):
        raise ValueError("Unknown PR or arm")
    result = copy.deepcopy(recipe)
    additional = json.loads(result["server_arguments"]["additional-config"])
    if additional.get("dspark_context_projection_sharding", False):
        raise ValueError("Source recipe already enables DSpark sharding")
    if result["server_environment"].get("VLLM_ROCM_KIMI_ATTN_RES_NORM_CACHE", "0") != "0":
        raise ValueError("Source recipe already enables AttnRes caching")
    if pr == 67372 and arm == "candidate":
        additional["dspark_context_projection_sharding"] = True
        result["server_arguments"]["additional-config"] = json.dumps(additional, separators=(",", ":"))
    result["server_environment"]["VLLM_ROCM_KIMI_ATTN_RES_NORM_CACHE"] = (
        "1" if pr == 67373 and arm == "candidate" else "0"
    )
    return result


PROBE = r'''
import hashlib, importlib.util, json, pathlib, subprocess, sys
root = pathlib.Path('/opt/ipat/vllm')
sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True)
if dirty:
    raise RuntimeError('Baked full-source checkout is dirty')
spec = importlib.util.find_spec('vllm')
installed = pathlib.Path(spec.origin).parent
tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', 'vllm'], text=True).splitlines()
hashes = {}
for name in tracked:
    if not name.endswith('.py'):
        continue
    source = root / name
    target = installed.parent / name
    if not target.is_file() or source.read_bytes() != target.read_bytes():
        raise RuntimeError('Installed/source mismatch: ' + name)
    hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
actual = {'vllm/' + str(p.relative_to(installed)).replace('\\', '/') for p in installed.rglob('*.py')}
generated = actual - set(hashes)
if generated - {'vllm/_version.py'}:
    raise RuntimeError('Unexpected generated/overlaid Python source: ' + repr(sorted(generated)))
for name in generated:
    hashes[name] = hashlib.sha256((installed.parent / name).read_bytes()).hexdigest()
test_hashes = {}
for name in subprocess.check_output(['git', '-C', str(root), 'ls-files', 'tests'], text=True).splitlines():
    if not name.endswith('.py'):
        continue
    target = pathlib.Path('/vllm-workspace') / name
    if not target.is_file() or (root / name).read_bytes() != target.read_bytes():
        raise RuntimeError('Baked test/source mismatch: ' + name)
    test_hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
native = {}
for target in installed.rglob('*.so'):
    with target.open('rb') as stream:
        native[str(target.relative_to(installed))] = hashlib.file_digest(stream, 'sha256').hexdigest()
if not native:
    raise RuntimeError('No compiled vLLM native artifacts found')
freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True)
deps = sorted(line.strip() for line in freeze.splitlines() if line.strip()
    and not line.startswith('#') and not line.lower().startswith(('vllm==', 'vllm @'))
    and line.strip() != '-e /opt/ipat/vllm')
identity = hashlib.sha256(('\n'.join(deps) + '\n').encode()).hexdigest()
print(json.dumps(dict(source_sha=sha, source_sha256=hashes, dependencies=deps,
    test_sha256=test_hashes, native_sha256=native, generated_python=sorted(generated),
    dependencies_sha256=identity, python=sys.executable, source_root=str(installed.parent))))
'''


def verify_probe(item, probe, labels):
    if probe["source_sha"] != item["source_sha"]:
        raise ValueError("Baked checkout revision mismatch")
    if labels.get("org.opencontainers.image.revision") != item["source_sha"]:
        raise ValueError("Image revision label mismatch")
    if probe["python"] != PYTHON:
        raise ValueError("Unexpected runtime interpreter")
    if probe["dependencies_sha256"] != item["dependencies_sha256"]:
        raise ValueError("Dependency identity mismatch")
    if not probe["source_sha256"]:
        raise ValueError("Missing installed-source inventory")
    return probe


def verify_image(item, output):
    image = item["image"]
    if not image.startswith("sha256:"):
        subprocess.run(["docker", "pull", image], check=True)
    inspect = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
    probe = probe_image(image, output)
    result = verify_probe(item, probe, inspect["Config"].get("Labels") or {})
    if image.startswith("sha256:") and inspect["Id"] != image:
        raise ValueError("Local immutable image ID mismatch")
    output.write_text(json.dumps(dict(image=image, image_id=inspect["Id"], **result), indent=2) + "\n")


def probe_image(image, output):
    name = "kimi-pr-probe-" + hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:16]
    base = Path("/share_nfs/ajith/benchmarks")
    relative = output.resolve().relative_to(base)
    owner = str(base / relative.parts[0])
    try:
        raw = subprocess.check_output([
            "docker", "run", "--rm", "--name", name, "--label", "ipat.pr.output=" + owner,
            "--network", "none", "--entrypoint", PYTHON, image, "-c", PROBE,
        ], text=True, timeout=900)
        return json.loads(raw)
    finally:
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=30, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--pr", type=int, choices=PAIRS, required=True)
    parser.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--recipe", type=Path)
    args = parser.parse_args()
    pair = load_pair(args.manifest, args.pr)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "runtime-manifest.json"
    if destination.exists():
        raise ValueError("Refusing to overwrite a prepared runtime record")
    destination.write_text(json.dumps(dict(pair=pair, arm=args.arm), indent=2) + "\n")
    (args.output / "runtime-image.txt").write_text(pair[args.arm]["image"] + "\n")
    (args.output / "runtime-suite.txt").write_text(SUITES[args.pr] + "\n")
    if args.recipe:
        recipe = candidate_recipe(json.loads(args.recipe.read_text()), args.pr, args.arm)
        recipe["image"] = pair[args.arm]["image"]
        recipe["runtime_identity"] = dict(pr=args.pr, arm=args.arm, source_sha=pair[args.arm]["source_sha"])
        (args.output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    if args.verify:
        verify_image(pair[args.arm], args.output / "runtime-verification.json")


if __name__ == "__main__":
    main()
