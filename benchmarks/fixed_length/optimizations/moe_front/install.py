#!/usr/bin/env python3
"""Prepare a verified run-local merged MoE-front overlay without editing installed vLLM."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(source, output):
    source, output = source.resolve(), output.resolve()
    bundle = Path(__file__).resolve().parent
    manifest = json.loads((bundle / "manifest.json").read_text())
    patch = bundle / manifest["patch"]
    if sha256(patch) != manifest["patch_sha256"]:
        raise ValueError("Candidate patch hash mismatch")
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from original source")
    if output.exists():
        raise ValueError("Output already exists; use a fresh run directory")
    expected = dict(manifest["native_dependency_sha256"])
    expected.update({entry["path"]: entry["baseline_sha256"] for entry in manifest["files"]})
    for path, digest in expected.items():
        if sha256(source / path) != digest:
            raise ValueError("Baseline source hash mismatch: " + path)
    output.mkdir(parents=True)
    for entry in manifest["files"]:
        for directory in ("original", "candidate"):
            target = output / directory / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / entry["path"], target)
    subprocess.run(
        ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch)],
        cwd=output / "candidate", check=True,
    )
    for entry in manifest["files"]:
        candidate = output / "candidate" / entry["path"]
        if sha256(candidate) != entry["candidate_sha256"]:
            raise ValueError("Patched source hash mismatch: " + entry["path"])
        ast.parse(candidate.read_text())
    for path, digest in expected.items():
        if sha256(source / path) != digest:
            raise ValueError("Original source changed during preparation: " + path)
    manifest["installation"] = {
        "source": str(source),
        "candidate": str(output / "candidate"),
        "original": str(output / "original"),
        "verified_source_sha256": expected,
        "status": "source hashes verified; full-model runtime qualification pending",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["installation"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Original site-packages root")
    parser.add_argument("--output", required=True, type=Path, help="Fresh run-local directory")
    args = parser.parse_args()
    install(args.source, args.output)


if __name__ == "__main__":
    main()
