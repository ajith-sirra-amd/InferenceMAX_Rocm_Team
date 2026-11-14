# Exact-revision Kimi PR runtime preparation

## Current MXFP4 qualification

Both frozen PR pairs have been built and inventory-verified. Original runtimes
reject Kimi MXFP4, so model checks use explicitly derived paired compatibility
variants. `compatibility.py` preserves original and syncfree variants and builds
the current `prefill_pinned` bundle: admission/page-stride support, global/local
DCP metadata, scheduler CPU scalar lengths and pinned MLA context-row copies.
Every declared source change is identical in both arms; inventories reject any
other source, dependency, native, test or vendor changes. No FP8 indexer
substitution or sync-check exemption is used.

Native MXFP4 cache/graph and independent scorer checks passed. Current metadata
qualification43606584068 passes4 metadata and2 pinned-row cases on8 ranks with
the native sync checker active across all three preparation paths. Model screen
43606789911 is active. AttnRes original suite43603528622 fails6 of29 bitwise
cases; diagnostic43604796100 reproduces differences in all12 native sequences
while prefix/residual-bank values stay identical. A successful diagnostic means
the failure reproduced, not that the PR passed. Neither full accuracy nor
end-to-end speedup is qualified. Sections below preserve earlier attempt history.

## GPU qualification checkpoint, 14 November

ROCm suite run43566518007 passed all32 cases, including the native loader.
Native TP8 projection run43567005492 passed10 cases per rank on all8 ranks:
BF16 projection/gather/RMSNorm, eager/graph and two synthetic weight loads.
These do not qualify full-model accuracy; the CPU triton-cpu setup remains unresolved.

`PR67372 model screen` runs the verified baseline first, then the candidate
only after a complete natural48-question screen. It preserves the natural
recipe and records the native loaded DSpark layer/source/weight shape and hash
on each rank via a read-only observational worker extension. Source and paired
runtime inventories are verified before exposing GPUs. The historical audit
is not reused for the new installed source layout. This diagnostic has loading
overhead and is not a performance measurement or full quality qualification.
Failures stop the pair and retain evidence; no gate is relaxed.

This is a staged addition to `IPAT-Script`. The existing fixed, natural and
agentic choices continue to run their historical runtime. They cannot qualify
upstream PR #67372 or #67373.

## Implemented locally

- `PR67372 prepare runtimes` and `PR67373 prepare runtimes`: build each full
  frozen baseline/head from its official `docker/docker-bake-rocm.hcl`.
  Build dependency images once from the paired baseline and reuse them for
  both arms. Reject differing dependency inputs or installed package identities.
- `PR67372 correctness` and `PR67373 correctness`: verify both prepared
  runtime identities without exposing GPUs, then run the candidate's complete
  component suite with a JUnit result. DSpark's handoff suite runs on CPU with
  no GPU devices; AttnRes waits for eight idle GPUs before its ROCm suite. Missing/skipped
  new cases and any failed cases fail the job.
- Source verification compares the installed Python package and baked tests
  with the clean full Git checkout. Generated `_version.py` is recorded
  separately. The official Dockerfile also vendors ROCm Triton kernels; these
  files are compared byte-for-byte with a clean reference checkout at the exact
  Dockerfile pin, retained in the adapter. Unexpected Python files fail.
  Native library hashes, package
  versions, image identity and actual interpreter are retained.
- Recipe preparation applies only the DSpark additional-config key or AttnRes
  environment flag, retaining all other SA settings. This prepared recipe is
  evidence for subsequent integration, not a launched server command.

The first authorized DSpark build completed shared dependency images, then
failed in the official baseline version-calculation stage: a depth-one checkout
had no reachable release tag. Run `43469338489` and its uploaded evidence are
preserved. Source fetching now includes full history and tags; an early check
records the pinned commit, release tag target and Git description in
`source-versions.json` before any dependency build. Early recovery run
`43491683215` demonstrated that the fork contains no tags, so release refs are
now explicitly fetched from `vllm-project/vllm`, while HEAD stays at the frozen
fork SHA. The official version gate is retained. A linked fresh attempt can
reuse Docker's completed build cache.
No correctness suite or server has been run with these additions.
Run `43492738381` built the baseline but its original inventory gate correctly
stopped on these previously unaccounted vendored files. The replacement probe
passed on that image using an explicit read-only pinned reference: all 47
vendored Python files matched. This diagnostic is not a qualified paired build;
a fresh adapter must bake the reference and complete both arm verifications.
The local tests validate manifest/delta/JUnit behavior; Bash/YAML validation
does not qualify Docker, compiler, GPU, graph or model behavior.

## Verified DSpark runtime build

Run [43495485285](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/43495485285)
passed both image verifications and artifact upload on `pit2-p03-g37`, using
team commit `3f774e55f1b36cbec6b8990fbe9461e71f110f5e`. The downloaded manifest
and verification records were checked again locally. Each arm verifies 3,415
Python files, 2,485 test files, 47 pinned vendored Python files and seven native
libraries. Installed package dependencies, vendor hashes and native hashes
match. Source differences are the intended DSpark file and generated version;
the test difference is `tests/models/test_dspark_mla.py`.

The retained host-bound pair manifest is
`/share_nfs/gguasti/benchmarks/pr-build-43495485285-1/runtime/pair.json`.
This establishes runtime preparation only. No GPU component suite, native
activation/graph/model validation or benchmark has run; PR67373's paired build
remains unstarted. The previous failed attempts and diagnostic results are
preserved separately.

## Frozen pairs

| PR | Baseline | Candidate |
| --- | --- | --- |
| #67372 | `0e2871508b20815f2e5969f1697a0de5b1b90881` | `c4aea1c4b27bc2a932a177ce8291ea1ba29d6365` |
| #67373 | `8a8cad0253ecf942f0c00c7e55a6bc7ddfee1dc1` | `d5d2c30af2599a164e7a18c7567aa401c5c40a8e` |

AttnRes's handoff lists an earlier candidate. These values were checked through
GitHub on 13 November 2026. A moved head requires an explicit new pair record
and fresh validation, not an automatic substitution.

## Build and runtime contract

The new jobs explicitly use `IPAT_PR_DATA_ROOT=/share_nfs/gguasti/benchmarks`.
The first run `43468920330` stopped before building because the Giovanni runner
cannot write Ajith's benchmark directory; its logs are preserved. Existing
Ajith data and directory permissions remain unchanged.

The runner must have Docker buildx using the `docker` driver, sufficient host
RAM/disk, network access for the upstream build and access to the shared output
directory. Builds may take hours. Build containers are not given GPU devices.
The official current source uses the TheRock ROCm stack; compatibility with the
host driver and the historical SA arguments must still be measured.

The adapter retains the entire clean source checkout at `/opt/ipat/vllm`, the
official built tests at `/vllm-workspace/tests`, and the compiled installed
package. It adds `/opt/runtime-venv/bin/python` and the native vLLM CLI entrypoint
without copying selected model files into the historical image.

Build results include a `runtime/pair.json`. Local images are pinned by Docker
content ID (`sha256:...`), are retained and are **bound to the build host**.
They are not uploaded with compact Actions artifacts or published to a registry.
The correctness job must land on that host with those images present. A
different host fails during preflight before GPU use. Registry-backed manifests
may instead use `repository@sha256:...`, after a separately authorized export.

The pair manifest schema is:

```json
{
  "schema_version": 1,
  "pr": 67372,
  "source_repository": "ajith-sirra-amd/vllm",
  "baseline": {
    "source_sha": "FULL_BASELINE_COMMIT",
    "image": "sha256:LOCAL_IMAGE_ID_OR_REPOSITORY_AT_DIGEST",
    "dependencies_sha256": "HASH_OF_NORMALIZED_PIP_FREEZE_EXCLUDING_VLLM"
  },
  "candidate": {
    "source_sha": "FULL_CANDIDATE_COMMIT",
    "image": "sha256:LOCAL_IMAGE_ID_OR_REPOSITORY_AT_DIGEST",
    "dependencies_sha256": "SAME_DEPENDENCY_HASH"
  }
}
```

This schema illustration contains placeholders and cannot pass validation.
Use the manifest produced by a successful build, never infer image identities.

Once these changes are published and execution is authorized, select the
matching preparation choice in `End-to-End Tests`. For the following correctness
job set `pr-runtime-manifest` to the resulting shared `runtime/pair.json` path.
Keep the documented runner labels and single-node concurrency group. Identify
the actual run by its descriptive name, preserve attempts separately and inspect
every five minutes through export/upload. No automatic dispatch is implemented.

The DSpark CPU command deliberately retains `VLLM_TARGET_DEVICE=cpu` from its
handoff. A ROCm test image may still lack native CPU support: that is a setup
failure to resolve, not a reason to skip its new native-loader test. Both suites
use `VLLM_GPU_SYNC_CHECK=error`; no timing measurement runs in this job.

## Ajith's diagnostic reference

Run [43254445132](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/43254445132)
at team SHA `19d35942380cddac5981b11cf57c4fc6f270ae7b` collected a short natural
verifier diagnostic using the historical SA image digest
`5b54f223a150a9baad41a5d511e159e7cbf43e308c6cd44ad3f6a2463c245b3c`.
It is not a prepared exact-PR runtime or a quality pass. Reuse its existing
`benchmarks/quality/diagnostic/wait_for_idle.py` and native worker-extension
pattern. Instrumented observations have timing/memory overhead and must stay
separate from performance measurements.

## Remaining serving integration gates

The shared serving launcher and its old audit/activation validator are not yet
adapted to these prepared runtimes. No exact-PR fixed/agentic serving choice is
enabled by this change. Next work, after runtime compatibility is established:

1. Qualify native worker-extension import/initialization at each exact revision;
   generate actual rank 0â€“7 loaded-source, DSpark weight/admission and AttnRes
   eligible-row evidence. A flag or historical marker is insufficient.
2. Adapt the shared launcher/audit to the prepared recipe and dynamic installed
   source path; preserve all other SA settings and stop on unsupported arguments.
3. Validate real-checkpoint projection/gather/norm, native graph lifetime and
   natural model accuracy against a valid fresh control. Preserve the earlier
   NaN baseline as failed evidence.
4. Add gated baseline-first fixed C48 comparisons, then repeated full native
   agentic pairs. Keep the existing client source/settings and separate the
   synthetic performance lane from natural verification.

No current-head accuracy or E2E gain is established, and neither PR is promoted.
