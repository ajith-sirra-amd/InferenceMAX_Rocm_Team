"""Startup memory inventory; no arithmetic, allocator or cache-policy changes."""
import functools
import json
import os
from pathlib import Path
import types


def snapshot(worker, phase):
    import torch

    runner = worker.model_runner
    device = worker.device
    free, total = torch.cuda.mem_get_info(device)
    seen = set()
    storages = {}
    pending = [(runner, "runner", 0)]
    scanned = 0
    while pending and scanned < 100000:
        value, name, depth = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        scanned += 1
        if isinstance(value, torch.Tensor):
            if value.device.type == "cuda" and value.numel():
                storage = value.untyped_storage()
                key = (str(value.device), storage.data_ptr())
                item = {"path": name, "storage_bytes": storage.nbytes(),
                        "shape": list(value.shape), "dtype": str(value.dtype)}
                if key not in storages:
                    storages[key] = item
            continue
        if depth >= 18 or isinstance(value, (str, bytes, int, float, bool, type(None), types.ModuleType, type)):
            continue
        if isinstance(value, dict):
            items = value.items()
        elif isinstance(value, (list, tuple)):
            items = enumerate(value)
        elif callable(value):
            continue
        elif hasattr(value, "__dict__") and type(value).__module__.startswith(("vllm.", "torch.nn.")):
            items = vars(value).items()
        else:
            continue
        for key, child in items:
            pending.append((child, name + "." + str(key), depth + 1))
    rows = sorted(storages.values(), key=lambda item: item["storage_bytes"], reverse=True)
    workspaces = [row for row in rows if any(word in row["path"].lower()
                  for word in ("workspace", "buffer", "metadata", "scratch"))]
    rank = int(worker.rank)
    report = {
        "rank": rank, "phase": phase, "device": str(device),
        "free_bytes": free, "total_bytes": total,
        "torch_allocated_bytes": torch.cuda.memory_allocated(device),
        "torch_reserved_bytes": torch.cuda.memory_reserved(device),
        "cumulative_torch_peak_bytes": torch.cuda.max_memory_allocated(device),
        "objects_scanned": scanned, "scan_truncated": bool(pending),
        "largest_storages": rows[:80], "workspace_storages": workspaces[:100],
        "scope": "Read-only tensor metadata and allocator counters. KV tensor storage can be virtual; storage sizes are not summed as physical residency.",
    }
    directory = Path(os.environ["REVIEW_MANIFEST"]).parent
    (directory / f"memory-audit-rank{rank}-{phase}.json").write_text(json.dumps(report, indent=2))
    print("PR64288_MEMORY_AUDIT", json.dumps({key: report[key] for key in (
        "rank", "phase", "free_bytes", "torch_allocated_bytes", "torch_reserved_bytes"
    )}), flush=True)


def register():
    if os.environ.get("REVIEW_MEMORY_AUDIT") != "1":
        return
    from vllm.v1.worker.gpu_worker import Worker

    if getattr(Worker, "_pr64288_memory_audit_installed", False):
        return
    Worker._pr64288_memory_audit_installed = True
    for name in ("load_model", "determine_available_memory", "_compile_or_warm_up_model"):
        original = getattr(Worker, name)
        def wrap(original, phase):
            @functools.wraps(original)
            def observed(self, *args, **kwargs):
                result = original(self, *args, **kwargs)
                snapshot(self, phase)
                return result
            return observed
        setattr(Worker, name, wrap(original, name))


def main():
    import pr64288_matched_serve as serving

    def command_transform(command):
        command = list(command)
        command[command.index("--max-model-len") + 1] = "1048576"
        command += ["--load-format", "auto", "--kv-transfer-config", json.dumps({
            "kv_connector": "SimpleCPUOffloadConnector", "kv_role": "kv_both",
            "kv_connector_extra_config": {"cpu_bytes_to_use_per_rank": 281125000000,
                                          "lazy_offload": False},
        })]
        spec_index = command.index("--speculative-config") + 1
        spec = json.loads(command[spec_index])
        spec.update(rejection_sample_method="synthetic", synthetic_acceptance_length=3.36)
        command[spec_index] = json.dumps(spec)
        assert command[command.index("--gpu-memory-utilization") + 1] == "0.90"
        return command

    def env_transform(env):
        env = dict(env)
        env["VLLM_PLUGINS"] += ",pr64288_memory_audit"
        env["REVIEW_MEMORY_AUDIT"] = "1"
        return env

    def exercise(arm, env, log, proc):
        records = [json.loads(p.read_text()) for p in (serving.EVIDENCE / arm).glob("memory-audit-rank*.json")]
        for phase in ("load_model", "determine_available_memory", "_compile_or_warm_up_model"):
            assert {record["rank"] for record in records if record["phase"] == phase} == set(range(8)), phase
        lines = log.read_text(errors="replace").splitlines()
        sizing = [line for line in lines if any(key in line for key in (
            "Memory after warmup:", "Extended KV cache", "GPU KV cache size:",
            "Available KV cache memory:", "Model loading took", "Graph capturing finished"))]
        return {"status": "passed", "snapshots": len(records), "cache_sizing_log": sizing,
                "scope": "Startup memory audit only; no throughput or numerical claim."}

    report = {"status": "running", "arms": {}, "candidate": serving.CANDIDATE,
              "base": serving.BASE, "scope": "Full-recipe startup memory inventory at GMU=0.90 and load-format=auto"}
    try:
        serving.main(report, arms=(("upstream_base", serving.BASE, "0"),
                                   ("candidate_on", serving.CANDIDATE, "1")),
                     exercise=exercise, command_transform=command_transform,
                     env_transform=env_transform)
        report["settings"].update(cpu_kv_offload=True, synthetic_acceptance=True)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (serving.EVIDENCE / "memory-validation.json").write_text(json.dumps(report, indent=2))
        print("MEMORY_VALIDATION", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
