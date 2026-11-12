#!/usr/bin/env python3
"""Bounded native three-projection versus merged MoE-front GPU comparison.

One GPU, synthetic BF16 weights/inputs, actual TP8 local widths. Includes SiTU
and native router top-k; excludes experts, shared down projection and collectives.
Never modifies installed packages or loads the model. Timing is not request E2E.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

HIDDEN, SHARED, EXPERTS, LATENT, TOPK = 7168, 768, 896, 3584, 16
WIDTHS = (2 * SHARED, EXPERTS, LATENT)
SOURCE_HASHES = {
    "vllm/model_executor/layers/linear.py": "d4b97d850eb8bdee5fd2459e7ac688d00623b54f5ec0ea590c08da0173f8b4d4",
    "vllm/model_executor/layers/utils.py": "571680e80668de63954f35830dd5f2dd617b0b56f339a7ebec314d00498a3c60",
    "vllm/model_executor/layers/activation.py": "bebeddbd4997f49f54471ee31ac42b22c3c11dfe25244e55d20c36f390489c05",
    "vllm/model_executor/layers/fused_moe/router/gate_linear.py": "d2a490dd1b084b2e783c2ca7d91b58164310e471ecc1f17df3ee9d6382e856f3",
    "vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py": "add3764cec2bb08d121610673b06a09e89e78c11ea45de8d4ab1c1cd24cfcf71",
    "vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py": "08c8fd04cfa8ce7bb7f33203cf38adeb4c815c0b9c868b6a1571bf94e46752ad",
    "aiter/ops/triton/moe/moe_situ_epilogue.py": "4d78e400fa743562e851c43c126a88154fccc7a156d76a3047f4035a761d68cf",
    "aiter/ops/triton/_triton_kernels/moe/moe_situ_epilogue.py": "8ba913f456776def5718459564816bfce328a4adb7fd9950c4c95933ffc4d502",
}
OUTPUT_NAMES = (
    "shared_bf16",
    "router_fp32",
    "routed_bf16",
    "route_weights_fp32",
    "route_ids_int32",
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", default="287,336")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--max-seconds", type=int, default=300)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Equivalent pinned site-packages for source-only checks",
    )
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.rows = [int(value) for value in args.rows.split(",")]
    except ValueError:
        parser.error("Rows must be comma-separated integers")
    if not (
        1 <= len(args.rows) <= 16
        and len(args.rows) == len(set(args.rows))
        and all(256 < value <= 4096 for value in args.rows)
    ):
        parser.error(
            "Choose one to sixteen unique rows in 257..4096; smaller rows retain native shared-stream overlap"
        )
    if not 6 <= args.rounds <= 60 or not 30 <= args.max_seconds <= 540:
        parser.error("Rounds must be 6..60 and max-seconds 30..540")
    if args.source_root and not args.source_only:
        parser.error(
            "--source-root is only for --source-only; GPU runs inspect their installed sources"
        )
    return args


def save(folder, result, event):
    result["elapsed_seconds"] = time.monotonic() - result["_started"]
    result.setdefault("progress", []).append(
        {"event": event, "elapsed_seconds": result["elapsed_seconds"]}
    )
    temporary = folder / "result.json.tmp"
    temporary.write_text(
        json.dumps(
            {k: v for k, v in result.items() if k != "_started"},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    temporary.replace(folder / "result.json")
    print(json.dumps(result["progress"][-1]), flush=True)


def inspect_sources(root, folder):
    if root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise RuntimeError("Pinned vLLM is not installed")
        root = Path(spec.origin).parent.parent
    evidence = {}
    for relative, expected in SOURCE_HASHES.items():
        path = root / relative
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError(f"Pinned source mismatch: {path}: {actual}")
        target = folder / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        evidence[relative] = {"path": str(path), "sha256": actual}
    return evidence


def differences(torch, actual, reference):
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise AssertionError("Output shape or dtype differs")
    a, b = actual.double(), reference.double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    result = {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "all_finite": finite,
        "bitwise_equal": bool(
            torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))
        ),
    }
    if finite:
        delta = a - b
        norm = b.square().sum().sqrt().item()
        result.update(
            max_abs=delta.abs().max().item(),
            relative_l2=delta.square().sum().sqrt().item() / max(norm, 1e-30),
            unequal_elements=int((a != b).sum().item()),
        )
    return result


def compare_outputs(torch, actual, reference):
    if len(actual) != len(OUTPUT_NAMES) or len(reference) != len(OUTPUT_NAMES):
        raise AssertionError("Incomplete front/router output tuple")
    report = {
        name: differences(torch, a, b)
        for name, a, b in zip(OUTPUT_NAMES, actual, reference)
    }
    aid, bid = actual[-1], reference[-1]
    report["router_selection"] = {
        "ordered_mismatch_rows": int((aid != bid).any(dim=1).sum().item()),
        "set_mismatch_rows": int(
            (aid.sort(dim=1).values != bid.sort(dim=1).values).any(dim=1).sum().item()
        ),
    }
    return report


def exact_outputs(report):
    return all(
        report[name]["all_finite"] and report[name]["bitwise_equal"]
        for name in OUTPUT_NAMES
    )


def summary(values):
    return {
        "samples_us": values,
        "count": len(values),
        "mean_us": statistics.mean(values),
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
        "std_us": statistics.pstdev(values),
    }


def paired_timing(torch, operations, rounds):
    samples = {name: [] for name in operations}
    events = {name: [] for name in operations}
    names = list(operations)
    orders = []
    for index in range(rounds):
        order = names if index % 2 == 0 else names[::-1]
        orders.append(order)
        for name in order:
            start, stop = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            operations[name]()
            stop.record()
            events[name].append((start, stop))
    torch.cuda.synchronize()
    for name, pairs in events.items():
        samples[name] = [start.elapsed_time(stop) * 1000 for start, stop in pairs]
    result = {name: summary(values) for name, values in samples.items()}
    result["orders"] = orders
    result["mean_reduction_pct"] = 100 * (
        1 - result["merged"]["mean_us"] / result["native"]["mean_us"]
    )
    result["median_reduction_pct"] = 100 * (
        1 - result["merged"]["median_us"] / result["native"]["median_us"]
    )
    return result


def run_gpu(args, result):
    import torch
    from aiter.ops.triton.moe.moe_situ_epilogue import moe_situ_epilogue
    from vllm import envs
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationConfig
    from vllm.model_executor.layers.activation import SituAndMul

    # Import registers the exact GateLinear router custom op used below.
    from vllm.model_executor.layers.fused_moe.router import gate_linear
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        GroupedTopk,
    )
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("A ROCm GPU is required")
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    if "MI355X" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError(
            "This probe is qualified only for the MI355X source/geometry"
        )
    if envs.VLLM_BATCH_INVARIANT:
        raise RuntimeError(
            "Native merged-front admission excludes batch-invariant mode"
        )
    if not envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM and any(
        row <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD for row in args.rows
    ):
        raise RuntimeError(
            "Selected rows use native shared-expert overlap; serial baseline would not match"
        )
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=0, cudagraph_mode="FULL_AND_PIECEWISE", custom_ops=["all"]
        )
    )
    config.kernel_config.set_platform_defaults(config)
    config.kernel_config.ir_op_priority.set_default()
    result["runtime"] = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0),
        "ir_priority": str(config.kernel_config.ir_op_priority),
        "compilation_mode": str(config.compilation_config.mode),
        "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
    }
    result["gpu_executed"] = True
    with set_current_vllm_config(config), torch.inference_mode():
        method = UnquantizedLinearMethod()
        activation = SituAndMul(beta=4, linear_beta=25, compile_native=False)
        router = GroupedTopk(
            topk=TOPK,
            renormalize=True,
            num_expert_group=1,
            topk_group=1,
            scoring_func="sigmoid",
            routed_scaling_factor=1.0,
        )
        result["native_dispatch"] = {
            "linear": method._gemm_impl.__qualname__,
            "activation": activation._forward_method.__qualname__,
            "router_topk": router._forward_method.__qualname__,
            "router_gemm": gate_linear.fp32_router_gemm_dispatch_impl.__qualname__,
        }
        save(args.output, result, "native dispatch initialized")
        gen = torch.Generator(device="cuda").manual_seed(66736)
        weights = [
            torch.randn(
                (width, HIDDEN), device="cuda", dtype=torch.bfloat16, generator=gen
            ).div_(math.sqrt(HIDDEN))
            for width in WIDTHS
        ]
        merged_weight = torch.cat(weights, dim=0)
        layers = [SimpleNamespace(weight=weight) for weight in weights]
        correction_bias = (
            torch.randn((EXPERTS,), device="cuda", dtype=torch.float32, generator=gen)
            * 0.01
        )
        for row in args.rows:
            case = {"rows": row, "status": "running", "checks": {}}
            result["cases"].append(case)
            save(args.output, result, f"M{row}: setup")
            x = torch.randn(
                (row, HIDDEN), device="cuda", dtype=torch.bfloat16, generator=gen
            )
            original = x.clone()

            def finish(shared, logits, latent):
                route_weights, route_ids = router(latent, logits, correction_bias)
                return shared, logits, latent, route_weights, route_ids

            def native(x=x):
                logits = torch.ops.vllm.fp32_router_gemm_dispatch(x, weights[1], False)
                latent = method.apply(layers[2], x)
                shared = activation(method.apply(layers[0], x))
                return finish(shared, logits, latent)

            def merged(x=x):
                projection = torch.mm(x, merged_weight.t(), out_dtype=torch.float32)
                return finish(
                    *moe_situ_epilogue(
                        projection,
                        shared_intermediate_size=SHARED,
                        num_experts=EXPERTS,
                        routed_latent_size=LATENT,
                        situ_beta=4.0,
                        situ_linear_beta=25.0,
                    )
                )

            operations = {"native": native, "merged": merged}
            for operation in operations.values():
                for _ in range(3):
                    operation()
            torch.cuda.synchronize()
            references = {name: operation() for name, operation in operations.items()}
            case["checks"]["eager_candidate_vs_native"] = compare_outputs(
                torch, references["merged"], references["native"]
            )
            case["checks"]["baseline_repeat"] = compare_outputs(
                torch, native(), references["native"]
            )
            case["checks"]["candidate_repeat"] = compare_outputs(
                torch, merged(), references["merged"]
            )
            baseline_scores = references["native"][1].sigmoid() + correction_bias
            ordered = baseline_scores.topk(TOPK + 1, dim=1).values
            case["minimum_router_16th_17th_margin"] = (
                (ordered[:, TOPK - 1] - ordered[:, TOPK]).min().item()
            )
            if not all(
                case["checks"]["eager_candidate_vs_native"][name]["all_finite"]
                for name in OUTPUT_NAMES
            ):
                raise RuntimeError(
                    "Nonfinite output; preserving evidence without timing"
                )
            save(args.output, result, f"M{row}: eager numerical/routing evidence saved")

            graphs, captured = {}, {}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            for name, operation in operations.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=side):
                    captured[name] = operation()
                graphs[name] = graph
            torch.cuda.synchronize()
            for name, graph in graphs.items():
                graph.replay()
            torch.cuda.synchronize()
            for name in operations:
                case["checks"][f"graph_original_{name}_vs_own_eager"] = compare_outputs(
                    torch, captured[name], references[name]
                )

            x.copy_((original.float() * -0.75 + 0.0625).to(torch.bfloat16))
            changed = {name: operation() for name, operation in operations.items()}
            case["checks"]["changed_candidate_vs_native"] = compare_outputs(
                torch, changed["merged"], changed["native"]
            )
            for name, graph in graphs.items():
                graph.replay()
            torch.cuda.synchronize()
            for name in operations:
                case["checks"][f"graph_changed_{name}_vs_own_eager"] = compare_outputs(
                    torch, captured[name], changed[name]
                )
                case["checks"][f"input_change_{name}"] = differences(
                    torch, changed[name][0], references[name][0]
                )
            own_keys = [
                key
                for key in case["checks"]
                if "vs_own_eager" in key or key.endswith("_repeat")
            ]
            case["own_eager_graph_exact"] = all(
                exact_outputs(case["checks"][key]) for key in own_keys
            )
            case["changed_input_observed"] = all(
                case["checks"][f"input_change_{name}"]["max_abs"] > 0
                for name in operations
            )
            case["strict_candidate_parity"] = all(
                exact_outputs(case["checks"][key])
                for key in ("eager_candidate_vs_native", "changed_candidate_vs_native")
            )
            save(args.output, result, f"M{row}: changed-input graph evidence saved")
            if not case["own_eager_graph_exact"] or not case["changed_input_observed"]:
                raise RuntimeError(
                    "Replay differs from its own eager implementation or ignores changed inputs"
                )
            x.copy_(original)
            torch.cuda.synchronize()
            case["eager_timing"] = paired_timing(torch, operations, args.rounds)
            case["graph_timing"] = paired_timing(
                torch,
                {name: graph.replay for name, graph in graphs.items()},
                args.rounds,
            )
            case["status"] = "completed"
            save(args.output, result, f"M{row}: complete timings saved")
            del graphs, captured, references, changed
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()


def main(argv=None):
    args = arguments(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        raise FileExistsError(
            "Use a fresh run directory; existing results are preserved"
        )
    sys.dont_write_bytecode = True
    result = {
        "status": "running",
        "_started": time.monotonic(),
        "scope": __doc__,
        "cases": [],
        "gpu_executed": False,
        "rows": args.rows,
        "rounds": args.rounds,
        "geometry": {"input_width": HIDDEN, "projection_widths": WIDTHS, "topk": TOPK},
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("VLLM_", "AITER_", "HSA_", "HIP_", "ROCPROF"))
            or key in ("LD_PRELOAD", "LD_AUDIT", "PYTHONHASHSEED")
        },
        "limits": [
            "Synthetic weights and activations, no captured model tensors or checkpoint loading",
            "One GPU; actual TP8 local projection sizes, no distributed expert or tail work",
            "Warm fixed-buffer event intervals; no profiler, HBM counters, or request E2E",
            "Collection success is not numerical equivalence: inspect candidate output and router differences",
            "No arbitrary tolerance or quality approval; graph replay must match its own eager outputs exactly",
            "Merged weight prepared once before timing; fixed weights, no reload/migration lifecycle test",
        ],
    }
    save(args.output, result, "created")

    def deadline(_signum, _frame):
        raise TimeoutError(
            f"Exceeded {args.max_seconds}s; blocked GPU operations may delay signal delivery"
        )

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        result["sources"] = inspect_sources(args.source_root, args.output)
        save(args.output, result, "pinned source hashes verified")
        if args.source_only:
            result["status"] = "source_check_passed"
        else:
            run_gpu(args, result)
            result["status"] = "completed"
            result["strict_candidate_parity"] = all(
                case["strict_candidate_parity"] for case in result["cases"]
            )
    except Exception as error:  # noqa: BLE001 -- Retain numerical/GPU failure evidence before exiting.
        result.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
    finally:
        signal.alarm(0)
        save(args.output, result, "finished")
    return 0 if result["status"] in ("completed", "source_check_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
