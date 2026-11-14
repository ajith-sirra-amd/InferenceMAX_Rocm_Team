"""Native TP8 BF16 projection/gather/RMSNorm check; not a model benchmark."""
import hashlib
import json
import os
from pathlib import Path

import torch


def main():
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    output = Path("/logs/projection")
    output.mkdir(exist_ok=True)
    record = dict(rank=rank, status="failed", cases=[], limits="Native layers only; not full model activation or accuracy")
    try:
        assert int(os.environ["WORLD_SIZE"]) == torch.cuda.device_count() == 8
        torch.cuda.set_device(local)
        torch.set_num_threads(4)
        props = torch.cuda.get_device_properties(local)
        assert props.gcnArchName.startswith("gfx950"), props
        from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config
        from vllm.distributed.parallel_state import (
            init_distributed_environment, initialize_model_parallel, get_tp_group,
        )
        from vllm.model_executor.layers.linear import ReplicatedLinear, ColumnParallelLinear
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.models.utils import AutoWeightsLoader
        from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM
        import vllm.models.kimi_k3.nvidia.dspark_mla as module

        record.update(source_sha=os.environ["EXPECTED_SOURCE_SHA"],
                      module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                      arch=props.gcnArchName, dtype="torch.bfloat16", input_width=35840, output_width=7168,
                      # Two BF16 machine epsilons, fixed before execution, for
                      # two rounded stages. Raw maximum errors are retained.
                      rtol=2 * torch.finfo(torch.bfloat16).eps,
                      atol=2 * torch.finfo(torch.bfloat16).eps)
        assert record["module_sha256"] == os.environ["EXPECTED_MODULE_SHA"]
        config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=8, decode_context_parallel_size=8))
        with set_current_vllm_config(config):
            init_distributed_environment(8, rank, "env://", local, backend="nccl")
            initialize_model_parallel(8, 1, 1, 8)
            with torch.device(f"cuda:{local}"):
                reference = ReplicatedLinear(35840, 7168, bias=False, return_bias=False, params_dtype=torch.bfloat16)
                candidate = ColumnParallelLinear(35840, 7168, bias=False, return_bias=False,
                                                 params_dtype=torch.bfloat16, gather_output=True)
                norm = RMSNorm(7168, eps=1e-6, dtype=torch.bfloat16)
            assert list(candidate.weight.shape) == [896, 35840]
            assert candidate.gather_output
            models = [torch.nn.ModuleDict({"model": torch.nn.ModuleDict({"context_proj": layer})})
                      for layer in (reference, candidate)]
            generator = torch.Generator(device="cpu").manual_seed(67372)
            weight = torch.randn((7168, 35840), generator=generator, device="cpu")
            weight.mul_(35840 ** -0.5)
            weight = weight.to(torch.bfloat16)
            for reload_index in range(2):
                checkpoint = weight if reload_index == 0 else weight.flip(0)
                for model, layer in zip(models, (reference, candidate)):
                    loaded = AutoWeightsLoader(model).load_weights(
                        [("context_proj.weight", checkpoint)], mapper=K3DSparkForCausalLM.hf_to_vllm_mapper)
                    assert loaded == {"model.context_proj.weight"}
                    layer.quant_method.process_weights_after_loading(layer)
                assert torch.equal(candidate.weight, checkpoint[rank * 896:(rank + 1) * 896].to(candidate.weight.device))
                for rows in (1, 7, 48, 280, 336):
                    generator = torch.Generator(device="cpu").manual_seed(rows + 91)
                    inputs = torch.randn((rows, 35840), generator=generator, device="cpu").to(
                        device=f"cuda:{local}", dtype=torch.bfloat16)
                    digest = hashlib.sha256(inputs.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                    digests = [None] * 8
                    torch.distributed.all_gather_object(digests, digest)
                    assert len(set(digests)) == 1, "Context rows differ across TP ranks"
                    for _ in range(3):
                        expected = norm(reference(inputs))
                        actual = norm(candidate(inputs))
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual, expected, rtol=record["rtol"], atol=record["atol"])
                    eager_error = (actual.float() - expected.float()).abs().max().item()
                    graph = torch.cuda.CUDAGraph()
                    with get_tp_group().graph_capture() as ctx:
                        with torch.cuda.graph(graph, stream=ctx.stream):
                            graph_expected = norm(reference(inputs))
                            graph_actual = norm(candidate(inputs))
                    for _ in range(2):
                        graph.replay()
                    torch.cuda.synchronize()
                    assert torch.isfinite(graph_actual).all().item()
                    torch.testing.assert_close(graph_actual, graph_expected, rtol=record["rtol"], atol=record["atol"])
                    torch.testing.assert_close(graph_actual, actual, rtol=0, atol=0)
                    record["cases"].append(dict(rows=rows, checkpoint=reload_index,
                        eager_max_abs_error=eager_error,
                        graph_max_abs_error=(graph_actual.float() - graph_expected.float()).abs().max().item(),
                        graph_eager_bitwise_equal=True, graph_recaptured_after_reload=True))
                    print(f"rank={rank} rows={rows} checkpoint={reload_index} passed", flush=True)
                    (output / f"rank-{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
            record["status"] = "passed"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / f"rank-{rank}.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
