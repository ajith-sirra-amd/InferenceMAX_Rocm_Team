"""Read-only loaded draft provenance via the native worker extension."""
class ModelWorkerExtension:
    def init_worker_extension(self):
        import hashlib
        import json
        import os
        from pathlib import Path
        import torch
        import vllm.models.kimi_k3.nvidia.dspark_mla as module
        from vllm.distributed import get_tensor_model_parallel_rank
        original = module.K3DSparkForCausalLM.load_weights

        def load_weights(model, *args, **kwargs):
            result = original(model, *args, **kwargs)
            rank = get_tensor_model_parallel_rank()
            layer = model.model.context_proj
            weight = layer.weight
            expected = [896, 35840] if os.environ['IPAT_ARM'] == 'candidate' else [7168, 35840]
            assert list(weight.shape) == expected, (rank, weight.shape, expected)
            assert weight.dtype == torch.bfloat16
            digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            assert digest == os.environ['EXPECTED_MODULE_SHA']
            record = dict(rank=rank, source_sha=os.environ['EXPECTED_SOURCE_SHA'],
                          module_path=module.__file__, module_sha256=digest,
                          layer_type=type(layer).__name__, weight_shape=list(weight.shape),
                          weight_dtype=str(weight.dtype), gather_output=getattr(layer, 'gather_output', False),
                          weight_sha256=hashlib.sha256(weight.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),
                          status='loaded', limits='Loaded draft provenance; no forward equivalence or performance claim')
            Path(f'/logs/rank-{rank}.json').write_text(json.dumps(record, indent=2)+'\n')
            return result

        module.K3DSparkForCausalLM.load_weights = load_weights
