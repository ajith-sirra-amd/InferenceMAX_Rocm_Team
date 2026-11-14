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
        import vllm.models.kimi_k3.amd.dsa as dsa
        indexer_init = dsa.KimiK3DSAIndexer.__init__
        def init_indexer(indexer, *args, **kwargs):
            indexer_init(indexer, *args, **kwargs)
            assert indexer.use_fp4, 'MXFP4 recipe silently fell back'
            record = dict(rank=get_tensor_model_parallel_rank(), use_fp4=indexer.use_fp4,
                          heads=indexer.index_num_heads, head_dim=indexer.index_head_dim,
                          source_sha256=hashlib.sha256(Path(dsa.__file__).read_bytes()).hexdigest())
            with Path(f"/logs/mxfp4-rank-{record['rank']}.jsonl").open('a') as sink:
                sink.write(json.dumps(record)+'\n')
        dsa.KimiK3DSAIndexer.__init__ = init_indexer

        def load_weights(model, *args, **kwargs):
            result = original(model, *args, **kwargs)
            rank = get_tensor_model_parallel_rank()
            layer = model.model.context_proj
            weight = layer.weight
            expected = [896, 35840] if os.environ['IPAT_PR_NUMBER'] == '67372' and os.environ['IPAT_ARM'] == 'candidate' else [7168, 35840]
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
        if os.environ['IPAT_PR_NUMBER'] == '67373':
            import vllm.models.kimi_k3.amd.linear as linear
            model_init = linear.KimiLinearModel.__init__
            def init_model(model, *args, **kwargs):
                model_init(model, *args, **kwargs)
                enabled = getattr(model, 'cache_attn_res_norms', False)
                assert enabled == (os.environ['IPAT_ARM'] == 'candidate')
                rank = get_tensor_model_parallel_rank()
                Path(f'/logs/attnres-rank-{rank}.json').write_text(json.dumps(dict(
                    rank=rank, cache_attn_res_norms=enabled,
                    source_sha256=hashlib.sha256(Path(linear.__file__).read_bytes()).hexdigest(),
                    limits='Constructor admission; does not establish graph execution coverage'))+'\n')
            linear.KimiLinearModel.__init__ = init_model
