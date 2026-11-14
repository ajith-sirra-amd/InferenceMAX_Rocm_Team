"""Locate native cached/uncached divergence without changing acceptance tests."""
import importlib.util,json
from pathlib import Path
import torch
from vllm.platforms import current_platform

spec=importlib.util.spec_from_file_location('native_test','/vllm-workspace/tests/models/kimi_k3/test_amd_attn_res.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
records=[]
for tokens in [280,336]:
    for quantized,mixed in [(False,False),(True,False),(True,True)]:
        torch.manual_seed(20261113)
        hidden=7168
        prefix=native._randn_with_row_padding(tokens,hidden,padding=7)
        reference_prefix=prefix.clone()
        delta=native._randn_with_row_padding(tokens,hidden,padding=11)
        blocks=native._randn_with_row_padding(tokens,8,hidden,padding=13)
        reference_blocks=blocks.clone()
        weights=torch.randn(8,3,hidden,device='cuda',dtype=torch.bfloat16)
        weights[:,1]/=hidden**0.5
        cache=torch.full((tokens,8),float('nan'),device='cuda')
        dtype=current_platform.fp8_dtype() if quantized else None
        for iteration in range(2):
            prefix.normal_();reference_prefix.copy_(prefix);cache[:,0]=float('nan')
            expected=native._norm_cache_sequence(reference_prefix,delta,reference_blocks,weights,None,dtype,mixed)
            actual=native._norm_cache_sequence(prefix,delta,blocks,weights,cache,dtype,mixed)
            differences=[]
            for index,(a,b) in enumerate(zip(actual,expected)):
                x,y=a.float(),b.float();mask=x!=y
                count=int(mask.sum().item())
                if count:
                    differences.append(dict(output_index=index,shape=list(a.shape),dtype=str(a.dtype),mismatches=count,elements=a.numel(),max_absolute=float((x-y).abs().max().item()),finite=bool(torch.isfinite(x).all())))
            records.append(dict(tokens=tokens,quantized=quantized,mixed_output=mixed,iteration=iteration,differences=differences,prefix_equal=torch.equal(prefix,reference_prefix),bank_equal=torch.equal(blocks,reference_blocks),cache_block0_nan=bool(torch.isnan(cache[:,0]).all()),cache_other_finite=bool(torch.isfinite(cache[:,1:]).all())))
assert len(records)==12 and all(r['differences'] for r in records), 'Original failure did not reproduce in all cases'
Path('/logs/attnres-diagnosis.json').write_text(json.dumps(dict(status='reproduced_numerical_divergence',records=records,acceptance='failed_original_bitwise_gate',limits='Diagnostic localization; no tolerance relaxation, model quality or performance claim'),indent=2)+'\n')
