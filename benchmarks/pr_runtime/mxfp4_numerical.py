"""MXFP4 paged scores versus an independent quantized Torch reference."""
import json
from pathlib import Path
import torch
from vllm.models.kimi_k3.amd.ops.fp4_indexer import score_paged,store_keys

def quantized(x):
    groups=x.float().reshape(*x.shape[:-1],4,32)
    scale=torch.pow(2.,torch.ceil(torch.log2((groups.abs().amax(-1)/6).clamp_min(1e-4))))
    y=groups/scale[...,None]
    # Prefer even E2M1 codes on equal-distance ties.
    levels=torch.tensor([0.,1.,2.,4.,.5,1.5,3.,6.],device=x.device)
    selected=(y.abs()[...,None]-levels).abs().argmin(-1)
    return (levels[selected]*torch.sign(y)*scale[...,None]).reshape(x.shape)

def main():
    torch.manual_seed(67372)
    records=[]
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx950')
    for heads in [16,32,64]:
        q=torch.randn(3,heads,128,device='cuda',dtype=torch.bfloat16)
        keys=torch.randn(320,128,device='cuda',dtype=torch.bfloat16)
        weights=torch.rand(3,heads,device='cuda')/heads
        table=torch.tensor([[3,1,0],[4,0,2]],device='cuda',dtype=torch.int32)
        starts=torch.tensor([0,1,3],device='cuda',dtype=torch.int32)
        ends=torch.tensor([129,70,131],device='cuda',dtype=torch.int32)
        refs=[]
        qref,kref=quantized(q),quantized(keys)
        for row,request in enumerate([0,1,1]):
            ordered=kref[(table[request,:,None].long()*64+torch.arange(64,device='cuda')[None,:]).reshape(-1)]
            refs.append((torch.relu(qref[row]@ordered.T)*weights[row,:,None]).sum(0))
        reference=torch.stack(refs)
        for stride,offset in [(4352,0),(46592,41984)]:
            backing=torch.full((5,stride),173,device='cuda',dtype=torch.uint8)
            cache=backing[:,offset:offset+4352].view(5,64,68)
            store_keys(keys,cache,torch.arange(320,device='cuda'))
            for mode in ['eager','graph']:
                for _ in range(3):score_paged(q,cache,weights,table,starts,ends,2)
                if mode=='graph':
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):actual=score_paged(q,cache,weights,table,starts,ends,2)
                    graph.replay()
                else:actual=score_paged(q,cache,weights,table,starts,ends,2)
                valid=torch.arange(192,device='cuda')[None,:]<ends[:,None]
                error=(actual[valid]-reference[valid]).abs()
                torch.testing.assert_close(actual[valid],reference[valid],rtol=1e-4,atol=2e-4)
                assert bool((backing[:,:offset]==173).all())
                assert bool((backing[:,offset+4352:]==173).all())
                records.append(dict(heads=heads,physical_page_stride=stride,offset=offset,mode=mode,max_abs_error=error.max().item(),rtol=1e-4,atol=2e-4,causal_bounds=ends.tolist(),status='passed'))
    Path('/logs/numerical-validation.json').write_text(json.dumps(dict(status='passed',cases=records,limits='One-GPU paged scorer correctness; distributed merge and full model quality remain separate'),indent=2)+'\n')

if __name__=='__main__':main()
