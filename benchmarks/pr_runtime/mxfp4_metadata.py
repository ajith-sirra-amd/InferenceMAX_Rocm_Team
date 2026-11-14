"""Qualify native global/local prefill bounds on each of eight GPU ranks."""
import json,os
from pathlib import Path
import torch
from vllm.config import VllmConfig,ParallelConfig,set_current_vllm_config
from vllm.v1.attention.backends.mla.indexer import build_prefill_chunk_metadata
from vllm.models.kimi_k3.amd.ops.fp4_indexer_prefill import prepare_prefill_metadata
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerPrefillMetadata

def main():
    rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    assert int(os.environ['WORLD_SIZE'])==8
    cpu=torch.tensor([0,3,8],dtype=torch.int32)
    qs=cpu.cuda();lens=torch.tensor([70,129],device='cuda',dtype=torch.int32)
    table=torch.tensor([[3,1,0],[4,0,2]],device='cuda',dtype=torch.int32)
    lens_cpu=lens.cpu()
    cases=[]
    config=VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=8,decode_context_parallel_size=8))
    with set_current_vllm_config(config):
        for interleave in [1,64]:
            counts=[]
            for length in [70,129]:
                base=length//(8*interleave)*interleave
                counts.append(base+max(0,min(interleave,length-base*8-rank*interleave)))
            for sliced in [False,True]:
                query_slice=slice(2,7) if sliced else None
                torch.cuda.set_sync_debug_mode('error')
                try:
                    chunk=build_prefill_chunk_metadata(0,2,qs,cpu,lens,lens,lens_cpu,table,1,query_slice=query_slice,dcp_rank=rank,dcp_world_size=8,cp_kv_cache_interleave_size=interleave,global_row_bounds=True)
                finally:
                    torch.cuda.set_sync_debug_mode('default')
                first,last=(2,7) if sliced else (0,8)
                expected_ks=[];expected_ke=[];local_ks=[];local_ke=[]
                for request,(length,qlen) in enumerate([(70,3),(129,5)]):
                    for j in range(qlen):
                        end=length-qlen+j+1
                        expected_ks.append(0 if request==0 else 70)
                        expected_ke.append(expected_ks[-1]+end)
                        base=end//(8*interleave)*interleave
                        local=base+max(0,min(interleave,end-base*8-rank*interleave))
                        local_ks.append(0 if request==0 else counts[0])
                        local_ke.append(local_ks[-1]+local)
                for actual,expected in [(chunk.global_cu_seqlen_ks,expected_ks),(chunk.global_cu_seqlen_ke,expected_ke),(chunk.cu_seqlen_ks,local_ks),(chunk.cu_seqlen_ke,local_ke)]:
                    assert actual.cpu().tolist()==expected[first:last],(rank,interleave,actual,expected)
                assert chunk.local_total_seq_lens==sum(counts)
                all_totals=[sum(length//(8*interleave)*interleave+max(0,min(interleave,length-(length//(8*interleave)*interleave)*8-other*interleave)) for length in [70,129]) for other in range(8)]
                assert chunk.max_local_total_seq_lens==max(all_totals)
                prepared=prepare_prefill_metadata(DeepseekV32IndexerPrefillMetadata([chunk]),8,rank,interleave)
                assert len(prepared.plans)==1
                cases.append(dict(interleave=interleave,sliced=sliced,status='passed'))
    out=Path('/logs/metadata');out.mkdir(exist_ok=True)
    (out/f'rank-{rank}.json').write_text(json.dumps(dict(status='passed',rank=rank,cases=cases,limits='Native metadata and FP4 plan on eight GPU ranks; full collective/model correctness remains pending'),indent=2)+'\n')

if __name__=='__main__':main()
