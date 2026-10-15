# Experimental context-only preparation; installed AITER attention kernels retained.
import torch
import triton
import triton.language as tl

@triton.jit
def _pack_qkv(Q,K,V,QO,KO,VO,TQ:tl.constexpr,TK:tl.constexpr,H:tl.constexpr,HP:tl.constexpr,DQ:tl.constexpr,DV:tl.constexpr,
              QS0:tl.constexpr,QS1:tl.constexpr,QS2:tl.constexpr,KS0:tl.constexpr,KS1:tl.constexpr,KS2:tl.constexpr,
              VS0:tl.constexpr,VS1:tl.constexpr,VS2:tl.constexpr,B:tl.constexpr):
    x=tl.program_id(0)*B+tl.arange(0,B)
    which=tl.program_id(1)
    if which==0:
        c=x%DQ;h=(x//DQ)%HP;t=x//(DQ*HP)
        val=tl.load(Q+t*QS0+(h%H)*QS1+c*QS2,t<TQ,other=0)
        tl.store(QO+x,val.to(QO.dtype.element_ty),t<TQ)
    elif which==1:
        c=x%DQ;h=(x//DQ)%HP;t=x//(DQ*HP)
        val=tl.load(K+t*KS0+(h%H)*KS1+c*KS2,t<TK,other=0)
        tl.store(KO+x,val.to(KO.dtype.element_ty),t<TK)
    else:
        c=x%DV;h=(x//DV)%HP;t=x//(DV*HP)
        val=tl.load(V+t*VS0+(h%H)*VS1+c*VS2,t<TK,other=0)
        tl.store(VO+x,val.to(VO.dtype.element_ty),t<TK)

def prepare_qkv(q,k,v,hp,dtype,cache):
    h=q.shape[1]
    if k.shape[1]!=h or v.shape[1]!=h or k.shape[0]!=v.shape[0]:
        raise ValueError('Q/K/V head counts and KV lengths must agree')
    result=[]
    for name,t in zip(('q_fp8','k_fp8','v_fp8'),(q,k,v)):
        shape=(t.shape[0],hp,t.shape[-1]);buf=cache.get(name)
        if buf is None or buf.shape[0]<shape[0] or buf.shape[1:]!=shape[1:] or buf.dtype!=dtype or buf.device!=t.device:
            buf=torch.empty(shape,device=t.device,dtype=dtype);cache[name]=buf
        result.append(buf[:shape[0]])
    qo,ko,vo=result
    extent=max(qo.numel(),ko.numel(),vo.numel())
    if extent:
        _pack_qkv[(triton.cdiv(extent,1024),3)](q,k,v,qo,ko,vo,q.shape[0],k.shape[0],h,hp,q.shape[-1],v.shape[-1],
                    *q.stride(),*k.stride(),*v.stride(),1024,num_warps=4)
    return qo,ko,vo

def prepare_plan(chunk,ps,nhead,total_q,total_kv,max_q_len,is_causal):
    from aiter import get_ps_metadata_v1
    # Chunk metadata is shared across model layers. Retain immutable CPU plans
    # on that batch object; avoid repeated device->host copies and CPU planning.
    key=(nhead,str(chunk.query_start_loc.device),total_q,total_kv,int(max_q_len),bool(is_causal))
    plans=getattr(chunk,'_fp8ctx_cpu_plans',None)
    if plans is None:
        plans={};chunk._fp8ctx_cpu_plans=plans
    plan=plans.get(key)
    if plan is None:
        qo=chunk.query_start_loc.detach().to(device='cpu',dtype=torch.int32)
        kv=chunk.cu_seq_lens.detach().to(device='cpu',dtype=torch.int32)
        if int(qo[-1])!=total_q or int(kv[-1])!=total_kv:
            raise ValueError('Context metadata does not match Q/KV lengths')
        host=[torch.empty_like(t,device='cpu',pin_memory=True) for t in ps['host']]
        get_ps_metadata_v1(qo,kv,kv[1:]-kv[:-1],1,nhead,ps['work_metadata'],*host,
                           qhead_granularity=1,qlen_granularity=256,kvlen_granularity=128,block_size=1,is_causal=is_causal,need_lse=True)
        plan=(host,qo,kv);plans[key]=plan
    # Host tensors remain alive and immutable until the batch completes.
    # Device destinations belong to this backend and are ordered on its stream.
    for dst,host in zip(ps['dev'],plan[0]):
        dst.copy_(host,non_blocking=True)
    return plan
