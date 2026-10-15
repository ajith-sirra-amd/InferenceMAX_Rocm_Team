"""Experimental FP8 cache -> BF16 dot arithmetic -> FP8 projected KV.
No BF16 latent workspace or expanded BF16 KV is materialized.
Installed AITER attention/reduction kernels are unchanged.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _project_fp8_cache(X,W,S,Y,ROPE,T:tl.constexpr,L:tl.constexpr,R:tl.constexpr,H:tl.constexpr,P:tl.constexpr,V:tl.constexpr,
                       XS:tl.constexpr,WS0:tl.constexpr,WS1:tl.constexpr,BM:tl.constexpr=16,BN:tl.constexpr=64,BK:tl.constexpr=64):
    mi=tl.program_id(0)*BM+tl.arange(0,BM)
    ni=tl.program_id(1)*BN+tl.arange(0,BN)
    ki=tl.arange(0,BK)
    scale=tl.load(S).to(tl.float32)
    acc=tl.zeros((BM,BN),tl.float32)
    for base in range(tl.cdiv(L,BK)):
        kk=base*BK+ki
        x=tl.load(X+mi[:,None]*XS+kk[None,:],(mi[:,None]<T)&(kk[None,:]<L),other=0).to(tl.float32)
        # Match the baseline gather's BF16 rounding before projection.
        x=(x*scale).to(tl.bfloat16)
        w=tl.load(W+ni[None,:]*WS0+kk[:,None]*WS1,(ni[None,:]<H*(P+V))&(kk[:,None]<L),other=0).to(tl.bfloat16)
        acc+=tl.dot(x,w)
    # Match the projection's BF16 output rounding before direct FP8 conversion.
    y=acc.to(tl.bfloat16)
    tl.store(Y+mi[:,None]*(H*(P+V))+ni[None,:],y.to(Y.dtype.element_ty),(mi[:,None]<T)&(ni[None,:]<H*(P+V)))
    if tl.program_id(1)==0:
        ri=tl.arange(0,triton.next_power_of_2(R))
        pe=tl.load(X+mi[:,None]*XS+L+ri[None,:],(mi[:,None]<T)&(ri[None,:]<R),other=0).to(tl.float32)
        pe=(pe*scale).to(tl.bfloat16)
        tl.store(ROPE+mi[:,None]*R+ri[None,:],pe.to(ROPE.dtype.element_ty),(mi[:,None]<T)&(ri[None,:]<R))

@triton.jit
def _concat_fp8_k(KV,PE,K,T:tl.constexpr,H:tl.constexpr,P:tl.constexpr,V:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    c=i%(P+R);h=(i//(P+R))%H;t=i//((P+R)*H)
    nope=tl.load(KV+t*(H*(P+V))+h*(P+V)+c,(t<T)&(c<P),other=0)
    rope=tl.load(PE+t*R+c-P,(t<T)&(c>=P),other=0)
    tl.store(K+i,tl.where(c<P,nope,rope),t<T)

def gather_workspace(impl,original,chunk):
    # Preserve original row budgeting; change only the element representation.
    key=(tuple(original.shape),str(original.device))
    if getattr(impl,'_fp8raw_workspace_key',None)!=key:
        impl._fp8raw_workspace=torch.empty(original.shape,dtype=torch.uint8,device=original.device)
        impl._fp8raw_workspace_key=key
    return impl._fp8raw_workspace

def project(impl,latent,pe,scale,chunk):
    from vllm.platforms import current_platform
    from vllm.model_executor.layers.quantization.utils.quant_utils import get_and_maybe_dequant_weights
    if scale.numel()!=1:
        raise ValueError('Raw FP8 prototype requires scalar cache scale')
    fp8=current_platform.fp8_dtype()
    # Reinterpret raw bytes; do not numerically cast uint8 into FP8.
    if scale.device != latent.device:
        raise ValueError('Cache scale must be GPU resident')
    x=latent.view(fp8).squeeze(1);rp=pe.view(fp8).squeeze(1)
    h,p,v=impl.num_heads,impl.qk_nope_head_dim,impl.v_head_dim
    t,l=x.shape;r=rp.shape[-1]
    w=getattr(impl,'_fp8raw_weight',None)
    if w is None:
        w=get_and_maybe_dequant_weights(impl.kv_b_proj,out_dtype=torch.bfloat16).contiguous()
        if w.shape!=(h*(p+v),l):raise ValueError(f'Unsupported projection weight shape {w.shape}')
        if getattr(impl.kv_b_proj,'bias',None) is not None:raise ValueError('Biased projection unsupported')
        impl._fp8raw_weight=w
    # Latent and positional slices arise from separate cat outputs. Join bytes
    # for the experimental projection; no BF16 intermediate is introduced.
    packed=torch.cat((x,rp),dim=-1)
    kv=torch.empty((t,h,p+v),dtype=fp8,device=x.device)
    rope=torch.empty((t,r),dtype=fp8,device=x.device)
    k=torch.empty((t,h,p+r),dtype=fp8,device=x.device)
    def candidate():
        _project_fp8_cache[(triton.cdiv(t,16),triton.cdiv(h*(p+v),64))](packed,w,scale,kv,rope,t,l,r,h,p,v,packed.stride(0),*w.stride())
        _concat_fp8_k[(triton.cdiv(k.numel(),1024),)](kv,rope,k,t,h,p,v,r,1024)
        return k,kv[...,p:]
    result=candidate()
    if not getattr(impl,'_fp8raw_validated',False):
        import time
        from vllm.logger import init_logger
        # Validate the installed quantized projection, not merely cached weights.
        bf_input=(x.float()*scale).to(torch.bfloat16)
        bf_pe=(rp.float()*scale).to(torch.bfloat16)
        ref=impl.kv_b_proj(bf_input)[0].view(t,h,p+v)
        ref_k=torch.cat((ref[...,:p],bf_pe[:,None,:].expand(-1,h,-1)),dim=-1)
        ref_v=ref[...,p:]
        def rms(a,b):return float((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12))
        ke,ve=rms(result[0],ref_k),rms(result[1],ref_v)
        finite=bool(torch.isfinite(result[0]).all() and torch.isfinite(result[1]).all())
        # Cached-weight projection must remain close to installed projection.
        if not finite or max(ke,ve)>0.1:raise RuntimeError(f'Raw projection gate failed: K RMS={ke}, V RMS={ve}')
        def elapsed(fn):
            torch.cuda.synchronize();begin=time.perf_counter()
            for _ in range(5):fn()
            torch.cuda.synchronize();return (time.perf_counter()-begin)*200
        opt_ms=elapsed(candidate)
        def original():
            a=impl.kv_b_proj((x.float()*scale).to(torch.bfloat16))[0].view(t,h,p+v)
            return torch.cat((a[...,:p],(rp.float()*scale).to(torch.bfloat16)[:,None,:].expand(-1,h,-1)),dim=-1),a[...,p:]
        bf_ms=elapsed(original)
        init_logger('vllm.fp8raw').info('[FP8-RAW-VALIDATION] tokens=%d K_rel_rms=%.6f V_rel_rms=%.6f original_projection_ms=%.4f raw_projection_ms=%.4f scale=%s weight_dtype=%s',t,ke,ve,bf_ms,opt_ms,scale,w.dtype)
        # Compare the full attention result against BF16 projected K/V.
        backend=impl._fp8raw_backend
        q=impl._fp8raw_q[chunk.token_slice].to(torch.bfloat16)
        bf_out,bf_lse=backend.flash_attn_varlen_func(
            q=q,k=ref_k,v=ref_v,cu_seqlens_q=chunk.query_start_loc,
            cu_seqlens_k=chunk.cu_seq_lens,max_seqlen_q=chunk.max_query_len,
            max_seqlen_k=chunk.max_seq_len,softmax_scale=backend.scale,
            causal=False,return_lse=True)
        from vllm.v1.attention.backends.mla.prefill import _fp8ctx
        backend._fp8ctx_current_chunk=chunk
        try:
            candidate_out,candidate_lse=_fp8ctx._run_fp8_asm(
                backend,q,*candidate(),chunk.query_start_loc,chunk.cu_seq_lens,
                chunk.cu_seq_lens[1:]-chunk.cu_seq_lens[:-1],chunk.max_query_len,False)
        finally:
            backend._fp8ctx_current_chunk=None
        ae=rms(candidate_out,bf_out)
        le=float((candidate_lse-bf_lse).abs().max())
        if not bool(torch.isfinite(candidate_out).all() and torch.isfinite(candidate_lse).all()) or ae>0.1 or le>0.5:
            raise RuntimeError(f'Raw full attention gate failed: RMS={ae}, LSE={le}')
        init_logger('vllm.fp8raw').info('[FP8-RAW-ATTENTION] output_rel_rms=%.6f lse_max_error=%.6f passed=True',ae,le)
        impl._fp8raw_validated=True
    return result
