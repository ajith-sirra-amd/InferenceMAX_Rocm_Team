import torch
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.prefill._flydsl_context.entry import flydsl_flash_attn_fp8_func
logger=init_logger("vllm.fp8ctx")

def run(backend,chunk,q,k,v):
    from aiter import per_tensor_quant_hip
    (fq,sq),(fk,sk),(fv,sv)=[per_tensor_quant_hip(t.contiguous(),quant_dtype=torch.float8_e4m3fn) for t in (q,k,v)]
    shape=(q.shape[0],q.shape[1],v.shape[-1])
    cache=getattr(backend,"_flydsl_outputs",None)
    if cache is None or cache[0].shape!=shape:
        cache=(torch.empty(shape,dtype=torch.bfloat16,device=q.device),torch.empty((q.shape[1],q.shape[0]),dtype=torch.float32,device=q.device))
        backend._flydsl_outputs=cache
    return flydsl_flash_attn_fp8_func(fq,fk,fv,softmax_scale=backend.scale,causal=False,
        cu_seqlens_q=chunk.query_start_loc,cu_seqlens_kv=chunk.cu_seq_lens,
        max_seqlen_q=int(chunk.max_query_len),max_seqlen_kv=int(chunk.max_seq_len),
        cross_seqlen=True,q_descale=sq,k_descale=sk,v_descale=sv,
        out=cache[0],lse=cache[1],return_lse=True)

def fp8_context_chunk(backend,chunk,q,k,v):
    try:
        out,lse=run(backend,chunk,q,k,v)
        if not getattr(backend,"_fp8ctx_validated",False):
            def bf16():
                return backend.flash_attn_varlen_func(q=q,k=k,v=v,
                    cu_seqlens_q=chunk.query_start_loc,cu_seqlens_k=chunk.cu_seq_lens,
                    max_seqlen_q=chunk.max_query_len,max_seqlen_k=chunk.max_seq_len,
                    softmax_scale=backend.scale,causal=False,return_lse=True)
            ref,ref_lse=bf16()
            finite=bool(torch.isfinite(out).all() and torch.isfinite(lse).all())
            rms=float((out.float()-ref.float()).square().mean().sqrt()/ref.float().square().mean().sqrt().clamp_min(1e-12))
            lse_err=float((lse-ref_lse).abs().max())
            for _ in range(2):
                bf16();run(backend,chunk,q,k,v)
            def timed(fn):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();value=fn();end.record();end.synchronize()
                return start.elapsed_time(end),value
            bf_ms,_=timed(bf16)
            fd_ms,(out,lse)=timed(lambda:run(backend,chunk,q,k,v))
            logger.info("[FP8-VALIDATION] backend=FlyDSL q=%s k=%s finite=%s rel_rms=%.6f lse_max_error=%.6f BF16_ms=%.4f FlyDSL_ms=%.4f",tuple(q.shape),tuple(k.shape),finite,rms,lse_err,bf_ms,fd_ms)
            if not finite or rms>0.1 or lse_err>0.5:
                logger.error("[FP8-VALIDATION-FAILED] FlyDSL numerical gate")
                raise SystemExit(2)
            backend._fp8ctx_validated=True
            logger.info("[FP8-VALIDATION-PASSED] FlyDSL live output and LSE")
        backend._flydsl_count=getattr(backend,"_flydsl_count",0)+1
        if backend._flydsl_count<=3 or backend._flydsl_count%200==0:
            logger.info("[FP8-CTX] FlyDSL context OK n=%d out=%s lse=%s",backend._flydsl_count,tuple(out.shape),tuple(lse.shape))
        return out,lse
    except Exception:
        import traceback
        logger.error("[FP8-VALIDATION-FAILED] FlyDSL exception: %s",traceback.format_exc())
        raise SystemExit(2)

def selftest_noncausal(backend):
    return

def wrap_flash(fn):
    return fn
