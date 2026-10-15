import torch
import triton
import triton.language as tl

@triton.jit
def _amax(X,P,N:tl.constexpr,B:tl.constexpr):
    idx=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+idx,idx<N,other=0).to(tl.float32)
    tl.store(P+tl.program_id(0),tl.max(tl.abs(x),0))

@triton.jit
def _scale(P,S,NP:tl.constexpr,B:tl.constexpr,MAX:tl.constexpr):
    idx=tl.arange(0,B)
    val=tl.max(tl.load(P+idx,idx<NP,other=0),0)
    scale=tl.exp2(tl.ceil(tl.log2(tl.maximum(val/MAX,1.0e-12))))
    tl.store(S,scale)

@triton.jit
def _quant(X,Y,S,N:tl.constexpr,B:tl.constexpr):
    idx=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+idx,idx<N,other=0).to(tl.float32)
    scale=tl.load(S)
    tl.store(Y+idx,(x/scale).to(Y.dtype.element_ty),idx<N)

def quant(t,cache,key):
    t=t.contiguous()
    n=t.numel();np=triton.cdiv(n,8192)
    buf=cache.get(key)
    if buf is None or buf[0].shape!=t.shape or buf[0].device!=t.device:
        buf=(torch.empty_like(t,dtype=torch.float8_e4m3fn),torch.empty(np,device=t.device,dtype=torch.float32),torch.empty(1,device=t.device,dtype=torch.float32))
        cache[key]=buf
    y,partial,scale=buf
    _amax[(np,)](t,partial,n,8192)
    _scale[(1,)](partial,scale,np,triton.next_power_of_2(np),448.0)
    _quant[(triton.cdiv(n,4096),)](t,y,scale,n,4096)
    return y,scale
