#!/usr/bin/env python3
"""Prepared-only 187-call GPU correctness/lifetime test, not a model benchmark.

Default is CPU validation. --execute explicitly starts a bounded child process.
No runtime source installation, checkpoint loading, distributed setup or dispatch.
"""
from __future__ import annotations
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace as NS

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
LINEAR = "vllm/models/kimi_k3/amd/linear.py"
NORM = "vllm/models/kimi_k3/amd/ops/attn_res.py"

def emit(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

def validate(source_root):
    manifest = json.loads((HERE / "manifest.json").read_text())
    for item in manifest["files"]:
        assert sha(source_root / item["path"]) == item["baseline_sha256"]
        assert sha(HERE / "candidate" / item["path"]) == item["candidate_sha256"]
    assert manifest["files"][1]["candidate_sha256"] == "59739607be72dc856b0a827d5bf82ab43eba13da181cff512216c2f6293ac021"
    tree = ast.parse((HERE / "candidate" / LINEAR).read_text())
    gate = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_attnres_norm_forward_ok")
    gate_hash = hashlib.sha256(ast.dump(gate).encode()).hexdigest()
    assert gate_hash == "7e0b6fdbbb42f1c81bd6ba36fd3a16c34f4c377aecbd83c7afc73c9c4ec6d14c"
    callmap = json.loads((HERE / "call-map.json").read_text())
    assert sha(HERE / "call-map.json") == manifest["call_map_sha256"]
    assert len(callmap["kda_layers_one_based"]) == 69 and len(callmap["dsa_layers_one_based"]) == 24
    assert set(callmap["kda_layers_one_based"]) | set(callmap["dsa_layers_one_based"]) == set(range(1,94))
    # Native source must still have mode NONE inside the actual outer graph.
    cgpath = source_root / "vllm/v1/worker/gpu/cudagraph_utils.py"
    assert sha(cgpath)=="3d6429f84a45e66f951392c86f620a5fe23fbd97a11ca8eac3e9ffc2e2a40067"
    cg = ast.parse(cgpath.read_text())
    owner = next(n for n in cg.body if isinstance(n,ast.ClassDef) and n.name=="CudaGraphManager")
    capture = next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=="capture")
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="forward_fn" and n.lineno>495 and ast.unparse(n.args[0])=="CUDAGraphMode.NONE" for n in ast.walk(capture))
    return {"status":"CPU source validation only; GPU not executed", "manifest":manifest,
        "unmodified_admission_gate_ast_sha256":gate_hash,"harness_sha256":sha(Path(__file__)),
        "rows":[280,336],"patterns":["bf16","synthetic-fp8"],"default_patterns":["bf16"],"calls_per_forward":187,
        "historical_reads_per_forward":816,"replay_rows":[280,336,280],
        "lanes":["explicit_full_descriptor", "native_none_capture_injected_diagnostic", "native_none_capture_no_injection_control"],
        "limits":["No actual attention/MLP/checkpoint computation or request grouping", "The injected lane bypasses admission only in the test harness", "No performance or production admission claim"]}

def extract(tree,name,ns,owner=None):
    if owner: tree=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    unit=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit),"<hash-verified caller AST>","exec"),ns)
    return ns[name]

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def worker(args):
    # This imports GPU packages only in the explicit execution child.
    import torch
    from vllm import envs
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym
    from vllm.compilation.cudagraph_pool import capture_pool
    from vllm.platforms.rocm import on_gfx950
    assert torch.cuda.is_available() and on_gfx950()
    assert kFp8DynamicTokenSym.dtype==torch.float8_e4m3fn
    assert not envs.VLLM_BATCH_INVARIANT
    torch.cuda.set_device(args.device)
    device=torch.device("cuda",args.device)
    provenance=validate(args.source_root)
    provenance["requested_patterns"]=args.patterns
    provenance["environment"]={k:v for k,v in os.environ.items() if k.startswith(("VLLM_","AITER_")) or k in ("SAFETENSORS_FAST_GPU","HSA_NO_SCRATCH_RECLAIM","THP_MEM_ALLOC_ENABLE","PYTHONNOUSERSITE","PYTHONHASHSEED")}
    provenance["torch_version"]=str(torch.__version__)
    provenance["gpu_name"]=torch.cuda.get_device_name(args.device)
    callmap=json.loads((HERE / "call-map.json").read_text())
    kda_layers=set(callmap["kda_layers_one_based"])
    baseline=load(args.source_root/NORM,"_norm187_native")
    candidate=load(HERE/"candidate"/NORM,"_norm187_v3")
    native_tree=ast.parse((args.source_root/LINEAR).read_text())
    cached_tree=ast.parse((HERE/"candidate"/LINEAR).read_text())
    ctx=NS(cudagraph_runtime_mode=CUDAGraphMode.NONE,batch_descriptor=None,ubatch_slices=None,attn_metadata={})
    config=NS(use_cumem_cudagraph_pool=False)
    pp=NS(is_first_rank=True,is_last_rank=True)
    started=time.monotonic(); events=[]
    def progress(phase,**values):
        value=dict(phase=phase,elapsed_s=time.monotonic()-started,**values)
        with (args.output/"progress.jsonl").open("a") as f: f.write(json.dumps(value)+"\n")
        print(json.dumps(value),flush=True)
    def exact(a,b):
        assert a.dtype==b.dtype and a.shape==b.shape
        return torch.eq(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)).all()
    def finite(*values):
        return torch.stack([torch.isfinite(v.float()).all() for v in values if v is not None]).all()
    def unpack(out):
        return (out.data,out.scale) if isinstance(out,QuantizedActivation) else (out,None)

    class Op:
        def __init__(self,scale,bias,quant=False): self.scale,self.bias,self.quant=scale,bias,quant
        def get_input_quant_key(self): return kFp8DynamicTokenSym if self.quant else None
        def __call__(self,hidden_states,**kwargs):
            x=hidden_states
            if isinstance(x,QuantizedActivation):
                assert x.quant_key==kFp8DynamicTokenSym and self.quant
                x=(x.data.float()*x.scale).to(x.orig_dtype)
            return (x.float()*self.scale+self.bias).to(torch.bfloat16)
    class KimiK3DeltaAttention(Op): pass
    class KimiMLAAttention(Op): pass
    class KimiK3DSAAttention(Op): pass
    class KimiMLP(Op): pass
    class KimiMoE(Op): pass
    class KimiDecoderLayer:
        def __call__(self,*a,**kw):
            lane=self.lane
            # Test-only injection below the unchanged whole-forward gate.
            # The bank, not row count or layer module, owns this diagnostic cache.
            if self.index==0:
                bank=a[2]
                bank.fill_(float("nan"))
                if lane.inject:
                    assert kw.get("block_rstd_cache") is None
                    lane.injected_cache=torch.full((bank.shape[0],8),float("nan"),dtype=torch.float32,device=bank.device)
                    lane.injected_bank=bank
                elif kw.get("block_rstd_cache") is not None:
                    kw["block_rstd_cache"].fill_(float("nan"))
            if lane.inject:
                assert a[2] is lane.injected_bank
                kw["block_rstd_cache"]=lane.injected_cache
            return self.forward(*a,**kw)
    class KimiLinearModel: pass

    def lane_factory(name,weights,mixed,observer):
        is_native=name=="baseline"
        lane=NS(name=name,inject=name=="native_none_capture_injected_diagnostic",
                injected_cache=None,injected_bank=None,observer=observer,calls=0,addresses=None,coverage={})
        # Each lane gets its own classes/method namespaces; no shared global
        # callback switches are baked into captured graphs.
        Decoder=type("KimiDecoderLayer",(KimiDecoderLayer,),{})
        Model=type("KimiLinearModel",(KimiLinearModel,),{})
        tree=native_tree if is_native else cached_tree
        ns=dict(torch=torch,envs=envs,CUDAGraphMode=CUDAGraphMode,
            get_forward_context=lambda:ctx,is_forward_context_available=lambda:True,
            get_pp_group=lambda:pp,cdiv=lambda x,y:(x+y-1)//y,
            KimiDecoderLayer=Decoder,KimiLinearModel=Model,
            KimiK3DeltaAttention=KimiK3DeltaAttention,KimiMLAAttention=KimiMLAAttention,
            KimiMLP=KimiMLP,QuantizedActivation=QuantizedActivation,
            kFp8DynamicTokenSym=kFp8DynamicTokenSym,cast=lambda typ,x:x,
            attn_res=baseline.attn_res if is_native else candidate.attn_res)
        apply=extract(tree,"_apply_attn_res",ns)
        def observed_apply(*a,**kw):
            final=a[4]==8 and kw.get("output_norm") is None
            if lane.inject and final:
                assert a[1] is lane.injected_bank
                kw["block_rstd_cache"]=lane.injected_cache
            cache=kw.get("block_rstd_cache")
            if lane.calls==0:
                lane.addresses=dict(bank=a[1].data_ptr(),cache=None if cache is None else cache.data_ptr())
            result=apply(*a,**kw)
            lane.observer(lane,a,kw,result)
            lane.coverage["historical_reads"]+=a[4]
            lane.coverage["cache_calls"]+=int(cache is not None)
            lane.coverage["writers"]+=int(kw.get("block_write_idx",-1)>=0)
            lane.coverage["delta_none"]+=int(kw.get("delta") is None)
            lane.coverage["fp8_calls"]+=int(isinstance(result,QuantizedActivation))
            lane.coverage["fp8_writers"]+=int(kw.get("block_write_idx",-1)>=0 and isinstance(result,QuantizedActivation))
            lane.calls+=1
            return result
        ns["_apply_attn_res"]=observed_apply
        if not is_native: extract(tree,"_attnres_norm_forward_ok",ns)
        for method in ("forward","forward_attn_residual","_run_self_attn"):
            setattr(Decoder,method,extract(tree,method,ns,"KimiDecoderLayer"))
        Model.forward=extract(tree,"forward",ns,"KimiLinearModel")
        m=Model(); m.config=NS(attn_res_block_size=12)
        m.start_layer,m.end_layer=0,93; m.layers=[]; m.aux_hidden_state_layers=[]
        m._maybe_add_hidden_state=lambda *a:[]
        m._attnres_norm_cache_ready=name in ("explicit_full_descriptor","native_none_capture_no_injection_control")
        m._attnres_norm_cache_device=device
        for i in range(93):
            d=Decoder(); d.index=i; d.lane=lane
            d.use_attn_residuals=True; d.attn_res_block_size=12
            d.prev_valid_blocks=(i+11)//12; d.is_block_write_layer=i%12==0; d.block_write_idx=i//12
            d.self_attention_res_norm,d.self_attention_res_proj,d.input_layernorm=weights[2*i]
            d.mlp_res_norm,d.mlp_res_proj,d.post_attention_layernorm=weights[2*i+1]
            typ=KimiK3DeltaAttention if i+1 in kda_layers else KimiK3DSAAttention
            d.self_attn=typ(.75,weights[-1][0].weight*.015,mixed)
            d.mlp=KimiMLP(.5,weights[-1][0].weight*.01,mixed) if i==0 else KimiMoE(.5,weights[-1][0].weight*.01,False)
            m.layers.append(d)
        m.output_attn_res_norm,m.output_attn_res_proj,_=weights[-1]
        lane.model=m
        def forward(x):
            lane.calls=0; lane.injected_cache=None; lane.injected_bank=None
            lane.coverage=dict(historical_reads=0,cache_calls=0,writers=0,delta_none=0,fp8_calls=0,fp8_writers=0)
            try:
                out=m.forward(None,None,None,inputs_embeds=x.clone())
                assert lane.calls==187
                assert lane.coverage["historical_reads"]==816 and lane.coverage["writers"]==8
                assert lane.coverage["delta_none"]==9
                assert lane.coverage["fp8_writers"]==(8 if mixed else 0)
                assert lane.coverage["fp8_calls"]==(70 if mixed else 0)
                return out
            finally:
                # No strong Python cache/bank owner survives capture/forward.
                lane.injected_cache=None; lane.injected_bank=None
        lane.forward=forward
        return lane

    gen=torch.Generator(device=device).manual_seed(20261112)
    weights=[]
    for i in range(187):
        norm=NS(weight=(1+.03*torch.randn(7168,device=device,generator=gen)).to(torch.bfloat16),variance_epsilon=1e-5)
        proj=NS(weight=(.01*torch.randn((1,7168),device=device,generator=gen)).to(torch.bfloat16))
        output=NS(weight=(1+.03*torch.randn(7168,device=device,generator=gen)).to(torch.bfloat16),variance_epsilon=1e-5)
        weights.append((norm,proj,output))
    def signal_for(rows,version):
        g=torch.Generator(device=device).manual_seed(9000+rows+version)
        x=torch.randn((rows,7168),device=device,generator=g,dtype=torch.float32).to(torch.bfloat16)
        # Padding-like rows remain finite; bank/cache poisoning is separate.
        x[-7:]=0 if version==1 else (0.125 if version==2 else x[-7:])
        return x

    checks=[]
    for pattern in args.patterns:
        mixed=pattern=="synthetic-fp8"
        progress("prepare_pattern",mixed_fp8=mixed)
        refs={}; cases={}; pool=torch.cuda.graph_pool_handle(); stream=torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream())
        for rows in (280,336):
            snapshots=[]; last_bank=[None]; finite_checks=[]
            def collect(lane,a,kw,out,snapshots=snapshots,last_bank=last_bank):
                data,scale=unpack(out)
                finite_checks.append(finite(a[0],data,scale))
                snapshots.append(dict(prefix=a[0].clone(),data=data.clone(),scale=None if scale is None else scale.clone(),
                    blocks=a[4],write=kw.get("block_write_idx",-1),delta=kw.get("delta") is not None))
                last_bank[0]=a[1]
            ref_lane=lane_factory("baseline",weights,mixed,collect)
            x=signal_for(rows,0)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                ref_lane.forward(x)
                bank=last_bank[0].clone(); last_bank[0]=None
            torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
            assert bool(torch.stack(finite_checks).all().item()) and bool(finite(bank).item()), (rows,"native reference nonfinite")
            refs[rows]=dict(lane=ref_lane,snapshots=snapshots,bank=bank,input=x,collector_bank=last_bank)
            # Ensure closures bind this shape's storage, not a later loop value.
            def comparator(reference,flags,expect_cache):
                def observe(lane,a,kw,out):
                    i=lane.calls; expected=reference["snapshots"][i]; data,scale=unpack(out)
                    assert (scale is None)==(expected["scale"] is None)
                    assert (a[4],kw.get("block_write_idx",-1),kw.get("delta") is not None)==(expected["blocks"],expected["write"],expected["delta"])
                    cache=kw.get("block_rstd_cache")
                    assert (cache is not None)==expect_cache
                    count=a[4]+int(kw.get("block_write_idx",-1)>=0)
                    flags[i,0].copy_(exact(a[0],expected["prefix"]))
                    flags[i,1].copy_(exact(data,expected["data"]))
                    if scale is None: flags[i,2].fill_(True)
                    else: flags[i,2].copy_(exact(scale,expected["scale"]))
                    flags[i,3].copy_(exact(a[1][:,:count],reference["bank"][:,:count]))
                    if cache is None: flags[i,4].fill_(True)
                    else: flags[i,4].copy_(torch.isfinite(cache[:,:count]).all())
                return observe
            names=("explicit_full_descriptor","native_none_capture_injected_diagnostic")
            if rows==280: names+=("native_none_capture_no_injection_control",)
            for name in names:
                flags_tensor=torch.empty((187,5),dtype=torch.bool,device=device)
                lane=lane_factory(name,weights,mixed,comparator(refs[rows],flags_tensor,"control" not in name))
                descriptor=BatchDescriptor(num_tokens=rows,num_reqs=rows//7,uniform=True)
                ctx.cudagraph_runtime_mode=(CUDAGraphMode.FULL if name=="explicit_full_descriptor"
                    else CUDAGraphMode.PIECEWISE if "control" in name else CUDAGraphMode.NONE)
                ctx.batch_descriptor=descriptor if name=="explicit_full_descriptor" else None
                # Eager compilation/wiring pass, no numerical tolerance.
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream): lane.forward(x)
                torch.cuda.synchronize()
                assert bool(flags_tensor.all().item()), (name,rows,"eager",torch.nonzero(~flags_tensor).cpu().tolist()[:12])
                progress("capture",lane=name,rows=rows,mixed_fp8=mixed)
                graph=torch.cuda.CUDAGraph()
                ctx.cudagraph_runtime_mode=CUDAGraphMode.FULL if name=="explicit_full_descriptor" else CUDAGraphMode.NONE
                with capture_pool(pool,config) as selected_pool, torch.cuda.graph(graph,pool=selected_pool,stream=stream):
                    result=lane.forward(x)
                    del result
                assert lane.injected_cache is None and lane.injected_bank is None
                assert lane.coverage["cache_calls"]==(0 if "control" in name else 187)
                cases[(name,rows)]=dict(graph=graph,lane=lane,flags=flags_tensor,addresses=lane.addresses,coverage=dict(lane.coverage))
                del graph,lane
                gc.collect()
        # Reference snapshots stay outside the graph pool. Refresh in place
        # for changed inputs so captured comparison pointers remain valid.
        def refresh_reference(rows,version):
            r=refs[rows]; old=r["snapshots"]
            saved_observer=r["lane"].observer
            finite_checks=[]
            def update(lane,a,kw,out):
                i=lane.calls; data,scale=unpack(out)
                finite_checks.append(finite(a[0],data,scale))
                old[i]["prefix"].copy_(a[0]); old[i]["data"].copy_(data)
                if scale is not None: old[i]["scale"].copy_(scale)
                if i==186: r["bank"].copy_(a[1])
            r["lane"].observer=update
            r["input"].copy_(signal_for(rows,version))
            r["lane"].forward(r["input"])
            r["lane"].observer=saved_observer
            assert bool(torch.stack(finite_checks).all().item()) and bool(finite(r["bank"]).item()), (rows,version,"refreshed native nonfinite")
        for version,rows in enumerate((280,336,280)):
            refresh_reference(rows,version)
            # Ordinary allocator pressure, after all Python forward locals died.
            pressure=torch.empty((args.pressure_mib*1024*1024,),dtype=torch.uint8,device=device)
            pressure.fill_(173); del pressure; gc.collect()
            for (name,m),case in cases.items():
                if m!=rows: continue
                case["flags"].fill_(False)
                case["graph"].replay(); torch.cuda.synchronize()
                bad=torch.nonzero(~case["flags"]).cpu().tolist()
                assert not bad,(name,rows,version,bad[:12])
                checks.append(dict(lane=name,rows=rows,version=version,mixed_fp8=mixed,
                    exact_call_checks=187,comparisons=935,addresses=case["addresses"]))
                checks[-1]["capture_coverage"]=case["coverage"]
                progress("replay_pass",lane=name,rows=rows,version=version,mixed_fp8=mixed)
        events.append(dict(mixed_fp8=mixed,pool=list(pool),graphs=len(cases)))
        cases.clear(); refs.clear(); gc.collect(); torch.cuda.empty_cache()
    return dict(status="passed",provenance=provenance,checks=checks,pools=events,
        elapsed_s=time.monotonic()-started,
        scope="Single-GPU synthetic full caller correctness and shared-pool lifetime only; no E2E or production admission",
        limitations=["Attention/MLP are deterministic BF16 stand-ins with the actual 69 KDA/24 DSA layer map. Default BF16 matches the source-derived frozen MXFP4 quant-key path; optional synthetic-fp8 is an artificial stress case. No checkpoint arithmetic or real request grouping.",
            "Explicit FULL descriptor lane tests unchanged prototype owner/kwargs allocation; native NONE injection lane is admission-bypassed diagnostic.",
            "Native capture_pool plus shared torch.cuda.graph pool is exercised; complete V2 manager/distributed/offload behavior is not.",
            "Comparison/poisoning operations are captured, so results provide no timing or instrumentation-free lifetime claim."])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--device",type=int,default=0)
    parser.add_argument("--timeout",type=int,default=150)
    parser.add_argument("--pressure-mib",type=int,default=128)
    parser.add_argument("--patterns",nargs="+",choices=["bf16","synthetic-fp8"],default=["bf16"])
    args=parser.parse_args()
    args.source_root=args.source_root.resolve(); args.output=args.output.resolve()
    assert 30<=args.timeout<=170 and 16<=args.pressure_mib<=256
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.execute and not args.worker:
        data=validate(args.source_root); emit(args.output/"source-validation.json",data)
        print(json.dumps(data,indent=2)); return
    if args.worker:
        try: result=worker(args)
        except BaseException as error:
            result=dict(status="failed",error=str(error),traceback=traceback.format_exc())
        emit(args.output/"result.json",result)
        print(json.dumps(result),flush=True)
        raise SystemExit(0 if result["status"]=="passed" else 1)
    validate(args.source_root)
    env=dict(os.environ)
    for key,subdir in (("TRITON_CACHE_DIR","triton"),("XDG_CACHE_HOME","xdg"),("TMPDIR","tmp")):
        path=args.output/"cache"/subdir; path.mkdir(parents=True,exist_ok=True); env[key]=str(path)
    argv=[sys.executable,"-B",str(Path(__file__).resolve()),"--worker","--source-root",str(args.source_root),"--output",str(args.output),"--device",str(args.device),"--pressure-mib",str(args.pressure_mib),"--patterns",*args.patterns]
    with (args.output/"worker.log").open("w") as log:
        proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try: returncode=proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL); proc.wait()
            emit(args.output/"result.json",dict(status="timed_out",deadline_seconds=args.timeout))
            raise SystemExit(124)
    raise SystemExit(returncode)

if __name__=="__main__": main()
