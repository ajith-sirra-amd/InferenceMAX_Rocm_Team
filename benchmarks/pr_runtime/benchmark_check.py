"""Run the pinned native fixed client against a verified isolated PR arm."""
import json,os,shlex,signal,subprocess,sys,time
from pathlib import Path
from urllib.request import urlopen
sys.path.insert(0,'/quality')
import launch
import runtime

def main():
    out=Path('/logs');pr=int(os.environ['IPAT_PR_NUMBER']);arm=os.environ['IPAT_ARM']
    workload=os.environ.get('IPAT_PR_WORKLOAD','fixed')
    assert workload in ('fixed','agentic')
    reference=Path('/agentic-work/recipe.json') if workload=='agentic' else Path('/fixed-tools/baseline.json')
    config=runtime.candidate_recipe(json.loads(reference.read_text()),pr,arm)
    config['image']=os.environ['EXPECTED_IMAGE']
    spec=json.loads(config['server_arguments']['speculative-config'])
    assert spec['rejection_sample_method']=='synthetic' and spec['synthetic_acceptance_length']==3.57
    if workload=='fixed':
        assert config['workload']==dict(input_tokens=8192,output_tokens=1024,concurrency=48,requests=192,warmup_requests=96,seed=0)
    else:
        assert pr==67372
        assert config['workload']['type']=='native-agentx' and config['workload']['concurrency']==48 and config['workload']['benchmark_duration_seconds']==3600
        (out/'client-recipe.json').write_bytes(reference.read_bytes())
        os.environ.update(IPAT_EXACT_PR_CLIENT='1',MODEL='moonshotai/Kimi-K3.1-Pro',RESULT_DIR='/logs',CONC='48',DURATION='3600')
    assert json.loads(config['server_arguments']['attention-config'])['indexer_kv_dtype']=='mxfp4'
    config['server_arguments']['worker-extension-cls']='model_worker.ModelWorkerExtension'
    os.environ.update(config['server_environment'])
    os.environ['PATH']='/opt/runtime-venv/bin:'+os.environ['PATH']
    os.environ['EVAL_ONLY']='false'
    assert os.environ.get('VLLM_GPU_SYNC_CHECK','')==''
    assert os.environ['PROFILE']=='0'
    argv=launch.command(config,'/model',os.environ['PORT'])
    (out/'recipe.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'command.txt').write_text(shlex.join(argv)+'\n')
    result=dict(status='failed',pr=pr,arm=arm,workload=workload,scope='Native workload with synthetic acceptance3.57; full accuracy remains unqualified')
    server=None
    def stop(signum,frame):raise RuntimeError(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        preflight=[sys.executable,'/agentic-tools/client.py','--check'] if workload=='agentic' else [sys.executable,'/fixed-tools/check_client.py']
        subprocess.run(preflight,check=True)
        with (out/'server.log').open('xb') as log:
            server=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+1800
            while time.monotonic()<deadline:
                if server.poll() is not None:raise RuntimeError(f'Server exited: {server.returncode}')
                try:
                    with urlopen('http://127.0.0.1:'+os.environ['PORT']+'/health',timeout=5) as r:
                        if r.status==200:break
                except OSError:pass
                time.sleep(2)
            else:raise TimeoutError('Readiness exceeded1800 seconds')
            records=[json.loads((out/f'rank-{rank}.json').read_text()) for rank in range(8)]
            assert [r['rank'] for r in records]==list(range(8))
            fp4=[[json.loads(line) for line in (out/f'mxfp4-rank-{rank}.jsonl').read_text().splitlines()] for rank in range(8)]
            assert all(rows and all(r['use_fp4'] and r['rank']==rank for r in rows) for rank,rows in enumerate(fp4))
            result['loaded_ranks']=8;result['mxfp4_indexers_per_rank']=[len(rows) for rows in fp4]
            profiling={k:v for k,v in os.environ.items() if k.startswith(('ROCPROF','HIP_TRACE','HSA_TOOLS','VLLM_TORCH_PROFILER','SGLANG_TORCH_PROFILER')) or k in ('LD_PRELOAD','PROFILE')}
            maps={}
            for process in Path('/proc').iterdir():
                if process.name.isdigit():
                    try:
                        lines=(process/'maps').read_text().splitlines()
                        found=sorted({line.split()[-1] for line in lines if 'rocprofiler' in line or 'roctracer' in line})
                        if found:maps[process.name]=found
                    except (OSError,PermissionError):pass
            (out/'profiling-observation.json').write_text(json.dumps(dict(environment=profiling,mapped_libraries=maps,limits='Library mapping is observation, not proof of active profiling or measured overhead'),indent=2)+'\n')
            with (out/'client.log').open('xb') as log:
                client=[sys.executable,'/agentic-tools/client.py'] if workload=='agentic' else ['bash','/pr-tools/benchmark_client.sh']
                subprocess.run(client,stdout=log,stderr=subprocess.STDOUT,check=True)
            if workload=='fixed':
                subprocess.run([sys.executable,'/fixed-tools/validate.py',str(out/'baseline.json'),'--output',str(out/'validation.json')],check=True)
            result['status']='collected'
    except Exception as error:result['error']=f'{type(error).__name__}: {error}'
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if server is not None:
            try:os.killpg(server.pid,signal.SIGTERM);server.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=15)
            except ProcessLookupError:pass
        (out/'performance-result.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['status']=='collected' else 1

if __name__=='__main__':raise SystemExit(main())
