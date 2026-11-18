"""Read-only startup observations for one label-bound owned benchmark arm.

Stops as soon as the native client-start evidence exists; never signals a process,
changes a cgroup or samples measured GPU execution. Permission gaps are explicit.
"""
import argparse,datetime,json,subprocess,time
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--arm-output',required=True,type=Path)
parser.add_argument('--container',required=True)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--timeout',type=int,default=1950)
args=parser.parse_args()
base=Path('/share_nfs/gguasti/benchmarks').resolve()
arm=args.arm_output.resolve()
assert arm.is_relative_to(base) and arm.parts[len(base.parts)].startswith('pr-perf-')
assert args.output.resolve().is_relative_to(base)
assert not args.output.exists(), 'Diagnostic evidence is immutable'
assert args.container.startswith('kimi-pr-perf-')
args.output.parent.mkdir(parents=True,exist_ok=True)
def read(path):
    try:return Path(path).read_text()
    except (OSError,PermissionError) as exc:return {'unavailable':str(exc)}
def emit(record):
    record['at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with args.output.open('a') as stream:stream.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)
def allocation():
    text=subprocess.check_output(['scontrol','show','job','29088'],text=True,timeout=15)
    assert all(s in text for s in ['JobState=RUNNING','NodeList=pit2-p03-g34','UserId=giovanni.guasti@amd.com'])
allocation()
deadline=time.monotonic()+args.timeout
emit(dict(event='startup_observer_started',arm=str(arm),container=args.container,scope='Host/proc/cgroup only; no mutation; stop before native client'))
while time.monotonic()<deadline:
    if (arm/'agentic-client-start.json').exists() or (arm/'performance-result.json').exists():
        emit(dict(event='stopped_before_client_or_terminal_arm'))
        break
    allocation()
    result=subprocess.run(['docker','inspect','--format','{{json .}}',args.container],capture_output=True,text=True,timeout=15)
    if result.returncode:
        emit(dict(event='waiting_for_owned_container'))
        time.sleep(15)
        continue
    container=json.loads(result.stdout)
    assert container['Config']['Labels'].get('ipat.pr.output')==str(arm), 'Container output binding differs'
    pid=int(container['State']['Pid'])
    if not container['State']['Running'] or pid<=0:
        emit(dict(event='owned_container_not_running',state=container['State']))
        break
    # Check again after inspect: avoid sampling a client-start transition.
    if (arm/'agentic-client-start.json').exists():
        emit(dict(event='stopped_before_client'))
        break
    processes=[];pending=[pid];seen=set()
    while pending:
        current=pending.pop()
        if current in seen:continue
        seen.add(current)
        proc=Path('/proc')/str(current)
        status=read(proc/'status')
        if isinstance(status,str):
            wanted=('Name:','State:','VmRSS:','VmHWM:','VmLck:','Threads:')
            status='\n'.join(line for line in status.splitlines() if line.startswith(wanted))
        processes.append(dict(pid=current,status=status,wchan=read(proc/'wchan')))
        children=read(proc/f'task/{current}/children')
        if isinstance(children,str):pending.extend(int(x) for x in children.split())
    cgroup=read(Path('/proc')/str(pid)/'cgroup');memory={}
    if isinstance(cgroup,str):
        for line in cgroup.splitlines():
            if line.startswith('0::'):
                cg=(Path('/sys/fs/cgroup')/line.split('::',1)[1].lstrip('/')).resolve()
                assert cg.is_relative_to(Path('/sys/fs/cgroup'))
                for name in ['memory.current','memory.max','memory.events','memory.stat','memory.pressure','memory.swap.current']:
                    memory[name]=read(cg/name)
    emit(dict(event='owned_arm_startup_sample',container_id=container['Id'],processes=processes,cgroup_memory=memory,
              host_memory=read('/proc/meminfo'),host_memory_pressure=read('/proc/pressure/memory')))
    time.sleep(15)
else:emit(dict(event='startup_observer_deadline'))
