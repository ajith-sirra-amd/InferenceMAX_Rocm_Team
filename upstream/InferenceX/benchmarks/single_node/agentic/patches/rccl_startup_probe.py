"""Install timed diagnostic stacks for serving communicator startup only."""
import pathlib
import vllm

root = pathlib.Path(vllm.__file__).parent
path = root / "distributed/device_communicators/pynccl.py"
source = path.read_text()
anchor = "        self.device = device\n"
if source.count(anchor) != 1:
    raise RuntimeError("Unexpected PyNccl device initialization layout")
source = source.replace(anchor, anchor + "        import faulthandler, os\n        print('[SERVING-RCCL-START]', 'pid=', os.getpid(), 'rank=', self.rank, 'device=', str(device), flush=True)\n        faulthandler.dump_traceback_later(90, repeat=True)\n", 1)
anchor = "            stream.synchronize()\n            del data\n"
if source.count(anchor) != 1:
    raise RuntimeError("Unexpected communicator warmup layout")
source = source.replace(anchor, "            print('[SERVING-RCCL-ALLREDUCE-QUEUED]', 'pid=', os.getpid(), 'rank=', self.rank, flush=True)\n" + anchor + "            faulthandler.cancel_dump_traceback_later()\n            print('[SERVING-RCCL-READY]', 'pid=', os.getpid(), 'rank=', self.rank, flush=True)\n", 1)
compile(source, str(path), "exec")
path.write_text(source)
print('[SERVING-RCCL-DIAGNOSTIC-INSTALLED]', flush=True)
