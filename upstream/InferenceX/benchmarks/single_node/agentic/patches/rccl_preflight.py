"""Check the serving image's actual PyNccl communicator before loading Kimi."""
import datetime
import os
import socket

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
if world != 8 or torch.cuda.device_count() != 8:
    raise RuntimeError("Kimi communication preflight requires exactly eight visible GPUs")
torch.cuda.set_device(rank)
print("[RCCL-PREFLIGHT-START]", "host=", socket.gethostname(), "rank=", rank,
      "gpu=", torch.cuda.get_device_name(rank), "torch=", torch.__version__,
      "hip=", torch.version.hip, flush=True)
dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=120))
comm = PyNcclCommunicator(dist.group.WORLD, device=rank)
if not comm.available or comm.disabled:
    raise RuntimeError("PyNccl communicator unavailable; refusing fallback")
for dtype in (torch.float32, torch.bfloat16):
    value = torch.full((4096,), rank + 1, device="cuda", dtype=dtype)
    result = comm.all_reduce(value)
    torch.cuda.synchronize()
    torch.testing.assert_close(result, torch.full_like(value, 36), rtol=0, atol=0)
print("[RCCL-PREFLIGHT-PASS]", "host=", socket.gethostname(), "rank=", rank,
      "world=", world, "nccl=", comm.nccl.ncclGetVersion(),
      "all_reduce_sum=36", flush=True)
dist.barrier()
dist.destroy_process_group()
