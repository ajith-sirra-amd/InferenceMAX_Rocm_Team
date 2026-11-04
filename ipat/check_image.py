"""Small GPU/library diagnostic; no model loading or system changes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch


def emit(key, value):
    print(json.dumps({key: value}), flush=True)


emit("torch", {"version": torch.__version__, "hip": torch.version.hip})
test_path = Path(os.environ['TMPDIR']) / 'ipat-name:check'
try:
    test_path.write_text('compiler filename check\n')
    test_path.unlink()
    emit('temporary_filename_check', {'status': 'passed'})
except OSError as error:
    emit('temporary_filename_check', {'status': 'failed', 'error': str(error)})
emit("environment", {key: value for key, value in os.environ.items()
                     if key.startswith(("ROCM", "HIP", "HSA", "ROCBLAS", "HIPBLAS", "TORCH_"))
                     or key in ("LD_LIBRARY_PATH", "LD_PRELOAD")})
results = []
for device in range(torch.cuda.device_count()):
    row = {"device": device, "properties": str(torch.cuda.get_device_properties(device))}
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        try:
            x = torch.ones((16, 16), dtype=dtype, device=f"cuda:{device}")
            y = torch.mm(x, x)
            torch.cuda.synchronize(device)
            row[str(dtype)] = {"ok": bool(torch.all(y == 16).item())}
        except Exception as error:
            row[str(dtype)] = {"ok": False, "error": str(error)}
    results.append(row)
    emit("gpu_math", row)

maps = Path('/proc/self/maps').read_text().splitlines()
libraries = sorted({line.split()[-1] for line in maps
                    if any(name in line for name in ("amdhip", "hsa-runtime", "hipblas", "rocblas"))})
emit("loaded_libraries", libraries)
for root in (Path('/usr/local/lib/python3.12/dist-packages/_rocm_sdk_libraries/lib/hipblaslt/library/gfx950'),
             Path('/opt/rocm/lib/hipblaslt/library')):
    for path in sorted(root.glob('*ID75a0_gfx950.co'))[:2]:
        emit("code_object", {"path": str(path), "bytes": path.stat().st_size,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for command in (["file", str(path)], ["readelf", "-h", str(path)]):
            try:
                p = subprocess.run(command, capture_output=True, text=True, timeout=20)
                emit("inspection", {"command": command, "returncode": p.returncode,
                                    "output": p.stdout, "stderr": p.stderr})
            except Exception as error:
                emit("inspection_error", str(error))
raise SystemExit(0 if results and all(row[str(dtype)]["ok"] for row in results
                                    for dtype in (torch.float32, torch.float16, torch.bfloat16)) else 1)
