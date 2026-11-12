#!/usr/bin/env python3
"""Prepare/compile a bounded GEMM2 owner-wave experiment; never launch kernels.

Compile-only artifacts require manual gfx950 ISA review before a separate GPU
correctness/timing experiment. Installed AITER sources and tuning stay intact.
"""

import argparse
import ast
import difflib
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import traceback


HASHES = {
    'ops/flydsl/kernels/mxmoe_dispatcher.py': '5b51a36a47be834249ef370d20b857144880bf44f674aa72045ea1734ffebf28',
    'ops/flydsl/kernels/mxmoe_gemm_v2.py': 'cb2b7e7b777edcda0f3c81c2bac502a0e3d8bf5fb775847175ecd61ac1b8e685',
    'ops/flydsl/kernels/mxfp4_gemm_common.py': 'a15e4ed13a21c4aa4f85d099a0449087947291f8114abc8a607b10dd19c61d7b',
    'ops/flydsl/kernels/tensor_shim.py': '89689a59ba36cc0fb127761ef32860daacab6b1a8fb8c711a39cfc6b4649c938',
    'aot/flydsl/dry_run.py': 'bf3304180fcfbe92c2d481b450002e35728dad317161592d9b83e234957442f6',
}
CONFIG = dict(BM=16, BN=128, BK=128, use_nt=True, HIDDEN_MAX=8192,
              epilog='atomic', INTER_MAX=384, a_dtype='fp4', b_dtype='fp4',
              topk=1, SBM=16, persist=False, cu_num=0, g2_bhoist=True,
              g2_ascale_pf=True, g2_spart=402, g2_bf16_lds=True,
              out_dtype='bf16', enable_bias=False, g2_prefetch_ids=False,
              use_valid_token_count=False)
DISPATCH = 'ops/flydsl/kernels/mxmoe_dispatcher.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def candidate_source(original):
    tree = ast.parse(original)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'compile_gemm2_a4w4_port')
    lines = original.splitlines(keepends=True)
    edits = []
    # Changes apply only to this isolated factory; native get_g2 is never used.
    arg = function.args.args[-1]
    assert arg.arg == 'use_valid_token_count'
    edits.append((arg.lineno, arg.lineno, '    _owner_wave=False,\n'))
    name_assignment = next(n for n in function.body if isinstance(n, ast.Assign)
                           and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'name')
    guard = '''
    if _owner_wave:
        if not (
            os.environ.get("FLYDSL_GPU_ARCH") == "gfx950"
            and BM == SBM == 16 and BN == 128 and BK == 128
            and INTER_MAX == 384 and HIDDEN_MAX == 8192
            and a_dtype == b_dtype == "fp4" and epilog == "atomic"
            and out_dtype == "bf16" and not persist and not enable_bias
            and g2_bf16_lds and g2_bhoist and g2_ascale_pf and g2_spart == 402
            and aStages == a_preload == 3 and g2_apre
            and _composition is None and _input_row_resolver is None
            and _output_n_range is None and _reduce_store_cache_modifier is None
            and not use_valid_token_count and not g2_prefetch_ids
        ):
            raise ValueError("Owner-wave experiment requires the exact native BM16/BN128/BK128 three-slot path")
        name += "_owner0_vmemwait"
'''
    edits.append((name_assignment.end_lineno, name_assignment.end_lineno, guard))
    loads = next(n for n in ast.walk(function) if isinstance(n, ast.FunctionDef)
                 and n.name == 'issue_all_a_loads')
    original_loads = ''.join(lines[loads.lineno - 1:loads.end_lineno])
    original_loads = original_loads.replace('def issue_all_a_loads(', 'def issue_all_a_loads_native(', 1)
    wrapper = '''
        def issue_all_a_loads(m_row0):
            if const_expr(_owner_wave):
                if wave == fx.Int32(0):
                    issue_all_a_loads_native(m_row0)
                    rocdl.s_waitcnt(vmcnt=0)
            else:
                issue_all_a_loads_native(m_row0)
'''
    edits.append((loads.lineno - 1, loads.end_lineno, original_loads + wrapper))
    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = [replacement]
    result = ''.join(lines)
    ast.parse(result)
    return result


def prepare(source, output):
    sources = {}
    for name, expected in HASHES.items():
        actual = digest(source / name)
        if actual != expected:
            raise ValueError('Pinned native source mismatch: ' + name)
        sources[name] = actual
    original = (source / DISPATCH).read_text()
    candidate = candidate_source(original)
    target = output / 'source'
    target.mkdir()
    (target / 'mxmoe_dispatcher.original.py').write_text(original)
    (target / 'mxmoe_dispatcher.owner.py').write_text(candidate)
    (target / 'owner.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True), candidate.splitlines(keepends=True),
        fromfile='a/aiter/' + DISPATCH, tofile='b/aiter/' + DISPATCH)))
    result = dict(status='compile_only', phase='prepared', launches=0,
        source_root=str(source), source_sha256=sources, config=CONFIG,
        candidate_sha256=digest(target / 'mxmoe_dispatcher.owner.py'),
        candidate_symbol_suffix='_owner0_vmemwait', isa_review='pending',
        probe_source_sha256=digest(Path(__file__)),
        restrictions='No GPU execution or timing implemented; no serving promotion',
        wait_contract='Owner wave completes its three A-to-LDS VMEM loads before branch exit; native all-wave barrier and later reuse waits/barriers remain unchanged',
        launcher_argument_order=['arg_aq','arg_ascale','arg_bq','arg_bscale','arg_eids',
            'arg_cumsum','arg_stids','arg_sweights','arg_bias','i32_M','i32_max_m_blocks',
            'i32_grid_blocks','i32_inter','i32_hidden','arg_out','arg_out_scale','stream'])
    save(output / 'result.json', result)
    return result


def worker(args):
    sys.dont_write_bytecode = True
    folder = args.output / args.worker
    folder.mkdir()
    cache = args.cache / args.worker
    cache.mkdir(parents=True, exist_ok=False)
    # Set before imports. Native dry_run additionally forbids engine creation
    # and GPU lazy initialization, even if a compile API unexpectedly dispatches.
    os.environ.update(COMPILE_ONLY='1', FLYDSL_GPU_ARCH='gfx950', ARCH='gfx950',
        GPU_ARCHS='gfx950', CU_NUM='256', AITER_AOT_IMPORT='1',
        FLYDSL_RUNTIME_CACHE_DIR=str(cache), PYTHONDONTWRITEBYTECODE='1')
    import torch
    import flydsl.compiler.jit_executor as executor
    import flydsl.utils.env as compiler_env
    from aiter.aot.flydsl.dry_run import dry_run
    from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled
    from aiter.ops.flydsl.kernels import mxmoe_dispatcher as native
    installed = Path(inspect.getsourcefile(native)).parents[3]
    for path, expected in HASHES.items():
        if digest(installed / path) != expected:
            raise ValueError('Installed native source mismatch: ' + path)
    modules = (native, executor, compiler_env)
    compiler_sources = {}
    for module in modules:
        path = Path(inspect.getsourcefile(module))
        compiler_sources[str(path)] = digest(path)
        shutil.copyfile(path, folder / (module.__name__.replace('.', '_') + '.py'))
    module = native
    if args.worker == 'owner':
        name = 'aiter.ops.flydsl.kernels._moe_owner_experiment'
        path = args.output / 'source/mxmoe_dispatcher.owner.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    with dry_run('gfx950', 256, 'AMD Instinct MI355X'):
        config = dict(CONFIG)
        if args.worker == 'owner':
            config['_owner_wave'] = True
        launch = module.compile_gemm2_a4w4_port(**config)
        # Same 17 native launcher positions and scalar types. Placeholder CPU
        # addresses are never dereferenced; stream uses native dry_run's stub.
        values = [0] * 9 + [280, 280, 280, 384, 3584, 0, 0, torch.cuda.current_stream()]
        _preload_compiled(launch, *values)
    save(folder / 'compile.json', dict(status='compile_only', compilation_succeeded=True,
        launches=0, isa_review='pending', config=config, cache=str(cache),
        compiler_sources=compiler_sources, torch=torch.__version__, hip=torch.version.hip,
        native_safety='COMPILE_ONLY=1 plus dry_run forbids GPU engine/lazy initialization'))


def elf_extent(data, offset):
    """Bound a standalone or embedded little-endian ELF64 AMDGPU object."""
    if data[offset:offset + 6] != b'\x7fELF\x02\x01' or len(data) - offset < 64:
        return None
    if struct.unpack_from('<H', data, offset + 18)[0] != 224:
        return None
    phoff, shoff = struct.unpack_from('<QQ', data, offset + 32)
    ehsize, phsize, phnum, shsize, shnum = struct.unpack_from('<HHHHH', data, offset + 52)
    extent = max(ehsize, phoff + phsize * phnum, shoff + shsize * shnum)
    if extent > len(data) - offset or (phnum and phsize < 56) or (shnum and shsize < 64):
        return None
    for index in range(phnum):
        pos = offset + phoff + index * phsize
        file_offset = struct.unpack_from('<Q', data, pos + 8)[0]
        file_size = struct.unpack_from('<Q', data, pos + 32)[0]
        extent = max(extent, file_offset + file_size)
    for index in range(shnum):
        pos = offset + shoff + index * shsize
        if struct.unpack_from('<I', data, pos + 4)[0] != 8:  # NOBITS has no file payload.
            file_offset, file_size = struct.unpack_from('<QQ', data, pos + 24)
            extent = max(extent, file_offset + file_size)
    return extent if 64 <= extent <= len(data) - offset else None


def artifacts(cache, output):
    folder = output / 'artifacts'
    folder.mkdir()
    choices = [shutil.which('llvm-objdump'), '/opt/rocm/llvm/bin/llvm-objdump',
        '/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/lib/llvm/bin/llvm-objdump']
    objdump = next((p for p in choices if p and Path(p).is_file()), None)
    records, gpu_hashes = [], set()
    total_saved = 0
    for path in sorted(cache.rglob('*')):
        if not path.is_file():
            continue
        size = path.stat().st_size
        row = dict(file=str(path.relative_to(cache)), bytes=size)
        records.append(row)
        if size > 32 * 1024 * 1024:
            row['inspection'] = 'size_limit'
            continue
        data = path.read_bytes()
        row['sha256'] = hashlib.sha256(data).hexdigest()
        if path.suffix in ('.s', '.ll', '.mlir') and total_saved + size <= 32 * 1024 * 1024:
            saved = folder / (str(len(records)) + '-' + path.name)
            saved.write_bytes(data)
            total_saved += size
            row['saved_ir'] = saved.name
        offset = data.find(b'\x7fELF')
        while offset >= 0:
            extent = elf_extent(data, offset)
            if extent:
                binary = data[offset:offset + extent]
                sha = hashlib.sha256(binary).hexdigest()
                if sha not in gpu_hashes and len(gpu_hashes) < 4 and total_saved + extent <= 32 * 1024 * 1024:
                    gpu_hashes.add(sha)
                    saved = folder / (path.parts[-2] + '-' + sha[:12] + '.hsaco')
                    saved.write_bytes(binary)
                    total_saved += extent
                    row.setdefault('gpu_objects', []).append(dict(file=saved.name, sha256=sha, offset=offset))
                    if objdump:
                        assembly = saved.with_suffix('.s')
                        with assembly.open('w') as stream:
                            completed = subprocess.run([objdump, '--disassemble', '--mcpu=gfx950', str(saved)],
                                stdout=stream, stderr=subprocess.STDOUT, timeout=15)
                        row['objdump_returncode'] = completed.returncode
            offset = data.find(b'\x7fELF', offset + 4)
    return dict(cache_files=records, gpu_objects=len(gpu_hashes), objdump=objdump,
                saved_bytes=total_saved, isa_review='pending')


def main(args):
    if args.worker:
        worker(args)
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    if args.source is None:
        os.environ['AITER_AOT_IMPORT'] = '1'
        spec = importlib.util.find_spec('aiter')
        if spec is None or not spec.origin:
            raise RuntimeError('Supply --source pointing to pinned aiter source directory')
        args.source = Path(spec.origin).parent
    result = prepare(args.source, args.output)
    if args.prepare_only:
        return 0
    if args.cache is None:
        raise ValueError('--cache must be a fresh run work directory for compile-only')
    args.cache.mkdir(parents=True, exist_ok=False)
    result['workers'] = []
    for variant in ('baseline', 'owner'):
        command = [sys.executable, __file__, '--compile-only', '--worker', variant,
                   '--output', str(args.output), '--cache', str(args.cache)]
        with (args.output / (variant + '.log')).open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            timed_out = False
            try:
                process.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        result['workers'].append(dict(variant=variant, returncode=process.returncode, timed_out=timed_out))
        save(args.output / 'result.json', result)
        if process.returncode:
            break
    result['artifacts'] = artifacts(args.cache, args.output)
    result['phase'] = 'compiled' if len(result['workers']) == 2 and all(w['returncode'] == 0 for w in result['workers']) else 'compile_failed'
    result['compilation_succeeded'] = result['phase'] == 'compiled'
    save(args.output / 'result.json', result)
    return 0 if result['compilation_succeeded'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--compile-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, help='Pinned aiter package directory; optional in image')
    parser.add_argument('--cache', type=Path, help='Fresh run work directory for compiler cache')
    parser.add_argument('--timeout-seconds', type=int, default=210)
    parser.add_argument('--worker', choices=('baseline','owner'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 30 <= args.timeout_seconds <= 240:
        parser.error('Per-variant timeout must be 30..240 seconds')
    try:
        raise SystemExit(main(args))
    except Exception:
        if args.output.is_dir():
            (args.output / ('error-' + (args.worker or 'parent') + '.txt')).write_text(traceback.format_exc())
        raise
