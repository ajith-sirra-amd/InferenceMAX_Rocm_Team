"""Transport stored FP8 MLA cache bytes; dequantize only after DCP all-gather."""
import torch
import triton
import triton.language as tl
from vllm import _custom_ops as ops
from vllm.platforms import current_platform


@triton.jit
def _decode(src, scales, dst, N: tl.constexpr, PER_RANK: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src + idx, idx < N, other=0).to(tl.float32)
    scale = tl.load(scales + idx // PER_RANK, idx < N, other=1)
    tl.store(dst + idx, value * scale, idx < N)


def gather(backend, cache, workspace, block_table, chunk, scale, manager):
    group = manager.group.device_group
    world = manager.group.world_size
    tokens = chunk.num_local_context_tokens
    width = workspace.shape[1]
    if backend.kv_cache_dtype != 'fp8' or width != 576:
        raise RuntimeError('Raw FP8 gather supports only ordinary fp8 MLA cache width576')
    if cache.element_size() != 1 or workspace.dtype != torch.bfloat16:
        raise RuntimeError('Unexpected FP8 cache / BF16 reconstruction workspace')
    buffers = getattr(backend, '_raw_fp8_gather_buffers', None)
    if buffers is None:
        buffers = {'bytes': torch.empty(workspace.shape, dtype=torch.uint8, device=workspace.device),
                   'scales': torch.empty(world, dtype=torch.float32, device=workspace.device),
                   'verified': False}
        backend._raw_fp8_gather_buffers = buffers
        torch.distributed.all_gather_into_tensor(buffers['scales'], scale.reshape(1).float(), group=group)
    offset = workspace.shape[0] // (world + 1)
    raw = buffers['bytes']
    local = raw[:tokens]
    gathered = raw[offset:offset + tokens * world]
    ops.cp_gather_cache(cache.view(torch.uint8), local, block_table,
                        chunk.padded_local_cu_seq_lens, chunk.num_requests, chunk.starts)
    if not buffers['verified']:
        ref = torch.empty_like(workspace[:tokens])
        ops.gather_and_maybe_dequant_cache(cache, ref, block_table,
            chunk.padded_local_cu_seq_lens, chunk.padded_local_token_to_seq,
            tokens, backend.kv_cache_dtype, scale, chunk.starts)
        check = torch.empty_like(ref)
        _decode[(triton.cdiv(tokens * width, 1024),)](
            local.view(current_platform.fp8_dtype()), scale.reshape(1), check,
            tokens * width, tokens * width, 1024)
        torch.testing.assert_close(check, ref, rtol=0, atol=0)
    torch.distributed.all_gather_into_tensor(gathered, local, group=group)
    output = workspace[offset:offset + tokens * world]
    _decode[(triton.cdiv(tokens * world * width, 1024),)](
        gathered.view(current_platform.fp8_dtype()), buffers['scales'], output,
        tokens * world * width, tokens * width, 1024)
    if not buffers['verified']:
        # Independent ordinary BF16 all-gather checks rank order and source scales.
        expected = torch.empty_like(output)
        torch.distributed.all_gather_into_tensor(expected, ref, group=group)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        buffers['verified'] = True
        print('[FP8-KV-ALLGATHER-VERIFIED]', 'device=', str(cache.device),
              'cache_encoding=', str(current_platform.fp8_dtype()),
              'transport=torch.uint8', 'shape=', tuple(local.shape),
              'bytes=', local.numel(), 'reference_bf16_bytes=', local.numel()*2,
              'dequantized_output=', str(output.dtype), 'exact_bf16_parity=True', flush=True)
    return output


def patch(source):
    method_anchor = '    def _context_parallel_compute_prefill_context('
    if source.count(method_anchor) != 1:
        raise ValueError('Expected exactly one DCP context method')
    method_start = source.index(method_anchor)
    next_method = source.find('\n    def ', method_start + len(method_anchor))
    method_end = next_method if next_method >= 0 else len(source)
    method = source[method_start:method_end]
    start_anchor = '            if self.kv_cache_dtype == "fp8_ds_mla":'
    end_anchor = '            dcp_manager.kv_gather(cur_allgather_kvcache, local_gathered_kvcache)'
    if method.count(start_anchor) != 1 or method.count(end_anchor) != 1:
        raise ValueError('Unexpected DCP cache gather layout')
    start = method.index(start_anchor)
    end = method.index(end_anchor, start) + len(end_anchor)
    old = method[start:end]
    new = ('            if self.kv_cache_dtype == "fp8":\n'
           '                from vllm.v1.attention.backends.mla.prefill import _fp8_kv_gather\n'
           '                cur_allgather_kvcache = _fp8_kv_gather.gather(\n'
           '                    self, kv_c_and_k_pe_cache, workspace, block_table, chunk, k_scale, chunked_context.dcp_manager)\n'
           '            else:\n') + '\n'.join('    ' + line for line in old.splitlines())
    result = source[:method_start] + method[:start] + new + method[end:] + source[method_end:]
    compile(result, '<fp8-kv-allgather>', 'exec')
    return result
