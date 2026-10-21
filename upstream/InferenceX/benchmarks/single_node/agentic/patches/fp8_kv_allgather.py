"""Transport stored FP8 MLA cache bytes; dequantize only after DCP all-gather."""
import torch
import triton
import triton.language as tl
from vllm import _custom_ops as ops
from vllm.platforms import current_platform


@triton.jit
def _decode(src, scales, dst, N: tl.constexpr, PER_RANK: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src + idx, idx < N).to(tl.float32)
    scale = tl.load(scales + idx // PER_RANK, idx < N, other=1.0)
    tl.store(dst + idx, value * scale, idx < N)


def validate_decode():
    """Exercise the real GPU decoder, including a masked tail and rank scales."""
    dtype = current_platform.fp8_dtype()
    for device_index in range(torch.cuda.device_count()):
        device = torch.device('cuda', device_index)
        per_rank = 1031
        raw = (torch.arange(2 * per_rank, device=device, dtype=torch.int32) % 256).to(torch.uint8)
        encoded = raw.view(dtype)
        scales = torch.tensor([0.5, 2.0], device=device, dtype=torch.float32)
        output = torch.empty(raw.numel(), device=device, dtype=torch.bfloat16)
        _decode[(triton.cdiv(raw.numel(), 1024),)](
            encoded, scales, output, raw.numel(), per_rank, 1024)
        expected = torch.cat([
            (encoded[:per_rank].float() * 0.5).bfloat16(),
            (encoded[per_rank:].float() * 2.0).bfloat16()])
        torch.testing.assert_close(output, expected, rtol=0, atol=0, equal_nan=True)
        torch.cuda.synchronize(device)
        print('[FP8-KV-DECODE-PREFLIGHT-PASSED]', 'device=', str(device),
              'encoding=', str(dtype), 'elements=', raw.numel(),
              'masked_tail=True', 'rank_scales_checked=True', flush=True)


def gather(backend, cache, workspace, block_table, chunk, scale, manager):
    from vllm.v1.attention.ops.rocm_aiter_mla_prefill import gather_compressed_context

    group = manager.group.device_group
    world = manager.group.world_size
    tokens = chunk.num_local_context_tokens
    width = workspace.shape[1]
    if backend.kv_cache_dtype != 'fp8' or width != 576:
        raise RuntimeError('Raw FP8 gather supports ordinary fp8 MLA cache width576')
    if cache.element_size() != 1 or workspace.dtype != torch.bfloat16:
        raise RuntimeError('Unexpected FP8 cache / BF16 reconstruction workspace')
    # Native byte gathering needs disjoint storage from the BF16 decode output.
    # Keep the buffer across calls so steady-state does not allocate.
    buffers = getattr(backend, '_raw_fp8_gather_buffers', None)
    if buffers is None:
        buffers = {'bytes': torch.empty(workspace.shape, dtype=torch.uint8, device=workspace.device),
                   'scales': torch.empty(world, dtype=torch.float32, device=workspace.device),
                   'verified': False}
        backend._raw_fp8_gather_buffers = buffers
        torch.distributed.all_gather_into_tensor(buffers['scales'], scale.reshape(1).float(), group=group)
    offset = workspace.shape[0] // (world + 1)
    output = workspace[offset:offset + tokens * world]
    if not buffers['verified']:
        ref = torch.empty_like(workspace[:tokens])
        ops.gather_and_maybe_dequant_cache(cache, ref, block_table,
            chunk.padded_local_cu_seq_lens, chunk.padded_local_token_to_seq,
            tokens, backend.kv_cache_dtype, scale, chunk.starts)
        expected = torch.empty_like(output)
        manager.kv_gather(expected, ref)
    gathered = gather_compressed_context(
        cache, buffers['bytes'], block_table, chunk, manager.kv_gather,
        current_platform.fp8_dtype())
    if tokens:
        _decode[(triton.cdiv(tokens * world * width, 1024),)](
            gathered, buffers['scales'], output,
            tokens * world * width, tokens * width, 1024)
    if not buffers['verified'] and tokens:
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        buffers['verified'] = True
        print('[FP8-KV-ALLGATHER-VERIFIED]', 'device=', str(cache.device),
              'cache_encoding=', str(current_platform.fp8_dtype()),
              'transport=torch.uint8', 'shape=', (tokens, width),
              'bytes=', tokens * width, 'reference_bf16_bytes=', tokens * width * 2,
              'native_gather_compressed_context=True',
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
