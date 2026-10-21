"""Install observation-only probes for native DCP context dispatch and KV transport."""
import pathlib
import sys


def patch_backend(source):
    start_anchor = '    def _context_parallel_compute_prefill_context('
    if source.count(start_anchor) != 1:
        raise ValueError('Unexpected native context method layout')
    start = source.index(start_anchor)
    stop = source.find('\n    def ', start + len(start_anchor))
    stop = len(source) if stop < 0 else stop
    method = source[start:stop]
    anchor = '        if not eligible:\n'
    if method.count(anchor) != 1:
        raise ValueError('Unexpected native eligibility layout')
    probe = """        if not getattr(self, '_native_dcp_probe_logged', False):
            self._native_dcp_probe_logged = True
            print('[NATIVE-DCP-ELIGIBILITY]', 'device=', str(q.device),
                  'eligible=', eligible, 'cache_dtype=', self.kv_cache_dtype,
                  'q_dtype=', str(q.dtype), 'prefill_q_dtype=', str(prefill.q_data_type),
                  'weight_dtype=', str(getattr(weight, 'dtype', None)),
                  'weight_shape=', tuple(weight.shape) if weight is not None else None,
                  'weight_contiguous=', weight.is_contiguous() if weight is not None else None,
                  'weight_scale_present=', getattr(self.kv_b_proj, 'weight_scale', None) is not None,
                  'bias_present=', getattr(self.kv_b_proj, 'bias', None) is not None,
                  'cache_contiguous=', kv_c_and_k_pe_cache.is_contiguous(),
                  'context_rows_present=', attn_metadata.dcp_context_row_indices is not None,
                  flush=True)
"""
    method = method.replace(anchor, probe + anchor)
    result = source[:start] + method + source[stop:]
    compile(result, '<native-dcp-probe>', 'exec')
    return result


def patch_collective(source):
    anchor = '        return self._kv_gather(gathered_kv, local_kv)'
    if source.count(anchor) != 1:
        raise ValueError('Unexpected native KV collective layout')
    probe = """        key = str(local_kv.dtype)
        seen = getattr(self, '_native_kv_probe_seen', None)
        if seen is None:
            seen = set()
            self._native_kv_probe_seen = seen
        if local_kv.numel() and key not in seen:
            seen.add(key)
            print('[NATIVE-KV-COLLECTIVE]', 'device=', str(local_kv.device),
                  'dtype=', key, 'shape=', tuple(local_kv.shape),
                  'local_bytes=', local_kv.numel() * local_kv.element_size(),
                  'gathered_dtype=', str(gathered_kv.dtype),
                  'gathered_bytes=', gathered_kv.numel() * gathered_kv.element_size(),
                  'transport_unmodified=True', flush=True)
"""
    result = source.replace(anchor, probe + anchor)
    compile(result, '<native-kv-probe>', 'exec')
    return result


def install(site):
    root = pathlib.Path(site)
    for relative, patcher in [
        ('v1/attention/backends/mla/rocm_aiter_mla.py', patch_backend),
        ('v1/attention/ops/dcp.py', patch_collective),
    ]:
        path = root / relative
        path.write_text(patcher(path.read_text()))
    print('[NATIVE-DCP-PROBE] installed; native dispatch and transport unchanged')


if __name__ == '__main__':
    import vllm
    install(pathlib.Path(vllm.__file__).parent)
