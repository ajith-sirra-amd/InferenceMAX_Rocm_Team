import json
import pathlib
import py_compile
import sys


def patch(source):
    anchor = ("        out, lse = self.flash_attn_varlen_func(\n"
              "            q=q,\n            k=k,\n            v=v,\n"
              "            cu_seqlens_q=chunk.query_start_loc,\n")
    if source.count(anchor) != 1:
        raise ValueError("Expected one context call anchor; refusing unverified patch")
    addition = (
        "        if not getattr(self, '_attnfp8_inputs_logged', False):\n"
        "            self._attnfp8_inputs_logged = True\n"
        "            import os\n"
        "            print('[ATTNFP8-CONTEXT-INPUT]', 'pid=', os.getpid(), "
        "'device=', str(q.device), 'q=', tuple(q.shape), str(q.dtype), "
        "'k=', tuple(k.shape), str(k.dtype), 'v=', tuple(v.shape), str(v.dtype), "
        "'causal=False', 'extra_quantization=False', flush=True)\n")
    result = source.replace(anchor, addition + anchor, 1)
    compile(result, '<context-probe>', 'exec')
    return result


def install(model_path, result_dir):
    import vllm
    config = json.loads((pathlib.Path(model_path) / 'config.json').read_text())
    record = {'model_path': model_path, 'vllm_version': vllm.__version__,
              'model_type': config.get('model_type'),
              'dtype': config.get('dtype', config.get('torch_dtype')),
              'quantization_config': config.get('quantization_config'),
              'context_extra_quantization': False}
    pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(result_dir, 'attnfp8_checkpoint_config.json').write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    backend = pathlib.Path(vllm.__file__).parent / 'v1/attention/backends/mla/prefill/aiter_flash_attn.py'
    backend.write_text(patch(backend.read_text()), encoding='utf-8')
    py_compile.compile(str(backend), doraise=True)
    print('[ATTNFP8-PROBE] checkpoint configuration retained; dtype probe installed')


if __name__ == '__main__':
    install(sys.argv[1], sys.argv[2])
