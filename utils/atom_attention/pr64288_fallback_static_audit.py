"""Structural audit of the mixed-context fallback refactor; no GPU claims."""
import ast
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
parent = "a03796b8e3bfe536e22f9db741109dbf80170079"
common = "vllm/model_executor/layers/attention/mla_attention.py"
fp8 = "vllm/v1/attention/ops/rocm_aiter_mla_fp8_prefill.py"
backend = "vllm/v1/attention/backends/mla/rocm_aiter_mla.py"


def before(path):
    return subprocess.check_output(
        ["git", "-C", str(root), "show", f"{parent}:{path}"]
    ).decode("utf-8")


def function(source, name):
    matches = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def signature(nodes):
    return [ast.dump(n, include_attributes=False) for n in nodes]


def find(nodes, predicate):
    return next(i for i, n in enumerate(nodes) if predicate(n))


def is_chunk_assert(node):
    return isinstance(node, ast.Assert) and (
        "chunk.padded_local_seq_lens" in ast.unparse(node)
    )


old_common = before(common)
new_common = (root / common).read_text()
old_fn = function(old_common, "_context_parallel_compute_prefill_context")
new_fn = function(new_common, "_context_parallel_compute_prefill_context")
old_loop = next(n for n in old_fn.body if isinstance(n, ast.For))
new_loop = next(n for n in new_fn.body if isinstance(n, ast.For))
helper = function(new_common, "_context_parallel_gather_prefill_kv")
old_start = find(old_loop.body, is_chunk_assert)
old_end = find(old_loop.body, lambda n: isinstance(n, ast.Assign)
               and "attn_output" in ast.unparse(n.targets[0]))
helper_start = find(helper.body, is_chunk_assert)
assert signature(old_loop.body[old_start:old_end]) == signature(
    helper.body[helper_start:-1]
), "Native gather/projection/packing operations changed"
assert ast.unparse(helper.body[-1]) == "return (k, v)"
assert signature(old_loop.body[old_end:]) == signature(new_loop.body[1:]), (
    "Native attention or accumulation changed"
)

old_fp8 = function(before(fp8), "compute_context")
new_fp8 = function((root / fp8).read_text(), "compute_context")
old_loop = next(n for n in old_fp8.body if isinstance(n, ast.For))
new_loop = next(n for n in new_fp8.body if isinstance(n, ast.For))
old_if = next(n for n in old_loop.body if isinstance(n, ast.If))
new_if = next(n for n in new_loop.body if isinstance(n, ast.If))
assert signature([old_if.test]) == signature([new_if.test])
assert signature([old_loop.body[0], *old_if.body]) == signature(new_if.body), (
    "Eligible FP8 gather/projection/attention changed"
)
assert "impl._context_parallel_gather_prefill_kv(" in ast.unparse(new_if.orelse[0])
assert signature(old_if.orelse[1:]) == signature(new_if.orelse[1:])
assert signature(old_loop.body[3:]) == signature(new_loop.body[2:]), (
    "FP8 context accumulation changed"
)
quant = "vllm/v1/attention/ops/rocm_aiter_mla_fp8_quant.py"
assert before(quant) == (root / quant).read_text()
for path in (common, fp8, backend):
    compile((root / path).read_text(), path, "exec")
print(json.dumps({
    "status": "passed",
    "scope": "syntax and AST equivalence, not numerical or GPU execution",
    "native_gather_projection_pack": "unchanged operations",
    "native_attention_and_merge": "unchanged operations",
    "eligible_fp8_pipeline": "unchanged operations",
    "quantization_kernels": "byte-identical",
}))
