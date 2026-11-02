"""Prepare reviewable comparison patches; do not deploy or publish branches."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path.cwd()
SOURCE = ROOT / "review-vllm"
OUT = ROOT / "pr-review-evidence"
BASE = "a1cfd2c1b584c4ac197db9cc9ac84a6d973d39df"
NATIVE = "492080779e9feea19308092b0aba0840e87c6115"
ROCM = "vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
PARENT = "vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py"
REF_BACKEND = "vllm/v1/attention/backends/mla/prefill/aiter_flydsl_fp8.py"
STATE = "vllm/v1/attention/ops/rocm_aiter_mla_fp8_prefill.py"


def run(args, cwd, data=None, check=True):
    result = subprocess.run(args, cwd=cwd, input=data, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(f"{args}: {result.stdout[-6000:]} {result.stderr[-4000:]}")
    return result


def once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one adaptation anchor, found {count}: {old!r}")
    return text.replace(old, new, 1)


def modify(root, path, fn):
    file = root / path
    file.write_text(fn(file.read_text()))


def adapt_59124(text):
    text = once(
        text,
        "from vllm.v1.attention.backends.mla.triton_mla import _compute_num_kv_splits",
        "from vllm.v1.attention.backends.mla.prefill.aiter_flydsl_fp8 import (\n"
        "    AiterFlyDSLFP8PrefillBackend,\n)\n"
        "from vllm.v1.attention.backends.mla.triton_mla import _compute_num_kv_splits",
    )
    helper = (
        "def _asm_fp8_prefill_superseded(prefill_backend: object) -> bool:\n"
        "    return isinstance(prefill_backend, AiterFlyDSLFP8PrefillBackend)\n\n\n"
    )
    anchor = "@functools.lru_cache(maxsize=1)\ndef _fused_mla_prefill_qkv_supported"
    text = once(text, anchor, helper + anchor)
    text = once(
        text,
        "            and vllm_config.model_config.dtype == torch.bfloat16\n"
        "            and AiterMLAHelper.is_valid_fp8_prefill_num_heads(self.num_heads)\n",
        "            and vllm_config.model_config.dtype == torch.bfloat16\n"
        "            and AiterMLAHelper.is_valid_fp8_prefill_num_heads(self.num_heads)\n"
        "            and not _asm_fp8_prefill_superseded(self._prefill_backend)\n",
    )
    text = once(
        text,
        "            or attn_metadata.fp8_prefill_qo_indptr is None\n",
        "            or attn_metadata.prefill is None\n"
        "            or _asm_fp8_prefill_superseded(attn_metadata.prefill.prefill_backend)\n"
        "            or attn_metadata.fp8_prefill_qo_indptr is None\n",
    )
    return text


def reference_query_state(text):
    text = once(text, "from dataclasses import dataclass",
                "from collections.abc import Callable\nfrom dataclasses import dataclass")
    text = once(
        text,
        "        self.gather_scales: torch.Tensor | None = None\n",
        "        self.gather_scales: torch.Tensor | None = None\n"
        "        self.query_provider: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None\n",
    )
    start = text.index("            key = ", text.index("    def run("))
    end = text.index("            k8, v8, ks, vs = kv8", start)
    body = text[start:end]
    body = "".join("    " + line if line.strip() else line for line in body.splitlines(True))
    return (
        text[:start]
        + "            if self.query_provider is not None:\n"
        + "                q8, qs = self.query_provider(chunk, q)\n"
        + "            else:\n" + body + text[end:]
    )


def combine_59124(text):
    return once(
        text,
        "        self._q_fp8 = (q, q_fp8, q_descale)\n",
        "        self._q_fp8 = (q, q_fp8, q_descale)\n"
        "        if self.fp8_context is not None:\n"
        "            self.fp8_context.gather_scales = torch.cat((k_descale, v_descale)).clamp_min(1e-6) * 2\n"
        "            self.fp8_context.query_provider = self._get_q_fp8\n",
    )


def restore_62447_asm(text):
    text = once(
        text,
        "        self._fp8_prefill_enabled = (\n"
        "            (\n"
        "                not envs.VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL_ATTN\n"
        "                or vllm_config.model_config.runner_type == \"draft\"\n"
        "            )\n"
        "            and _fp8_mla_prefill_supported()\n"
        "        ) and (\n",
        "        self._fp8_prefill_enabled = _fp8_mla_prefill_supported() and (\n",
    )
    return once(
        text,
        "        self._fp8_prefill_enabled = (\n"
        "            not self._use_flydsl_fp8_prefill and _fp8_mla_prefill_supported()\n"
        "        ) and (\n",
        "        self._fp8_prefill_enabled = _fp8_mla_prefill_supported() and (\n",
    )


def combine_62447(reference, native_parent):
    reference = once(
        reference, "from typing import TYPE_CHECKING",
        "from collections.abc import Iterator\n"
        "from contextlib import contextmanager\n"
        "from typing import TYPE_CHECKING",
    )
    reference = once(
        reference, "    from vllm.platforms.interface import DeviceCapability\n",
        "    from vllm.platforms.interface import DeviceCapability\n"
        "    from vllm.v1.attention.ops.rocm_aiter_mla_fp8_prefill import FP8PrefillContext\n",
    )
    reference = once(
        reference, "        self.flash_attn_varlen_func = flash_attn_varlen_func\n",
        "        self.flash_attn_varlen_func = flash_attn_varlen_func\n"
        "        self.fp8_context: FP8PrefillContext | None = None\n",
    )
    scope_start = native_parent.index("    @contextmanager\n")
    scope_end = native_parent.index("    def run_prefill_new_tokens(", scope_start)
    scope = native_parent[scope_start:scope_end]
    reference = once(reference, "    def _fp8_attn(", scope + "    def _fp8_attn(")
    reference = once(
        reference, "        assert self.fp8_attn_func is not None\n",
        "        if self.fp8_context is not None and kwargs.get(\"causal\", False):\n"
        "            self.fp8_context.gather_scales = torch.cat((k_descale, v_descale)).clamp_min(1e-6) * 2\n"
        "            self.fp8_context.query_provider = self._get_direct_context_q\n"
        "        assert self.fp8_attn_func is not None\n",
    )
    query_method = (
        "    def _get_direct_context_q(self, chunk, q):\n"
        "        quant = _quant_fp8_qkv_per_tensor(q, q[:0], q[:0, :, :128])\n"
        "        return quant[0], quant[3]\n\n"
    )
    reference = once(reference, "    def run_prefill_new_tokens(",
                     query_method + "    def run_prefill_new_tokens(")
    reference = once(
        reference, "        result = self.flash_attn_varlen_func(\n",
        "        if self.fp8_context is not None:\n"
        "            self.fp8_context.prepare(q, k, v)\n"
        "        result = self.flash_attn_varlen_func(\n",
    )
    return reference


native_parent = run(["git", "show", f"{NATIVE}:{PARENT}"], SOURCE).stdout
reference_parent = run(
    ["gh", "api", "repos/vllm-project/vllm/contents/" + PARENT
     + "?ref=cefacabced64eda821bcb49493cc970056d15429",
     "-H", "Accept: application/vnd.github.raw+json"], ROOT
).stdout
manifest = {"common_base": BASE, "native_head": NATIVE, "arms": {},
            "scope": "Prepared patches and static checks only; no GPU execution",
            "limitations": [
                "Direct arms retain new-token-derived K/V ranges; references use actual-context maxima.",
                "Runtime backend selection, numerical tolerances and eight-worker execution are unvalidated.",
                "Small chunks in mixed direct forwards project to BF16 then retain reference attention.",
            ]}
for number in (59124, 62447):
    for direct in (False, True):
        label = f"pr{number}" + ("-direct" if direct else "-alone")
        root = ROOT / "review-arms" / label
        source_sha = NATIVE if direct else BASE
        run(["git", "worktree", "add", "--detach", str(root), source_sha], SOURCE)
        patch = (OUT / f"pr-{number}.diff").read_text()
        args = ["git", "apply"]
        if number == 59124:
            args.append("--exclude=" + ROCM)
        elif direct:
            args.append("--exclude=" + PARENT)
        args.append("-")
        run(args, root, data=patch)
        if number == 59124:
            modify(root, ROCM, adapt_59124)
            if direct:
                modify(root, REF_BACKEND, combine_59124)
        elif direct:
            modify(root, ROCM, restore_62447_asm)
            (root / PARENT).write_text(combine_62447(reference_parent, native_parent))
        if direct:
            modify(root, STATE, reference_query_state)
        run(["git", "add", "-A"], root)
        changed = run(["git", "diff", "--cached", "--name-only", BASE], root).stdout.splitlines()
        pyfiles = [p for p in changed if p.endswith(".py")]
        if pyfiles:
            run([str(ROOT / ".venv/bin/ruff"), "check", "--fix", *pyfiles], root)
            run([str(ROOT / ".venv/bin/ruff"), "format", *pyfiles], root)
        for p in pyfiles:
            ast.parse((root / p).read_text(), filename=p)
        run(["git", "add", "-A"], root)
        run(["git", "diff", "--cached", "--check"], root)
        patch_text = run(["git", "diff", "--cached", BASE], root).stdout
        (OUT / f"{label}.patch").write_text(patch_text)
        arm_out = OUT / label
        for path in changed:
            src = root / path
            if src.is_file():
                dst = arm_out / path
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())
        manifest["arms"][label] = {
            "source_sha": source_sha,
            "files": {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                      for p in changed if (root / p).is_file()},
            "attention": "ROCM_AITER_FLYDSL_FP8" if number == 59124 else "ROCM_AITER_FA",
            "direct_projection": direct,
            "q_behavior": "reuse full-forward Q slices" if number == 59124 else "per-call Q quantization",
            "causal_routing": "reference retained",
            "checks": ["AST", "Ruff", "format", "diff whitespace"],
        }
        (OUT / "comparison-arms.json").write_text(json.dumps(manifest, indent=2)+"\n")
print(json.dumps(manifest, indent=2))
