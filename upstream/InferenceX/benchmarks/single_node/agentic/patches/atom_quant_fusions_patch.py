# SPDX-License-Identifier: Apache-2.0
"""Install narrow producer-quant fusions into the exact pinned ROCm Kimi sources."""
import pathlib
import shutil


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Unexpected pinned source anchor: {old!r}")
    return source.replace(old, new)


def patch_sources(kda, mla, linear, pytorch):
    kda = replace_once(kda, "        g1 = self.f_b_proj(f_a)[0]",
        "        from vllm.models.kimi_k3.amd._infx_atom_quant import strided_input\n"
        "        g1 = self.f_b_proj(strided_input(f_a, self.f_b_proj))[0]")
    old = '''            if rocm_aiter_ops.is_enabled():
                from aiter.ops.triton.fusions.fused_sigmoid_mul import fused_sigmoid_mul

                attn_out = fused_sigmoid_mul(attn_out, gate)
            else:
                attn_out = attn_out * gate.sigmoid()'''
    mla = replace_once(mla, old,
        "            from vllm.models.kimi_k3.amd._infx_atom_quant import gated_input\n"
        "            attn_out = gated_input(attn_out, gate, self.o_proj)")
    linear = replace_once(linear,
        "        x = self.act_fn(gate_up)\n        x, _ = self.down_proj(x)",
        "        from vllm.models.kimi_k3.amd._infx_atom_quant import activation_input\n"
        "        x = activation_input(gate_up, self.act_fn, self.down_proj)\n"
        "        x, _ = self.down_proj(x)")
    # Rowwise torch GEMM already consumes supplied (M,1) scales. Expose that
    # capability only when its usual token-padding requirement is absent.
    anchor = "class RowWiseTorchFP8ScaledMMLinearKernel(TorchFP8ScaledMMLinearKernel):\n"
    pytorch = replace_once(pytorch, anchor, anchor + '''
    def input_quant_key(self) -> QuantKey | None:
        from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym
        if (self.config.activation_quant_key == kFp8DynamicTokenSym
                and self.get_output_padding() is None):
            return kFp8DynamicTokenSym
        return None

''')
    result = (kda, mla, linear, pytorch)
    for name, text in zip(("kda", "mla", "linear", "pytorch"), result):
        compile(text, f"<infx-atom-{name}>", "exec")
    return result


def install():
    import vllm
    root = pathlib.Path(vllm.__file__).parent
    paths = (
        root / "models/kimi_k3/amd/kda.py",
        root / "models/kimi_k3/amd/mla.py",
        root / "models/kimi_k3/amd/linear.py",
        root / "model_executor/kernels/linear/scaled_mm/pytorch.py",
    )
    texts = patch_sources(*(path.read_text() for path in paths))
    shutil.copyfile(pathlib.Path(__file__).with_name("atom_quant_fusions.py"),
                    root / "models/kimi_k3/amd/_infx_atom_quant.py")
    shutil.copyfile(pathlib.Path(__file__).with_name("atom_quant_fusions_probe.py"),
                    root / "models/kimi_k3/amd/_infx_atom_probe.py")
    for path, text in zip(paths, texts):
        path.write_text(text)
    print("[ATOM-QUANT-FUSION-INSTALLED] modes=0:strided,1:sigmoid_mul,2:SiTU; rowwise_prequantized_input=True", flush=True)


if __name__ == "__main__":
    install()
