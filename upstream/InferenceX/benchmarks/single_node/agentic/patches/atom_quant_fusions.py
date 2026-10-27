# SPDX-License-Identifier: Apache-2.0
"""ATOM-inspired producer-side FP8 quantization for the pinned Kimi ROCm model."""
import torch
import triton
import triton.language as tl
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation, get_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym

_seen = set()


@triton.jit
def _producer_quant(X, G, Y, S, D: tl.constexpr, SX0: tl.constexpr,
                    SX1: tl.constexpr, SG0: tl.constexpr, SG1: tl.constexpr,
                    MODE: tl.constexpr, BETA: tl.constexpr, LBETA: tl.constexpr,
                    FMAX: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row * SX0 + col * SX1, col < D, 0).to(tl.float32)
    if MODE == 1:
        gate = tl.load(G + row * SG0 + col * SG1, col < D, 0).to(tl.float32)
        value = value / (1.0 + tl.exp(-gate))
    elif MODE == 2:
        up = tl.load(X + row * SX0 + (col + D) * SX1, col < D, 0).to(tl.float32)
        gate_tanh = 2.0 / (1.0 + tl.exp(-2.0 * value / BETA)) - 1.0
        gate = BETA * gate_tanh / (1.0 + tl.exp(-value))
        if LBETA > 0:
            up = LBETA * (2.0 / (1.0 + tl.exp(-2.0 * up / LBETA)) - 1.0)
        value = gate * up
    # Match the BF16 producer rounding before quantizing, without an HBM buffer.
    value = value.to(X.dtype.element_ty).to(tl.float32)
    value = tl.where(col < D, value, 0.0)
    scale = tl.maximum(tl.max(tl.abs(value), 0) / FMAX, 1.0e-12)
    quantized = tl.minimum(tl.maximum(value / scale, -FMAX), FMAX)
    tl.store(Y + row * D + col, quantized, col < D)
    tl.store(S + row, scale)


def producer_quant(x: torch.Tensor, gate: torch.Tensor, mode: int,
                   beta: float, linear_beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    if mode not in (0, 1, 2):
        raise ValueError(f"Unsupported producer mode: {mode}")
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise ValueError(f"Expected two-dimensional BF16 producer, got {x.shape} {x.dtype}")
    width = x.shape[1] // 2 if mode == 2 else x.shape[1]
    if width <= 0:
        raise ValueError("Producer width must be positive")
    if mode == 2 and x.shape[1] % 2:
        raise ValueError("SiTU input must contain equal gate and up halves")
    if gate.device != x.device or gate.dtype != x.dtype:
        raise ValueError("Producer and gate must have identical device and dtype")
    if mode == 1 and gate.shape != x.shape:
        raise ValueError("Sigmoid gate and attention output shapes differ")
    if mode == 2 and beta <= 0:
        raise ValueError("SiTU beta must be positive")
    dtype = current_platform.fp8_dtype()
    output = torch.empty((x.shape[0], width), dtype=dtype, device=x.device)
    scales = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
    if x.shape[0]:
        # Triton resolves the launch stream on the current device. A caller may
        # hold tensors on another GPU (the all-GPU preflight does exactly that).
        with torch.cuda.device(x.device):
            _producer_quant[(x.shape[0],)](
                x, gate, output, scales, width, x.stride(0), x.stride(1),
                gate.stride(0), gate.stride(1), mode, beta, linear_beta,
                float(torch.finfo(dtype).max), triton.next_power_of_2(width),
                num_warps=4 if width <= 2048 else 8, enable_fp_fusion=False,
            )
    key = (mode, str(x.device))
    if x.shape[0] and key not in _seen:
        _seen.add(key)
        print("[ATOM-QUANT-FUSION-EXECUTED]", "device=", str(x.device),
              "mode=", mode, "input_dtype=", str(x.dtype),
              "output_dtype=", str(output.dtype), "width=", width,
              "input_stride=", x.stride(), "bf16_intermediate_buffer=False",
              flush=True)
    return output, scales


def producer_quant_fake(x: torch.Tensor, gate: torch.Tensor, mode: int,
                        beta: float, linear_beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    width = x.shape[1] // 2 if mode == 2 else x.shape[1]
    return (
        x.new_empty((x.shape[0], width), dtype=current_platform.fp8_dtype()),
        x.new_empty((x.shape[0], 1), dtype=torch.float32),
    )


direct_register_custom_op(
    op_name="infx_atom_producer_quant",
    op_func=producer_quant,
    mutates_args=[],
    fake_impl=producer_quant_fake,
)


def _wrap(x, layer, mode, gate=None, beta=1.0, linear_beta=-1.0):
    key = get_input_quant_key(layer)
    if key != kFp8DynamicTokenSym:
        raise RuntimeError(f"Producer quant requires per-token FP8 input capability: {type(layer.quant_method).__name__} {key}")
    data, scale = torch.ops.vllm.infx_atom_producer_quant(
        x, x if gate is None else gate, mode, beta, linear_beta,
    )
    shape = torch.Size((x.shape[0], x.shape[1] // 2 if mode == 2 else x.shape[1]))
    return QuantizedActivation(data, scale, x.dtype, shape, key)


def strided_input(x, layer):
    return _wrap(x, layer, 0)


def gated_input(x, gate, layer):
    return _wrap(x, layer, 1, gate=gate)


def activation_input(x, activation, layer):
    if get_input_quant_key(layer) is None:
        # Untargeted BF16 draft consumers keep their existing producer.
        weight = getattr(layer, "weight", None)
        if weight is not None and weight.dtype in (torch.bfloat16, torch.float32):
            return activation(x)
        raise RuntimeError("FP8 MLP consumer does not advertise compatible fused input")
    if not hasattr(activation, "beta"):
        raise RuntimeError(f"Expected Kimi SiTU activation, got {type(activation).__name__}")
    linear_beta = activation.linear_beta
    return _wrap(x, layer, 2, beta=float(activation.beta),
                 linear_beta=-1.0 if linear_beta is None else float(linear_beta))


def preflight():
    # Actual GPU execution, not model accuracy evaluation. Real TP8 producer widths.
    if torch.cuda.device_count() != 8:
        raise RuntimeError("Producer preflight requires eight visible GPUs")
    for device in range(8):
        torch.manual_seed(17)
        dev = f"cuda:{device}"
        print("[ATOM-PREFLIGHT-DEVICE]", "tensor_device=", dev,
              "caller_current_device=", torch.cuda.current_device(), flush=True)
        cases = 0
        for rows in (0, 1, 33, 256):
            for mode, width in ((0, 128), (1, 1536), (2, 768), (2, 4224)):
                storage = torch.randn((rows, (2 if mode == 2 else 1) * width + 32),
                                      device=dev, dtype=torch.bfloat16)
                x = storage[:, : (2 if mode == 2 else 1) * width]
                gate = torch.randn_like(x)
                for beta, linear_beta in ((4.0, 25.0), (4.0, -1.0)):
                    for zero in (False, True):
                        if zero:
                            x.zero_()
                        else:
                            x.copy_(torch.randn_like(x))
                        q, scale = torch.ops.vllm.infx_atom_producer_quant(
                            x, gate, mode, beta, linear_beta)
                        raw = x.float()
                        if mode == 1:
                            ref = (raw * gate.float().sigmoid()).to(x.dtype).float()
                        elif mode == 2:
                            a, b = raw.chunk(2, dim=-1)
                            clipped = linear_beta * (b / linear_beta).tanh() if linear_beta > 0 else b
                            ref = (beta * (a / beta).tanh() * a.sigmoid() * clipped).to(x.dtype).float()
                        else:
                            ref = raw
                        decoded = q.float() * scale
                        if rows:
                            relative = (decoded - ref).square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-6)
                            if not torch.isfinite(decoded).all() or relative.item() > 0.04:
                                raise RuntimeError(f"Producer numerical failure: device={device} mode={mode} relative_rms={relative.item()}")
                        elif q.shape != (0, width) or scale.shape != (0, 1):
                            raise RuntimeError("Empty producer shape mismatch")
                        cases += 1
                if rows == 33:
                    # Exercise the actual hipBLASLt FP8 scaled-MM consumer contract.
                    x.copy_(torch.randn_like(x))
                    q, scale = torch.ops.vllm.infx_atom_producer_quant(x, gate, mode, 4.0, 25.0)
                    w = torch.randn((64, width), device=dev, dtype=torch.bfloat16).float()
                    ws = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) / torch.finfo(q.dtype).max
                    wq = (w / ws).to(q.dtype)
                    out = torch._scaled_mm(q, wq.t(), scale_a=scale, scale_b=ws.t(), out_dtype=torch.bfloat16)
                    if isinstance(out, tuple):
                        out = out[0]
                    ref = (q.float() * scale) @ (wq.float() * ws).t()
                    relative = (out.float() - ref).square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-6)
                    if out.dtype != torch.bfloat16 or not torch.isfinite(out).all() or relative.item() > 0.02:
                        raise RuntimeError(f"FP8 consumer failure: device={device} mode={mode} relative_rms={relative.item()}")
        torch.cuda.synchronize(device)
        print("[ATOM-QUANT-FUSION-PREFLIGHT]", "device=", dev,
              "modes=0,1,2", "cases=", cases, "strided_zero_empty_cases=True",
              "fp8_scaled_mm=True", "passed=True", flush=True)


if __name__ == "__main__":
    preflight()
