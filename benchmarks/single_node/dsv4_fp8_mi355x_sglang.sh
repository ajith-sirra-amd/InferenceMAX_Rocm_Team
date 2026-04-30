#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL"

# Transformers in the container doesn't recognize the `deepseek_v4` model_type.
# PR #23608's fallback in hf_transformers_utils.get_config tries to handle this
# by writing a patched config to /tmp, but in practice isn't catching the error
# in this image. Patch the cached config.json directly instead: set model_type
# to `deepseek_v3` so AutoConfig.from_pretrained succeeds, and keep
# architectures=['DeepseekV4ForCausalLM'] so SGLang dispatches to its native
# DSv4 model class (python/sglang/srt/models/deepseek_v4.py).
python3 << PYEOF
import json
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="$MODEL", filename="config.json")
with open(path) as f:
    config = json.load(f)
if config.get("model_type") == "deepseek_v4":
    config["model_type"] = "deepseek_v3"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Patched {path}: model_type deepseek_v4 -> deepseek_v3")
else:
    print(f"No patch needed: model_type is {config.get('model_type')!r}")
PYEOF

PATCH_FILE="/tmp/pr_24127_layernorm_dsv4.patch"
TARGET_FILE="/sgl-workspace/sglang-dsv4-0426/python/sglang/srt/layers/layernorm.py"

# Write and apply layernorm patch from PR #24127 https://github.com/sgl-project/sglang/pull/24127/changes
cat > $PATCH_FILE << 'PATCH_EOF'
--- /sgl-workspace/sglang-dsv4-0426/python/sglang/srt/layers/layernorm.py	2026-04-30 10:05:36.665047183 +0000
+++ /sgl-workspace/sglang-dsv4-0426/python/sglang/srt/layers/layernorm.py	2026-04-30 10:06:28.805281594 +0000
@@ -200,6 +200,10 @@
         )
         if _use_aiter:
             self._forward_method = self.forward_aiter
+        # if get_bool_env_var("SGLANG_USE_NATIVE_LAYERNORM"):
+        #    self._forward_method = self.forward_native
+        # elif _use_aiter:
+        #    self._forward_method = self.forward_aiter
 
     def forward_cuda(
         self,
@@ -284,6 +288,13 @@
         residual: Optional[torch.Tensor] = None,
         post_residual_addition: Optional[torch.Tensor] = None,
     ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
+
+        # Fix dsv4 dp attenton issue
+        # the symptom is torch.AcceleratorError: HIP error: invalid configuration argument
+        if x.shape[0] == 0:
+            if residual is not None:
+                return x, residual
+            return x
         if residual is not None:
             residual_out = torch.empty_like(x)
             output = torch.empty_like(x)

PATCH_EOF

PATCH_MARKER="Fix dsv4 dp attenton issue"

if [[ ! -f "$TARGET_FILE" ]]; then
    echo "ERROR: target file not found: $TARGET_FILE"
    exit 1
fi

if grep -Fq "$PATCH_MARKER" "$TARGET_FILE"; then
    echo "Layernorm patch already present; skipping apply."
else
    if patch --forward "$TARGET_FILE" < "$PATCH_FILE"; then
        echo "Patch command completed."
    else
        echo "ERROR: failed to apply patch file: $PATCH_FILE"
        exit 1
    fi

    if grep -Fq "$PATCH_MARKER" "$TARGET_FILE"; then
        echo "Layernorm patch applied successfully."
    else
        echo "ERROR: patch command ran but expected marker not found in $TARGET_FILE"
        exit 1
    fi
fi

export SGLANG_REASONING_EFFORT=max

export SGLANG_OPT_USE_FUSED_COMPRESS=false #use PyTorch implemented compressor
export SGLANG_OPT_USE_OLD_COMPRESSOR=true #use old compressor
export SGLANG_OPT_USE_TILELANG_SWA_PREPARE=false #use old prepare
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=false #use old topk
export SGLANG_OPT_USE_FUSED_HASH_TOPK=false #AMD: hash_topk JIT needs CUDA toolchain

export SGLANG_HACK_FLASHMLA_BACKEND=torch
export SGLANG_HACK_FLASHMLA_BACKEND=tilelang
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false #use old prenorm

export SGLANG_OPT_USE_TILELANG_MHC_PRE=false #use torch hc_pre
export SGLANG_OPT_USE_TILELANG_MHC_POST=false #use torch hc_post

export SGLANG_ENABLE_THINKING=1
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
export SGLANG_TOPK_TRANSFORM_512_TORCH=1
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1

export SGLANG_DSV4_FP4_EXPERTS=false

export SGLANG_OPT_DPSK_V4_RADIX=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false #non-radix backend has no store_cache method
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false #fused_store_cache JIT needs CUDA toolchain

export SGLANG_FORCE_TRITON_MOE_FP8=1  # this is required to apply swiglu_limit clamp in fused_moe_triton

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi
# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

export PYTHONDONTWRITEBYTECODE=1
python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --dp $TP \
    --enable-dp-attention \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend compressed \
    --max-running-request 256 \
    --page-size 256 \
    --chunked-prefill-size 8192 \
    --disable-shared-experts-fusion \
    --disable-cuda-graph \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    --watchdog-timeout 1800 $EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x