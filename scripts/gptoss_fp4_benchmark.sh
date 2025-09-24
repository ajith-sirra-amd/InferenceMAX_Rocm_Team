#!/bin/bash

# InferenceMAX benchmark script for MI300X and MI325X GPUs
# Both GPUs use identical settings for GPT-OSS models

# Usage: ./gptoss_fp4_benchmark.sh <gpu_type> <tp> <model_path>
# Example: ./gptoss_fp4_benchmark.sh mi300x 1 /home/amd/models/gpt-oss-120b
# Example: ./gptoss_fp4_benchmark.sh mi300x 8 /home/amd/models/gpt-oss-120b
# Example: ./gptoss_fp4_benchmark.sh mi325x 1 /data/models/gpt-oss-120b
# Example: ./gptoss_fp4_benchmark.sh mi325x 8 /data/models/gpt-oss-120b

GPU_TYPE="${1:-mi300x}"
TP="${2:-1}"
MODEL_PATH="${3:-/home/amd/models/gpt-oss-120b}"

# Validate GPU type
if [[ "$GPU_TYPE" != "mi300x" && "$GPU_TYPE" != "mi325x" ]]; then
    echo "ERROR: Invalid GPU type. Use 'mi300x' or 'mi325x'"
    echo "Usage: $0 <gpu_type> <tp> <model_path>"
    exit 1
fi

# Validate model path
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path does not exist: $MODEL_PATH"
    echo "Usage: $0 <gpu_type> <tp> <model_path>"
    echo "Example: $0 mi300x 1 /home/amd/models/gpt-oss-120b"
    exit 1
fi

# Extract model name from path
MODEL_NAME=$(basename "$MODEL_PATH")

# Docker image and port
IMAGE="${IMAGE:-rocm/7.0:rocm7.0_ubuntu_22.04_vllm_0.10.1_instinct_20250915}"
PORT=8888

# Results directory with current date
DATE=$(date +%Y%m%d)
RESULTS_DIR="./inferencemax_results_${DATE}"
mkdir -p $RESULTS_DIR

echo "=========================================="
echo "InferenceMAX Benchmarks"
echo "GPU: $GPU_TYPE | TP: $TP"
echo "Model: $MODEL_PATH"
echo "Results: $RESULTS_DIR"
echo "=========================================="

# No need to clone bench_serving - using vllm bench serve directly

# Pull Docker image if not present
if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${IMAGE}$"; then
    echo "Pulling Docker image: $IMAGE"
    docker pull $IMAGE
fi

# Test configurations: (ISL, OSL) combinations
ISL_OSL_CONFIGS=(
    "1024 1024"
    "8192 1024"
    "1024 8192"
)

# Concurrency levels
CONCURRENCY_LEVELS=(4 8 16 32 64)

# Set environment variables (same for both MI300X and MI325X)
set_env_vars() {
    export HF_HUB_OFFLINE=1
    export HSA_NO_SCRATCH_RECLAIM=1
    export NCCL_MIN_NCHANNELS=112
    export VLLM_ROCM_USE_AITER=1
    export VLLM_USE_AITER_UNIFIED_ATTENTION=1
    export VLLM_ROCM_USE_AITER_MHA=0
    export VLLM_ROCM_USE_AITER_TRITON_BF16_GEMM=0
    export ROCM_TRITON_MOE_PRESHUFFLE_SCALES=0
}


for ISL_OSL in "${ISL_OSL_CONFIGS[@]}"; do
    ISL=$(echo $ISL_OSL | cut -d' ' -f1)
    OSL=$(echo $ISL_OSL | cut -d' ' -f2)

    echo ""
    echo "=========================================="
    echo "Testing ISL=${ISL}, OSL=${OSL}"
    echo "=========================================="

    for CONC in "${CONCURRENCY_LEVELS[@]}"; do
        echo ""
        echo "[TP=$TP, ISL=$ISL, OSL=$OSL, CONC=$CONC] Starting..."

        # Filename following oobx convention: VLLM_gpu_model_tp_isl_osl_c_s_mnbt.json
        RESULT_FILENAME="VLLM_${GPU_TYPE}_${MODEL_NAME}_tp${TP}_isl${ISL}_osl${OSL}_c${CONC}_s${CONC}_mnbt8192.json"
        RESULT_FILE="$RESULTS_DIR/$RESULT_FILENAME"

        # Skip if results already exist
        if [ -f "$RESULT_FILE" ] && [ -s "$RESULT_FILE" ]; then
            echo "[TP=$TP, ISL=$ISL, OSL=$OSL, CONC=$CONC] SKIPPED - Results already exist: $RESULT_FILENAME"
            continue
        fi

        # Clean up any existing containers
        docker stop vllm-server >/dev/null 2>&1
        docker rm vllm-server >/dev/null 2>&1

        # Set environment variables
        set_env_vars

        # Calculate max model length based on ISL+OSL
        MAX_MODEL_LEN=$((ISL + OSL))

        # Build environment variables for Docker
        ENV_VARS=""
        for var in HF_HUB_OFFLINE HSA_NO_SCRATCH_RECLAIM NCCL_MIN_NCHANNELS \
                   VLLM_ROCM_USE_AITER VLLM_USE_AITER_UNIFIED_ATTENTION \
                   VLLM_ROCM_USE_AITER_MHA VLLM_ROCM_USE_AITER_TRITON_BF16_GEMM \
                   ROCM_TRITON_MOE_PRESHUFFLE_SCALES; do
            ENV_VARS="$ENV_VARS -e $var=${!var}"
        done

        # Start vLLM server (exactly like reference scripts)
        docker run --rm -d \
            --name vllm-server \
            --ipc=host \
            --shm-size=16g \
            --privileged \
            --cap-add=CAP_SYS_ADMIN \
            --device=/dev/kfd \
            --device=/dev/dri \
            --device=/dev/mem \
            --cap-add=SYS_PTRACE \
            --security-opt seccomp=unconfined \
            -v $MODEL_PATH:/model:ro \
            -v $RESULTS_DIR:/results \
            -p ${PORT}:${PORT} \
            $ENV_VARS \
            $IMAGE \
            bash -c "set -x && vllm serve /model --port ${PORT} \
                --tensor-parallel-size=${TP} \
                --gpu-memory-utilization 0.95 \
                --max-model-len ${MAX_MODEL_LEN} \
                --max-seq-len-to-capture ${MAX_MODEL_LEN} \
                --compilation-config '{\"cudagraph_mode\": \"FULL_AND_PIECEWISE\"}' \
                --block-size=64 \
                --no-enable-prefix-caching \
                --disable-log-requests \
                --async-scheduling" \
            >/dev/null 2>&1

        # Check if container started
        sleep 5
        if ! docker ps | grep -q vllm-server; then
            echo "[ERROR] Server failed to start"
            continue
        fi

        # Wait for server to be ready (watch logs for startup message, with timeout + fallback)
        WAIT_TIMEOUT=600      # seconds total to wait before giving up (adjust as needed)
        SLEEP_INTERVAL=5      # check interval
        WAIT_ELAPSED=0
        STARTED=0

        # Useful startup patterns to look for (add others if your vllm build prints different text)
        START_PATTERNS=(
          "Application startup complete."
          "Application startup complete"   # tolerant of punctuation
          "Server ready"                   # generic fallback
          "Serving on"                     # some versions show a 'serving on' line
        )

        echo "[TP=$TP, ISL=$ISL, OSL=$OSL, CONC=$CONC] Server initialization (waiting for startup)..."

        # tail logs in background so operator can see them; capture pid so we can kill later
        docker logs -f vllm-server 2>&1 | sed 's/^/[vllm] /' &
        LOG_PID=$!

        while true; do
          # Check if container still exists
          if ! docker ps --format '{{.Names}}' | grep -q '^vllm-server$'; then
            echo ""
            echo "[ERROR] Server container died while starting."
            kill $LOG_PID 2>/dev/null || true
            continue 2
          fi

          # Check logs for any of the startup patterns
          for pat in "${START_PATTERNS[@]}"; do
            if docker logs vllm-server 2>&1 | grep -Fq "$pat"; then
              echo ""
              echo "[TP=$TP, ISL=$ISL, OSL=$OSL, CONC=$CONC] Detected startup message: '$pat'"
              kill $LOG_PID 2>/dev/null || true
              # brief pause to let server finish warm internal init
              sleep 1
              STARTED=1
              break 2
            fi
          done

          # Fallback: probe the HTTP readiness endpoint (if vllm exposes /v1/models)
          if curl -s "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
            echo ""
            echo "[TP=$TP, ISL=$ISL, OSL=$OSL, CONC=$CONC] HTTP readiness check passed (/v1/models)"
            kill $LOG_PID 2>/dev/null || true
            STARTED=1
            break
          fi

          sleep $SLEEP_INTERVAL
          WAIT_ELAPSED=$((WAIT_ELAPSED + SLEEP_INTERVAL))

          if [ "$WAIT_ELAPSED" -ge "$WAIT_TIMEOUT" ]; then
            echo ""
            echo "[ERROR] Timeout ($WAIT_TIMEOUT s) waiting for vllm server to start. Dumping last 200 lines of logs:"
            docker logs --tail 200 vllm-server || true
            kill $LOG_PID 2>/dev/null || true
            continue 2
          fi
        done

        if [ "$STARTED" -ne 1 ]; then
            echo ""
            echo "[ERROR] Server startup failed"
            continue
        fi

        # Run benchmark client using vllm bench serve
        echo "[TP=$TP, ISL=$ISL, OSL=$OSL, CONC=$CONC] Running benchmark..."

        docker run --rm \
            --network host --ipc host \
            --privileged --cap-add=CAP_SYS_ADMIN \
            --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
            --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 32G \
            -v $MODEL_PATH:/model \
            -v $(pwd):/workspace/ -w /workspace/vllm/benchmarks/ \
            -v $RESULTS_DIR:/workspace/results/ \
            -e ROCM_TRITON_MOE_PRESHUFFLE_SCALES=0 \
            $IMAGE \
            vllm bench serve \
                --model /model \
                --backend vllm \
                --host 0.0.0.0 \
                --port ${PORT} \
                --dataset-name "random" \
                --random-range-ratio 0.8 \
                --random-input-len ${ISL} \
                --random-output-len ${OSL} \
                --random-prefix-len 0 \
                --num-prompts $(( ${CONC} * 10 )) \
                --max-concurrency ${CONC} \
                --request-rate "inf" \
                --ignore-eos \
                --save-result \
                --result-dir "/workspace/results/" \
                --result-filename "$(basename $RESULT_FILE)" \
                --percentile-metrics "ttft,tpot,itl,e2el"

        # Post-process the results file to keep only metrics (like oobx reference)
        if [ -f "$RESULT_FILE" ] && [ -s "$RESULT_FILE" ]; then
            echo "[TP=$TP, ISL=$ISL, OSL=$OSL, CONC=$CONC] Filtering results to keep only metrics..."

            # Fix file permissions first
            sudo chown $(whoami):$(whoami) "$RESULT_FILE" 2>/dev/null || true
            chmod 664 "$RESULT_FILE" 2>/dev/null || true

            # Create a temporary filtered file with only the metrics fields
            python3 -c "
import json
import sys

try:
    with open('$RESULT_FILE', 'r') as f:
        data = json.load(f)

    # Keep only the metrics fields (same as oobx reference file)
    filtered_data = {
        'date': data.get('date'),
        'endpoint_type': data.get('endpoint_type'),
        'label': data.get('label'),
        'model_id': data.get('model_id'),
        'tokenizer_id': data.get('tokenizer_id'),
        'num_prompts': data.get('num_prompts'),
        'request_rate': data.get('request_rate'),
        'burstiness': data.get('burstiness'),
        'max_concurrency': data.get('max_concurrency'),
        'duration': data.get('duration'),
        'completed': data.get('completed'),
        'total_input_tokens': data.get('total_input_tokens'),
        'total_output_tokens': data.get('total_output_tokens'),
        'request_throughput': data.get('request_throughput'),
        'request_goodput': data.get('request_goodput'),
        'output_throughput': data.get('output_throughput'),
        'total_token_throughput': data.get('total_token_throughput'),
        'mean_ttft_ms': data.get('mean_ttft_ms'),
        'median_ttft_ms': data.get('median_ttft_ms'),
        'std_ttft_ms': data.get('std_ttft_ms'),
        'p99_ttft_ms': data.get('p99_ttft_ms'),
        'mean_tpot_ms': data.get('mean_tpot_ms'),
        'median_tpot_ms': data.get('median_tpot_ms'),
        'std_tpot_ms': data.get('std_tpot_ms'),
        'p99_tpot_ms': data.get('p99_tpot_ms'),
        'mean_itl_ms': data.get('mean_itl_ms'),
        'median_itl_ms': data.get('median_itl_ms'),
        'std_itl_ms': data.get('std_itl_ms'),
        'p99_itl_ms': data.get('p99_itl_ms'),
        'mean_e2el_ms': data.get('mean_e2el_ms'),
        'median_e2el_ms': data.get('median_e2el_ms'),
        'std_e2el_ms': data.get('std_e2el_ms'),
        'p99_e2el_ms': data.get('p99_e2el_ms')
    }

    with open('$RESULT_FILE', 'w') as f:
        json.dump(filtered_data, f)

    print('Results filtered successfully')
except Exception as e:
    print(f'Warning: Could not filter results file: {e}')
"
        fi

        echo "[TP=$TP, ISL=$ISL, OSL=$OSL, CONC=$CONC] Benchmark completed!"

        # Stop server
        docker stop vllm-server >/dev/null 2>&1

        sleep 5
    done
done

echo ""
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="
echo "All results saved in: $RESULTS_DIR"
echo ""
echo "Results files:"
ls -la $RESULTS_DIR/*.json 2>/dev/null || echo "No results files found"
echo "=========================================="