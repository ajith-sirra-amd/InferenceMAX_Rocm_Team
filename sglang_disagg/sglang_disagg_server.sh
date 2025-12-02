#!/bin/bash
# SGLang Disaggregated Server Launcher with Model-Specific Configurations
# =============================================================================

# =============================================================================
# Environment Configuration
# =============================================================================

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23731}"
NODE_RANK="${NODE_RANK:-0}"
MODEL_PATH=$MODEL_PATH
MODEL_NAME="${MODEL_NAME:-}"
xP="${xP:-1}"
yD="${yD:-1}"
IPADDRS="${IPADDRS:-localhost}"
PROFILER_ARGS="${PROFILER_ARGS:-}"

# =============================================================================
# Dependencies and Environment Setup
# =============================================================================

source $MOONCAKE_COOKBOOK_PATH/set_env_vars.sh

host_ip=$(ip route get 1.1.1.1 | awk '/src/ {print $7}')
host_name=$(hostname)

# =============================================================================
# Model-Specific Configuration Maps
# =============================================================================

declare -A MODEL_PREFILL_CONFIGS=(
    ["DeepSeek-V3"]="--moe-a2a-backend none --enable-dp-attention --decode-log-interval 1 --load-balance-method round_robin --moe-dense-tp-size 1 --enable-dp-lm-head --disable-radix-cache --watchdog-timeout 1000000 --mem-fraction-static 0.9 --chunked-prefill-size 131076 --max-running-requests 4096 --context-length 12288 --max-total-tokens 131024 --disable-cuda-graph --attention-backend aiter"
    ["DeepSeek-R1"]="--moe-a2a-backend mori --enable-dp-attention --decode-log-interval 1 --load-balance-method round_robin --moe-dense-tp-size 1 --enable-dp-lm-head --disable-radix-cache --watchdog-timeout 1000000 --mem-fraction-static 0.7 --chunked-prefill-size 8192 --max-running-requests 4096 --context-length 12288 --max-total-tokens 131024 --disable-cuda-graph --attention-backend aiter"
)

declare -A MODEL_DECODE_CONFIGS=(
    ["DeepSeek-V3"]="--moe-a2a-backend none --enable-dp-attention --decode-log-interval 1 --prefill-round-robin-balance --moe-dense-tp-size 1 --enable-dp-lm-head --watchdog-timeout 1000000 --mem-fraction-static 0.9 --max-running-requests 4096 --cuda-graph-bs 128 256 --attention-backend aiter"
    ["DeepSeek-R1"]="--moe-a2a-backend mori --enable-dp-attention --decode-log-interval 1 --prefill-round-robin-balance --moe-dense-tp-size 1 --enable-dp-lm-head --watchdog-timeout 1000000 --mem-fraction-static 0.7 --max-running-requests 4096 --cuda-graph-bs 128 256 --attention-backend aiter"
)

# =============================================================================
# Configuration Selection Functions
# =============================================================================

get_model_config() {
    local mode="$1"
    local model_name="$2"
    
    if [[ "$mode" == "prefill" ]]; then
        if [[ -n "${MODEL_PREFILL_CONFIGS[$model_name]}" ]]; then
            echo "${MODEL_PREFILL_CONFIGS[$model_name]}"
        else
            echo "--tp-size 4"
        fi
    elif [[ "$mode" == "decode" ]]; then
        if [[ -n "${MODEL_DECODE_CONFIGS[$model_name]}" ]]; then
            echo "${MODEL_DECODE_CONFIGS[$model_name]}"
        else
            echo "--tp-size 4"
        fi
    fi
}

if [[ -z "$MODEL_NAME" ]]; then
    echo "Warning: MODEL_NAME not set, using default configurations"
    PREFILL_MODEL_CONFIG="--tp-size 4"
    DECODE_MODEL_CONFIG="--tp-size 4"
else
    PREFILL_MODEL_CONFIG=$(get_model_config "prefill" "$MODEL_NAME")
    DECODE_MODEL_CONFIG=$(get_model_config "decode" "$MODEL_NAME")
    echo "Using model-specific configuration for: $MODEL_NAME"
fi

# =============================================================================
# Container Synchronization
# =============================================================================

echo "Waiting at the container creation barrier on $host_name"
python $MOONCAKE_COOKBOOK_PATH/socket_barrier.py \
    --local-ip ${host_ip} \
    --local-port 5000 \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports 5000

# =============================================================================
# Cluster Topology Configuration
# =============================================================================

IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

PREFILL_ARGS=""
DECODE_ARGS=""

PREFILL_ARGS="http://${IP_ARRAY[0]}:8000"
DECODE_ARGS="http://${IP_ARRAY[$xP]}:8000"

PREFILL_PARALELL=$((xP * 8))
DECODE_PARALELL=$((yD * 8))

# =============================================================================
# Node Role Assignment and Server Launch
# =============================================================================

if [ "$NODE_RANK" -eq 0 ]; then
    echo "NODE INFO ======================================="
    echo "================================================"
    echo "Node List : ${SLURM_JOB_NODELIST}"
    echo "Node IPs : ${IPADDRS}"
    echo "Model Name : ${MODEL_NAME:-'Not specified'}"
    echo "================================================"

    echo "CLUSTER INFO ===================================="
    echo "================================================"
    echo "${host_name}:${host_ip} is Proxy Node and Prefill Node"
    echo "Using prefill config: $PREFILL_MODEL_CONFIG"
    echo "${PREFILL_ARGS} are Proxy's Prefill"
    echo "${DECODE_ARGS} are Proxy's Decode"
    echo "================================================"

    set -x 
    python -m sglang_router.launch_router \
    --pd-disaggregation \
    --mini-lb \
    --port 30000 \
    --prefill ${PREFILL_ARGS} \
    --decode ${DECODE_ARGS} \
    2>&1 | tee /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null &
    set +x
    
    proxy_pid=$!
    
    # start the head prefill server
    PREFILL_CMD="MC_TE_METRIC=true python3 -m sglang.launch_server \
        --model-path $MODEL_PATH/$MODEL_NAME \
        --disaggregation-mode prefill \
        --disaggregation-ib-device ${IBDEVICES} \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        --node-rank 0 \
        --nnodes $xP \
        --dist-init-addr ${IP_ARRAY[0]}:20000 \
        --tp-size $PREFILL_PARALELL \
        --dp-size $PREFILL_PARALELL \
        --ep-size $PREFILL_PARALELL"
    
    if [[ -n "$PREFILL_MODEL_CONFIG" ]]; then
        PREFILL_CMD="$PREFILL_CMD $PREFILL_MODEL_CONFIG"
    fi
    
    set -x 
    eval "$PREFILL_CMD" \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/prefill_NODE${NODE_RANK}.log >/dev/null &
    set +x

    prefill0_pid=$!
    
    echo "Waiting for all prefill and decode servers to be up . . ."
    python $MOONCAKE_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${IPADDRS} \
        --node-ports 8000

    echo "Proxy Server Ready for benchmarking on ${host_name}:${host_ip}"

    echo "Benchmarking on ${host_name}:${host_ip}"
    cd /opt/mooncake-cookbook
    # todo: put bench.sh in sglang folder
    # n_prefill n_decode prefill_gpus decode_gpus model_path model_name log_path isl osl concurrency_list req_rate
    bash /opt/mooncake-cookbook/bench.sh 1 1 8 8 $MODEL_PATH $MODEL_NAME /run_logs/${SLURM_JOB_ID} ${PROFILER_ARGS}
 
    echo "Killing the proxy server and prefill server"
    kill $proxy_pid
    kill $prefill0_pid

elif [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -lt "$xP" ]; then
    echo "${host_name}:${host_ip} is Prefill Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using prefill config: $PREFILL_MODEL_CONFIG"

    PREFILL_CMD="MC_TE_METRIC=true python3 -m sglang.launch_server \
        --model-path $MODEL_PATH/${MODEL_NAME} \
        --disaggregation-mode prefill \
        --disaggregation-ib-device ${IBDEVICES} \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        --nnodes $xP \
        --node-rank $NODE_RANK \
        --dist-init-addr ${IP_ARRAY[0]}:20000 \
        --tp-size $PREFILL_PARALELL \
        --dp-size $PREFILL_PARALELL \
        --ep-size $PREFILL_PARALELL"
    
    if [[ -n "$PREFILL_MODEL_CONFIG" ]]; then
        PREFILL_CMD="$PREFILL_CMD $PREFILL_MODEL_CONFIG"
    fi

    set -x 
    
    eval "$PREFILL_CMD" \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/prefill_NODE${NODE_RANK}.log >/dev/null &
    set +x

    prefill_pid=$!

    echo "Waiting for proxy server to be up..."
    python $MOONCAKE_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${MASTER_ADDR} \
        --node-ports 30000
    
    echo "Waiting until proxy server closes..."
    python $MOONCAKE_COOKBOOK_PATH/socket_wait.py \
        --remote-ip ${MASTER_ADDR} \
        --remote-port 30000

    echo "Killing the prefill server"
    kill $prefill_pid

else
    RANK=$((NODE_RANK - xP))
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using decode config: $DECODE_MODEL_CONFIG"
    echo "Decode node rank: $RANK"
    
    DECODE_CMD="MC_TE_METRIC=true python3 -m sglang.launch_server \
        --model-path ${MODEL_PATH}/${MODEL_NAME} \
        --disaggregation-mode decode \
        --disaggregation-ib-device ${IBDEVICES} \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        --nnodes $yD \
        --dist-init-addr ${IP_ARRAY[$xP]}:20000 \
        --node-rank $RANK \
        --tp-size $DECODE_PARALELL \
        --dp-size $DECODE_PARALELL \
        --ep-size $DECODE_PARALELL"

    if [[ -n "$DECODE_MODEL_CONFIG" ]]; then
        DECODE_CMD="$DECODE_CMD $DECODE_MODEL_CONFIG "
    fi

    set -x 
    eval "$DECODE_CMD" \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/decode_NODE${NODE_RANK}.log >/dev/null &
    
    decode_pid=$!
    set +x 

    echo "Waiting for proxy server to be up..."
    python $MOONCAKE_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${MASTER_ADDR} \
        --node-ports 30000
    
    echo "Waiting until proxy server closes..."
    python $MOONCAKE_COOKBOOK_PATH/socket_wait.py \
        --remote-ip ${MASTER_ADDR} \
        --remote-port 30000

    echo "Killing the decode server"
    kill $decode_pid

fi

echo "Script completed successfully"
exit 0
