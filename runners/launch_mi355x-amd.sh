#!/usr/bin/env bash

# === Workflow-defined Env Vars ===
# IMAGE
# MODEL
# TP
# HF_HUB_CACHE
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# CONC
# GITHUB_WORKSPACE
# RESULT_FILENAME
# HF_TOKEN
# RUN_ACCURACY_TEST
# ACCURACY_MODEL
# LM_EVAL_TASKS
# LM_EVAL_BATCH_SIZE
# LM_EVAL_NUM_FEWSHOT
# LM_EVAL_NUM_CONCURRENT
# LM_EVAL_MAX_RETRIES
# LM_EVAL_MAX_GEN_TOKS
# LM_EVAL_OUTPUT_BASENAME

HF_HUB_CACHE_MOUNT="/data/hf_hub_cache/"  # Temp solution
#VLLM_CACHE_MOUNT="/data/.vllm_cache-mi355x/" # Temp solution

if [[ "$MODEL" == *"DeepSeek-R1"* && "$FRAMEWORK" == "sglang" ]]; then
    FRAMEWORK_SUFFIX="_sglang"
else
    FRAMEWORK_SUFFIX=""
fi
echo $FRAMEWORK_SUFFIX

PORT=8777

network_name="bmk-net"
server_name="bmk-server"
client_name="bmk-client"

run_benchmark=${RUN_BENCHMARK:-true}
run_benchmark=${run_benchmark,,}

# CUSTOM
for CONTAINER_NAME in $server_name; do
    running_container=$(docker ps -a -q --filter "name=$CONTAINER_NAME")
    if [ $running_container ]; then
        echo "Terminating the already running $CONTAINER_NAME container"
        docker stop $CONTAINER_NAME
        sleep 5
        docker rm $CONTAINER_NAME
        sleep 5
        docker network rm $network_name
    fi
done

docker network create $network_name

# Turn off now (dsr1 have huge cache variation)
#-v $VLLM_CACHE_MOUNT:/root/.cache/vllm/ \
set -x
docker run --rm -d --ipc=host --shm-size=16g --network=$network_name --name=$server_name \
--privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e PORT=$PORT \
-e ISL -e OSL \
--entrypoint=/bin/bash \
$IMAGE \
benchmarks/"${EXP_NAME%%_*}_${PRECISION}_mi355x${FRAMEWORK_SUFFIX}_docker.sh"

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ Application\ startup\ complete ]]; then
        break
    fi
done < <(docker logs -f --tail=0 $server_name 2>&1)

if [[ "$MODEL" == "amd/DeepSeek-R1-0528-MXFP4-Preview" || "$MODEL" == "deepseek-ai/DeepSeek-R1-0528" ]]; then
  if [[ "$OSL" == "8192" ]]; then
    NUM_PROMPTS=$(( CONC * 20 ))
  else
    NUM_PROMPTS=$(( CONC * 50 ))
  fi
else
  NUM_PROMPTS=$(( CONC * 10 ))
fi

if [[ "$run_benchmark" != "false" ]]; then
  git clone https://github.com/kimbochen/bench_serving.git

  set -x
  docker run --rm --network=$network_name --name=$client_name \
  -v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
  -e HF_TOKEN -e PYTHONPYCACHEPREFIX=/tmp/pycache/ \
  --entrypoint=python3 \
  $IMAGE \
  bench_serving/benchmark_serving.py \
  --model=$MODEL --backend=vllm --base-url="http://$server_name:$PORT" \
  --dataset-name=random \
  --random-input-len=$ISL --random-output-len=$OSL --random-range-ratio=$RANDOM_RANGE_RATIO \
  --num-prompts=$NUM_PROMPTS \
  --max-concurrency=$CONC \
  --request-rate=inf --ignore-eos \
  --save-result --percentile-metrics="ttft,tpot,itl,e2el" \
  --result-dir=/workspace/ --result-filename=$RESULT_FILENAME.json
  set +x
fi

if [[ "${RUN_ACCURACY_TEST,,}" == "true" ]]; then
  accuracy_client_name="${client_name}-accuracy"
  accuracy_timestamp=$(date +"%Y%m%d_%H%M%S")
  sanitized_model=${ACCURACY_MODEL:-$MODEL}
  sanitized_model=${sanitized_model//\//-}
  accuracy_basename=${LM_EVAL_OUTPUT_BASENAME:-"VLLM_${sanitized_model}_${accuracy_timestamp}"}
  accuracy_output_relative="results_accuracy/${accuracy_basename}"
  eval_tasks=${LM_EVAL_TASKS:-gsm8k}
  eval_batch_size=${LM_EVAL_BATCH_SIZE:-auto}
  eval_num_fewshot=${LM_EVAL_NUM_FEWSHOT:-5}
  eval_num_concurrent=${LM_EVAL_NUM_CONCURRENT:-256}
  eval_max_retries=${LM_EVAL_MAX_RETRIES:-10}
  eval_max_gen_toks=${LM_EVAL_MAX_GEN_TOKS:-2048}
  accuracy_model_args=$(cat <<EOF
{"model": "${ACCURACY_MODEL:-$MODEL}", "base_url": "http://$server_name:$PORT/v1/completions", "num_concurrent": $eval_num_concurrent, "max_retries": $eval_max_retries, "max_gen_toks": $eval_max_gen_toks}
EOF
)
  mkdir -p "$GITHUB_WORKSPACE/results_accuracy"
  set -x
  docker run --rm --network=$network_name --name=$accuracy_client_name \
  -v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
  -e LM_EVAL_MODEL_ARGS="$accuracy_model_args" \
  --entrypoint=/bin/bash \
  $IMAGE \
  -lc "mkdir -p /workspace/results_accuracy && lm_eval --model local-completions --model_args \"\$LM_EVAL_MODEL_ARGS\" --tasks $eval_tasks --batch_size $eval_batch_size --num_fewshot $eval_num_fewshot --output_path /workspace/$accuracy_output_relative"
  set +x
fi
set -x


if ls gpucore.* 1> /dev/null 2>&1; then
  echo "gpucore files exist. not good"
  rm -f gpucore.*
fi


#while [ -n "$(docker ps -aq)" ]; do
#    docker stop $server_name
#    docker network rm $network_name
#    sleep 5
#done

# CUSTOM
for CONTAINER_NAME in $server_name; do
    running_container=$(docker ps -a -q --filter "name=$CONTAINER_NAME")
    if [ $running_container ]; then
        echo "Terminating the already running $CONTAINER_NAME container"
        docker stop $CONTAINER_NAME
        sleep 5
        docker rm $CONTAINER_NAME
        sleep 5
        docker network rm $network_name
    fi
done
