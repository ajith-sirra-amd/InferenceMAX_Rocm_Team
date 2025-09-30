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

HF_HUB_CACHE_MOUNT="/data/hf_hub_cache/"  # Temp solution
if [[ "$FRAMEWORK" == "vllm_dsr1" ]]; then
    DEVICE_FRAMEWORK_SUFFIX="mi355xvllm_dsr1"
else; then
    DEVICE_FRAMEWORK_SUFFIX="mi355x"
fi

echo $DEVICE_FRAMEWORK_SUFFIX
PORT=8888

network_name="bmk-net"
server_name="bmk-server"
client_name="bmk-client"

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
benchmarks/"${EXP_NAME%%_*}_${PRECISION}_${DEVICE_FRAMEWORK_SUFFIX}_docker.sh"

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ Application\ startup\ complete ]]; then
        break
    fi
done < <(docker logs -f --tail=0 $server_name 2>&1)

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
--num-prompts=$(( $CONC * 10 )) \
--max-concurrency=$CONC \
--request-rate=inf --ignore-eos \
--save-result --percentile-metrics="ttft,tpot,itl,e2el" \
--result-dir=/workspace/ --result-filename=$RESULT_FILENAME.json

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
