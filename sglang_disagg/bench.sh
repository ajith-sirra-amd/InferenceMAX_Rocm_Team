#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

n_prefill=$1
n_decode=$2
prefill_gpus=$3
decode_gpus=$4
model_path=$5
model_name=$6
MODEL_PATH="${model_path}/${model_name}"
log_path=$7

chosen_isl=${8:-1024}
chosen_osl=${9:-1024}
concurrency_list=${10:-"512x1"}
chosen_req_rate=${11:-1}
random_range_ratio=${12:-1}
num_prompts_multiplier=${13:-10}

IFS='x' read -r -a chosen_concurrencies <<< "$concurrency_list"

# echo "Config ${chosen_isl}; ${chosen_osl}; ${chosen_concurrencies[@]}; ${chosen_req_rate}"
# TODO only useing the first concurrency for now
echo "Config ${chosen_isl}; ${chosen_osl}; ${chosen_concurrencies[0]}; ${chosen_req_rate}"

head_node="localhost"
head_port="30000"

# MODEL_NAME="DeepSeek-R1"
# MODEL_PATH="/shared-inference"

source benchmark_utils.sh

# wait_for_model $head_node $head_port $n_prefill $n_decode 5 900 60

# sleep 300

set -e
# warmup_model $head_node $head_port $MODEL_PATH $MODEL_PATH "${chosen_isl}x${chosen_osl}x10000x10000x250"
set +e

profile_folder="${log_path}/sglang_isl_${chosen_isl}_osl_${chosen_osl}"
mkdir -p $profile_folder

# for max_concurrency in ${chosen_concurrencies[@]}; do

#     chosen_n_requests=$((5*max_concurrency))

#     export_file="${profile_folder}/concurrency_${max_concurrency}_req_rate_${chosen_req_rate}.json"

#     command=(
#         python3 -m sglang.bench_serving
#         --base-url "http://${head_node}:${head_port}"
#         --model ${MODEL_PATH} --tokenizer ${MODEL_PATH}
#         --backend sglang-oai
#         --dataset-name random --random-input ${chosen_isl} --random-output ${chosen_osl}
#         --random-range-ratio 1
#         --num-prompts ${chosen_n_requests} --request-rate ${chosen_req_rate} --max-concurrency ${max_concurrency}
#         --output-file $export_file
#     )

#     echo "Running command ${command[@]}"

#     ${command[@]}

#     echo "-----------------------------------------"
# done

source "$(dirname "$0")/benchmark_lib.sh"

max_concurrency=${chosen_concurrencies[0]}
export_file="${profile_folder}/concurrency_${max_concurrency}_req_rate_${chosen_req_rate}.json"

echo "=== debug info ==="

echo "profile_folder: $profile_folder"
echo "max_concurrency: $max_concurrency"
echo "chosen_req_rate: $chosen_req_rate"
echo "MODEL_PATH: $MODEL_PATH"
echo "head_port: $head_port"
echo "chosen_isl: $chosen_isl"
echo "chosen_osl: $chosen_osl"
echo "export_file: $export_file"

run_benchmark_serving \
    --model  ${MODEL_PATH} \
    --port ${head_port} \
    --backend openai \
    --input-len ${chosen_isl} \
    --output-len ${chosen_osl} \
    --random-range-ratio ${random_range_ratio} \
    --num-prompts $(( $max_concurrency * $num_prompts_multiplier )) \
    --max-concurrency "$max_concurrency" \
    --result-filename "$export_file" \
    --result-dir /workspace/
