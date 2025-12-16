#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

warmup_model() {
    service_host=$1
    service_port=$2
    served_model_name=$3
    model_path=$4
    config=$5

    IFS='x' read -r -a config_list <<< "$config"
    isl=${config_list[0]}
    osl=${config_list[1]}
    num_prompts=${config_list[2]}
    concurrency=${config_list[3]}
    request_rate=${config_list[4]}

    command=(
        python3 -m sglang.bench_serving
        --base-url "http://${service_host}:${service_port}"
        --model ${served_model_name} --tokenizer ${model_path}
        --backend sglang-oai
        --dataset-name random --random-input ${isl} --random-output ${osl}
        --random-range-ratio 1
        --num-prompts ${num_prompts} --request-rate ${request_rate} --max-concurrency ${concurrency}
    )

    echo "Config ${config}. Running command ${command[@]}"

    ${command[@]}
}