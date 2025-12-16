#!/bin/bash

# Before running this code - request 5 nodes from salloc 
#salloc -N 5 --ntasks-per-node=1 --nodelist=<Nodes> --gres=gpu:8 -p <partition> -t 12:00:00
#Sample Commands
# Parallelism Configuration:
# PREFILL_TP_SIZE: Tensor Parallelism size for Prefill nodes (default: 8)
# PREFILL_ENABLE_EP: Enable Expert Parallelism for Prefill nodes (true/false, default: true)
# PREFILL_ENABLE_DP: Enable Data Parallelism for Prefill nodes (true/false, default: true)
# DECODE_TP_SIZE: Tensor Parallelism size for Decode nodes (default: 8)
# DECODE_ENABLE_EP: Enable Expert Parallelism for Decode nodes (true/false, default: true)
# DECODE_ENABLE_DP: Enable Data Parallelism for Decode nodes (true/false, default: true)
# Note: When EP/DP is enabled, its size will be set equal to TP_SIZE

# Benchmark Configuration:
# BENCH_INPUT_LEN: Input sequence length for benchmark (default: 1024)
# BENCH_OUTPUT_LEN: Output sequence length for benchmark (default: 1024)
# BENCH_RANDOM_RANGE_RATIO: Random range ratio for benchmark (default: 1)
# BENCH_NUM_PROMPTS_MULTIPLIER: Number of prompts = max_concurrency * multiplier (default: 10)
# BENCH_MAX_CONCURRENCY: Maximum concurrency for benchmark (default: 512) [can be a single value or a list like "512x128", only using the first value for now]

export xP=1
export yD=2
export MODEL_NAME=DeepSeek-R1
export PREFILL_TP_SIZE=8
export PREFILL_ENABLE_EP=true
export PREFILL_ENABLE_DP=true
export DECODE_TP_SIZE=8
export DECODE_ENABLE_EP=true
export DECODE_ENABLE_DP=true
export BENCH_INPUT_LEN=1024
export BENCH_OUTPUT_LEN=1024
export BENCH_RANDOM_RANGE_RATIO=1
export BENCH_NUM_PROMPTS_MULTIPLIER=10
export BENCH_MAX_CONCURRENCY=2048
bash run_xPyD_models.slurm 2>&1 | tee log_${MODEL_NAME}_xP${xP}_yD${yD}.log

# export xP=2; export yD=2; export MODEL_NAME=Qwen3-30B-A3B;                      bash run_xPyD_models.slurm 2>&1 | tee log_${MODEL_NAME}_xP${xP}_yD${yD}.log
# export xP=2; export yD=2; export MODEL_NAME=Mixtral-8x7B-v0.1;                   bash run_xPyD_models.slurm 2>&1 | tee log_${MODEL_NAME}_xP${xP}_yD${yD}.log
# export xP=2; export yD=2; export MODEL_NAME=Llama-3.1-8B-Instruct;               bash run_xPyD_models.slurm 2>&1 | tee log_${MODEL_NAME}_xP${xP}_yD${yD}.log
# export xP=2; export yD=2; export MODEL_NAME=Llama-3.1-405B-Instruct-FP8-KV;      bash run_xPyD_models.slurm 2>&1 | tee log_${MODEL_NAME}_xP${xP}_yD${yD}.log
# export xP=2; export yD=2; export MODEL_NAME=amd-Llama-3.3-70B-Instruct-FP8-KV;   bash run_xPyD_models.slurm 2>&1 | tee log_${MODEL_NAME}_xP${xP}_yD${yD}.log


#Or directly run with sbatch command
#export DOCKER_IMAGE_NAME=<DOCKER IMAGE NAME>
#export xP=<num_prefill_nodes>; export yD=<num_decode_nodes>; export MODEL_NAME=Llama-3.1-8B-Instruct; sbatch -N <num_nodes> -n <num_nodes> --nodelist=<Nodes> run_xPyD_models.slurm

