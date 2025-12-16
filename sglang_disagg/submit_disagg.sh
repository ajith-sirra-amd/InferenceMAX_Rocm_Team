#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

usage() {
    cat << 'USAGE'
This script aims to provide a one-liner call to the submit_job_script.py,
so that the deployment process can be further simplified.

To use this script, fill in the following script and run it under your `slurm_jobs` directory:
======== begin script area ========
export SLURM_ACCOUNT=
export SLURM_PARTITION=
export TIME_LIMIT=

# Add path to your DSR1-FP8 model directory here
export MODEL_PATH=

# Add path to your container image here, either as a link or as a cached file
export CONTAINER_IMAGE=

bash submit_disagg.sh \
$PREFILL_NODES $PREFILL_WORKERS $DECODE_NODES $DECODE_WORKERS \
$ADDITIONAL_FRONTENDS \
$ISL $OSL $CONCURRENCIES $REQUEST_RATE
======== end script area ========
USAGE
}

check_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "Error: ${name} not specified" >&2
        usage >&2
        exit 1
    fi
}

check_env SLURM_ACCOUNT
check_env SLURM_PARTITION
check_env TIME_LIMIT

check_env MODEL_PATH
check_env MODEL_NAME
# check_env CONFIG_DIR
check_env CONTAINER_IMAGE
check_env PREFILL_NODES
check_env DECODE_NODES
check_env PREFILL_WORKERS
check_env DECODE_WORKERS


# GPU_TYPE="mi300x"
GPUS_PER_NODE=8
# : "${NETWORK_INTERFACE:=enP6p9s0np0}"

# COMMAND_LINE ARGS
PREFILL_NODES=$1
PREFILL_WORKERS=${2:-1}
DECODE_NODES=$3
DECODE_WORKERS=${4:-1}
ISL=$5
OSL=$6
CONCURRENCIES=$7
REQUEST_RATE=$8

NUM_NODES=$((PREFILL_NODES + DECODE_NODES))
# Should not need retries

# isl osl concurrency_list req_rate
profiler_args="${ISL} ${OSL} ${CONCURRENCIES} ${REQUEST_RATE}"

# ... (lines 1-66)

NUM_NODES=$((PREFILL_NODES + DECODE_NODES))
profiler_args="${ISL} ${OSL} ${CONCURRENCIES} ${REQUEST_RATE}"

# Export variables for the SLURM job
export MODEL_NAME=$MODEL_NAME
export MODEL_DIR=$MODEL_PATH
export DOCKER_IMAGE_NAME=$CONTAINER_IMAGE
export xP=$PREFILL_NODES
export yD=$DECODE_NODES
export PROFILER_ARGS=$profiler_args

# Construct the sbatch command
sbatch_cmd=(
    sbatch 
    -N "$NUM_NODES" 
    -n "$NUM_NODES" 
    --time "$TIME_LIMIT" 
    --partition "$SLURM_PARTITION" 
    --account "$SLURM_ACCOUNT" 
    run_xPyD_models.slurm
)

echo "Running: ${sbatch_cmd[*]}"
"${sbatch_cmd[@]}"