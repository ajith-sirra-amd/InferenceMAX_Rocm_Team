#!/bin/bash

timestamp=$(date "+%Y%m%d_%H%M%S")
LOG_PATH="${LOG_PATH:-/run_logs}"
LOG="/${LOG_PATH}/${SLURM_JOB_ID}/benchmark_${SLURM_JOB_ID}_${timestamp}_xP${xP}_yD${yD}_${MODEL_NAME}"
  
### Concurrency Sweep Test ###
{
    echo "==== Benchmark Serving Concurrency Sweep Test ${LOG} ====="
    echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"
    echo ""
} | tee -a "${LOG}_CONCURRENCY.log"

CON="32 64 128 256 512 1024"
COMBINATIONS=("1024/1024" "8196/1024")

for i in {1..1}; do
    echo "Running the benchserving script for iter: $i" | tee -a "${LOG}_CONCURRENCY.log"
    for combo in "${COMBINATIONS[@]}"; do
       IFS="/" read -r isl osl <<< "$combo"
    #    for con in $CON; do
    #        p_con=$(($con * 2))
    #        if [ "$p_con" -lt 16 ]; then
    #            p_con=16
    #        fi
           echo "[RUNNING] prompts $prompts isl $isl osl $osl con $con model ${MODEL_NAME} xP=${xP} yD=${yD} job=${SLURM_JOB_ID}" | tee -a "${LOG}_CONCURRENCY.log"
           set -x 
        #    timeout 300s python3 -m sglang.bench_one_batch_server \
           python3 -m sglang.bench_one_batch_server \
            --model-path ${MODEL_PATH}/${MODEL_NAME} \
            --base-url http://0.0.0.0:30000 \
            --batch-size 2048 \
            --input-len $isl \
            --output-len $osl \
            --skip-warmup \
            2>&1 | tee -a "${LOG}_CONCURRENCY.log"
           sleep 1
           set +x
    #    done
    done
done

### Concurrency Sweep End Time ###
{
    echo "==== Benchmark Serving Concurrency End Time ${LOG} ====="
    echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"
    echo ""
} | tee -a "${LOG}_CONCURRENCY.log"
