#!/bin/bash
# Note: This script should be run from the InferenceMAX directory, e.g. run-scripts/run-all-mi355x.sh

export RUNNER=b200-perfteam-power
export HF_HUB_CACHE=/data/huggingface-cache/hub
export HF_HUB_OFFLINE=1
export MODEL=nvidia/Llama-3.3-70B-Instruct-FP4
export RANDOM_RANGE_RATIO=0.8
export IMAGE=docker.gpuperf:5000/nvidia/tensorrt-llm/release:1.1.0rc2.post2-zeus0.8
export FRAMEWORK=trt
export INFERENCEMAX_HOME=$(pwd)
export GITHUB_WORKSPACE=${INFERENCEMAX_HOME}
export RESULTS_BASE_DIR="${HOME}/inferencemax_results/results_$(date +%Y%m%d%H%M%S)"
export FREQ_BASE_DIR="${RESULTS_BASE_DIR}/freq"

mkdir -p ${RESULTS_BASE_DIR}
mkdir -p ${FREQ_BASE_DIR}

models=("70b fp4 nvidia/Llama-3.3-70B-Instruct-FP4")
seqlens=("1024 1024" "1024 8192" "8192 1024")

for model in "${models[@]}"; do
	set -- $model
	export EXP_NAME=$1
	export PRECISION=$2
	export MODEL=$3

	for seqlen in "${seqlens[@]}"; do
		set -- $seqlen
		export ISL=$1
		export OSL=$2
		export MAX_MODEL_LEN=$((ISL + OSL))
		export RESULTS_DIR=${RESULTS_BASE_DIR}/${EXP_NAME}_${ISL}_${OSL}
		mkdir -p ${RESULTS_DIR}

		for TP in 1 ; do
		#for TP in 1 8 ; do
			export TP

			for CONC in 4 8 16 32 64; do
				export CONC
				export RESULT_FILENAME="${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}_conc${CONC}_${RUNNER}"
				export FREQ_DIR="${FREQ_BASE_DIR}/${EXP_NAME}_${ISL}_${OSL}/${RESULT_FILENAME}"
                                mkdir -p ${FREQ_DIR}
				bash ./runners/launch_${RUNNER}.sh
				echo "RESULT_FILENAME=${RESULT_FILENAME}"
				pushd ${RESULTS_DIR}
				python3 ${INFERENCEMAX_HOME}/utils/process_result.py ${RUNNER} $TP $RESULT_FILENAME $FRAMEWORK $PRECISION
				popd
			done
		done
		pushd ${RESULTS_DIR}
		mkdir temp
		cp agg_* temp
		python3 ${INFERENCEMAX_HOME}/utils/collect_results.py temp ${EXP_NAME}
		rm -rf temp
		popd
	done
done
