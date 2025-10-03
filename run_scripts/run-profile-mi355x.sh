#!/bin/bash
# Note: This script should be run from the InferenceMAX directory, e.g. run-scripts/run-all-mi355x.sh

export RUNNER=mi355x-perfteam-profile
export HF_HUB_CACHE=/data/huggingface-cache/hub
export HF_HUB_OFFLINE=1
export MODEL=amd/Llama-3.3-70B-Instruct-FP8-KV
export RANDOM_RANGE_RATIO=0.8
export IMAGE=rocm/7.0:rocm7.0_ubuntu_22.04_vllm_0.10.1_instinct_20250927_rc1
export FRAMEWORK=vllm
export INFERENCEMAX_HOME=$(pwd)
export GITHUB_WORKSPACE=${INFERENCEMAX_HOME}
export RESULTS_BASE_DIR="${HOME}/inferencemax_results/results_$(date +%Y%m%d%H%M%S)"
export PROFILE_BASE_DIR="${RESULTS_BASE_DIR}/profiles"

mkdir -p ${RESULTS_BASE_DIR}
mkdir -p ${PROFILE_BASE_DIR}

#models=("70b fp4 amd/Llama-3.3-70B-Instruct-MXFP4-Preview" "70b fp8 amd/Llama-3.3-70B-Instruct-FP8-KV" "dsr1 fp4 amd/DeepSeek-R1-0528-MXFP4-Preview" "dsr1 fp8 deepseek-ai/DeepSeek-R1-0528" "gptoss fp4 openai/gpt-oss-120b")
models=("70b fp4 amd/Llama-3.3-70B-Instruct-MXFP4-Preview")
#seqlens=("1024 1024" "1024 8192" "8192 1024")
seqlens=("1024 1024" "8192 1024")

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

		for TP in 1 2 4 8 ; do
			export TP

			for CONC in 4 8 16 32 64; do
				export CONC
				export RESULT_FILENAME="${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}_conc${CONC}_${RUNNER}"
				export PROFILE_DIR="${PROFILE_BASE_DIR}/${RESULT_FILENAME}"
				mkdir -p ${PROFILE_DIR}
				bash ./runners/launch_${RUNNER}.sh
				echo "RESULT_FILENAME=${RESULT_FILENAME}"
				pushd ${RESULTS_DIR}
				python3 ${INFERENCEMAX_HOME}/utils/process_result.py ${RUNNER} $TP $RESULT_FILENAME-warmup $FRAMEWORK $PRECISION
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
