export RUNNER=b200-perfteam
export HF_HUB_CACHE=/data/huggingface-cache/hub
export HF_HUB_OFFLINE=1
export MODEL=nvidia/Llama-3.3-70B-Instruct-FP4
export RANDOM_RANGE_RATIO=0.8
export IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc2.post2
export FRAMEWORK=trt
export GITHUB_WORKSPACE=${HOME}/InferenceMAX_rocm
export RESULTS_DIR="results/results_$(date +%Y%m%d%H%M%S)"

mkdir -p ${RESULTS_DIR}

#models=("70b fp4 amd/Llama-3.3-70B-Instruct-MXFP4-Preview" "70b fp8 amd/Llama-3.3-70B-Instruct-FP8-KV" "dsr1 fp4 amd/DeepSeek-R1-0528-MXFP4-Preview" "dsr1 fp8 deepseek-ai/DeepSeek-R1-0528" "gptoss fp4 openai/gpt-oss-120b")
#models=("70b fp4 amd/Llama-3.3-70B-Instruct-MXFP4-Preview" "70b fp8 amd/Llama-3.3-70B-Instruct-FP8-KV")
models=("70b fp4 nvidia/Llama-3.3-70B-Instruct-FP4")
seqlens=("1024 1024" "1024 8192" "8192 1024")
#seqlens=("1024 1024" "8192 1024")

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
		for TP in 1 2 4 8 ; do
			export TP

			for CONC in 4 8 16 32 64; do
				export CONC
				export RESULT_FILENAME="${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}_conc${CONC}_${RUNNER}"
				bash ./runners/launch_${RUNNER}.sh
				echo "RESULT_FILENAME=${RESULT_FILENAME}"
				python3 utils/process_result.py ${RUNNER} $TP $RESULT_FILENAME $FRAMEWORK $PRECISION
			done
		done
		mkdir -p ${RESULTS_DIR}/${EXP_NAME}_${ISL}_${OSL}
		mv agg_*.json ${RESULTS_DIR}/${EXP_NAME}_${ISL}_${OSL}
		#python3 utils/collect_results.py ${RESULTS_DIR}/${EXP_NAME}_${ISL}_${OSL} ${EXP_NAME}
		mv ${EXP_NAME}*.json ${RESULTS_DIR}/${EXP_NAME}_${ISL}_${OSL}
	done
done
