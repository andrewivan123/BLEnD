#!/bin/bash

# =============================================================================
# model_inference_vllm.sh - vLLM-based inference for BLEnD short-answer questions
#
# All prompts across every country/language/prompt_no are compiled first, then
# a single llm.generate() call is made per model for maximum GPU throughput.
#
# To overwrite existing output files, set OVERWRITE="--overwrite" below.
# =============================================================================

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOGDIR="logs"
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/model_inference_vllm_${TIMESTAMP}.log"

{

echo "=========================================="
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname)"
echo "Args: $@"
nvidia-smi
echo "=========================================="

export HF_HOME="/home/ec2-user/efs/huggingface"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Set to "--overwrite" to overwrite existing output files; leave empty to skip them.
OVERWRITE=""

# Models to evaluate
MODEL_KEYS=(
    "/home/ec2-user/efs/CS6207/output/models/grpo_20260219_143022/grpo_merged"
)

# Country -> native language mapping (must match utils.py COUNTRY_LANG)
declare -A COUNTRY_LANG
COUNTRY_LANG["UK"]="English"
COUNTRY_LANG["US"]="English"
COUNTRY_LANG["South_Korea"]="Korean"
COUNTRY_LANG["Algeria"]="Arabic"
COUNTRY_LANG["China"]="Chinese"
COUNTRY_LANG["Indonesia"]="Indonesian"
COUNTRY_LANG["Spain"]="Spanish"
COUNTRY_LANG["Iran"]="Persian"
COUNTRY_LANG["Mexico"]="Spanish"
COUNTRY_LANG["Assam"]="Assamese"
COUNTRY_LANG["Greece"]="Greek"
COUNTRY_LANG["Ethiopia"]="Amharic"
COUNTRY_LANG["Northern_Nigeria"]="Hausa"
COUNTRY_LANG["Azerbaijan"]="Azerbaijani"
COUNTRY_LANG["North_Korea"]="Korean"
COUNTRY_LANG["West_Java"]="Sundanese"

# Prompt IDs (comma-separated; all are passed to Python in one call)
PROMPT_NOS="inst-4,pers-3"

# ---------------------------------------------------------------------------
# Build comma-separated LANGUAGES and COUNTRIES lists.
# For each non-English country we add two entries:
#   1. native language  (uses Translation column, native prompts)
#   2. English          (uses Question column, English prompts)
# English-speaking countries get one entry only.
# ---------------------------------------------------------------------------
LANGUAGES=""
COUNTRIES=""

for country in "${!COUNTRY_LANG[@]}"; do
    language="${COUNTRY_LANG[$country]}"

    [ -n "$LANGUAGES" ] && LANGUAGES="${LANGUAGES},"
    [ -n "$COUNTRIES" ] && COUNTRIES="${COUNTRIES},"
    LANGUAGES="${LANGUAGES}${language}"
    COUNTRIES="${COUNTRIES}${country}"

    if [ "$language" != "English" ]; then
        LANGUAGES="${LANGUAGES},English"
        COUNTRIES="${COUNTRIES},${country}"
    fi
done

# ---------------------------------------------------------------------------
# Run inference: ONE Python call per model covers all countries/languages/prompts
# ---------------------------------------------------------------------------
for model_key in "${MODEL_KEYS[@]}"; do
    echo ""
    echo ">>> Model: ${model_key}"
    echo ">>> Countries: ${COUNTRIES}"
    echo ">>> Languages: ${LANGUAGES}"
    echo ">>> Prompts  : ${PROMPT_NOS}"

    python model_inference_vllm.py \
        --model        "$model_key" \
        --language     "$LANGUAGES" \
        --country      "$COUNTRIES" \
        --prompt_no    "$PROMPT_NOS" \
        --question_dir "./data/questions" \
        --prompt_dir   "./data/prompts" \
        --id_col       ID \
        --output_dir   "./model_inference_results_vllm" \
        --model_cache_dir "/home/ec2-user/efs/huggingface" \
        --temperature  0 \
        --top_p        1 \
        --max_length   512 \
        $OVERWRITE
done

echo "=========================================="
echo "Finish time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

} 2>&1 | tee "$LOGFILE"
