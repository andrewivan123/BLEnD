#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
HOST=$(hostname)
LOGFILE="../logs/evaluate_all_${TIMESTAMP}.log"

echo "============================================" | tee -a "$LOGFILE"
echo "START:  $(date)"                              | tee -a "$LOGFILE"
echo "HOST:   $HOST"                                | tee -a "$LOGFILE"
echo "SCRIPT: evaluate_all.sh"                      | tee -a "$LOGFILE"
echo "============================================" | tee -a "$LOGFILE"
nvidia-smi                                          | tee -a "$LOGFILE"
echo "============================================" | tee -a "$LOGFILE"

export HF_HOME=/home/ec2-user/efs/huggingface

python evaluate_all.py \
    --models \
        "google/gemma-3-1b-it" \
        "/home/ec2-user/efs/CS6207/output/models/grpo_20260219_143022/grpo_merged" \
    --country_lang \
        "UK:English" \
        "US:English" \
        "South_Korea:Korean" \
        "Algeria:Arabic" \
        "China:Chinese" \
        "Indonesia:Indonesian" \
        "Spain:Spanish" \
        "Iran:Persian" \
        "Mexico:Spanish" \
        "Assam:Assamese" \
        "Greece:Greek" \
        "Ethiopia:Amharic" \
        "Northern_Nigeria:Hausa" \
        "Azerbaijan:Azerbaijani" \
        "North_Korea:Korean" \
        "West_Java:Sundanese" \
    --prompt_nos "inst-4" "pers-3" \
    --response_dir "../model_inference_results_vllm" \
    --annotation_dir "../data/annotations" \
    --id_col ID \
    --question_col Translation \
    --response_col response \
    --annotation_filename "{country}_data.json" \
    --annotations_key "annotations" \
    --evaluation_result_file "evaluation_results.csv" \
    --skip_mcq \
    2>&1 | tee -a "$LOGFILE"

echo "============================================" | tee -a "$LOGFILE"
echo "FINISH: $(date)"                              | tee -a "$LOGFILE"
echo "============================================" | tee -a "$LOGFILE"
