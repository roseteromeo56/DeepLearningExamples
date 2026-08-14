#!/usr/bin/env bash

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

: "${DATASET_DIR:=/datasets/LibriSpeech}"
: "${VALID_SUBSET:=test-other}"
: "${OUTPUT_DIR:=results/inference}"
: "${NUM_GPUS:=1}"
: "${BATCH_SIZE:=8}"
: "${AMP:=false}"
: "${BF16:=false}"
: "${FP16:=false}"
: "${EMA:=0.0}"
: "${SEED:=1}"
: "${FINETUNED_MODEL:=results/finetune_base_960h/wav2vec2_update320000.pt}"
: "${MASK_PROB:=0.5}"
: "${MASK_CHANNEL_PROB:=0.25}"
: "${DISTRIBUTED:=-m torch.distributed.launch --nproc_per_node=$NUM_GPUS}"
# inference
: "${MAX_DURATION:=}"
: "${NUM_STEPS:=0}"
: "${NUM_WARMUP_STEPS:=0}"
: "${CPU:=false}"
: "${LOGITS_FILE:=}"
: "${PREDICTION_FILE:=${OUTPUT_DIR}/${DATASET}.predictions}"
: "${TORCHSCRIPT:=false}"
: "${TORCHSCRIPT_SAVE:=false}"
: "${LOG_FILE:=${OUTPUT_DIR}/nvlog.json}"

mkdir -p "$OUTPUT_DIR"

ARGS=(
    --w2v_path "$FINETUNED_MODEL"
    --data "$DATASET_DIR"
    --valid_subset "$VALID_SUBSET"
    --output_dir "$OUTPUT_DIR"
    --ema "$EMA"
    --seed "$SEED"
    --skip_invalid_size_inputs_valid_test
    --apply_mask
    --mask_prob "$MASK_PROB"
    --mask_channel_prob "$MASK_CHANNEL_PROB"
    --mask_channel_length 64
    --encoder_layerdrop 0.1  # NOTE This is called `layerdrop` in fairseq finetuning yamls
    --activation_dropout 0.1
    --feature_grad_mult 0.0
    "--batch_size=$BATCH_SIZE"
    --steps "$NUM_STEPS"
    --warmup_steps "$NUM_WARMUP_STEPS"
)

[ "$AMP" = true ] &&                 ARGS+=(--amp --fp16)
[ "$BF16" = true ] &&                ARGS+=(--bf16)
[ "$TORCHSCRIPT" = true ] &&         ARGS+=(--torchscript)
[ "$TORCHSCRIPT_SAVE" = true ] &&    ARGS+=(--torchscript_export)
[ -n "$LOG_FILE" ] &&                ARGS+=(--log_file "$LOG_FILE")
[ "$CPU" == "true" ] &&              ARGS+=(--cpu)
[ -n "$MAX_DURATION" ] &&            ARGS+=(--max_duration "$MAX_DURATION")

set -x
if [ "$NUM_GPUS" -gt 1 ]; then
    python3 -m torch.distributed.launch "--nproc_per_node=$NUM_GPUS" inference.py "${ARGS[@]}" "$@"
else
    python3 inference.py "${ARGS[@]}" "$@"
fi
