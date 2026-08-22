#!/bin/bash

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

set -a

: "${OUTPUT_DIR:=${3:-/results}}"
: "${CUDNN_BENCHMARK:=true}"
: "${PAD_TO_MAX_DURATION:=true}"
: "${NUM_WARMUP_STEPS:=10}"
: "${NUM_STEPS:=500}"

: "${AMP:=false}"
: "${DALI_DEVICE:=cpu}"
: "${BATCH_SIZE_SEQ:=1 2 4 8 16}"
: "${MAX_DURATION_SEQ:=2 7 16.7}"

read -r -a MAX_DURATIONS <<< "$MAX_DURATION_SEQ"
read -r -a BATCH_SIZES <<< "$BATCH_SIZE_SEQ"

for MAX_DURATION in "${MAX_DURATIONS[@]}"; do
  for BATCH_SIZE in "${BATCH_SIZES[@]}"; do

    LOG_FILE="$OUTPUT_DIR/perf-infer_dali-${DALI_DEVICE}_amp-${AMP}_dur${MAX_DURATION}_bs${BATCH_SIZE}.json"
    bash ./scripts/inference.sh "$@"

  done
done
