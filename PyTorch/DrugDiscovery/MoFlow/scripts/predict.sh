#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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


bs=${1:-512}
prec=${2:-amp}
flags=("${@:3}")

set -x

# Build command array
cmd=(
    "python"
    "/workspace/moflow_pyt/moflow/runtime/generate.py"
    "--batch_size" "${bs}"
    "--jit"
    "--correct_validity"
) dd/fix-sast-vulnerability-predict-sh
# Add flags
cmd+=("${flags[@]}")

# Add amp flag if specified
if [ "$prec" == "amp" ]; then
    cmd+=("--amp")
fi

# Execute command directly
exec "${cmd[@]}"
set -x
python \
    /workspace/moflow_pyt/moflow/runtime/generate.py \
    --batch_size ${bs} \
    --jit \
    --correct_validity \
    ${flags} \
    "$@"

if [ $prec == "amp" ]; then
    python \
        /workspace/moflow_pyt/moflow/runtime/generate.py \
        --batch_size ${bs} \
        --jit \
        --correct_validity \
        --amp \
        ${flags} \
        "$@"
fi master
