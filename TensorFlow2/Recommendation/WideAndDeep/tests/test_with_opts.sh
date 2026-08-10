#!/bin/bash
set -e
set -x
NAMES=${1:-'*.yaml'}
TARGET_PATTERN="/wd/tests/feature_specs/${NAMES}"
OPTIONS=${2-""}
read -r -a OPTIONS_ARRAY <<< "${OPTIONS}"
mapfile -t TARGETS < <(compgen -G "${TARGET_PATTERN}")
if [ "${#TARGETS[@]}" -eq 0 ]; then
  TARGETS=("${TARGET_PATTERN}")
fi

for file in "${TARGETS[@]}";
do
  echo "${file}";
done
for fspec_file in "${TARGETS[@]}";
do
  CSV_DIR=/tmp/generated_data/csv_dir
  TRANS_DIR=/tmp/generated_data/trans_dir
  # generate data based on fspec
  python gen_csv.py --feature_spec_in "${fspec_file}" --output "${CSV_DIR}" --size 393216 #131072*3, bsize*3
  cp "${fspec_file}" "${CSV_DIR}/feature_spec.yaml"

  python transcode.py --input "${CSV_DIR}" --output "${TRANS_DIR}" --chunk_size 16384 # to get 8 partitions out of 131072 rows

  EMBEDDING_SIZES_FILE="${TRANS_DIR}/embedding_sizes.json"
  python gen_embedding_sizes.py --feature_spec_in "${fspec_file}" --output "${EMBEDDING_SIZES_FILE}"

  #horovodrun -np 1 sh hvd_wrapper.sh python main.py --dataset_path ${TRANS_DIR} ${OPTIONS} --disable_map_calculation --embedding_sizes_file ${EMBEDDING_SIZES_FILE} --num_epochs 10

  horovodrun -np 8 sh hvd_wrapper.sh python main.py --dataset_path "${TRANS_DIR}" "${OPTIONS_ARRAY[@]}" --disable_map_calculation --embedding_sizes_file "${EMBEDDING_SIZES_FILE}" --num_epochs 6 --xla --amp

  rm -r "${CSV_DIR}"
  rm -r "${TRANS_DIR}"
done

#usage: bash tests/test_with_opts.sh
