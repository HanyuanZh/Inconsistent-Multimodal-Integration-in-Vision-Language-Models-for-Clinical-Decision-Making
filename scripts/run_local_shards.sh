#!/usr/bin/env bash
# Run one local model on N GPUs of one node: one process per GPU, each on 1/N of
# the units, then merge the shard files.
#
#   scripts/run_local_shards.sh <experiment> <dataset> <model> <data_root> [n_gpus]
#   scripts/run_local_shards.sh main prostate qwen36 /path/to/data 4
#
# Models that do not fit on one GPU (e.g. Lingshu-32B, MedGemma-27B, Qwen3.6-35B-A3B
# on small cards) can instead run as a single process over several GPUs:
#   CUDA_VISIBLE_DEVICES=0,1 python -m vlmbench run -e main -d prostate -m lingshu32 --data-root ...
set -euo pipefail
exp=$1; ds=$2; model=$3; data=$4; n=${5:-$(nvidia-smi -L | wc -l)}
for ((i = 0; i < n; i++)); do
  CUDA_VISIBLE_DEVICES=$i python -m vlmbench run -e "$exp" -d "$ds" -m "$model" \
    --data-root "$data" --shard "$i" --num-shards "$n" &
done
wait
python -m vlmbench merge -e "$exp" -d "$ds" -m "$model"
