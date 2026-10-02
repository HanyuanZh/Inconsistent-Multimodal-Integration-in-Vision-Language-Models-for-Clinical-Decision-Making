#!/usr/bin/env bash
# Slurm array: one single-GPU task per shard.  Adjust partition/account/time to
# your cluster, then:
#   sbatch --array=0-7 scripts/slurm_array.sbatch main prostate qwen36 /path/to/data
# and, when all tasks have finished:
#   python -m vlmbench merge -e main -d prostate -m qwen36
#SBATCH --job-name=vlmbench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
exp=$1; ds=$2; model=$3; data=$4
python -m vlmbench run -e "$exp" -d "$ds" -m "$model" --data-root "$data" \
  --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$SLURM_ARRAY_TASK_COUNT"
