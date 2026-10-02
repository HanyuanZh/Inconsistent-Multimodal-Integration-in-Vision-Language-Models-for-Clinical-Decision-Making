#!/usr/bin/env bash
# Batch API run for gemini or gpt.  `build` is offline; `submit` costs money and
# asks for confirmation; `fetch` works once every job has finished (usually
# within 24 h).  Keys come from the environment: GEMINI_API_KEY / OPENAI_API_KEY.
#
#   scripts/run_api_batch.sh <experiment> <dataset> <gemini|gpt> <data_root> <build|submit|status|fetch>
set -euo pipefail
exp=$1; ds=$2; model=$3; data=$4; action=$5
extra=()
if [[ $action == submit ]]; then
  read -r -p "Submit $exp/$ds to $model? This incurs API charges [y/N] " ok
  [[ $ok == y ]] || exit 1
  extra=(--yes)
fi
python -m vlmbench batch "$action" -e "$exp" -d "$ds" -m "$model" --data-root "$data" "${extra[@]}"
