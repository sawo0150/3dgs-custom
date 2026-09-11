#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 DATASET_DIR OUTPUT_ROOT RUN_LABEL" >&2
    exit 2
fi

task_data=$(realpath "$1")
task_root=$(realpath -m "$2")
task_label=$3
task_python=${PYTHON:-python}
task_iterations=$($task_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["total_iterations"])' "$task_data/causal_arrivals.json")

if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
    echo "GPU compute process is already active; refusing to start." >&2
    nvidia-smi >&2
    exit 3
fi

mkdir -p "$task_root"
declare -a task_schedulers=(
    "causal_rr:0.0:rr"
    "normalized_interval_size_softmax_rr:0.22314355131420976:normalized_odds125_K8"
    "relative_floor_interval_softmax_rr:1.0986122886681098:relative_half_odds3_K8"
)

for task_seed in 0 1; do
    for task_spec in "${task_schedulers[@]}"; do
        IFS=: read -r task_scheduler task_beta task_suffix <<<"$task_spec"
        task_output="$task_root/${task_label}_zerotail_${task_suffix}_r4_s${task_seed}"
        if [[ -e "$task_output" || -e "$task_output.train.log" ]]; then
            echo "Refusing to overwrite existing run: $task_output" >&2
            exit 4
        fi
        echo "[$(date --iso-8601=seconds)] START $task_label $task_suffix seed=$task_seed T=$task_iterations"
        "$task_python" train.py \
            -s "$task_data" -m "$task_output" -r 4 --eval \
            --iterations "$task_iterations" --test_iterations 999999 \
            --save_iterations "$task_iterations" \
            --view_schedule "$task_data/causal_arrivals.json" \
            --view_scheduler "$task_scheduler" \
            --scheduler_seed "$task_seed" --scheduler_beta "$task_beta" \
            --scheduler_block_size 8 --quiet --disable_viewer \
            >"$task_output.train.log" 2>&1
        "$task_python" render.py \
            -s "$task_data" -m "$task_output" -r 4 --eval \
            --iteration "$task_iterations" --skip_train --quiet \
            >"$task_output.render.log" 2>&1
        "$task_python" metrics.py -m "$task_output" \
            >"$task_output.metrics.log" 2>&1
        echo "[$(date --iso-8601=seconds)] DONE  $task_label $task_suffix seed=$task_seed"
    done
done

"$task_python" scripts/incremental/summarize_ercb_benchmark.py "$task_root" "$task_label"
