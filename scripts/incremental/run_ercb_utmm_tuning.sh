#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 DATASET_ROOT OUTPUT_ROOT [SEED]" >&2
    exit 2
fi

task_data_root=$(realpath "$1")
task_output_root=$(realpath -m "$2")
task_seed=${3:-0}
task_python=${PYTHON:-python}

if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
    echo "GPU compute process is already active; refusing to start." >&2
    nvidia-smi >&2
    exit 3
fi

declare -a task_sequences=(
    ego-centric-1
    ego-centric-2
    ego-drive
    fast-straight
    slow-straight-2
    square-1
)

# One bundle-wide grid.  The baseline is evaluated once per seed and scene;
# every ERCB setting shares K=8 while rho and exp(gamma) are tuned.
declare -a task_specs=(
    "causal_rr:0:0.5:8:rr"
    "relative_floor_interval_softmax_rr:0.09531017980432493:0.25:8:ercb_rho025_odds11_K8"
    "relative_floor_interval_softmax_rr:0.22314355131420976:0.25:8:ercb_rho025_odds125_K8"
    "relative_floor_interval_softmax_rr:0.4054651081081644:0.25:8:ercb_rho025_odds15_K8"
    "relative_floor_interval_softmax_rr:0.6931471805599453:0.25:8:ercb_rho025_odds2_K8"
    "relative_floor_interval_softmax_rr:1.0986122886681098:0.25:8:ercb_rho025_odds3_K8"
    "relative_floor_interval_softmax_rr:0.09531017980432493:0.5:8:ercb_rho050_odds11_K8"
    "relative_floor_interval_softmax_rr:0.22314355131420976:0.5:8:ercb_rho050_odds125_K8"
    "relative_floor_interval_softmax_rr:0.4054651081081644:0.5:8:ercb_rho050_odds15_K8"
    "relative_floor_interval_softmax_rr:0.6931471805599453:0.5:8:ercb_rho050_odds2_K8"
    "relative_floor_interval_softmax_rr:1.0986122886681098:0.5:8:ercb_rho050_odds3_K8"
    "relative_floor_interval_softmax_rr:0.09531017980432493:0.75:8:ercb_rho075_odds11_K8"
    "relative_floor_interval_softmax_rr:0.22314355131420976:0.75:8:ercb_rho075_odds125_K8"
    "relative_floor_interval_softmax_rr:0.4054651081081644:0.75:8:ercb_rho075_odds15_K8"
    "relative_floor_interval_softmax_rr:0.6931471805599453:0.75:8:ercb_rho075_odds2_K8"
    "relative_floor_interval_softmax_rr:1.0986122886681098:0.75:8:ercb_rho075_odds3_K8"
)

if [[ ${ERCB_TUNING_SHORTLIST:-0} == 1 ]]; then
    task_specs=(
        "causal_rr:0:0.5:8:rr"
        "relative_floor_interval_softmax_rr:0.09531017980432493:0.75:8:ercb_rho075_odds11_K8"
        "relative_floor_interval_softmax_rr:0.4054651081081644:0.25:8:ercb_rho025_odds15_K8"
        "relative_floor_interval_softmax_rr:0.4054651081081644:0.75:8:ercb_rho075_odds15_K8"
    )
fi

if [[ ${ERCB_TUNING_FINAL:-0} == 1 ]]; then
    task_specs=(
        "causal_rr:0:0.5:8:rr"
        "relative_floor_interval_softmax_rr:0.4054651081081644:0.75:8:ercb_rho075_odds15_K8"
    )
fi

if [[ ${ERCB_TUNING_K:-0} == 1 ]]; then
    task_specs=(
        "causal_rr:0:0.5:8:rr"
        "relative_floor_interval_softmax_rr:0.4054651081081644:0.75:4:ercb_rho075_odds15_K4"
        "relative_floor_interval_softmax_rr:0.4054651081081644:0.75:16:ercb_rho075_odds15_K16"
    )
fi

mkdir -p "$task_output_root"
for task_sequence in "${task_sequences[@]}"; do
    task_data="$task_data_root/$task_sequence"
    task_iterations=$($task_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["total_iterations"])' "$task_data/causal_arrivals.json")
    task_sequence_root="$task_output_root/$task_sequence"
    mkdir -p "$task_sequence_root"

    for task_spec in "${task_specs[@]}"; do
        IFS=: read -r task_scheduler task_gamma task_rho task_k task_suffix <<<"$task_spec"
        task_output="$task_sequence_root/${task_suffix}_r4_s${task_seed}"
        if [[ -f "$task_output/screening_metrics.json" && -f "$task_output/view_scheduler_summary.json" ]]; then
            echo "[$(date --iso-8601=seconds)] SKIP complete $task_sequence $task_suffix seed=$task_seed"
            continue
        fi
        if [[ -e "$task_output" || -e "$task_output.train.log" ]]; then
            echo "Refusing to overwrite partial run: $task_output" >&2
            exit 4
        fi

        echo "[$(date --iso-8601=seconds)] START $task_sequence $task_suffix seed=$task_seed T=$task_iterations"
        "$task_python" train.py \
            -s "$task_data" -m "$task_output" -r 4 --eval \
            --iterations "$task_iterations" --test_iterations 999999 \
            --save_iterations "$task_iterations" \
            --view_schedule "$task_data/causal_arrivals.json" \
            --view_scheduler "$task_scheduler" \
            --scheduler_seed "$task_seed" --scheduler_beta "$task_gamma" \
            --scheduler_relative_floor_ratio "$task_rho" \
            --scheduler_block_size "$task_k" --quiet --disable_viewer \
            >"$task_output.train.log" 2>&1
        "$task_python" render.py \
            -s "$task_data" -m "$task_output" -r 4 --eval \
            --iteration "$task_iterations" --skip_train --quiet \
            >"$task_output.render.log" 2>&1
        "$task_python" scripts/incremental/compute_heldout_psnr.py "$task_output" \
            >"$task_output.screening.log" 2>&1
        "$task_python" scripts/incremental/write_ercb_run_manifest.py "$task_output" "$task_data"
        echo "[$(date --iso-8601=seconds)] DONE  $task_sequence $task_suffix seed=$task_seed"
    done
done

"$task_python" scripts/incremental/summarize_ercb_utmm_tuning.py "$task_output_root"
