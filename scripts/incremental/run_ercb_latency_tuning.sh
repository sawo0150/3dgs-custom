#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 DATASET SCHEDULE OUTPUT_ROOT SEED" >&2
    exit 2
fi
task_data=$(realpath "$1")
task_schedule=$(realpath "$2")
task_root=$(realpath -m "$3")
task_seed=$4
task_python=${PYTHON:-python}
task_iterations=$($task_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["total_iterations"])' "$task_schedule")

if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
    echo "GPU compute process is already active; refusing to start." >&2
    exit 3
fi

# scheduler:fraction:deadline:temperature:block:candidates:label
declare -a task_specs=(
    "ticket_archive_rr:.25:128:.25:8:24:ticket_f025_d128"
    "ticket_archive_rr:.25:256:.25:8:24:ticket_f025_d256"
    "ticket_archive_rr:.50:1000000:.25:8:24:ticket_f050_free"
    "ticket_archive_rr:.65:1000000:.25:8:24:ticket_f065_free"
    "ticket_archive_rr:.75:1000000:.25:8:24:ticket_f075_free"
)
if [[ ${ERCB_EXP02_POLICY_TUNE:-0} == 1 ]]; then
    task_specs=(
        "ticket_debt_utility:.75:1000000:.10:8:16:debt_t010_c16"
        "ticket_debt_utility:.75:1000000:.50:8:32:debt_t050_c32"
        "ticket_service_field:.75:1000000:.10:8:16:service_t010_c16"
        "ticket_service_field:.75:1000000:.25:8:24:service_t025_c24"
        "ticket_service_field:.75:1000000:.50:8:32:service_t050_c32"
        "ticket_prefix_balance:.75:1000000:.10:4:16:prefix_qfix_t010_b4_c16"
        "ticket_prefix_balance:.75:1000000:.25:8:24:prefix_qfix_t025_b8_c24"
        "ticket_prefix_balance:.75:1000000:.50:16:32:prefix_qfix_t050_b16_c32"
    )
fi

mkdir -p "$task_root"
for task_spec in "${task_specs[@]}"; do
    IFS=: read -r scheduler fraction deadline temperature block candidates label <<<"$task_spec"
    output="$task_root/${label}_r4_s${task_seed}"
    if [[ -f "$output/screening_metrics.json" && -f "$output/service_latency_metrics.json" ]]; then
        echo "SKIP $label"
        continue
    fi
    if [[ -e "$output" || -e "$output.train.log" ]]; then
        echo "Refusing to overwrite partial $output" >&2
        exit 4
    fi
    echo "[$(date --iso-8601=seconds)] START $label"
    /usr/bin/time -f '%e' -o "$output.wall_seconds.txt" "$task_python" train.py \
        -s "$task_data" -m "$output" -r 4 --eval --iterations "$task_iterations" \
        --test_iterations 999999 --save_iterations "$task_iterations" \
        --view_schedule "$task_schedule" --view_scheduler "$scheduler" \
        --scheduler_seed "$task_seed" --scheduler_beta "$temperature" \
        --scheduler_block_size "$block" --scheduler_ticket_fraction "$fraction" \
        --scheduler_deadline_steps "$deadline" --scheduler_candidate_size "$candidates" \
        --apply_final_optimizer_step --quiet --disable_viewer >"$output.train.log" 2>&1
    "$task_python" render.py -s "$task_data" -m "$output" -r 4 --eval \
        --iteration "$task_iterations" --skip_train --quiet >"$output.render.log" 2>&1
    "$task_python" scripts/incremental/compute_heldout_psnr.py "$output" >"$output.screening.log" 2>&1
    "$task_python" scripts/incremental/compute_service_latency.py "$output" "$task_schedule" >"$output.service_latency.log" 2>&1
    echo "[$(date --iso-8601=seconds)] DONE  $label"
done
