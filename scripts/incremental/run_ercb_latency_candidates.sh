#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 DATASET_ROOT SCHEDULE_ROOT OUTPUT_ROOT SEED" >&2
    exit 2
fi

task_data_root=$(realpath "$1")
task_schedule_root=$(realpath "$2")
task_output_root=$(realpath -m "$3")
task_seed=$4
task_python=${PYTHON:-python}

if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
    echo "GPU compute process is already active; refusing to start." >&2
    nvidia-smi >&2
    exit 3
fi

declare -a task_scenes=(slow-straight-2 ego-drive)
if [[ ${ERCB_EXP02_ALL_SCENES:-0} == 1 ]]; then
    task_scenes=(ego-centric-1 ego-centric-2 ego-drive fast-straight slow-straight-2 square-1)
fi

# scheduler:ticket_fraction:deadline_steps:temperature:block:candidates:label
declare -a task_specs=(
    "ticket_archive_rr:0:1000000:.25:8:24:capture_rr"
    "ticket_archive_rr:.75:32:.25:8:24:ticket_archive_rr"
    "ticket_moment_compensation:.75:32:.25:8:24:dense_moment"
    "ticket_service_field:.75:32:.25:8:24:dense_service_field"
    "ticket_debt_utility:.75:32:.25:8:24:debt_utility_A"
    "ticket_prefix_balance:.75:32:.25:8:24:prefix_balance_B"
    "ticket_approx_mir:.75:32:.25:8:24:approx_mir_C"
    "ticket_pair_mean:.75:32:.25:8:24:pair_mean_dense3"
    "ticket_pair_safe:.75:32:.25:8:24:pair_safe_D"
)

# Predeclared final gate after the two-scene screen: retain the two highest
# held-out-PSNR latency methodologies (B and A), plus capture-time RR control.
if [[ ${ERCB_EXP02_FINAL:-0} == 1 ]]; then
    task_specs=(
        "ticket_archive_rr:0:1000000:.25:8:24:capture_rr"
        "ticket_debt_utility:.75:1000000:.50:8:32:debt_final_t050_c32"
        "ticket_prefix_balance:.75:1000000:.10:4:16:prefix_final_t010_b4_c16"
    )
fi
if [[ ${ERCB_EXP02_SERVICE_FINAL:-0} == 1 ]]; then
    task_specs=(
        "ticket_service_field:.75:1000000:.25:8:24:service_final_t025_c24"
    )
fi
if [[ ${ERCB_EXP02_MOMENT_PAIR_AUDIT:-0} == 1 ]]; then
    task_specs=(
        "ticket_archive_rr:0:1000000:.25:8:24:capture_rr"
        "ticket_moment_compensation:.75:32:.25:8:24:dense_moment"
        "ticket_pair_mean:.75:32:.25:8:24:pair_mean_dense3"
        "ticket_pair_safe:.75:32:.25:8:24:pair_safe_D"
    )
fi

mkdir -p "$task_output_root"
for task_scene in "${task_scenes[@]}"; do
    task_data="$task_data_root/$task_scene"
    task_schedule="$task_schedule_root/$task_scene.json"
    task_iterations=$($task_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["total_iterations"])' "$task_schedule")
    for task_spec in "${task_specs[@]}"; do
        IFS=: read -r task_scheduler task_fraction task_deadline task_temperature task_block task_candidates task_label <<<"$task_spec"
        task_output="$task_output_root/$task_scene/${task_label}_r4_s${task_seed}"
        if [[ -f "$task_output/screening_metrics.json" && -f "$task_output/service_latency_metrics.json" ]]; then
            echo "[$(date --iso-8601=seconds)] SKIP $task_scene $task_label"
            continue
        fi
        if [[ -e "$task_output" || -e "$task_output.train.log" ]]; then
            echo "Refusing to overwrite partial run: $task_output" >&2
            exit 4
        fi
        mkdir -p "$(dirname "$task_output")"
        echo "[$(date --iso-8601=seconds)] START $task_scene $task_label T=$task_iterations"
        /usr/bin/time -f '%e' -o "$task_output.wall_seconds.txt" \
            "$task_python" train.py \
                -s "$task_data" -m "$task_output" -r 4 --eval \
                --iterations "$task_iterations" --test_iterations 999999 \
                --save_iterations "$task_iterations" --view_schedule "$task_schedule" \
                --view_scheduler "$task_scheduler" --scheduler_seed "$task_seed" \
                --scheduler_beta "$task_temperature" --scheduler_block_size "$task_block" \
                --scheduler_ticket_fraction "$task_fraction" \
                --scheduler_deadline_steps "$task_deadline" \
                --scheduler_candidate_size "$task_candidates" --apply_final_optimizer_step \
                --quiet --disable_viewer >"$task_output.train.log" 2>&1
        "$task_python" render.py -s "$task_data" -m "$task_output" -r 4 --eval \
            --iteration "$task_iterations" --skip_train --quiet \
            >"$task_output.render.log" 2>&1
        "$task_python" scripts/incremental/compute_heldout_psnr.py "$task_output" \
            >"$task_output.screening.log" 2>&1
        "$task_python" scripts/incremental/compute_service_latency.py "$task_output" "$task_schedule" \
            >"$task_output.service_latency.log" 2>&1
        $task_python - "$task_output" "$task_data" "$task_schedule" "$task_label" <<'PY'
import hashlib, json, pathlib, sys
run, data, schedule, label = map(pathlib.Path, sys.argv[1:])
payload = {
    "protocol": "ercb_latency_exp02_v1", "arm": label.name,
    "dataset": str(data.resolve()), "schedule": str(schedule.resolve()),
    "clock": "physical RGB capture timestamp", "tail_optimizer_updates": 0,
    "git_head": None,
}
try:
    import subprocess
    payload["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    pass
(run / "run_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
        echo "[$(date --iso-8601=seconds)] DONE  $task_scene $task_label"
    done
done
