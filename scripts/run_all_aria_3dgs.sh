#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/home/wosas/Desktop/26-1_RPM/Datas/CustomData}"
CONVERT_ROOT="${CONVERT_ROOT:-${REPO_ROOT}/datasets/aria_3dgs}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/output/aria_3dgs}"
LOG_ROOT="${LOG_ROOT:-${RESULT_ROOT}/logs}"

PYTHON="${PYTHON:-python3}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
ITERATIONS="${ITERATIONS:-30000}"
TEST_ITERATIONS="${TEST_ITERATIONS:-7000 30000}"
SAVE_ITERATIONS="${SAVE_ITERATIONS:-7000 30000}"
TRAIN_DATA_DEVICE="${TRAIN_DATA_DEVICE:-cpu}"
TRAIN_RESOLUTION="${TRAIN_RESOLUTION:--1}"

ROTATE="${ROTATE:-0}"
ALPHA_MASK="${ALPHA_MASK:-0}"
SKIP_CONVERT_EXISTING="${SKIP_CONVERT_EXISTING:-1}"
SKIP_TRAIN_EXISTING="${SKIP_TRAIN_EXISTING:-1}"
CONVERT_ONLY="${CONVERT_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

EXTRA_CONVERT_ARGS="${EXTRA_CONVERT_ARGS:-}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

mkdir -p "${CONVERT_ROOT}" "${RESULT_ROOT}" "${LOG_ROOT}"

sanitize_relpath() {
    local path="$1"
    local rel="${path#${DATA_ROOT}/}"
    rel="${rel// /_}"
    rel="${rel//\//__}"
    echo "${rel}"
}

find_vrs_file() {
    local aria_dir="$1"
    local mps_dir="$2"
    local mps_name
    mps_name="$(basename "${mps_dir}")"

    local stem="${mps_name#mps_}"
    stem="${stem%_vrs}"

    if [[ -f "${aria_dir}/${stem}.vrs" ]]; then
        echo "${aria_dir}/${stem}.vrs"
        return 0
    fi

    local matches=()
    while IFS= read -r -d '' candidate; do
        matches+=("${candidate}")
    done < <(find "${aria_dir}" -maxdepth 1 -type f -name "*.vrs" -print0 | sort -z)

    if [[ "${#matches[@]}" -eq 1 ]]; then
        echo "${matches[0]}"
        return 0
    fi

    return 1
}

run_cmd() {
    echo "+ $*"
    if [[ "${DRY_RUN}" == "0" ]]; then
        "$@"
    fi
}

has_converted_scene() {
    local data_out="$1"

    if [[ -f "${data_out}/.conversion_done" ]]; then
        return 0
    fi

    [[ -f "${data_out}/sparse/0/cameras.txt" ]] || return 1
    [[ -f "${data_out}/sparse/0/images.txt" ]] || return 1
    [[ -f "${data_out}/sparse/0/points3D.txt" ]] || return 1

    find "${data_out}/images" -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.png" \) -print -quit 2>/dev/null | grep -q .
}

convert_args=()
if [[ "${ROTATE}" == "1" ]]; then
    convert_args+=(--rotate)
fi
if [[ "${ALPHA_MASK}" == "1" ]]; then
    convert_args+=(--alpha_mask)
fi

scene_count=0
failed_count=0

while IFS= read -r -d '' traj_csv; do
    slam_dir="$(dirname "${traj_csv}")"
    mps_dir="$(dirname "${slam_dir}")"
    aria_dir="$(dirname "${mps_dir}")"
    points_csv="${slam_dir}/semidense_points.csv.gz"

    if [[ ! -f "${points_csv}" ]]; then
        echo "[skip] missing semidense points: ${slam_dir}" >&2
        continue
    fi

    if ! vrs_file="$(find_vrs_file "${aria_dir}" "${mps_dir}")"; then
        echo "[skip] could not uniquely match .vrs for: ${aria_dir}" >&2
        failed_count=$((failed_count + 1))
        continue
    fi

    scene_name="$(sanitize_relpath "${aria_dir}")"
    data_out="${CONVERT_ROOT}/${scene_name}"
    train_out="${RESULT_ROOT}/${scene_name}"
    convert_log="${LOG_ROOT}/${scene_name}_convert.log"
    train_log="${LOG_ROOT}/${scene_name}_train.log"

    scene_count=$((scene_count + 1))
    echo
    echo "== [${scene_count}] ${scene_name}"
    echo "aria_dir : ${aria_dir}"
    echo "vrs_file : ${vrs_file}"
    echo "mps_dir  : ${mps_dir}"

    if [[ "${SKIP_CONVERT_EXISTING}" == "1" ]] && has_converted_scene "${data_out}"; then
        echo "[convert] existing converted scene found, skip: ${data_out}"
    else
        if [[ "${DRY_RUN}" == "1" ]]; then
            run_cmd "${PYTHON}" "${REPO_ROOT}/aria_to_3dgs.py" \
                --aria_dir "${aria_dir}" \
                --vrs_file "${vrs_file}" \
                --trajectory_csv "${traj_csv}" \
                --points_csv "${points_csv}" \
                --output_dir "${data_out}" \
                --width "${WIDTH}" \
                --height "${HEIGHT}" \
                "${convert_args[@]}" \
                ${EXTRA_CONVERT_ARGS}
        else
            "${PYTHON}" "${REPO_ROOT}/aria_to_3dgs.py" \
                --aria_dir "${aria_dir}" \
                --vrs_file "${vrs_file}" \
                --trajectory_csv "${traj_csv}" \
                --points_csv "${points_csv}" \
                --output_dir "${data_out}" \
                --width "${WIDTH}" \
                --height "${HEIGHT}" \
                "${convert_args[@]}" \
                ${EXTRA_CONVERT_ARGS} 2>&1 | tee "${convert_log}"
            touch "${data_out}/.conversion_done"
        fi
    fi

    if [[ "${CONVERT_ONLY}" == "1" ]]; then
        continue
    fi

    if [[ "${SKIP_TRAIN_EXISTING}" == "1" && -f "${train_out}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply" ]]; then
        echo "[train] existing final model found, skip: ${train_out}"
        continue
    fi

    mkdir -p "${train_out}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        run_cmd "${PYTHON}" "${REPO_ROOT}/train.py" \
            -s "${data_out}" \
            -m "${train_out}" \
            --iterations "${ITERATIONS}" \
            --test_iterations ${TEST_ITERATIONS} \
            --save_iterations ${SAVE_ITERATIONS} \
            --data_device "${TRAIN_DATA_DEVICE}" \
            --resolution "${TRAIN_RESOLUTION}" \
            --disable_viewer \
            ${EXTRA_TRAIN_ARGS}
    else
        "${PYTHON}" "${REPO_ROOT}/train.py" \
            -s "${data_out}" \
            -m "${train_out}" \
            --iterations "${ITERATIONS}" \
            --test_iterations ${TEST_ITERATIONS} \
            --save_iterations ${SAVE_ITERATIONS} \
            --data_device "${TRAIN_DATA_DEVICE}" \
            --resolution "${TRAIN_RESOLUTION}" \
            --disable_viewer \
            ${EXTRA_TRAIN_ARGS} 2>&1 | tee "${train_log}"
    fi
done < <(find "${DATA_ROOT}" -path "*/slam/closed_loop_trajectory.csv" -print0 | sort -z)

echo
echo "Done. scenes=${scene_count}, unmatched_or_failed=${failed_count}"
echo "Converted data: ${CONVERT_ROOT}"
echo "Training runs : ${RESULT_ROOT}"
echo "Logs          : ${LOG_ROOT}"
