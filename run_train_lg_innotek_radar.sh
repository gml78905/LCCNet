#!/bin/bash

set -e

GPU_IDS="0"
WANDB_NAME="LCCNet_Radar"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_TS="$(date '+%Y%m%d_%H%M%S')"
LOG_ROOT="$SCRIPT_DIR/run_logs/lg_innotek_radar/$RUN_TS"
mkdir -p "$LOG_ROOT"

STDOUT_LOG="$LOG_ROOT/train.log"
GPU_LOG="$LOG_ROOT/nvidia_smi.log"
META_LOG="$LOG_ROOT/meta.log"
CMD_LOG="$LOG_ROOT/command.sh"

cat > "$CMD_LOG" <<EOF
python3 -u train_with_sacred.py with \\
    dataset='lg_innotek' \\
    data_folder='/workspace/data/LG_Innotek/CustomDataset/LG_Innotek' \\
    sensor_mode='radar' \\
    lg_train_scene='["afternoon_parking_lot_1","afternoon_campus_1"]' \\
    lg_val_scene='["afternoon_campus_2"]' \\
    lg_val_frame_limit=3000 \\
    epochs=120 \\
    num_worker=4 \\
    batch_size=120 \\
    max_t=0.5 \\
    max_r=5.0 \\
    checkpoint_name='lg_innotek_radar' \\
    wandb_enabled=True \\
    wandb_entity='LGIT_calib' \\
    wandb_project='LCCNet' \\
    wandb_name="$WANDB_NAME" \\
    debug_timing=True
EOF

{
    echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "hostname=$(hostname)"
    echo "cwd=$SCRIPT_DIR"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "python_unbuffered=$PYTHONUNBUFFERED"
    echo "log_root=$LOG_ROOT"
} > "$META_LOG"

cleanup() {
    if [ -n "${MONITOR_PID:-}" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
    (
        echo "timestamp,index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit"
        while true; do
            nvidia-smi \
                --query-gpu=timestamp,index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit \
                --format=csv,noheader,nounits
            sleep 2
        done
    ) >> "$GPU_LOG" 2>&1 &
    MONITOR_PID=$!
fi

set +e
bash "$CMD_LOG" 2>&1 | tee "$STDOUT_LOG"
EXIT_CODE=${PIPESTATUS[0]}
set -e

{
    echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "exit_code=$EXIT_CODE"
} >> "$META_LOG"

echo "Logs saved to: $LOG_ROOT"
exit "$EXIT_CODE"
