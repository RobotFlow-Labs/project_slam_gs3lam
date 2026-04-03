#!/bin/bash
# Independent training launcher — survives Claude session restarts
cd /mnt/forge-data/modules/03_wave7/project_slam_gs3lam
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=""
export CUDA_VISIBLE_DEVICES=2

MODULE="project_slam_gs3lam"
LOGFILE="/mnt/artifacts-datai/logs/$MODULE/train_office2_$(date +%Y%m%d_%H%M).log"

exec python scripts/train_replica.py \
    --scene office2 \
    --config configs/l4_replica.toml \
    --max-frames -1 \
    --ckpt-every 100 \
    --eval-every 10 \
    --device cuda \
    > "$LOGFILE" 2>&1
