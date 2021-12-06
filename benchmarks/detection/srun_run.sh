#!/usr/bin/env bash

set -e
set -x

PARTITION=$1
DET_CFG=$2
WEIGHTS=$3
OUT=$4
JOB_NAME="run"
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}

# train
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=1 \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python $(dirname "$0")/train_net.py --config-file $DET_CFG \
        --num-gpus 8 MODEL.WEIGHTS $WEIGHTS OUTPUT_DIR $OUT
