#!/bin/bash

set -x

PARTITION=$1
CFG=$2
OUTPUT=$3
GPUS=1  # support single-gpu inference only
PY_ARGS=${@:4} # --port
JOB_NAME="selective_search"
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

# test
GLOG_vmodule=MemcachedClient=-1 \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/selective_search.py \
        $CFG \
        $OUTPUT \
        --work_dir $WORK_DIR --launcher="slurm" $PY_ARGS
