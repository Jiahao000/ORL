#!/usr/bin/env bash
set -e
set -x

PARTITION=$1
CFG=$2
EPOCH=$3
FEAT_LIST=$4 # e.g.: "feat5", "feat4 feat5". If leave empty, the default is "feat5"
GPUS=${5:-8}
PY_ARGS=${@:6}
JOB_NAME="openselfsup"
SRUN_ARGS=${SRUN_ARGS:-""}
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

if [ ! -f $WORK_DIR/epoch_${EPOCH}.pth ]; then
    echo "ERROR: File not exist: $WORK_DIR/epoch_${EPOCH}.pth"
    exit
fi

mkdir -p $WORK_DIR/logs
echo "Testing checkpoint: $WORK_DIR/epoch_${EPOCH}.pth" 2>&1 | tee -a $WORK_DIR/logs/eval_svm.log

bash tools/srun_extract.sh $PARTITION $CFG $GPUS $WORK_DIR --checkpoint $WORK_DIR/epoch_${EPOCH}.pth ${PY_ARGS}

srun -p $PARTITION --job-name=${JOB_NAME} ${SRUN_ARGS} bash benchmarks/svm_tools/eval_svm_full.sh $WORK_DIR "$FEAT_LIST"

srun -p $PARTITION --job-name=${JOB_NAME} ${SRUN_ARGS} bash benchmarks/svm_tools/eval_svm_lowshot.sh $WORK_DIR "$FEAT_LIST"
