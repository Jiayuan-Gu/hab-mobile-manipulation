#!/usr/bin/env bash
export MAGNUM_LOG=quiet GLOG_minloglevel=2 HABITAT_SIM_LOG=quiet

evaluate_run () {
    if [[ -z "$SPLIT" ]]; then
        SPLIT=val
        echo "set split to $SPLIT"
    fi
    if [[ -z "$SEED" ]]; then
        SEED=100
        echo "set seed to $SEED"
    fi
    if [[ -z "${TRAIN_SEED}" ]]; then
        TRAIN_SEED=100
        echo "set train seed to ${TRAIN_SEED}"
    fi
    if [[ -z "$DATE" ]]; then
        DATE=$(date +'%y%m%d')
        echo "set date to $DATE"
    fi
    if [[ -z "$PREFIX" ]]; then
        PREFIX="train_seed=${TRAIN_SEED}.seed=${SEED}"
        echo "set prefix to $PREFIX"
    fi
    PREFIX="${SPLIT}.${DATE}${PREFIX2}/${PREFIX}"
    
    python mobile_manipulation/eval_composite.py --cfg ${CONFIG} --split ${SPLIT} --no-rgb --save-log --train-seed ${TRAIN_SEED} ${ARGS} PREFIX ${PREFIX} TASK_CONFIG.SEED ${SEED} ${OPTS}
}

evaluate_runs () {
    for SEED in {100..102}
    do
        if [ $BG -eq 1 ]; then
            echo background
            # evaluate_run
            evaluate_run > /dev/null 2>&1 &
            # evaluate_run > /dev/null &
        else
            evaluate_run
        fi
    done
}

CONFIG=$1
if [[ -z "$CONFIG" ]]; then
    echo "No config provided!"
    exit 1
fi

for TRAIN_SEED in 100 101 102; do
    evaluate_runs
done
wait
