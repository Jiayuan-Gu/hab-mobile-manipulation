#!/usr/bin/env bash
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

# -------------------------------------------------------------------------- #
# 220417
# -------------------------------------------------------------------------- #
gen_train () {
    macro_id=$1
    for micro_id in {0..15}
    do
        scene_id="v3_sc${macro_id}_staging_$(printf %02d ${micro_id})"
        seed="${macro_id}$(printf %02d ${micro_id})"
        echo $scene_id $seed
        python -m habitat.datasets.rearrange.rearrange_generator --run --config scripts/datasets/configs/${TASK_NAME}.yaml --num-episodes 100 --out data/datasets/rearrange/v3/${TASK_NAME}_train.$scene_id.json.gz --seed $seed scene_sampler.params.scene $scene_id
    done
}

gen_val () {
    for macro_id in {0..3}
    do
        for micro_id in {16..20}
        do
            scene_id="v3_sc${macro_id}_staging_$(printf %02d ${micro_id})"
            seed="${macro_id}$(printf %02d ${micro_id})"
            echo $scene_id $seed
            python -m habitat.datasets.rearrange.rearrange_generator --run --config scripts/datasets/configs/${TASK_NAME}.yaml --num-episodes 5 --out data/datasets/rearrange/v3/${TASK_NAME}_val.$scene_id.json.gz --seed $seed scene_sampler.params.scene $scene_id
        done
    done
}

gen_eval () {
    for macro_id in {4..4}
    do
        for micro_id in {0..19}
        do
            scene_id="v3_sc${macro_id}_staging_$(printf %02d ${micro_id})"
            seed="${macro_id}$(printf %02d ${micro_id})"
            echo $scene_id $seed
            python -m habitat.datasets.rearrange.rearrange_generator --run --config scripts/datasets/configs/${TASK_NAME}.yaml --num-episodes 5 --out data/datasets/rearrange/v3/${TASK_NAME}_eval.$scene_id.json.gz --seed $seed scene_sampler.params.scene $scene_id
        done
    done
}

TASK_NAMES=(
    tidy_house_220417
    prepare_groceries_220417
    set_table_220417
)

for TASK_NAME in ${TASK_NAMES[*]}; do
    gen_train 0 &
    gen_train 1 > /dev/null 2>&1 &
    gen_train 2 > /dev/null 2>&1 &
    gen_train 3 > /dev/null 2>&1 &
    gen_val
    # gen_eval
    wait
done