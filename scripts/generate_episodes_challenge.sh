#!/usr/bin/env bash
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

for i in {0..15}
do
    for j in {0..9}
    do
        k=$((i*10+j))
        seed=$((k + 2022))
        name="$(printf %03d ${k})"
        echo $seed $name
        python -m habitat.datasets.rearrange.run_episode_generator --run --config habitat-lab/habitat/datasets/rearrange/configs/hab/rearrange_easy.yaml --num-episodes 100 --out data/datasets/replica_cad/rearrange/v1/val/rearrange_easy.raw.$name.json.gz --seed $seed scene_sampler.params.scene_sets "['scene_val_split']" &
    done
    wait
done
