# Challenges

## Installation

```bash
conda create -n rearrange-challenge python=3.7 cmake=3.14.0
conda activate rearrange-challenge
conda install -y habitat-sim-rearrange-challenge-2022  withbullet  headless -c conda-forge -c aihabitat
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets
```

## Notes

- remove largest island check
- hardcode when `SPLIT_DATASET=True`
- `episode_id` is not unique in training data
- remove ik feasibility check
- `snap_to_object(force=False)` to avoid potential bugs

## Issues

- <https://github.com/facebookresearch/habitat-lab/pull/952>

## Differences between M3 and challenge

- sliding is disabled
- arm initialization
- suction gripper
- camera placement
- async rendering
- robot's fixed joints are not updated per internal step
