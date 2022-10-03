# Challenges

## Installation

```bash
conda create -n rearrange-challenge python=3.7 cmake=3.14.0
conda activate rearrange-challenge
conda install -y habitat-sim-rearrange-challenge-2022  withbullet  headless -c conda-forge -c aihabitat
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets
```

## Issues

- <https://github.com/facebookresearch/habitat-lab/pull/952>

## Differences

- sliding is disabled
- arm initialization is without noise
