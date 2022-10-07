from collections import defaultdict
import gzip
import json
from habitat import Config, Env, RLEnv, make_dataset

import habitat_extensions.tasks.rearrange.env

dataset_config = Config(
    dict(
        TYPE="RearrangeDataset-v0",
        SPLIT="train",
        DATA_PATH="data/datasets/replica_cad/rearrange/v1/{split}/rearrange_easy.json.gz",
        SCENES_DIR="",
    )
)

dataset = make_dataset("RearrangeDataset-v0", config=dataset_config)
# print(dataset.num_episodes)
# print(len(dataset.scene_ids))
# datasets = dataset.get_splits(
#     63, sort_by_episode_id=True, allow_uneven_splits=True
# )
# for d in datasets:
#     print(d.scene_ids)

scene_id_episode_ids = defaultdict(list)
for episode in dataset.episodes:
    episode.markers = {}
    episode.name_to_receptacle = {}
    scene_id_episode_ids[episode.scene_id].append(episode)

n_ep = 0
scene_ids = sorted(dataset.scene_ids)
scene_ids.append(scene_ids[-1])
for i, scene_id in enumerate(scene_ids):
    dataset.episodes = scene_id_episode_ids[scene_id]
    for j, episode in enumerate(dataset.episodes):
        episode.episode_id = str(n_ep + j)
    json_str = dataset.to_json()
    datasetfile_path = f"data/datasets/replica_cad/rearrange/v1/train/rearrange_easy.{i:02d}.json.gz"
    with gzip.open(datasetfile_path, "wt") as f:
        f.write(json_str)
    n_ep += dataset.num_episodes
