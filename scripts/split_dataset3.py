import gzip
import json
from collections import defaultdict
import tqdm
import random


def load_json_gz(filename):
    with gzip.open(filename, "rt") as f:
        return json.loads(f.read())


def dump_json_gz(filename, obj):
    with gzip.open(filename, "wt") as f:
        json.dump(obj, f)


scene_id_episode_ids = defaultdict(list)
for i in range(160):
    print(i)
    in_path = f"data/datasets/replica_cad/rearrange/v1/val/rearrange_easy.raw.{i:03d}.json.gz"
    dataset = load_json_gz(in_path)
    for episode in dataset["episodes"]:
        scene_id_episode_ids[episode["scene_id"]].append(episode)


n_ep = 50000
scene_ids = sorted(scene_id_episode_ids.keys())
for i, scene_id in enumerate(scene_ids):
    episodes = scene_id_episode_ids[scene_id]
    for j, episode in enumerate(episodes):
        episode["episode_id"] = str(n_ep + j)
    n_ep += len(episodes)

# n_ep = 50000
# scene_ids = sorted(scene_id_episode_ids.keys())
# for i in range(64):
#     print(i)
#     scene_id = scene_ids[i % len(scene_ids)]
#     episodes = scene_id_episode_ids[scene_id]
#     sub_scene_id = (i // len(scene_ids)) % 3
#     sub_cnt = len(episodes) // 3
#     episodes = episodes[: (sub_scene_id + 1) * sub_cnt]
#     for j, episode in enumerate(episodes):
#         episode["episode_id"] = str(n_ep + j)
#     dump_json_gz(
#         f"data/datasets/replica_cad/rearrange/v1/val/rearrange_easy.v1.{i:02d}.json.gz",
#         {"episodes": episodes},
#     )
#     n_ep += len(episodes)


for i in range(63):
    print(i)
    in_path = f"data/datasets/replica_cad/rearrange/v1/train/rearrange_easy.{i:02d}.json.gz"
    dataset = load_json_gz(in_path)
    for episode in dataset["episodes"]:
        scene_id_episode_ids[episode["scene_id"]].append(episode)


random.seed(2022)
scene_ids = sorted(scene_id_episode_ids.keys())
for i in range(64):
    episodes = []
    if i < 42:
        for j in range(2):
            scene_id = scene_ids[(i * 2 + j) % len(scene_ids)]
            print(i, scene_id)
            episodes.extend(scene_id_episode_ids[scene_id])
    else:
        random.shuffle(scene_ids)
        for scene_id in scene_ids[:2]:
            print(i, scene_id)
            episodes.extend(scene_id_episode_ids[scene_id])
    dump_json_gz(
        f"data/datasets/replica_cad/rearrange/v1/train/rearrange_easy.v2.{i:02d}.json.gz",
        {"episodes": episodes},
    )
