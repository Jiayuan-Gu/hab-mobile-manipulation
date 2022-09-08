import gzip
import json
from pathlib import Path


def load_dataset(dataset_path):
    with gzip.open(dataset_path, "rt") as f:
        return json.loads(f.read())


def remove_markers(dataset_path, new_dataset_path=None):
    deserialized = load_dataset(dataset_path)
    for episode in deserialized["episodes"]:
        episode["markers"] = []
    if new_dataset_path is None:
        new_dataset_path = dataset_path
    with gzip.open(new_dataset_path, "wt") as f:
        json.dump(deserialized, f)


def merge_datasets(
    root_dir,
    prefix,
    update_episode_id=True,
    output_fname=None,
    num_episodes=None,
):
    root_dir = Path(root_dir)
    new_dataset = None
    if output_fname is None:
        new_dataset_path = root_dir / (prefix + ".json.gz")
    else:
        new_dataset_path = root_dir / (output_fname + ".json.gz")
    for dataset_path in sorted(Path(root_dir).glob(prefix + "*")):
        if dataset_path == new_dataset_path:
            print("Ignore", dataset_path)
            continue
        deserialized = load_dataset(dataset_path)
        if num_episodes is not None:
            deserialized["episodes"] = deserialized["episodes"][:num_episodes]
        if new_dataset is None:
            new_dataset = deserialized
        else:
            if update_episode_id:
                offset = len(new_dataset["episodes"])
                for episode in deserialized["episodes"]:
                    episode["episode_id"] = str(
                        int(episode["episode_id"]) + offset
                    )
            new_dataset["episodes"].extend(deserialized["episodes"])
    print("New dataset size", len(new_dataset["episodes"]))
    with gzip.open(new_dataset_path, "wt") as f:
        json.dump(new_dataset, f)


def main():
    for name in ["tidy_house", "prepare_groceries", "set_table"]:
        for split in ["train", "val"]:
            merge_datasets(
                "data/datasets/rearrange/v3",
                f"{name}_220417_{split}",
            )


if __name__ == "__main__":
    main()
