import gzip
import json

import tqdm


def load_json_gz(filename):
    with gzip.open(filename, "rt") as f:
        return json.loads(f.read())


def dump_json_gz(filename, obj):
    with gzip.open(filename, "wt") as f:
        json.dump(obj, f)


for i in tqdm.tqdm(range(64)):
    i0 = i % 63
    i1 = (i + 1) % 63
    in_path0 = f"data/datasets/replica_cad/rearrange/v1/train/rearrange_easy.{i0:02d}.json.gz"
    in_path1 = f"data/datasets/replica_cad/rearrange/v1/train/rearrange_easy.{i1:02d}.json.gz"
    out_path = f"data/datasets/replica_cad/rearrange/v1/train/rearrange_easy.v1.{i:02d}.json.gz"
    dataset0 = load_json_gz(in_path0)
    dataset1 = load_json_gz(in_path1)
    dataset0["episodes"].extend(dataset1["episodes"])
    dump_json_gz(out_path, dataset0)
