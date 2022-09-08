import argparse
import json
import pprint
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import sem


def merge_dicts(ds: List[Dict], keys=None, asarray=False):
    if keys is None:
        keys = list(ds[0].keys())
    ret = {k: [d[k] for d in ds] for k in keys}
    if asarray:
        ret = {k: np.concatenate(v) for k, v in ret.items()}
    return ret


def collect_results(root_dir, keys=None, pattern="**/result.json"):
    root_dir = Path(root_dir)
    json_files = root_dir.glob(pattern)
    all_results = []

    for path in sorted(json_files):
        rel_path = path.relative_to(root_dir)
        with open(path, "rt") as f:
            results = json.load(f)
        all_results.extend(results)

        metrics = merge_dicts(results, keys=keys)
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        print(",".join([str(rel_path)] + list(map(str, avg_metrics.values()))))

    merged_metrics = merge_dicts(all_results, keys=keys)
    stats = {
        k: (np.mean(v), np.std(v, ddof=1), sem(v))
        for k, v in merged_metrics.items()
    }
    pprint.pprint(
        {k: "{:.4f}({:.4f})".format(v[0], v[2]) for k, v in stats.items()}
    )
    return stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", "-d", type=str, nargs="*")
    parser.add_argument(
        "--pattern",
        "-p",
        type=str,
        default="**/result.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for root_dir in args.root_dir:
        if not Path(root_dir).is_dir():
            continue
        print(root_dir)
        if "tidy_house" in root_dir:
            if "so" in root_dir:
                keys = [
                    "gripper_status.pick_correct",
                    "place_obj_success",
                    "rearrange_place_success",
                    "elapsed_steps",
                ]
            elif "mo" in root_dir:
                keys = []
                for i in range(5):
                    keys.append(f"stage_success.pick_{i}")
                    keys.append(f"stage_success.place_{i}")
                keys.append("elapsed_steps")
        elif "prepare_groceries" in root_dir:
            keys = []
            for i in range(3):
                keys.append(f"stage_success.pick_{i}")
                keys.append(f"stage_success.place_{i}")
            keys.append("elapsed_steps")
        elif "set_table" in root_dir:
            keys = []
            for i in range(2):
                keys.append(f"stage_success.open_{i}")
                keys.append(f"stage_success.pick_{i}")
                keys.append(f"stage_success.place_{i}")
                keys.append(f"stage_success.close_{i}")
        else:
            keys = None
        collect_results(root_dir, keys=keys, pattern=args.pattern)


if __name__ == "__main__":
    main()
