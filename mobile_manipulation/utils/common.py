import os
import socket
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import git
import numpy as np
from gym import spaces
from gym.utils.colorize import colorize
from habitat_baselines.utils.common import get_checkpoint_id


def get_run_name():
    """Get a runtime name: {timestamp}.{hostname}"""
    timestamp = time.strftime("%m-%d_%H-%M-%S")
    hostname = socket.gethostname()
    run_name = "{:s}.{:s}".format(timestamp, hostname)
    return run_name


def get_latest_checkpoint(ckpt_folder: str, filter: bool):
    """Get the latest checkpoint.

    Args:
        ckpt_folder: folder storing checkpoints.
        filter: whether to ignore checkpoints without checkpoint id.

    Returns:
        str or None: the latest checkpoint path if found.
    """
    ckpt_folder = Path(ckpt_folder)
    assert ckpt_folder.is_dir(), f"{ckpt_folder} is not a folder"

    ckpt_paths = []
    for ckpt_path in ckpt_folder.glob("*.pth"):
        ckpt_id = get_checkpoint_id(ckpt_path)
        if filter and ckpt_id is None:
            continue
        ckpt_paths.append(ckpt_path)

    if len(ckpt_paths) == 0:
        return None
    else:
        ckpt_paths.sort(key=os.path.getmtime)
        return ckpt_paths[-1]


def extract_scalars_from_info(
    info: Dict[str, Any],
    blacklist=(),
) -> Dict[str, float]:
    result = {}
    for k, v in info.items():
        if k in blacklist:
            continue

        if isinstance(v, dict):
            sub_result = extract_scalars_from_info(v, blacklist=blacklist)
            for subk, subv in sub_result.items():
                fullk = f"{k}.{subk}"
                if fullk not in blacklist:
                    result[fullk] = subv
        elif v is None:
            continue
        # Things that are scalar-like will have an np.size of 1.
        # Strings also have an np.size of 1, so explicitly ban those
        elif np.size(v) == 1 and not isinstance(v, str):
            result[k] = float(v)

    return result


class Timer:
    def __init__(self) -> None:
        self.t_starts = dict()
        self.t_ends = dict()
        self.elapsed_times = defaultdict(float)

    def start(self, key):
        self.t_starts[key] = time.time()
        return self.t_starts[key]

    def stop(self, key, add=False):
        self.t_ends[key] = time.time()
        dt = self.t_ends[key] - self.t_starts[key]
        if add:
            self.elapsed_times[key] += dt
        return dt

    @contextmanager
    def timeit(self, key=None):
        try:
            yield self.start(key)
        finally:
            self.elapsed_times[key] += self.stop(key)


def warn(string):
    print(colorize(string, "red", bold=True, highlight=True))


def get_git_commit_id(path="."):
    try:
        repo = git.Repo(path, search_parent_directories=True)
        # https://github.com/gitpython-developers/GitPython/issues/718#issuecomment-360267779
        repo.__del__()
        return repo.head.commit
    except git.InvalidGitRepositoryError as err:
        return None


def get_state_dict_by_prefix(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def get_flat_space_names(dict_space: spaces.Dict):
    return [
        k
        for k, v in dict_space.spaces.items()
        if isinstance(v, spaces.Box) and len(v.shape) == 1
    ]
