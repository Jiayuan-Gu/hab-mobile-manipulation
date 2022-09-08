import os

from habitat.config import Config as HabitatConfig
from yacs.config import CfgNode, _assert_with_logging, _load_module_from_file


class Config(HabitatConfig):
    @classmethod
    def _load_cfg_py_source(cls, filename):
        """Load a config from a Python source file."""
        module = _load_module_from_file("yacs.config.override", filename)
        _assert_with_logging(
            hasattr(module, "cfg"),
            "Python module from file {} must have 'cfg' attr".format(filename),
        )
        # NOTE(jigu): allow subclasses of valid types
        VALID_ATTR_TYPES = (dict, CfgNode)
        _assert_with_logging(
            isinstance(module.cfg, VALID_ATTR_TYPES),
            "Imported module 'cfg' attr must be in {} but is {} instead".format(
                VALID_ATTR_TYPES, type(module.cfg)
            ),
        )
        return cls(module.cfg)


def load_config(config_path: str):
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = Config.load_cfg(f)
        print(f"Loaded {config_path}")
    if "__BASE__" in config:
        base_config_path: str = config.pop("__BASE__")
        base_config_path = base_config_path.format(
            fileDirname=os.path.dirname(config_path)
        )
        print(f"Merging {base_config_path} into {config_path}")
        base_config: Config = load_config(base_config_path)
        base_config.merge_from_other_cfg(config)
        config = base_config
    return config


def get_config(config_path, opts=None) -> Config:
    config = load_config(config_path)

    # Load task config into `TASK_CONFIG` from `BASE_TASK_CONFIG_PATH`
    task_config = load_config(config.BASE_TASK_CONFIG_PATH)
    task_config.merge_from_other_cfg(config.TASK_CONFIG)
    config.TASK_CONFIG = task_config

    if opts:
        config.CMD_TRAILING_OPTS = config.CMD_TRAILING_OPTS + opts
        config.merge_from_list(config.CMD_TRAILING_OPTS)

    config.freeze()
    return config
