from omegaconf import OmegaConf
from types import SimpleNamespace
from typing import Any, Dict, Union, Tuple
import os

def dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    """
    Converts a dictionary to a SimpleNamespace object for easier attribute access.

    Args:
        d (Dict[str, Any]): Dictionary to convert.

    Returns:
        SimpleNamespace: The converted dictionary as a namespace.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)

def add_root_paths(d):
    root_dir = os.getenv('ROOT_DIR')
    if root_dir is None:
        raise EnvironmentError("The ROOT_DIR environment variable is not set. Please set this variable to the path of your root directory. Instructions can be found in the README.md file.")
    if 'paths' in d:
        for k, v in d['paths'].items():
            if isinstance(v, str):
                d['paths'][k] = os.path.join(root_dir, v.lstrip('/'))
            elif isinstance(v, dict):
                update_paths(v)
    return d

def get_config_dir() -> str:
    """
    Determines the configuration base directory based on whether the code is running inside a Docker container.

    Returns:
        str: The path to the configuration directory.

    Raises:
        EnvironmentError: If the ROOT_DIR environment variable is not set.
    """
#    root_dir = os.getenv('ROOT_DIR')
    return os.getcwd()
#    if root_dir is None:
#        raise EnvironmentError("The ROOT_DIR environment variable is not set. Please set this variable to the path of your root directory. Instructions can be found in the README.md file.")
#    return os.path.join(root_dir, 'config')
