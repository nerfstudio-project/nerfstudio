# Copyright 2022 The Plenoptix Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Input/output utils.
"""
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Tuple


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def write_to_json(filename: Path, content: dict):
    """Write data to a JSON file.

    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.suffix == ".json"
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(content, file)


def load_from_pkl(filename: Path):
    """Load from a pickle file.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".pkl"
    with open(filename, "rb") as f:
        return pickle.load(f)


def write_to_pkl(filename: Path, content: Any):
    """Write to a pickle file.

    Args:
        filename (str): The filename to write to.
        content (Any): The data to write.
    """
    assert filename.suffix == ".pkl"
    with open(filename, "wb") as f:
        pickle.dump(content, f)


def get_git_root(path: str, dirs: Tuple[str] = (".git",), default: str = "") -> str:
    """Find the root of the git repository.
    'https://stackoverflow.com/questions/22081209/find-the-root-of-the-git-repository-where-the-file-lives'

    Args:
        path: The path to start from.
        dirs: The directories to look for. Defaults to (".git",).
        default: The default value to return if no git root is found.

    Returns:
        The root of the git repository. Returns None if not found.
    """
    prev, test = None, os.path.abspath(path)
    while prev != test:
        if any(os.path.isdir(os.path.join(test, directory)) for directory in dirs):
            return test
        prev, test = test, os.path.abspath(os.path.join(test, os.pardir))
    logging.info("Couldn't find git root")
    return default


def get_project_root(path: str) -> Path:
    """Return the project root directory from an environment variable.
    # TODO: handle this better and report user error

    Args:
        path: The path to start from.

    Returns:
        The project root directory.
    """
    project_root = os.getenv("PROJECT_ROOT")
    if project_root is None:
        logging.info(
            "Please set PROJECT_ROOT to the root directory of this repo. "
            "Going to try calling get_git_root(path) instead."
        )
        project_root = get_git_root(path)
    return Path(project_root)


def get_absolute_path(path, proj_root_func=get_project_root) -> Path:
    """
    Returns the full, absolute path.
    Relative paths are assumed to start at the repo directory.
    """
    str_absolute_path = str(path)
    if str_absolute_path == "" or str_absolute_path[0] == "/":
        return path
    absolute_path = proj_root_func(str_absolute_path) / path
    return absolute_path
