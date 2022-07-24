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
from typing import Any, Tuple, Optional


def load_from_json(filename: str):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.endswith(".json")
    with open(filename, "r", encoding="UTF-8") as file:
        return json.load(file)


def write_to_json(filename: str, content: dict):
    """Write data to a JSON file.

    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.endswith(".json")
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(content, file)


def load_from_pkl(filename: str):
    """Load from a pickle file.

    Args:
        filename: The filename to load from.
    """
    assert filename.endswith(".pkl")
    with open(filename, "rb") as f:
        return pickle.load(f)


def write_to_pkl(filename: str, content: Any):
    """Write to a pickle file.

    Args:
        filename (str): The filename to write to.
        content (Any): The data to write.
    """
    assert filename.endswith(".pkl")
    with open(filename, "wb") as f:
        pickle.dump(content, f)


def get_git_root(path: str, dirs: Tuple[str] = (".git",), default=None) -> Optional[str]:
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
    return default


def get_project_root(path: str):
    """Return the project root directory from an environment variable.
    # TODO: handle this better and report user error

    Args:
        path: The path to start from.
    """
    project_root = os.getenv("PROJECT_ROOT")
    if project_root is None:
        logging.info(
            (
                "Please set PROJECT_ROOT to the root directory of this repo. "
                "Going to try calling get_git_root(path) instead."
            )
        )
        project_root = get_git_root(path)
    return project_root


def get_absolute_path(path, proj_root_func=get_project_root):
    """
    Returns the full, absolute path.
    Relative paths are assumed to start at the repo directory.
    """
    if path == "":
        return ""
    absolute_path = path
    if absolute_path[0] != "/":
        absolute_path = os.path.join(proj_root_func(path), absolute_path)
    return absolute_path


def make_dir(filename_or_folder: str) -> str:
    """Make the directory for either the filename or folder.
    Note that filename_or_folder currently needs to end in / for it to be recognized as a folder.

    Args:
        filename_or_folder (str): The filename or folder to make.
    """
    if filename_or_folder[-1] != "/" and filename_or_folder.find(".") < 0:
        folder = os.path.dirname(filename_or_folder + "/")
    else:
        folder = os.path.dirname(filename_or_folder)
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except Exception as e:  # pylint: disable=broad-except
            print(f"Couldn't create folder: {folder}. Maybe due to a parallel process?")
            print(e)
    return filename_or_folder
