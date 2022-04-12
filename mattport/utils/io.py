"""
Input/output utils.
"""

import json
import os


def load_from_json(filename: str):
    """_summary_

    Args:
        filename (str): _description_

    Returns:
        _type_: _description_
    """
    assert filename.endswith(".json")
    with open(filename, "r", encoding="UTF-8") as file:
        return json.load(file)


def write_to_json(filename: str, content: dict):
    """_summary_

    Args:
        filename (str): _description_
        content (dict): _description_
    """
    assert filename.endswith(".json")
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(content, file)


def get_git_root(path, dirs=(".git",), default=None):
    """_summary_
    "https://stackoverflow.com/questions/22081209/find-the-root-of-the-git-repository-where-the-file-lives

    Args:
        path (_type_): _description_
        dirs (tuple, optional): _description_. Defaults to (".git",).
        default (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    prev, test = None, os.path.abspath(path)
    while prev != test:
        if any(os.path.isdir(os.path.join(test, directory)) for directory in dirs):
            return test
        prev, test = test, os.path.abspath(os.path.join(test, os.pardir))
    return default


def get_absolute_path(path, proj_root_func=get_git_root):
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


def make_dir(filename_or_folder):
    """Make the directory for either the filename or folder.
    Note that filename_or_folder currently needs to end in / for it to be recognized as a folder.
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
