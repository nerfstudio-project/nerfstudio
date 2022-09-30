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
import pickle
from pathlib import Path
from typing import Any


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
