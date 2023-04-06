from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn

from nerfstudio.models.base_model import Model
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.viewer.server.viewer_param import ViewerElement


def parse_object(
    obj,
    type_check,
    tree_stub,
):
    if not hasattr(obj, "__dict__"):
        return []
    ret = []
    for k in dir(obj):
        if k[0] == "_":
            continue
        v = getattr(obj, k)
        new_tree_stub = f"{tree_stub}/{k}"
        if isinstance(v, type_check):
            ret.append((new_tree_stub, v))
        elif isinstance(v, nn.Module):
            ret.extend(parse_object(v, type_check, new_tree_stub))
    return ret


def get_viewer_elements(pipeline: Pipeline) -> List[Tuple[str, ViewerElement]]:
    """
    Recursively parse the pipeline object and return a tree of all the ViewerElements contained

    returns a list of (path/to/object, param), which represents the path down the object tree
    """
    ret = parse_object(pipeline, ViewerElement, "pipeline")
    return ret
