import json
from pathlib import Path

import numpy as np

from nerfstudio.process_data.record3d_utils import record3d_to_json


def test_record3d_preserves_distinct_focal_lengths(tmp_path: Path):
    intrinsics = np.array([[500, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=float)
    metadata = {"K": intrinsics.T.flatten().tolist(), "h": 480, "w": 640, "poses": [[0, 0, 0, 1, 0, 0, 0]]}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    record3d_to_json([Path("frame.png")], path, tmp_path, np.array([0]), None, None)

    output = json.loads((tmp_path / "transforms.json").read_text())
    assert (output["fl_x"], output["fl_y"]) == (500, 525)
