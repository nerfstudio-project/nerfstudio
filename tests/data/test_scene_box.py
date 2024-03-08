import torch

from nerfstudio.data.scene_box import SceneBox


def test_scene_box_within():
    """Test if a set of points is within a scene box."""
    scene_box = SceneBox(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    )
    query_points = torch.tensor([[0.5, 0.5, 0.5], [2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]])
    is_within = scene_box.within(query_points)
    assert is_within[0]
    assert not is_within[1]
    assert not is_within[2]
