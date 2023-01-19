"""
Test if temporal grid run properly (forward and backward)
"""
import torch

from nerfstudio.field_components import temporal_grid


def test_temporal_grid():
    """Test temporal grid"""
    if not torch.cuda.is_available():
        print("Unable to test temporal grid without GPU, since CUDA kernels involved.")
        return

    params = dict(
        temporal_dim=2,
        input_dim=1,
        num_levels=1,
        level_dim=1,
        per_level_scale=1,
        base_resolution=1,
        log2_hashmap_size=2,
        desired_resolution=None,
        gridtype="tiled",
        align_corners=False,
    )
    model = temporal_grid.TemporalGridEncoder(**params).cuda()
    random_embedding = torch.rand_like(model.embeddings)
    random_embedding[:, 0] = torch.arange(8).to(model.embeddings)
    model.embeddings = torch.nn.Parameter(random_embedding, requires_grad=True)

    x = torch.zeros([1024, 1]).cuda()
    t = torch.zeros([1024, 1]).cuda()
    out = model(x, t)
    weight = torch.randn_like(out)
    (out * weight).sum().backward()
    assert torch.all(out == 0.5)
    assert model.embeddings.grad.sum() - weight.sum() < 0.01
    assert torch.all(model.embeddings.grad[2:, :] == 0)
    assert torch.all(model.embeddings.grad[:, 1:] == 0)
    model.get_temporal_tv_loss()
    # TODO: Any better way to numerically test it? Especially for the gradients.
    #       (maybe add some randomness and multiple cases for the testing?)


if __name__ == "__main__":
    test_temporal_grid()
