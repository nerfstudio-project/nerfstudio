import torch
from einops import rearrange

from pyrad.cuda_pl.backend import _C as vren

from .custom_functions import RayAABBIntersector, RayMarcher, VolumeRenderer

MAX_SAMPLES = 1024  # fixed!


def render(model, rays, **kwargs):
    """
    Render rays by
    1. Compute the intersection of the rays with the scene bounding box
    2. Follow the process in @render_func (different for train/test)

    Inputs:
        model: NGP
        rays: (N_rays, 3+3), ray origins and directions

    Outputs:
        result: dictionary containing final rgb and depth
    """

    rays_o, rays_d = rays[:, 0:3].contiguous(), rays[:, 3:6].contiguous()
    _, hits_t, _ = RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)

    if kwargs.get("test_time", False):
        render_func = __render_rays_test
    else:
        render_func = __render_rays_train

    results = render_func(model, rays_o, rays_d, hits_t, **kwargs)
    for k, v in results.items():
        results[k] = v.cpu() if kwargs.get("to_cpu", False) else v
    return results


@torch.no_grad()
def __render_rays_test(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by

    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples
           and evaluate the properties (sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    results = {}

    # output tensors to be filled in
    N_rays = len(rays_o)
    device = rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)

    samples = 0
    alive_indices = torch.arange(N_rays, device=device)

    while samples < MAX_SAMPLES:
        N_alive = len(alive_indices)
        if N_alive == 0:
            break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays // N_alive, 64), 1)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = vren.raymarching_test(
            rays_o,
            rays_d,
            hits_t[:, 0],
            alive_indices,
            model.density_bitfield,
            model.scale,
            kwargs.get("exp_step_factor", 0.0),
            model.grid_size,
            MAX_SAMPLES,
            N_samples,
        )
        xyzs = rearrange(xyzs, "n1 n2 c -> (n1 n2) c")
        dirs = rearrange(dirs, "n1 n2 c -> (n1 n2) c")
        valid_mask = ~torch.all(dirs == 0, dim=1)
        if valid_mask.sum() == 0:
            break

        sigmas = torch.zeros(len(xyzs), device=device)
        rgbs = torch.zeros(len(xyzs), 3, device=device)
        _sigmas, _rgbs = model(xyzs[valid_mask], dirs[valid_mask])
        sigmas[valid_mask], rgbs[valid_mask] = _sigmas.float(), _rgbs.float()
        sigmas = rearrange(sigmas, "(n1 n2) -> n1 n2", n2=N_samples)
        rgbs = rearrange(rgbs, "(n1 n2) c -> n1 n2 c", n2=N_samples)

        vren.composite_test_fw(
            sigmas,
            rgbs,
            deltas,
            ts,
            hits_t[:, 0],
            alive_indices,
            kwargs.get("T_threshold", 1e-4),
            N_eff_samples,
            opacity,
            depth,
            rgb,
        )
        alive_indices = alive_indices[alive_indices >= 0]  # remove converged rays

    rgb_bg = torch.ones(3, device=device)  # TODO: infer env map from network
    results["opacity"] = opacity
    results["depth"] = depth
    results["rgb"] = rgb + rgb_bg * rearrange(1 - opacity, "n -> n 1")

    return results


@torch.cuda.amp.autocast()
def __render_rays_train(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by
    1. March the rays along their directions, querying @density_bitfield
       to skip empty space, and get the effective sample points (where
       there is object)
    2. Infer the NN at these positions and view directions to get properties
       (currently sigmas and rgbs)
    3. Use volume rendering to combine the result (front to back compositing
       and early stop the ray if its transmittance is below a threshold)
    """
    results = {}

    rays_a, xyzs, dirs, deltas, ts = RayMarcher.apply(
        rays_o,
        rays_d,
        hits_t[:, 0],
        model.density_bitfield,
        model.scale,
        kwargs.get("exp_step_factor", 0.0),
        model.grid_size,
        MAX_SAMPLES,
    )

    sigmas, rgbs = model(xyzs, dirs)

    rgb_bg = torch.ones(3, device=rays_o.device)  # TODO: infer env map from network
    results["opacity"], results["depth"], rgb = VolumeRenderer.apply(
        sigmas, rgbs.contiguous(), deltas, ts, rays_a, kwargs.get("T_threshold", 1e-4)
    )

    results["rgb"] = rgb + rgb_bg * rearrange(1 - results["opacity"], "n -> n 1")

    return results
