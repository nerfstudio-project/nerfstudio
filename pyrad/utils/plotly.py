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
Visualization code for plotly.
"""

from typing import List, Union

import numpy as np
import plotly.graph_objects as go
import torch
from plotly import express as ex
from torchtyping import TensorType

from pyrad.cameras.rays import Frustums, RayBundle
from pyrad.utils.math import Gaussians


def color_str(color):
    """Plotly color string.

    Args:
        color: list [r, g, b] in [0, 1] range

    Returns:
        str: plotly-formatted color string
    """
    color = list((np.array(color) * 255.0).astype("int"))
    return f"""rgb({color[0]}, {color[1]}, {color[2]})"""


def get_line_segments_from_lines(
    lines: TensorType["num_rays", 2, 3],
    color: str = color_str((1, 0, 0)),
    marker_color: str = color_str((1, 0, 0)),
    colors: List[str] = None,
    draw_marker: bool = True,
    draw_line: bool = True,
    marker_size: float = 4,
    line_width: float = 10,
) -> go.Scatter3d:
    """Returns a list of Scatter3D objects for creating lines with plotly.
    # TODO(ethan): make this function more efficient instead of having a list of objects.

    Args:
        lines: Tensor of lines.
        color: Color of the lines. Defaults to red.
        marker_color: Color of the markers. Defaults to red.
        colors: List of colors for each line. Defaults to None.
        draw_marker: Whether to draw markers. Defaults to True.
        draw_line: Whether to draw lines. Defaults to True.
        marker_size: Size of the markers. Defaults to 4.
        line_width: Width of the lines. Defaults to 10.

    Returns:
        Scatter3D object on lines.
    """
    data = []
    for idx, line in enumerate(lines):
        thiscolor = color if draw_line else "rgba(0, 0, 0, 0)"
        if colors is not None:
            marker_color = colors[idx]
            thiscolor = colors[idx]
        data.append(
            go.Scatter3d(
                x=line[:, 0],
                y=line[:, 1],
                z=line[:, 2],
                showlegend=False,
                marker=dict(
                    size=marker_size,
                    color=marker_color,
                    colorscale="Viridis",
                )
                if draw_marker
                else dict(color="rgba(0, 0, 0, 0)"),
                line=dict(color=thiscolor, width=line_width),
            )
        )
    return data


def visualize_dataset(camera_origins: TensorType["num_cameras", 3], ray_bundle: RayBundle) -> go.FigureWidget:
    """Visualize a dataset with plotly using our cameras and generated rays.

    Args:
        camera_origins: Tensor of camera origins.
        ray_bundle: Ray bundle.

    Returns:
        plotly figure.
    """

    skip = 1
    size = 8
    assert len(ray_bundle) < 500, "Let's not break plotly by plotting too many rays!"

    data = []
    data += [
        go.Scatter3d(
            x=camera_origins[::skip, 0],
            y=camera_origins[::skip, 1],
            z=camera_origins[::skip, 2],
            mode="markers",
            name="camera origins",
            marker=dict(color="rgba(0, 0, 0, 1)", size=size),
        )
    ]

    length = 2.0
    lines = torch.stack(
        [ray_bundle.origins, ray_bundle.origins + ray_bundle.directions * length], dim=1
    )  # (num_rays, 2, 3)

    data += get_line_segments_from_lines(lines)

    layout = go.Layout(
        autosize=False,
        width=1000,
        height=1000,
        margin=go.layout.Margin(l=50, r=50, b=100, t=100, pad=4),
        scene=go.layout.Scene(
            aspectmode="data",
            camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.25, y=1.25, z=1.25)),
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    return fig


def get_random_color(colormap: List[str] = None, idx: int = None) -> str:
    """Get a random color from a colormap.

    Args:
        colormap: List of colors. Defaults to Plotly colors.
        idx: Index of color to return. Defaults to None.

    Returns:
        random color string
    """
    if colormap is None:
        colormap = ex.colors.qualitative.Plotly
    if idx is None:
        return colormap[np.random.randint(0, len(colormap))]
    return colormap[idx % len(colormap)]


def get_sphere(
    radius: float, center: TensorType[3] = None, color: str = "black", opacity: float = 1.0, resolution: int = 32
) -> go.Mesh3d:
    """Returns a sphere object for plotting with plotly.

    Args:
        radius: radius of sphere.
        center: center of sphere. Defaults to origin.
        color: color of sphere. Defaults to "black".
        opacity: opacity of sphere. Defaults to 1.0.
        resolution: resolution of sphere. Defaults to 32.

    Returns:
        sphere object.
    """
    phi = torch.linspace(0, 2 * torch.pi, resolution)
    theta = torch.linspace(-torch.pi / 2, torch.pi / 2, resolution)
    phi, theta = torch.meshgrid(phi, theta, indexing="ij")

    x = torch.cos(theta) * torch.sin(phi)
    y = torch.cos(theta) * torch.cos(phi)
    z = torch.sin(theta)
    pts = torch.stack((x, y, z), dim=-1)

    pts *= radius
    if center is not None:
        pts += center

    return go.Mesh3d(
        {
            "x": pts[:, :, 0].flatten(),
            "y": pts[:, :, 1].flatten(),
            "z": pts[:, :, 2].flatten(),
            "alphahull": 0,
            "opacity": opacity,
            "color": color,
        }
    )


def get_gaussian_ellipsiod(
    mean: TensorType[3],
    cov: TensorType[3, 3],
    n_std: int = 2,
    color="lightblue",
    opacity: float = 0.5,
    resolution: int = 20,
    name: str = "ellipse",
) -> go.Mesh3d:
    """Get a plotly ellipsoid for a Gaussian.

    Args:
        mean: mean of the Gaussian.
        cov: covariance of the Gaussian.
        n_std: Standard devation to visualize. Defaults to 2 (95% confidence).
        color: Color of the ellipsoid. Defaults to None.
        opacity: Opacity of the ellipsoid. Defaults to 0.5.
        resolution: Resolution of the ellipsoid. Defaults to 20.
        name: Name of the ellipsoid. Defaults to "ellipse".

    Returns:
        ellipsoid object.
    """

    phi = torch.linspace(0, 2 * torch.pi, resolution)
    theta = torch.linspace(-torch.pi / 2, torch.pi / 2, resolution)
    phi, theta = torch.meshgrid(phi, theta, indexing="ij")

    x = torch.cos(theta) * torch.sin(phi)
    y = torch.cos(theta) * torch.cos(phi)
    z = torch.sin(theta)
    pts = torch.stack((x, y, z), dim=-1)

    eigenvals, eigenvecs = torch.linalg.eigh(cov)
    idx = torch.sum(cov, dim=0).argsort()
    idx = eigenvals[idx].argsort()
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    scaling = torch.sqrt(eigenvals) * n_std
    pts = pts * scaling

    pts = pts @ eigenvecs.t()

    pts += mean

    return go.Mesh3d(
        {
            "x": pts[:, :, 0].flatten(),
            "y": pts[:, :, 1].flatten(),
            "z": pts[:, :, 2].flatten(),
            "alphahull": 0,
            "opacity": opacity,
            "color": color,
            "name": name,
        }
    )


def get_gaussian_ellipsoids_list(
    gaussians: Gaussians, opacity: float = 0.5, color: str = "random", resolution: int = 20
) -> List[Union[go.Mesh3d, go.Scatter3d]]:
    """Get a list of plotly meshes for frustums.

    Args:
        gaussians (Gaussians): Gaussians to visualize.
        opacity (float, optional): Opacity of the mesh. Defaults to 0.3.
        color (str, optional): Color of the mesh. Defaults to "random".
        resolution: Resolution of the mesh. Defaults to 20.

    Returns:
        List of plotly meshes
    """
    data = []

    vis_means = go.Scatter3d(
        x=gaussians.mean[:, 0],
        y=gaussians.mean[:, 1],
        z=gaussians.mean[:, 2],
        mode="markers",
        marker=dict(size=2, color="black"),
        name="Means",
    )
    data.append(vis_means)

    for i in range(gaussians.mean.shape[0]):
        if color == "random":
            c = get_random_color()
        else:
            c = color
        ellipse = get_gaussian_ellipsiod(
            gaussians.mean[i],
            cov=gaussians.cov[i],
            color=c,
            opacity=opacity,
            resolution=resolution,
        )
        data.append(ellipse)

    return data


def get_frustum_mesh(
    frustum: Frustums, opacity: float = 0.3, color: str = "#DC203C", resolution: int = 20
) -> go.Mesh3d:
    """Get a plotly mesh for a single frustum.

    Args:
        frustum: Single frustum
        opacity: Opacity of the mesh. Defaults to 0.3.
        color: Color of the mesh. Defaults to "#DC203C".
        resolution: Resolution of the mesh. Defaults to 20.

    Returns:
        Plotly mesh
    """

    if frustum.ndim > 1:
        raise ValueError("Frustum must be a single Frustum object.")

    base_radius = torch.sqrt(frustum.pixel_area / torch.pi)
    f_radius = frustum.starts * base_radius
    b_radius = frustum.ends * base_radius

    x = torch.cat([torch.ones(resolution) * frustum.starts, torch.ones(resolution) * frustum.ends])
    pts = torch.linspace(0, 2 * torch.pi, resolution)

    y = torch.sin(pts)
    z = torch.cos(pts)
    y = torch.cat([y * f_radius, y * b_radius])
    z = torch.cat([z * f_radius, z * b_radius])

    pts = torch.stack([x, y, z], dim=-1)

    forward = frustum.directions
    up = torch.nn.functional.normalize(torch.cross(forward, torch.tensor([0.0, 0, 1])), dim=-1)
    right = torch.nn.functional.normalize(torch.cross(forward, up), dim=-1)
    rotation = torch.stack([forward, up, right], dim=1)
    pts = torch.einsum("kj,ij->ki", pts, rotation)

    pts += frustum.origins
    return go.Mesh3d(
        x=pts[..., 0],
        y=pts[..., 1],
        z=pts[..., 2],
        opacity=opacity,
        alphahull=0,
        color=color,
        flatshading=True,
        name="Frustums",
    )


def get_frustums_mesh_list(
    frustums: Frustums, opacity: float = 1.0, color: str = "random", resolution: int = 20
) -> List[go.Mesh3d]:
    """Get a list of plotly meshes for a list of frustums.

    Args:
        frustums: List of frustums
        opacity: Opacity of the mesh. Defaults to 0.3.
        color: Color of the mesh. Defaults to "random".
        resolution: Resolution of the mesh. Defaults to 20.

    Returns:
        List of plotly meshes
    """
    data = []
    for i, frustum in enumerate(frustums.flatten()):
        if color == "random":
            c = get_random_color(idx=i)
        else:
            c = color
        data.append(get_frustum_mesh(frustum, opacity=opacity, color=c, resolution=resolution))
    return data


def get_frustum_points(
    frustum: Frustums, opacity: float = 1.0, color: str = "forestgreen", size: float = 5
) -> go.Scatter3d:
    """Get a set plotly points for frustums centers.

    Args:
        frustum: Frustums to visualize.
        opacity: Opacity of the points. Defaults to 0.3.
        color: Color of the poinst. Defaults to "forestgreen".
        size: Size of points. Defaults to 10.

    Returns:
        Plotly points
    """

    frustum = frustum.flatten()
    pts = frustum.get_positions()

    return go.Scatter3d(
        x=pts[..., 0],
        y=pts[..., 1],
        z=pts[..., 2],
        mode="markers",
        marker=dict(
            size=size,
            color=color,
            opacity=opacity,
        ),
        name="Frustums -> Positions",
    )


def get_ray_bundle_lines(
    ray_bundle: RayBundle, length: float = 1.0, color: str = "#DC203C", width: float = 1
) -> go.Scatter3d:
    """Get a plotly line for a ray bundle.

    Args:
        ray_bundle: Ray bundle
        length: Length of the line. Defaults to 1.0.
        color: Color of the line.
        width: Width of the line. Defaults to 1.

    Returns:
        Plotly lines
    """

    origins = ray_bundle.origins.view(-1, 3)
    directions = ray_bundle.directions.view(-1, 3)

    lines = torch.empty((origins.shape[0] * 2, 3))
    lines[0::2] = origins
    lines[1::2] = origins + directions * length
    return go.Scatter3d(
        x=lines[..., 0],
        y=lines[..., 1],
        z=lines[..., 2],
        mode="lines",
        name="Ray Bundle",
        line=dict(color=color, width=width),
    )
