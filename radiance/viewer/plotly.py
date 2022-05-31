"""
Visualization code for plotly.
"""

import numpy as np
import plotly.graph_objects as go
import torch

from radiance.structures.rays import RayBundle


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
    lines,
    color=color_str((1, 0, 0)),
    marker_color=color_str((1, 0, 0)),
    colors=None,
    draw_marker=True,
    draw_line=True,
    marker_size=4,
    line_width=10,
):
    """Returns a list of Scatter3D objects for creating lines with plotly.
    # TODO(ethan): make this function more efficient instead of having a list of objects.

    Args:
        lines (_type_): list of [np.array(2, 3), ...], line is defined by a (2, 3) array
        color (_type_, optional): _description_. Defaults to color_str((1, 0, 0)).
        marker_color (_type_, optional): _description_. Defaults to color_str((1, 0, 0)).
        colors (_type_, optional): _description_. Defaults to None.
        draw_marker (bool, optional): _description_. Defaults to True.
        draw_line (bool, optional): _description_. Defaults to True.
        marker_size (int, optional): _description_. Defaults to 4.
        line_width (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
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


def visualize_dataset(camera_origins, ray_bundle: RayBundle):
    """Visualize a dataset with plotly using our cameras and generated rays."""

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
