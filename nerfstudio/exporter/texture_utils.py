# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Texture utils.
"""

from __future__ import annotations

import math
from pathlib import Path

import mediapy as media
import torch
from rich.console import Console
from rich.progress import track
from torchtyping import TensorType

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.exporter.exporter_utils import Mesh
from nerfstudio.pipelines.base_pipeline import Pipeline

CONSOLE = Console(width=120)


def get_parallelogram_area(
    p: TensorType["bs":..., 2], v0: TensorType["bs":..., 2], v1: TensorType["bs":..., 2]
) -> TensorType["bs":...]:
    """Given three 2D points, return the area defined by the parallelogram. I.e., 2x the triangle area."""
    return (p[..., 0] - v0[..., 0]) * (v1[..., 1] - v0[..., 1]) - (p[..., 1] - v0[..., 1]) * (v1[..., 0] - v0[..., 0])


def export_textured_mesh(mesh: Mesh, pipeline: Pipeline, px_per_uv_triangle: int, output_dir: Path):
    """Textures a mesh using the radiance field from the Pipeline.
    The mesh is written to an OBJ file in the output directory,
    along with the corresponding material and texture files.
    Operations will occur on the same device as the Pipeline.

    Args:
        mesh: The mesh to texture.
        pipeline: The pipeline to use for texturing.
        px_per_uv_triangle: The number of pixels per side of UV triangle.
        output_dir: The directory to write the textured mesh to.
    """

    # pylint: disable=too-many-statements

    device = pipeline.device

    vertices = mesh.vertices.to(device)
    faces = mesh.faces.to(device)
    vertex_normals = mesh.normals.to(device)
    assert len(vertices) == len(vertex_normals), "Number of vertices and vertex normals must be equal"

    # calculate the number of rectangles needed
    triangle_padding = 3
    num_squares = math.ceil(len(faces) / 2)
    squares_per_side_w = math.ceil(math.sqrt(num_squares))
    squares_per_side_h = math.ceil(num_squares / squares_per_side_w)
    px_per_square_w = px_per_uv_triangle + triangle_padding
    px_per_square_h = px_per_uv_triangle
    num_pixels_w = squares_per_side_w * px_per_square_w
    num_pixels_h = squares_per_side_h * px_per_square_h

    # Construct what one square would look like
    # The height is equal to px_per_uv_triangle pixels.
    # The width is equal to px_per_uv_triangle + 3 pixels.
    # v0---------------v1------------------------v2
    # --Triangle 1---------------------------------
    # -----------------3px gap---------------------
    # --------------------------------Triangle 2---
    # v2-----------------------v1----------------v0

    lr_w = (px_per_uv_triangle + triangle_padding) / num_pixels_w
    lr_h = (px_per_uv_triangle) / num_pixels_h
    lr = torch.tensor([lr_w, lr_h], device=device)
    px_w = 1.0 / num_pixels_w
    px_h = 1.0 / num_pixels_h
    px = torch.tensor([px_w, px_h], device=device)
    edge_len_w = px_per_uv_triangle / num_pixels_w
    edge_len_h = px_per_uv_triangle / num_pixels_h
    scalar = (px_per_uv_triangle - 1) / px_per_uv_triangle
    # uv coords (upper left and lower right)
    uv_coords_upper_left = torch.tensor([[0, 0], [edge_len_w, 0], [0, edge_len_h]], device=device)
    # scale for bilinear interpolation reasons
    uv_coords_upper_left = uv_coords_upper_left * scalar + px / 2
    lower_right = [lr_w, lr_h]
    uv_coords_lower_right = torch.tensor(
        [
            lower_right,  # lower right
            [3 * px_w, lr_h],  # lower left
            [lr_w, 0],  # upper right
        ],
        device=device,
    )
    # scale for bilinear interpolation reasons
    uv_coords_lower_right = (
        (uv_coords_lower_right - torch.tensor(lower_right, device=device)) * scalar
        + torch.tensor(lower_right, device=device)
        - px / 2
    )

    # Tile this pattern across the entire texture
    uv_coords_square = torch.stack([uv_coords_upper_left, uv_coords_lower_right], dim=0)  # (2, 3, 2)
    uv_coords_square = uv_coords_square.reshape(1, 1, 6, 2)  # (6, 2)
    square_offsets = (
        torch.stack(
            torch.meshgrid(
                torch.arange(squares_per_side_w, device=device),
                torch.arange(squares_per_side_h, device=device),
                indexing="xy",
            ),
            dim=-1,
        )
        * lr
    )
    uv_coords_square = uv_coords_square + square_offsets.view(
        squares_per_side_h, squares_per_side_w, 1, 2
    )  # (num_squares_h, num_squares_w, 6, 2)
    texture_coordinates = uv_coords_square.view(-1, 3, 2)[: len(faces)]  # (num_faces, 3, 2)

    # Now find the triangle indices for every pixel and the barycentric coordinates
    # which can be used to interpolate the XYZ and normal values to then query with NeRF
    uv_indices = torch.stack(
        torch.meshgrid(
            torch.arange(num_pixels_w, device=device), torch.arange(num_pixels_h, device=device), indexing="xy"
        ),
        dim=-1,
    )
    linspace_h = torch.linspace(px_h / 2, 1 - px_h / 2, num_pixels_h, device=device)
    linspace_w = torch.linspace(px_w / 2, 1 - px_w / 2, num_pixels_w, device=device)
    uv_coords = torch.stack(
        torch.meshgrid(linspace_w, linspace_h, indexing="xy"), dim=-1
    )  # (num_pixels_h, num_pixels_w, 2)

    u_index = torch.div(uv_indices[..., 0], px_per_square_w, rounding_mode="floor")
    v_index = torch.div(uv_indices[..., 1], px_per_square_h, rounding_mode="floor")
    square_index = v_index * squares_per_side_w + u_index
    u_offset = uv_indices[..., 0] % px_per_square_w
    v_offset = uv_indices[..., 1] % px_per_square_h
    lower_right = (u_offset + v_offset) >= (px_per_square_w - 2)
    triangle_index = square_index * 2 + lower_right
    triangle_index = torch.clamp(triangle_index, min=0, max=len(faces) - 1)

    nearby_uv_coords = texture_coordinates[triangle_index]  # (num_pixels_h, num_pixels_w, 3, 2)
    nearby_vertices = vertices[faces[triangle_index]]  # (num_pixels_h, num_pixels_w, 3, 3)
    nearby_normals = vertex_normals[faces[triangle_index]]  # (num_pixels_h, num_pixels_w, 3, 3)

    # compute barycentric coordinates
    v0 = nearby_uv_coords[..., 0, :]  # (num_pixels, num_pixels, 2)
    v1 = nearby_uv_coords[..., 1, :]  # (num_pixels, num_pixels, 2)
    v2 = nearby_uv_coords[..., 2, :]  # (num_pixels, num_pixels, 2)
    p = uv_coords  # (num_pixels, num_pixels, 2)
    epsilon = 1e-8
    area = get_parallelogram_area(v2, v0, v1) + epsilon  # 2 x face area.
    w0 = get_parallelogram_area(p, v1, v2) / area
    w1 = get_parallelogram_area(p, v2, v0) / area
    w2 = get_parallelogram_area(p, v0, v1) / area

    # TODO: clean this up
    # compute the length of the rays we want to render
    # we can make a good guess for this by looking at the average distance between vertices on the triangle faces
    offset = torch.mean(torch.norm(nearby_vertices[..., 0, :] - nearby_vertices[..., 1, :], dim=-1)).float()

    origins = (
        nearby_vertices[..., 0, :] * w0[..., None]
        + nearby_vertices[..., 1, :] * w1[..., None]
        + nearby_vertices[..., 2, :] * w2[..., None]
    ).float()
    directions = -(
        nearby_normals[..., 0, :] * w0[..., None]
        + nearby_normals[..., 1, :] * w1[..., None]
        + nearby_normals[..., 2, :] * w2[..., None]
    ).float()
    # normalize the direction vector to make it a unit vector
    directions = torch.nn.functional.normalize(directions, dim=-1)

    origins = origins - offset * directions
    pixel_area = torch.ones_like(origins[..., 0:1])
    camera_indices = torch.zeros_like(origins[..., 0:1])
    nears = torch.zeros_like(origins[..., 0:1])
    fars = torch.ones_like(origins[..., 0:1]) * offset * 2.0
    camera_ray_bundle = RayBundle(
        origins=origins,
        directions=directions,
        pixel_area=pixel_area,
        camera_indices=camera_indices,
        nears=nears,
        fars=fars,
    )

    CONSOLE.print("Creating texture image by rendering with NeRF...")
    with torch.no_grad():
        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

    # save the texture image
    texture_image = outputs["rgb"].cpu().numpy()
    media.write_image(str(output_dir / "material_0.png"), texture_image)

    CONSOLE.print("Writing relevant OBJ information to files...")
    # create the .mtl file
    lines_mtl = [
        "# Generated with nerfstudio",
        "newmtl material_0",
        "Ka 1.000 1.000 1.000",
        "Kd 1.000 1.000 1.000",
        "Ks 0.000 0.000 0.000",
        "d 1.0",
        "illum 2",
        "Ns 1.00000000",
        "map_Kd material_0.png",
    ]
    lines_mtl = [line + "\n" for line in lines_mtl]
    file_mtl = open(output_dir / "material_0.mtl", "w", encoding="utf-8")  # pylint: disable=consider-using-with
    file_mtl.writelines(lines_mtl)
    file_mtl.close()

    # create the .obj file
    lines_obj = ["# Generated with nerfstudio", "mtllib material_0.mtl", "usemtl material_0"]
    lines_obj = [line + "\n" for line in lines_obj]
    file_obj = open(output_dir / "mesh.obj", "w", encoding="utf-8")  # pylint: disable=consider-using-with
    file_obj.writelines(lines_obj)

    # write the geometric vertices
    # move vertices back to cpu
    vertices = vertices.cpu().numpy()
    for i in track(range(len(vertices)), description="Writing vertices"):
        vertex = vertices[i]
        line = f"v {vertex[0]} {vertex[1]} {vertex[2]}\n"
        file_obj.write(line)

    # write the texture coordinates
    # move texture coordinates back to cpu
    texture_coordinates = texture_coordinates.cpu().numpy()
    for i in track(range(len(faces)), description="Writing texture coordinates"):
        for uv in texture_coordinates[i]:
            line = f"vt {uv[0]} {1.0 - uv[1]}\n"
            file_obj.write(line)

    # write the vertex normals
    # move vertex normals back to cpu
    vertex_normals = vertex_normals.cpu().numpy()
    for i in track(range(len(vertex_normals)), description="Writing vertex normals"):
        normal = vertex_normals[i]
        line = f"vn {normal[0]} {normal[1]} {normal[2]}\n"
        file_obj.write(line)

    # write the faces
    # move faces back to cpu
    faces = faces.cpu().numpy()
    for i in track(range(len(faces)), description="Writing faces"):
        face = faces[i]
        v1 = face[0] + 1
        v2 = face[1] + 1
        v3 = face[2] + 1
        vt1 = i * 3 + 1
        vt2 = i * 3 + 2
        vt3 = i * 3 + 3
        vn1 = v1
        vn2 = v2
        vn3 = v3
        line = f"f {v1}/{vt1}/{vn1} {v2}/{vt2}/{vn2} {v3}/{vt3}/{vn3}\n"
        file_obj.write(line)

    file_obj.close()
