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

# pylint: disable=no-member

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, Union

import mediapy as media
import numpy as np
import torch
import xatlas
from rich.console import Console
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.exporter.exporter_utils import Mesh
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.rich_utils import get_progress

CONSOLE = Console(width=120)

TORCH_DEVICE = Union[torch.device, str]  # pylint: disable=invalid-name


def get_parallelogram_area(
    p: TensorType["bs":..., 2], v0: TensorType["bs":..., 2], v1: TensorType["bs":..., 2]
) -> TensorType["bs":...]:
    """Given three 2D points, return the area defined by the parallelogram. I.e., 2x the triangle area.

    Args:
        p: The origin of the parallelogram.
        v0: The first vector of the parallelogram.
        v1: The second vector of the parallelogram.

    Returns:
        The area of the parallelogram.
    """
    return (p[..., 0] - v0[..., 0]) * (v1[..., 1] - v0[..., 1]) - (p[..., 1] - v0[..., 1]) * (v1[..., 0] - v0[..., 0])


def get_texture_image(
    num_pixels_w: int, num_pixels_h: int, device: TORCH_DEVICE
) -> Tuple[TensorType["num_pixels_h", "num_pixels_w", 2], TensorType["num_pixels_h", "num_pixels_w", 2]]:
    """Get a texture image."""
    px_w = 1.0 / num_pixels_w
    px_h = 1.0 / num_pixels_h
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

    return uv_coords, uv_indices


def unwrap_mesh_per_uv_triangle(
    vertices: TensorType["num_verts", 3],
    faces: TensorType["num_faces", 3],
    vertex_normals: TensorType["num_verts", 3],
    px_per_uv_triangle: int,
) -> Tuple[
    TensorType["num_faces", 3, 2],
    TensorType["num_pixels", "num_pixels", 3],
    TensorType["num_pixels", "num_pixels", "num_pixels"],
]:
    """Unwrap a mesh to a UV texture. This is done by making a grid of rectangles in the UV texture map
    and then having two triangles per rectangle. Then the texture image is rasterized and uses barycentric
    interpolation to get the origins and directions, per pixel, that are needed to render the NeRF with.

    Args:
        vertices: The vertices of the mesh.
        faces: The faces of the mesh.
        vertex_normals: The vertex normals of the mesh.
        px_per_uv_triangle: The number of pixels per UV triangle.
    """

    # pylint: disable=too-many-statements

    assert len(vertices) == len(vertex_normals), "Number of vertices and vertex normals must be equal"
    device = vertices.device

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
    uv_coords, uv_indices = get_texture_image(num_pixels_w, num_pixels_h, device)

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
    area = get_parallelogram_area(v2, v0, v1)  # 2x face area.
    w0 = get_parallelogram_area(p, v1, v2) / area
    w1 = get_parallelogram_area(p, v2, v0) / area
    w2 = get_parallelogram_area(p, v0, v1) / area

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

    return texture_coordinates, origins, directions


def unwrap_mesh_with_xatlas(
    vertices: TensorType["num_verts", 3],
    faces: TensorType["num_faces", 3, torch.long],
    vertex_normals: TensorType["num_verts", 3],
    num_pixels_per_side=1024,
    num_faces_per_barycentric_chunk=10,
) -> Tuple[
    TensorType["num_faces", 3, 2],
    TensorType["num_pixels", "num_pixels", 3],
    TensorType["num_pixels", "num_pixels", "num_pixels"],
]:
    """Unwrap a mesh using xatlas. We use xatlas to unwrap the mesh with UV coordinates.
    Then we rasterize the mesh with a square pattern. We interpolate the XYZ and normal
    values for every pixel in the texture image. We return the texture coordinates, the
    origins, and the directions for every pixel.

    Args:
        vertices: Tensor of mesh vertices.
        faces: Tensor of mesh faces.
        vertex_normals: Tensor of mesh vertex normals.
        num_pixels_per_side: Number of pixels per side of the texture image. We use a square.
        num_faces_per_barycentric_chunk: Number of faces to use for barycentric chunk computation.

    Returns:
        texture_coordinates: Tensor of texture coordinates for every face.
        origins: Tensor of origins for every pixel.
        directions: Tensor of directions for every pixel.
    """

    # pylint: disable=unused-variable
    # pylint: disable=too-many-statements

    device = vertices.device

    # unwrap the mesh
    vertices_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy()
    vertex_normals_np = vertex_normals.cpu().cpu().numpy()
    vmapping, indices, uvs = xatlas.parametrize(  # pylint: disable=c-extension-no-member
        vertices_np, faces_np, vertex_normals_np
    )

    # vertices texture coordinates
    vertices_tc = torch.from_numpy(uvs.astype(np.float32)).to(device)

    # render uv maps
    vertices_tc = vertices_tc * 2.0 - 1.0  # uvs to range [-1, 1]
    vertices_tc = torch.cat(
        (vertices_tc, torch.zeros_like(vertices_tc[..., :1]), torch.ones_like(vertices_tc[..., :1])), dim=-1
    )  # [num_verts, 4]

    texture_coordinates = torch.from_numpy(uvs[indices]).to(device)  # (num_faces, 3, 2)

    # Now find the triangle indices for every pixel and the barycentric coordinates
    # which can be used to interpolate the XYZ and normal values to then query with NeRF
    uv_coords, _ = get_texture_image(num_pixels_per_side, num_pixels_per_side, device)
    uv_coords_shape = uv_coords.shape
    p = uv_coords.reshape(1, -1, 2)  # (1, N, 2)
    num_vertices = p.shape[1]
    num_faces = texture_coordinates.shape[0]
    triangle_distances = torch.ones_like(p[..., 0]) * torch.finfo(torch.float32).max  # (1, N)
    triangle_indices = torch.zeros_like(p[..., 0]).long()  # (1, N)
    triangle_w0 = torch.zeros_like(p[..., 0])  # (1, N)
    triangle_w1 = torch.zeros_like(p[..., 0])  # (1, N)
    triangle_w2 = torch.zeros_like(p[..., 0])  # (1, N)
    arange_list = torch.arange(num_vertices, device=device)
    progress = get_progress("Chunking faces for rasterization")
    with progress:
        for i in progress.track(range(num_faces // num_faces_per_barycentric_chunk)):
            s = i * num_faces_per_barycentric_chunk
            e = min((i + 1) * num_faces_per_barycentric_chunk, num_faces)
            v0 = texture_coordinates[s:e, 0:1, :]  # (F, 1, 2)
            v1 = texture_coordinates[s:e, 1:2, :]  # (F, 1, 2)
            v2 = texture_coordinates[s:e, 2:3, :]  # (F, 1, 2)
            # NOTE: could try clockwise vs counter clockwise
            area = get_parallelogram_area(v2, v0, v1)  # 2x face area.
            w0 = get_parallelogram_area(p, v1, v2) / area  # (num_faces_per_barycentric_chunk, N)
            w1 = get_parallelogram_area(p, v2, v0) / area
            w2 = get_parallelogram_area(p, v0, v1) / area
            # get distance from center of triangle
            dist_to_center = torch.abs(w0) + torch.abs(w1) + torch.abs(w2)
            d_values, d_indices = torch.min(dist_to_center, dim=0, keepdim=True)
            d_indices_with_offset = d_indices + s  # add offset
            condition = d_values < triangle_distances
            triangle_distances = torch.where(condition, d_values, triangle_distances)
            triangle_indices = torch.where(condition, d_indices_with_offset, triangle_indices)
            w0_selected = w0[d_indices[0], arange_list].unsqueeze(0)  # (1, N)
            w1_selected = w1[d_indices[0], arange_list].unsqueeze(0)  # (1, N)
            w2_selected = w2[d_indices[0], arange_list].unsqueeze(0)  # (1, N)
            triangle_w0 = torch.where(condition, w0_selected, triangle_w0)
            triangle_w1 = torch.where(condition, w1_selected, triangle_w1)
            triangle_w2 = torch.where(condition, w2_selected, triangle_w2)

    nearby_vertices = vertices[faces[triangle_indices[0]]]  # (N, 3, 3)
    nearby_normals = vertex_normals[faces[triangle_indices[0]]]  # (N, 3, 3)

    origins = (
        nearby_vertices[..., 0, :] * triangle_w0[0, :, None]
        + nearby_vertices[..., 1, :] * triangle_w1[0, :, None]
        + nearby_vertices[..., 2, :] * triangle_w2[0, :, None]
    ).float()
    directions = -(
        nearby_normals[..., 0, :] * triangle_w0[0, :, None]
        + nearby_normals[..., 1, :] * triangle_w1[0, :, None]
        + nearby_normals[..., 2, :] * triangle_w2[0, :, None]
    ).float()

    origins = origins.reshape(uv_coords_shape[0], uv_coords_shape[1], 3)
    directions = directions.reshape(uv_coords_shape[0], uv_coords_shape[1], 3)

    # normalize the direction vector to make it a unit vector
    directions = torch.nn.functional.normalize(directions, dim=-1)

    return texture_coordinates, origins, directions


def export_textured_mesh(
    mesh: Mesh,
    pipeline: Pipeline,
    output_dir: Path,
    px_per_uv_triangle: Optional[int] = None,
    unwrap_method: Literal["xatlas", "custom"] = "xatlas",
    raylen_method: Literal["edge", "none"] = "edge",
    num_pixels_per_side=1024,
) -> None:
    """Textures a mesh using the radiance field from the Pipeline.
    The mesh is written to an OBJ file in the output directory,
    along with the corresponding material and texture files.
    Operations will occur on the same device as the Pipeline.

    Args:
        mesh: The mesh to texture.
        pipeline: The pipeline to use for texturing.
        output_dir: The directory to write the textured mesh to.
        px_per_uv_triangle: The number of pixels per side of UV triangle.
        unwrap_method: The method to use for unwrapping the mesh.
        offset_method: The method to use for computing the ray length to render.
        num_pixels_per_side: The number of pixels per side of the texture image.
    """

    # pylint: disable=too-many-statements

    device = pipeline.device

    vertices = mesh.vertices.to(device)
    faces = mesh.faces.to(device)
    vertex_normals = mesh.normals.to(device)

    summary_log = []
    summary_log.append(f"Unwrapped mesh using {unwrap_method} method.")
    summary_log.append(f"Mesh has {len(vertices)} vertices and {len(faces)} faces.")

    if unwrap_method == "xatlas":
        CONSOLE.print("Unwrapping mesh with xatlas method... this may take a while.")
        texture_coordinates, origins, directions = unwrap_mesh_with_xatlas(
            vertices, faces, vertex_normals, num_pixels_per_side=num_pixels_per_side
        )
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Unwrapped mesh with xatlas method")
    elif unwrap_method == "custom":
        CONSOLE.print("Unwrapping mesh with custom method...")
        texture_coordinates, origins, directions = unwrap_mesh_per_uv_triangle(
            vertices, faces, vertex_normals, px_per_uv_triangle
        )
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Unwrapped mesh with custom method")
    else:
        raise ValueError(f"Unwrap method {unwrap_method} not supported.")

    if raylen_method == "edge":
        face_vertices = vertices[faces]
        # compute the length of the rays we want to render
        # we make a reasonable approximation by using the mean length of one edge per face
        raylen = 2.0 * torch.mean(torch.norm(face_vertices[:, 1, :] - face_vertices[:, 0, :], dim=-1)).float()
    elif raylen_method == "none":
        raylen = 0.0
    else:
        raise ValueError(f"Ray length method {raylen_method} not supported.")

    summary_log.append(f"Length of rendered rays to compute texture values: {raylen}")

    origins = origins - 0.5 * raylen * directions
    pixel_area = torch.ones_like(origins[..., 0:1])
    camera_indices = torch.zeros_like(origins[..., 0:1])
    nears = torch.zeros_like(origins[..., 0:1])
    fars = torch.ones_like(origins[..., 0:1]) * raylen
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
    vertices = vertices.cpu().numpy()
    progress = get_progress("Writing vertices to file", suffix="lines-per-sec")
    with progress:
        for i in progress.track(range(len(vertices))):
            vertex = vertices[i]
            line = f"v {vertex[0]} {vertex[1]} {vertex[2]}\n"
            file_obj.write(line)

    # write the texture coordinates
    texture_coordinates = texture_coordinates.cpu().numpy()
    with progress:
        progress = get_progress("Writing texture coordinates to file", suffix="lines-per-sec")
        for i in progress.track(range(len(faces))):
            for uv in texture_coordinates[i]:
                line = f"vt {uv[0]} {1.0 - uv[1]}\n"
                file_obj.write(line)

    # write the vertex normals
    vertex_normals = vertex_normals.cpu().numpy()
    progress = get_progress("Writing vertex normals to file", suffix="lines-per-sec")
    with progress:
        for i in progress.track(range(len(vertex_normals))):
            normal = vertex_normals[i]
            line = f"vn {normal[0]} {normal[1]} {normal[2]}\n"
            file_obj.write(line)

    # write the faces
    faces = faces.cpu().numpy()
    progress = get_progress("Writing faces to file", suffix="lines-per-sec")
    with progress:
        for i in progress.track(range(len(faces))):
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

    summary_log.append(f"OBJ file saved to {output_dir / 'mesh.obj'}")
    summary_log.append(f"MTL file saved to {output_dir / 'material_0.mtl'}")
    summary_log.append(
        f"Texture image saved to {output_dir / 'material_0.png'} "
        f"with resolution {texture_image.shape[1]}x{texture_image.shape[0]} (WxH)"
    )

    CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")
    for summary in summary_log:
        CONSOLE.print(summary, justify="center")
    CONSOLE.rule()
