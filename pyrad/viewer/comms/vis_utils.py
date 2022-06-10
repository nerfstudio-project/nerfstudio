"""Code to interface with the `vis/` (the JS visualizer).
"""

import copy
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

import pyrad.viewer.comms.cameras as c
import pyrad.viewer.comms.geometry as g
import pyrad.viewer.comms.transformations as tf
from pyrad.viewer.comms import ViewerWindow, Visualizer


def get_vis():
    """Returns the vis object."""
    window = ViewerWindow(zmq_url="tcp://0.0.0.0:6000")
    vis = Visualizer(window=window)
    return vis


def show_box_test(vis):
    """Simple test to draw a box and make sure everything is working."""
    vis["box"].set_object(g.Box([1.0, 1.0, 1.0]), material=g.MeshPhongMaterial(color=0xFF0000))


def get_random_color():
    color = np.random.rand(3) * 255.0
    color = tuple([int(x) for x in color])
    return color


def plot_correspondences(pair, plot=True):
    """Draw what the pair looks like, with the correspondences as lines."""
    image0 = (pair["data0"]["image"] * 255).astype("uint8").transpose((1, 2, 0))
    image1 = (pair["data1"]["image"] * 255).astype("uint8").transpose((1, 2, 0))
    original_image = np.hstack([image0, image1])
    h, w, _ = image0.shape

    matches_image = original_image.copy()
    # draw lines
    correspondences = list(pair["correspondences"])
    num = min(20, len(correspondences))
    for x0, y0, x1, y1 in np.array(random.sample(correspondences, k=num)).astype("uint64"):
        color = get_random_color()
        thickness = 2
        matches_image = cv2.line(matches_image, (x0, y0), (int(w + x1), y1), color, thickness)
    # show image
    if plot:
        print("Correspondences:")
        plt.figure(figsize=(20, 10))
        plt.imshow(matches_image)
        plt.show()

    # ----------
    # example training data
    equal_distances_image = original_image.copy()
    for i in range(num):
        c0, c1 = np.array(random.sample(correspondences, k=2)).astype("uint64")
        color = get_random_color()
        thickness = 2
        p0, p1 = c0[:2], c1[:2]
        equal_distances_image = cv2.line(equal_distances_image, tuple(p0), tuple(p1), color, thickness)
        p0, p1 = c0[2:], c1[2:]
        p0[0] += w
        p1[0] += w
        equal_distances_image = cv2.line(equal_distances_image, tuple(p0), tuple(p1), color, thickness)
    if plot:
        print("Example training data:")
        plt.figure(figsize=(20, 10))
        plt.imshow(equal_distances_image)
        plt.show()

    return matches_image, equal_distances_image


def show_ply(vis, ply_path, name="ply", color=None):
    """Show the PLY file in the 3D viewer. Specify the full filename as input."""
    assert ply_path.endswith(".ply")
    if color:
        material = g.MeshPhongMaterial(color=color)
    else:
        material = g.MeshPhongMaterial(vertexColors=True)
    vis[name].set_object(g.PlyMeshGeometry.from_file(ply_path), material)


def show_obj(vis, obj_path, name="obj", color=None):
    """Show the PLY file in the 3D viewer. Specify the full filename as input."""
    assert obj_path.endswith(".obj")
    if color:
        material = g.MeshPhongMaterial(color=color)
    else:
        material = g.MeshPhongMaterial(vertexColors=True)
    vis[name].set_object(g.ObjMeshGeometry.from_file(obj_path), material)


def show_pair(vis, pair, name="pair"):
    """Call show_data for both data points in the pair."""
    show_data(vis, pair["data0"], name=f"{name}/data0")
    show_data(vis, pair["data1"], name=f"{name}/data1")


# TODO: bring back the functions below here


def show_prediction_in_viewer(vis, data, pred_depth, name="data"):
    d = copy.deepcopy(data)
    d["image"] = data["image"][0].cpu().numpy()
    d["depth"] = pred_depth[0].cpu().numpy()
    d["pose"] = data["pose"][0].cpu().numpy().astype("float64")
    d["intrinsics"] = data["intrinsics"][0].cpu().numpy()
    show_data(vis, d, name=name)


def show_experiment(vis, inputs, left_depth, right_depth, model_name="temp"):
    show_prediction_in_viewer(vis, inputs["data0"], left_depth, name="{}/left_pred".format(model_name))
    show_prediction_in_viewer(vis, inputs["data0"], inputs["data0"]["depth"], name="{}/left_gt".format(model_name))

    show_prediction_in_viewer(vis, inputs["data1"], right_depth, name="{}/right_pred".format(model_name))
    show_prediction_in_viewer(vis, inputs["data1"], inputs["data1"]["depth"], name="{}/right_gt".format(model_name))
    return


def pair_from_inputs(inputs, b=0):
    """Return a pair without the batch dimension.
    Note that b specifies which batch index to use.
    Also note that inputs has a nested dictionary structure 2 layers deep.
    """
    pair = {}
    for key, value in inputs.items():
        if isinstance(value, dict):
            pair[key] = {}
            for key2, value2 in value.items():
                pair[key][key2] = copy.deepcopy(value2[b])
        else:
            pair[key] = copy.deepcopy(value[b])
    return pair


def pair_outputs_from_outputs(outputs, b=0):
    pair_outputs = pair_from_inputs(outputs, b=b)
    return pair_outputs


# COLMAP helper functions
# camera drawing helper functions


def draw_camera_frustum(
    vis,
    image=np.random.rand(100, 100, 3) * 255.0,
    pose=tf.translation_matrix([0, 0, 0]),
    K=None,
    name="0000000",
    displayed_focal_length=None,
    shift_forward=None,
    height=None,
    realistic=True,
):
    """Draw the camera in the scene."""

    assert K[0, 0] == K[1, 1]
    focal_length = K[0, 0]
    pp_w = K[0, 2]
    pp_h = K[1, 2]

    if displayed_focal_length:
        assert height is None or not realistic
    if height:
        assert displayed_focal_length is None or not realistic

    if height:
        dfl = height / (2.0 * (pp_h / focal_length))
        width = 2.0 * (pp_w / focal_length) * dfl
        if displayed_focal_length is None:
            displayed_focal_length = dfl
    elif displayed_focal_length:
        width = 2.0 * (pp_w / focal_length) * displayed_focal_length
        height = 2.0 * (pp_h / focal_length) * displayed_focal_length
    else:
        assert not realistic

    if pose.shape == (3, 4):
        pose = np.concatenate([pose, np.zeros_like(pose[:1])], axis=0)
        pose[3, 3] = 1.0

    # draw the frustum
    g_frustum = c.frustum(scale=1.0, focal_length=displayed_focal_length, width=width, height=height)
    vis[name + "/frustum"].set_object(g_frustum)
    if not realistic:
        vis[name + "/frustum"].set_transform(tf.translation_matrix([0, 0, displayed_focal_length]))

    # draw the image plane
    g_image_plane = c.ImagePlane(image, width=width, height=height)
    vis[name + "/image_plane"].set_object(g_image_plane)
    if realistic:
        vis[name + "/image_plane"].set_transform(tf.translation_matrix([0, 0, -displayed_focal_length]))

    if shift_forward:
        matrix = tf.translation_matrix([0, 0, displayed_focal_length])
        matrix2 = tf.translation_matrix([0, 0, -shift_forward])
        vis[name + "/frustum"].set_transform(matrix2 @ matrix)
        vis[name + "/image_plane"].set_transform(matrix2)

    # set the transform of the camera
    vis[name].set_transform(pose)


def set_camera_render(vis, intrinsics=None, pose=None, name="renderer"):
    """Place a three.js camera in the scene.
    This can be used to render an image from.
    """
    full_name_str = f"/Cameras/{name}/rotated"
    g_camera = c.PerspectiveCamera(fov=120, aspect=1.0, near=0.01, far=1000)
    g_camera_helper = c.CameraHelper(g_camera)
    # vis[full_name_str].set_object(g_camera)
    vis[full_name_str].set_object(g_camera_helper)


def set_persp_camera(vis, pose, K, colmap=True):
    """Assumes simple pinhole model for intrinsics.
    Args:
        colmap: whether to use the colmap camera coordinate convention or not
    """
    pose_processed = copy.deepcopy(pose)
    if colmap:
        pose_processed[:, 1:3] *= -1
    pp_w = K[0, 2]
    pp_h = K[1, 2]
    assert K[0, 0] == K[1, 1]
    focal_length = K[0, 0]
    x = pp_h / (focal_length)
    fov = 2.0 * np.arctan(x) * (180.0 / np.pi)
    vis["/Cameras/Main Camera R/<object>"].set_property("fov", fov)
    vis["/Cameras/Main Camera R/<object>"].set_property("aspect", float(pp_w / pp_h))  # three.js expects width/height
    vis["/Cameras/Main Camera R"].set_transform(pose_processed)


def set_orth_camera(vis, pose, width, height, colmap=True):
    """ """
    pose_processed = copy.deepcopy(pose)
    if colmap:
        pose_processed[:, 1:3] *= -1
    vis["/Cameras/Main Camera Orth"].set_transform(pose_processed)
    vis["/Cameras/Main Camera Orth/<object>"].set_property("left", -width / 2.0)
    vis["/Cameras/Main Camera Orth/<object>"].set_property("right", width / 2.0)
    vis["/Cameras/Main Camera Orth/<object>"].set_property("top", height / 2.0)
    vis["/Cameras/Main Camera Orth/<object>"].set_property("bottom", -height / 2.0)
