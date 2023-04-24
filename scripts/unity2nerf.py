import os
import sys
import csv
import cv2
import math
import json
import argparse
import numpy as np

from math import cos, sin
from scipy.spatial.transform import Rotation

# Unity camera intrinsics
CAMERA_ANGLE_X = 0.9942381084084299
CAMERA_ANGLE_Y = 1.248833026050588
FL_X = 2830.983620611989
FL_Y = 2830.983620611989
K1 = 0.0005688416121690353
K2 = 0
P1 = 0
P2 = 0
CX = 1536.0
CY = 2040.0

def parse_args():
    parser = argparse.ArgumentParser(description="convert a Unity simulation export to nerf format transforms.json")
    parser.add_argument("--flip_images", help="flip input images on disk")
    parser.add_argument("--input", help="path to Unity simulation output")
    parser.add_argument("--aabb_scale", default=16, choices=["1", "2", "4", "8", "16"], help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16")
    parser.add_argument("--subsample", default=1, choices=["1", "2", "3", "4", "5", "6"], help="subsample scene by picking every nth image. n=1 means pick every image")
    args = parser.parse_args()
    return args

def flip_images(image_dir):
    images = os.listdir(image_dir)
    for image in images:
        img = cv2.imread(os.path.join(image_dir, image))
        flipped_img = cv2.flip(img, 1)

        # NOTE: This overwrites images on disk
        cv2.imwrite(os.path.join(image_dir, image), flipped_img)

def _create_translation_matrix(x, y, z):
    translation_matrix = np.identity(4)
    translation_matrix[0:3, 3] = [x, y, z]
    return translation_matrix

def _create_x_rotation_matrix(rads):
    c = cos(rads)
    s = sin(rads)
    return np.array([
        [1., 0., 0., 0.],
        [0., +c, -s, 0.],
        [0., +s, +c, 0.],
        [0., 0., 0., 1.]
    ])

def _create_y_rotation_matrix(rads):
    c = cos(rads)
    s = sin(rads)
    return np.array([
        [+c, 0., +s, 0.],
        [0., 1., 0., 0.],
        [-s, 0., +c, 0.],
        [0., 0., 0., 1.]
    ])

def _create_z_rotation_matrix(rads):
    c = cos(rads)
    s = sin(rads)
    return np.array([
        [+c, -s, 0., 0.],
        [+s, +c, 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ])

def convert_unity_pose_to_colmap_transform(unity_pos_xyz, unity_euler_xyz):
    # This conversion corresponds to the successful experiment variant code "xyZ_XYZ_tyxz".
    # It is known to work when unity_euler_z (roll) is 0.
    # It is expected to work (but not yet tested) when unity_euler_z is non-0.
    unity_pos_x, unity_pos_y, unity_pos_z = unity_pos_xyz
    unity_euler_x, unity_euler_y, unity_euler_z = unity_euler_xyz
    translation_matrix = _create_translation_matrix(unity_pos_x, unity_pos_y, -unity_pos_z)
    x_rotation_matrix = _create_x_rotation_matrix(-unity_euler_x)
    y_rotation_matrix = _create_y_rotation_matrix(-unity_euler_y)
    z_rotation_matrix = _create_z_rotation_matrix(-unity_euler_z)
    colmap_transform = translation_matrix @ y_rotation_matrix @ x_rotation_matrix @ z_rotation_matrix
    return colmap_transform

def is_so3(mat):

    # Check if transpose is equal to inverse
    if np.mean(np.abs(mat.T - np.linalg.inv(mat))) > 1e-4:
        print(np.mean(np.abs(mat.T - np.linalg.inv(mat))))
        return False
    elif np.abs(np.linalg.det(mat) - 1) > 1e-4:
        print(np.abs(np.linalg.det(mat) - 1))
        return False

    return True

def load_settings(settings_file):
    try:
        with open(settings_file, 'r') as fp:
            json_data = json.load(fp)
            xy = json_data['imageDimensions']
            return xy['x'], xy['y']

    except FileNotFoundError:
        print('SimulationSettings.json missing')

def load_transforms(transforms_file):
    df = pd.read_csv(transforms_file)
    return df

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

def parse_image_path(image_path):
    frame = image_path[5:8]
    cam = image_path[12:15]
    return int(frame), int(cam)

def read_csv(csv_file):
    data = []
    with open(csv_file, 'r') as fp:
        reader = csv.reader(fp)
        for row in reader:
            data.append(row)
    return data[1:]

def row_to_path(row):

    # Params: FlightNumber, Time, ImageName, FrameNumber, ...
    # ...,    PosX, PosY, PosZ, RotX, RotY, RotZ, QuatX, QuatY, QuatZ, QuatW
    fname = row[2]
    return f'{fname}.jpg'

def parse_csv(args):
    try:
        data = read_csv(os.path.join(args.input, 'CameraPoseInfo.csv'))[1:]
    except FileNotFoundError:
        print("Invalid input directory. Must contain CameraPoseInfo.csv")
        sys.exit(1)

    W, H = load_settings('./SimulationSettings.json')

    # Hardcoded intrinsics
    out = {
        "camera_angle_x": CAMERA_ANGLE_X,
        "camera_angle_y": CAMERA_ANGLE_Y,
        "fl_x": FL_X,
        "fl_y": FL_Y,
        "k1": K1,
        "k2": K2,
        "p1": P1,
        "p2": P2,
        "cx": CX,
        "cy": CY,
        "w": W,
        "h": H,
        "aabb_scale": args.aabb_scale,
        "frames": [],
    }

    # Main data loop
    up = np.zeros(3)
    for i, row in enumerate(data):

        if i % int(args.subsample) != 0:
            continue
        img_dir = os.path.join(args.input, 'images')
        img_path = row_to_path(row)
        b = sharpness(os.path.join(img_dir, img_path))  # TODO: figure out reasonable threshold

        c2w = np.eye(4)
        tvec = np.float_(row[4:7])
        euler_angles = np.float_(row[7:10])
        qvec = np.float_(row[10:14])

        # Construct c2w
        R = Rotation.from_quat(qvec).as_matrix()
        c2w[:3, :3] = -R
        c2w[:3, 3] = tvec

        c2w[0:3,1] *= -1 # flip y
        c2w = c2w[[0,2,1,3],:]  # swap y and z
        c2w[2,:] *= -1 # flip whole world upside down
        up += c2w[0:3,1]

        if not is_so3(c2w[:3, :3]):
            raise ValueError("Invalid rotation matrix")

        out_img_path = os.path.join(img_dir, img_path)
        frame={"file_path": out_img_path, "sharpness": b, "transform_matrix": c2w}
        out["frames"].append(frame)

    nframes = len(out["frames"])

    up = up / np.linalg.norm(up)
    print(f'up vector was: {up}')
    R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    R = np.pad(R,[0,1])
    R[-1, -1] = 1

    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in out["frames"]:
        mf = f["transform_matrix"][0:3,:]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3,:]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.00001:
                totp += p*w
                totw += w
        if totw > 0.0:
            totp /= totw
    print(totp) # the cameras are looking at totp
    for f in out["frames"]:
        f["transform_matrix"][0:3,3] -= totp

    avglen = 0.
    for f in out["frames"]:
        avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
    avglen /= nframes
    print("avg camera distance from origin", avglen)
    for f in out["frames"]:
        f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()

    print(nframes,"frames")
    print("writing tforms.json")

    with open(os.path.join(args.input, 'transforms.json'), 'w') as outfile:
        json.dump(out, outfile, indent=2)

def main(args):
    if args.flip_images:
        print("flipping input images...")
        flip_images(os.path.join(args.input, 'images'))
    parse_csv(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)
