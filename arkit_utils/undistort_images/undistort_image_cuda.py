import cv2
import numpy as np
import json
import os
import argparse
from numba import cuda
import tqdm
class Data:
    def __init__(self, intrinsic_matrix, 
                 intrinsic_matrix_reference_dimensions, 
                 lens_distortion_center, 
                 inverse_lens_distortion_lookup_table, 
                 lens_distortion_lookup_table):
        self.intrinsic_matrix = intrinsic_matrix
        self.intrinsic_matrix_reference_dimensions = intrinsic_matrix_reference_dimensions
        self.lens_distortion_center = lens_distortion_center
        self.inverse_lens_distortion_lookup_table = inverse_lens_distortion_lookup_table
        self.lens_distortion_lookup_table = lens_distortion_lookup_table

def readCalibrationJson(path: str) -> Data:
    with open(path, "r") as f:
        data = json.load(f)

    intrinsic_matrix = data["calibration_data"]["intrinsic_matrix"]
    intrinsic_matrix_reference_dimensions = data["calibration_data"]["intrinsic_matrix_reference_dimensions"]
    lens_distortion_center = data["calibration_data"]["lens_distortion_center"]
    inverse_lut = data["calibration_data"]["inverse_lens_distortion_lookup_table"]
    lut = data["calibration_data"]["lens_distortion_lookup_table"]

    data = Data(intrinsic_matrix, intrinsic_matrix_reference_dimensions, lens_distortion_center, inverse_lut, lut)
    return data

@cuda.jit
def rectify_image_kernel(image, rectified_image, lookup_table, distortion_center_x, distortion_center_y, width, height, ratio_x, ratio_y):
    x, y = cuda.grid(2)
    if x < width and y < height:
        radius_max_x = min(distortion_center_x, width - distortion_center_x)
        radius_max_y = min(distortion_center_y, height - distortion_center_y)
        radius_max = (radius_max_x**2 + radius_max_y**2)**0.5

        radius_point = ((x - distortion_center_x)**2 + (y - distortion_center_y)**2)**0.5
        magnification = lookup_table[-1]
        if radius_point < radius_max:
            relative_position = radius_point / radius_max * (len(lookup_table) - 1)
            frac = relative_position - int(relative_position)
            lower_lookup = lookup_table[int(relative_position)]
            upper_lookup = lookup_table[int(relative_position) + 1]
            magnification = lower_lookup * (1.0 - frac) + upper_lookup * frac

        mapped_x = int(distortion_center_x + (x - distortion_center_x) * (1.0 + magnification))
        mapped_y = int(distortion_center_y + (y - distortion_center_y) * (1.0 + magnification))

        if 0 <= mapped_x < width and 0 <= mapped_y < height:
            rectified_image[y, x, 0] = image[mapped_y, mapped_x, 0]
            rectified_image[y, x, 1] = image[mapped_y, mapped_x, 1]
            rectified_image[y, x, 2] = image[mapped_y, mapped_x, 2]

def rectify_single_image(image_path: str, output_path: str, distortion_param_json_path: str, crop_x: int, crop_y: int):
    image = cv2.imread(image_path)
    height, width, channel = image.shape
    rectified_image = np.zeros((height, width, channel), dtype=image.dtype)

    data = readCalibrationJson(distortion_param_json_path)
    lookup_table = np.array(data.inverse_lens_distortion_lookup_table, dtype=np.float32)
    distortion_center = np.array(data.lens_distortion_center, dtype=np.float32)
    reference_dimensions = data.intrinsic_matrix_reference_dimensions
    ratio_x = width / reference_dimensions[0]
    ratio_y = height / reference_dimensions[1]

    threads_per_block = (16, 16)
    blocks_per_grid_x = int(np.ceil(width / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(height / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    d_image = cuda.to_device(image)
    d_rectified_image = cuda.to_device(rectified_image)
    d_lookup_table = cuda.to_device(lookup_table)
    d_distortion_center_x = distortion_center[0] * ratio_x
    d_distortion_center_y = distortion_center[1] * ratio_y

    rectify_image_kernel[blocks_per_grid, threads_per_block](d_image, d_rectified_image, d_lookup_table, d_distortion_center_x, d_distortion_center_y, width, height, ratio_x, ratio_y)
    rectified_image = d_rectified_image.copy_to_host()

    u_shift = crop_x
    v_shift = crop_y
    crop_image = rectified_image[v_shift:height - v_shift, u_shift:width - u_shift]
    cv2.imwrite(output_path, crop_image)

def rectify_all_images(image_folder_path, distortion_param_json_path, output_image_folder_path, crop_x, crop_y):
    for filename in tqdm.tqdm(os.listdir(image_folder_path), desc="Processing:"):
        image_path = os.path.join(image_folder_path, filename)
        output_path = os.path.join(output_image_folder_path, filename)
        rectify_single_image(image_path, output_path, distortion_param_json_path, crop_x, crop_y)

def rectified_intrinsic(input_path, output_path, crop_x, crop_y):
    num_intrinsic = 0
    with open(input_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_intrinsic += 1
    # print(num_intrinsic)

    camera_ids = np.empty((num_intrinsic, 1))
    widths = np.empty((num_intrinsic, 1))
    heights = np.empty((num_intrinsic, 1))
    paramss = np.empty((num_intrinsic, 4))

    count = 0
    with open(input_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])-crop_x*2
                height = int(elems[3])-crop_y*2
                params = np.array(tuple(map(float, elems[4:])))
                params[2] = params[2] - crop_x
                params[3] = params[3] - crop_y

                camera_ids[count] = camera_id
                widths[count] = width
                heights[count] = height
                paramss[count] = params

                count = count+1

    with open(output_path, "w") as f:
        for i in range(num_intrinsic):
            line = str(int(camera_ids[i])) + " " + "PINHOLE" + " " + str(int(widths[i]))+ " " + str(int(heights[i]))+ " " + str(paramss[i][0]) + " " + str(paramss[i][1])+ " " + str(paramss[i][2])+ " " +  str(paramss[i][3])
            f.write(line  + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="undistort ARKit image using distortion params get from AVfoundation")
    parser.add_argument("--input_base", type=str)
    parser.add_argument("--crop_x", type=int, default=10)
    parser.add_argument("--crop_y", type=int, default=8)


    args = parser.parse_args()
    base_folder_path = args.input_base
    input_image_folder_path = base_folder_path + "/distort_images"
    distortion_param_json_path = base_folder_path + "/sparse/0/calibration.json"
    output_image_folder_path = base_folder_path + "/post/images/"
    crop_x = args.crop_x
    crop_y = args.crop_y
    input_camera = base_folder_path + "/sparse/0/distort_cameras.txt"

    if not os.path.exists(output_image_folder_path):
        os.makedirs(output_image_folder_path)

    rectify_all_images(input_image_folder_path, distortion_param_json_path, output_image_folder_path, crop_x, crop_y)
    
    output_cameras = [
        base_folder_path + "/post/sparse/online/cameras.txt",
        base_folder_path + "/post/sparse/online_loop/cameras.txt"
    ]

    for output_camera in output_cameras:
        rectified_intrinsic(input_camera, output_camera, crop_x, crop_y)
