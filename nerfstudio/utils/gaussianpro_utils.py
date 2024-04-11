import os
import struct
from typing import Dict, List
import cv2
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras

import ctypes
import threading

# thread-safe singleton class
class Singleton(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances: 
            with cls._lock:
                # Another thread could have created the instance
                # before we acquired the lock. So check that the
                # instance is still nonexistent.
                if cls not in cls._instances: 
                    cls._instances[cls] = super(type(cls), cls).__call__(*args, **kwargs)
        return cls._instances[cls]

# class to implement depth propagation by incorporating the shared object
class _Propagation:
    def __init__(self, lib_path: str = None) -> None:
        # get the path of the propagation lib
        path = lib_path if lib_path is not None else 'submodules/Propagation/build/libpropagation.so'
        
        # load the shared library
        self.propagation_lib = ctypes.CDLL(path)

        # set the argument types for run_exec call
        self.propagation_lib.run_exec.argtypes = [ctypes.c_char_p,
                           ctypes.c_char_p,
                           ctypes.c_bool,
                           ctypes.c_int,
                           ctypes.c_int]
        
    def run_exec(self, 
                 dense_folder: str, 
                 src_ids_str: str, 
                 patch_size: int = 20, 
                 ref_id: int = 0, 
                 geom_consistency: bool = False):
        # execute the function
        self.propagation_lib.run_exec(dense_folder.encode('utf-8'),
                                      src_ids_str.encode('utf-8'),
                                      geom_consistency,
                                      patch_size,
                                      ref_id)

# create a thread-safe singleton version
class Propagation(_Propagation, metaclass=Singleton):
    pass

def write_cam_txt(cam_path, K, w2c, depth_range):
    with open(cam_path, "w") as file:
        file.write("extrinsic\n")
        for row in w2c:
            file.write(" ".join(str(element) for element in row))
            file.write("\n")

        file.write("\nintrinsic\n")
        for row in K:
            file.write(" ".join(str(element) for element in row))
            file.write("\n")
        
        file.write("\n")
        
        file.write(" ".join(str(element) for element in depth_range))
        file.write("\n")

def depth_propagation(propagation_cameras: Dict[int, Cameras],
                      projected_depth: torch.Tensor, 
                      ref_idx: int,
                      src_idxs: List[int], 
                      patch_size: int,
                      ref_img: torch.Tensor, 
                      gt_images: List,
                      depth_max: float,):
    # pass data to c++ api for mvs
    cdata_image_path = './cache/images'
    cdata_camera_path = './cache/cams'
    cdata_depth_path = './cache/depths'

    # get the reference camera
    camera = propagation_cameras[ref_idx]

    # set the depth min
    depth_min = 0.1

    #scale it for float type
    projected_depth = projected_depth * 100

    # get the 
    ref_K = camera.get_intrinsics_matrices()
    ref_w2c = camera.world_to_cameras.transpose(0, 1)

    # convert image to numpy
    ref_img = ref_img * 255
    ref_img = ref_img.detach().cpu().numpy().astype(np.uint8)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(cdata_image_path, "0.jpg"), ref_img)
    cv2.imwrite(os.path.join(cdata_depth_path, "0.png"), projected_depth.detach().cpu().numpy().astype(np.uint16))
    write_cam_txt(cam_path=os.path.join(cdata_camera_path, "0.txt"), 
                  K=ref_K.squeeze().detach().cpu().numpy(), 
                  w2c=ref_w2c.squeeze().detach().cpu().numpy(),
                  depth_range=[depth_min, (depth_max-depth_min)/192.0, 192.0, depth_max])
    for idx, src_idx in enumerate(src_idxs):
        cur_camera = propagation_cameras[src_idx]
        src_w2c = cur_camera.world_to_cameras.transpose(0, 1)
        src_K = cur_camera.get_intrinsics_matrices()

        src_img = gt_images[idx]
        src_img = src_img * 255
        src_img = src_img.detach().cpu().numpy().astype(np.uint8)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(cdata_image_path, str(idx+1)+".jpg"), src_img)
        write_cam_txt(cam_path=os.path.join(cdata_camera_path, str(idx+1)+".txt"), 
                      K=src_K.squeeze().detach().cpu().numpy(), 
                      w2c=src_w2c.squeeze().detach().cpu().numpy(), 
                      depth_range=[depth_min, (depth_max-depth_min)/192.0, 192.0, depth_max])
        
    # c++ api for depth propagation
    pro = Propagation("/home/ubuntu/GaussianPro/submodules/Propagation/build/libpropagation.so")
    
    # run the command
    pro.run_exec('./cache', '1 2 3 4', patch_size, 0, False)
    
def load_pairs_relation(path):
    pairs_relation = []
    num = 0
    with open(path, 'r') as file:
        num_images = int(file.readline())
        for i in range(num_images):

            ref_image_id = int(file.readline())
            if i != ref_image_id:
                print(ref_image_id)
                print(i)

            src_images_infos = file.readline().split()
            num_src_images = int(src_images_infos[0])
            src_images_infos = src_images_infos[1:]
            
            pairs = []
            #only fetch the first 4 src images
            for j in range(num_src_images):
                id, score = int(src_images_infos[2*j]), int(src_images_infos[2*j+1])
                #the idx needs to align to the training images
                if score <= 0.0 or id % 8 == 0:
                    continue
                id = (id // 8) * 7 + (id % 8) - 1
                pairs.append(id)
                
                if len(pairs) > 3:
                    break
                
            if ref_image_id % 8 != 0:
                #only load the training images
                pairs_relation.append(pairs)
            else:
                num = num + 1
            
    return pairs_relation

def readDepthDmb(file_path):
    inimage = open(file_path, "rb")
    if not inimage:
        print("Error opening file", file_path)
        return -1

    type = -1

    type = struct.unpack("i", inimage.read(4))[0]
    h = struct.unpack("i", inimage.read(4))[0]
    w = struct.unpack("i", inimage.read(4))[0]
    nb = struct.unpack("i", inimage.read(4))[0]

    if type != 1:
        inimage.close()
        return -1

    dataSize = h * w * nb

    depth = np.zeros((h, w), dtype=np.float32)
    depth_data = np.frombuffer(inimage.read(dataSize * 4), dtype=np.float32)
    depth_data = depth_data.reshape((h, w))
    np.copyto(depth, depth_data)

    inimage.close()
    return depth

def readNormalDmb(file_path):
    try:
        with open(file_path, 'rb') as inimage:
            type = np.fromfile(inimage, dtype=np.int32, count=1)[0]
            h = np.fromfile(inimage, dtype=np.int32, count=1)[0]
            w = np.fromfile(inimage, dtype=np.int32, count=1)[0]
            nb = np.fromfile(inimage, dtype=np.int32, count=1)[0]

            if type != 1:
                print("Error: Invalid file type")
                return -1

            dataSize = h * w * nb

            normal = np.zeros((h, w, 3), dtype=np.float32)
            normal_data = np.fromfile(inimage, dtype=np.float32, count=dataSize)
            normal_data = normal_data.reshape((h, w, nb))
            normal[:, :, :] = normal_data[:, :, :3]

            return normal

    except IOError:
        print("Error opening file", file_path)
        return -1

def read_propagted_depth(path):    
    cost = readDepthDmb(os.path.join(path, 'costs.dmb'))
    cost[np.isnan(cost)] = 2
    cost[cost < 0] = 2
    # mask = cost > 0.5

    depth = readDepthDmb(os.path.join(path, 'depths.dmb'))
    depth[np.isnan(depth)] = 300
    depth[depth < 0] = 300
    depth[depth > 300] = 300
    
    normal = readNormalDmb(os.path.join(path, 'normals.dmb'))

    return depth, cost, normal

def bilinear_sampler(img, coords, mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = torch.nn.functional.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img




def build_quaternion(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]
    return q



# project the reference point cloud into the source view, then project back
#extrinsics here refers c2w
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    batch, height, width = depth_ref.shape
    
    ## step1. project reference pixels to the source view
    # reference view x, y
    y_ref, x_ref = torch.meshgrid(torch.arange(0, height).to(depth_ref.device), torch.arange(0, width).to(depth_ref.device))
    x_ref = x_ref.unsqueeze(0).repeat(batch,  1, 1)
    y_ref = y_ref.unsqueeze(0).repeat(batch,  1, 1)
    x_ref, y_ref = x_ref.reshape(batch, -1), y_ref.reshape(batch, -1)
    # reference 3D space

    A = torch.inverse(intrinsics_ref)
    B = torch.stack((x_ref, y_ref, torch.ones_like(x_ref).to(x_ref.device)), dim=1) * depth_ref.reshape(batch, 1, -1)
    xyz_ref = torch.matmul(A, B)

    # source 3D space
    xyz_src = torch.matmul(torch.matmul(torch.inverse(extrinsics_src), extrinsics_ref),
                        torch.cat((xyz_ref, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1))[:, :3]
    # source view x, y
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:, :2] / K_xyz_src[:, 2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[:, 0].reshape([batch, height, width]).float()
    y_src = xy_src[:, 1].reshape([batch, height, width]).float()

    # print(x_src, y_src)
    sampled_depth_src = bilinear_sampler(depth_src.view(batch, 1, height, width), torch.stack((x_src, y_src), dim=-1).view(batch, height, width, 2))

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = torch.matmul(torch.inverse(intrinsics_src),
                        torch.cat((xy_src, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1) * sampled_depth_src.reshape(batch, 1, -1))
    # reference 3D space
    xyz_reprojected = torch.matmul(torch.matmul(torch.inverse(extrinsics_ref), extrinsics_src),
                                torch.cat((xyz_src, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1))[:, :3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[:, 2].reshape([batch, height, width]).float()
    K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:, :2] / K_xyz_reprojected[:, 2:3]
    x_reprojected = xy_reprojected[:, 0].reshape([batch, height, width]).float()
    y_reprojected = xy_reprojected[:, 1].reshape([batch, height, width]).float()

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src

def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, thre1=1, thre2=0.01):
    batch, height, width = depth_ref.shape
    y_ref, x_ref = torch.meshgrid(torch.arange(0, height).to(depth_ref.device), torch.arange(0, width).to(depth_ref.device))
    x_ref = x_ref.unsqueeze(0).repeat(batch,  1, 1)
    y_ref = y_ref.unsqueeze(0).repeat(batch,  1, 1)
    inputs = [depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src]
    outputs = reproject_with_depth(*inputs)
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = outputs
    # check |p_reproj-p_1| < 1
    dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = torch.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = torch.logical_and(dist < thre1, relative_depth_diff < thre2)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff
