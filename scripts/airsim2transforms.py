import numpy as np
import math
import os
import json
import csv
import imageio
from sklearn.metrics import pairwise_distances

def get_intrinsic(imgdir):
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png') or f.endswith('jpeg')]

    H, W, C = imageio.imread(imgfiles[0]).shape
    vfov = 40

    focal_y = H / 2  / np.tan(np.deg2rad(vfov/2))
    focal_x = H / 2  / np.tan(np.deg2rad(vfov/2))

    return H, W, focal_x, focal_y
    
def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


def pad_rot(rot):
    padh = lambda x: np.hstack([x, np.zeros((x.shape[0], 1))])
    padv = lambda x: np.vstack([x, np.zeros((1, x.shape[1]))])

    rot_mat = padv(padh(rot))
    rot_mat[-1,-1] = 1
    return rot_mat


def quaternion_to_euler(q_w, q_x, q_y, q_z):
    # Convert quaternion to rotation matrix
    rotation_matrix = np.array([
        [1 - 2*q_y*q_y - 2*q_z*q_z, 2*q_x*q_y - 2*q_w*q_z, 2*q_x*q_z + 2*q_w*q_y],
        [2*q_x*q_y + 2*q_w*q_z, 1 - 2*q_x*q_x - 2*q_z*q_z, 2*q_y*q_z - 2*q_w*q_x],
        [2*q_x*q_z - 2*q_w*q_y, 2*q_y*q_z + 2*q_w*q_x, 1 - 2*q_x*q_x - 2*q_y*q_y]
    ])

    # Extract Euler angles from rotation matrix
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Convert angles to degrees
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)

    return roll, pitch, yaw

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("datadir", type=str, help='path to your meta')
    parser.add_argument("filename", type=str, help='file name')
    parser.add_argument("--imgdir", type=str, default='images', help='image directory name')
    
    return parser
    

if __name__ == '__main__':

    parser = config_parser()
    args = parser.parse_args()
 
    data = {}

    with open(os.path.join(args.datadir, args.filename), 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  # Skip the header row

        data['cameraFrames'] = []

        for row in reader:
            vehicle_name = row[0]
            timestamp = int(row[1])
            pos_x = float(row[2])
            pos_y = float(row[3])
            pos_z = float(row[4])
            q_w = float(row[5])
            q_x = float(row[6])
            q_y = float(row[7])
            q_z = float(row[8])
            image_file = row[9]

            roll, pitch, yaw = quaternion_to_euler(q_w, q_x, q_y, q_z)
            
            data['cameraFrames'].append({
                'position': {
                    'x': pos_x,
                    'y': pos_y,
                    'z': pos_z
                },
                'rotation': {
                    'x': roll,
                    'y': pitch,
                    'z': yaw
                },
                'image': image_file
                ,
                'timestamp': timestamp
            })
            



    GES_pos = np.array([[data['cameraFrames'][i]['position']['x'], 
                            data['cameraFrames'][i]['position']['y'],
                            data['cameraFrames'][i]['position']['z']] 
                        for i in range(len(data['cameraFrames']))])

    H, W, focal_x, focal_y = get_intrinsic(os.path.join(args.datadir, args.imgdir))

    # rescale the whole range if you want
    scale = 2**3 * np.pi / max(GES_pos.max(), -GES_pos.min())
    SS = np.eye(4)
    SS[0,0] = scale
    SS[1,1] = scale
    SS[2,2] = scale
    
    rot_ECEF2ENUV = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])

    nxyz = []
    frames = []
    
    image_list = np.sort(os.listdir(os.path.join(args.datadir, args.imgdir)))

    print("Found images...", len(image_list))
    
    print("Found entries...", len(data['cameraFrames']))
    for i in range(len(data['cameraFrames'])):
        position = data['cameraFrames'][i]['position']
        pos_x = position['x'] / 50
        pos_y = position['y'] / 50
        pos_z = position['z'] / 50
        xyz = np.array([pos_x, pos_y, pos_z])
        # [pos_e,pos_n,pos_u] = np.dot(rot_ECEF2ENUV, xyz)
        [pos_e,pos_n,pos_u] = xyz

        rotation = data['cameraFrames'][i]['rotation']

        x = np.radians(rotation['x'])
        y = np.radians(rotation['y'])
        z = np.radians(rotation['z'])

        rot_mat = np.linalg.inv(eulerAnglesToRotationMatrix([x, y, z]))
        rot_mat = np.dot(rot_ECEF2ENUV, rot_mat)
        GES_rotmat = pad_rot(rot_mat)

        xyz  = np.array([pos_e,pos_n,pos_u,1])[None,:]
        nx,ny,nz = np.dot(SS, xyz.T)[:3,0]
        nxyz.append([nx,ny,nz])
        GES_rotmat[:3,3] = np.array([nx,ny,nz])

        c2w = np.concatenate([GES_rotmat[:3,:4], np.array([[0, 0, 0, 1]])], 0)
        
        if not os.path.exists(os.path.join(args.datadir, args.imgdir, data['cameraFrames'][i]['image'])):
            print("Image not found", os.path.join(args.imgdir, data['cameraFrames'][i]['image']))
            continue

        frame = {
            "file_path": os.path.join(args.imgdir, data['cameraFrames'][i]['image']),
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": i,
        }
        
        frames.append(frame)

    print("Frames: ", len(frames))
    nxyz = np.array(nxyz)
    dists = np.sqrt(np.sum(nxyz**2, -1))

    out = {
        "w": W,
        "h": H,
    }
    
    out["fl_x"] = focal_x
    out["fl_y"] = focal_y
    out["cx"] = W/2
    out["cy"] = H/2
    out["k1"] = 0.0
    out["k2"] = 0.0
    out["p1"] = 0.0
    out["p2"] = 0.0
    out["scale"] = scale
        
    out["frames"] = frames
    # applied_transform = np.eye(4)[:3, :]
    # out["applied_transform"] = applied_transform.tolist()

    print("Saving...", os.path.join(args.datadir, 'transforms.json'))
    with open(os.path.join(args.datadir, 'transforms.json'), 'w', encoding="utf-8") as f: 
        json.dump(out, f, indent=4)
    
