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


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("datadir", type=str, help='path to your meta')
    parser.add_argument("filename", type=str, help='file name')
    parser.add_argument("--imgdir", type=str, default='footage', help='image directory name')
    parser.add_argument("lat", type=float, help='lat of center building')
    parser.add_argument("lon", type=float, help='lon of center building')
    
    return parser
    

if __name__ == '__main__':

    parser = config_parser()
    args = parser.parse_args()
 
    import json
    import math
    import os

    import imageio
    import numpy as np  

    with open(os.path.join(args.datadir, args.filename), 'r') as f:
        data = json.load(f)

    GES_pos = np.array([[data['cameraFrames'][i]['position']['x'], 
                            data['cameraFrames'][i]['position']['y'],
                            data['cameraFrames'][i]['position']['z']] 
                        for i in range(len(data['cameraFrames']))])

    H, W, focal_x, focal_y = get_intrinsic(os.path.join(args.datadir, args.imgdir))

    # # rescale the whole range if you want
    # scale = 2**3 * np.pi / max(GES_pos.max(), -GES_pos.min())
    scale = 1.0
    SS = np.eye(4)
    SS[0,0] = scale
    SS[1,1] = scale
    SS[2,2] = scale

    
    rclat, rclng = np.radians(args.lat), np.radians(args.lon) 
    rot_ECEF2ENUV = np.array([[-math.sin(rclng),                math.cos(rclng),                              0],
                              [-math.sin(rclat)*math.cos(rclng), -math.sin(rclat)*math.sin(rclng), math.cos(rclat)],
                              [math.cos(rclat)*math.cos(rclng),  math.cos(rclat)*math.sin(rclng),  math.sin(rclat)]])

    nxyz = []
    frames = []
    
    image_list = np.sort(os.listdir(os.path.join(args.datadir, args.imgdir)))
    
    print("Found entries...", len(data['cameraFrames']))
    for i in range(len(data['cameraFrames'])):
        position = data['cameraFrames'][i]['position']
        pos_x = position['x']
        pos_y = position['y']
        pos_z = position['z']
        xyz = np.array([pos_x, pos_y, pos_z])
        [pos_e,pos_n,pos_u] = np.dot(rot_ECEF2ENUV, xyz)
        pos_u = pos_u - 6371106 # earth radius

        rotation = data['cameraFrames'][i]['rotation']

        x = np.radians(-rotation['x'])
        y = np.radians(180-rotation['y'])
        z = np.radians(180+rotation['z'])

        rot_mat = np.linalg.inv(eulerAnglesToRotationMatrix([x, y, z]))
        rot_mat = np.dot(rot_ECEF2ENUV, rot_mat)
        GES_rotmat = pad_rot(rot_mat)

        xyz  = np.array([pos_e,pos_n,pos_u,1])[None,:]
        nx,ny,nz = np.dot(SS, xyz.T)[:3,0]
        nxyz.append([nx,ny,nz])
        GES_rotmat[:3,3] = np.array([nx,ny,nz])

        c2w = np.concatenate([GES_rotmat[:3,:4], np.array([[0, 0, 0, 1]])], 0)

        img_name_i = image_list[0][:-9] + '_' + str(i).zfill(3) + '.jpeg'
        img_path = os.path.join(args.imgdir, img_name_i)
        if img_name_i not in image_list:
            print("Image not found, skipping...", img_path)
            continue

        frame = {
            "file_path": img_path,
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": i,
        }
        
        frames.append(frame)

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
    out["lat"] = args.lat
    out["lon"] = args.lon
        
    out["frames"] = frames
    # applied_transform = np.eye(4)[:3, :]
    # out["applied_transform"] = applied_transform.tolist()

    print("Saving...", os.path.join(args.datadir, 'transforms.json'))
    with open(os.path.join(args.datadir, 'transforms.json'), 'w', encoding="utf-8") as f: 
        json.dump(out, f, indent=4)
    
