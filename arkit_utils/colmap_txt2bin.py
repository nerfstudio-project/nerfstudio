  
from hloc.utils.read_write_model import write_model, read_model
from hloc.utils.read_write_model import Camera, Image, Point3D
import numpy as np

if __name__ == "__main__":
    images = {}
    with open("/home/dennis.chuang/3DGS/data/homee/sai_test/colmap/sparse/0/images.txt", "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = 1 #int(elems[8])

                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                        tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))

                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    cameras = {}
    with open("/home/dennis.chuang/3DGS/data/homee/sai_test/colmap/sparse/0/cameras.txt", "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = 1 #int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    points3D={}
    write_model(cameras, images, points3D, "/home/dennis.chuang/3DGS/data/homee/sai_test/colmap/sparse/1", ext=".txt")
    _, _, points3D = read_model("/home/dennis.chuang/3DGS/data/homee/sai_test/colmap/sparse/0", ext=".txt")
    write_model(cameras, images, points3D, "/home/dennis.chuang/3DGS/data/homee/sai_test/colmap/sparse/0", ext=".bin")