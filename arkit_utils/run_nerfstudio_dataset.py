import os
import shutil
import argparse
from pathlib import Path

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def copy_directory(src, dst, ext: str = "txt"):
    if os.path.exists(src):
        os.makedirs(dst, exist_ok=True)
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isfile(s) and item.endswith(f'.{ext}'):
                shutil.copy2(s, d)
    else:
        print(f"Warning: Source directory {src} does not exist. Skipping this copy operation.")

def main(args):
    root_path = Path(args.input_path).resolve()
    parent_name = root_path.parent.name
    output_root = root_path.parent / f"{parent_name}_nerfstudio"
    pose_method = args.method

    cmds = [
        'ns-train splatfacto ',
        f'--data {output_root} ',
        '--max-num-iterations 50000',
        '--pipeline.model.use-mesh-initialization True',
        '--pipeline.model.rasterize-mode antialiased',
        '--pipeline.model.camera-optimizer.mode SO3xR3',
        '--viewer.make-share-url True',
        'colmap',
        f'--colmap_path "colmap/{pose_method}/0"',
        '--auto_scale_poses False',
        '--center_method none',
        '--orientation_method none',
    ]

    full_cmds = ' '.join(cmds)  
    print("run nerfstudio with command: ", full_cmds)  
    os.system(full_cmds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ARKit 3DGS output for nerfstudio training.")
    parser.add_argument("--input_path", help="Path to the root directory of run_arkit_3dgs.sh output")
    parser.add_argument("--method", type=str,  default=['arkit'], help="Choose pose optimization methods")
    args = parser.parse_args()
    
    main(args)