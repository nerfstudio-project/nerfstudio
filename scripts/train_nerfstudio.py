def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("datadir", type=str, help='path to your data')
    parser.add_argument("--checkpoint", type=str, default=None, help='path to checkpoint directory (E.g. outputs/<dataset>/nerfacto/<date>/nerfstudio_models/)')
    parser.add_argument("--port", type=int, default=51224, help='port')
    parser.add_argument("--num_sample", type=int, default=500, help='Number of training images to sample from')
    parser.add_argument("--save_latest", type=bool, default=True, help='Save only the latest checkpoint')
    parser.add_argument("--downscale_factor", type=float, default=None, help='Downscale factor')
    parser.add_argument("--pose_optimizer", type=str, default="off", help='Pose optimizer mode')
    parser.add_argument("--near_plane", type=float, default=0.1, help='Near plane')
    parser.add_argument("--far_plane", type=float, default=1000.0, help='Far plane')

    return parser

def construct_command(data_folder, checkpoint, downscale_factor, port, num_sample, save_latest, pose_optimizer, near_plane, far_plane):
    command_list = ["ns-train nerfacto"]
    if port is not None:
        command_list.append(f"--vis viewer --viewer.websocket-port={port}")
    if num_sample is not None:
        command_list.append(f"--pipeline.datamanager.train-num-images-to-sample-from {num_sample}")
    command_list.append(f"--data {data_folder}")
    if save_latest is not None:
        command_list.append(f"--save-only-latest-checkpoint {save_latest}")
    if downscale_factor is not None:
        command_list.append(f"--downscale-factor {downscale_factor}")
    if checkpoint is not None:
        command_list.append(f"--load-dir {checkpoint}")
    command_list.append(f"--pipeline.datamanager.camera-optimizer.mode {pose_optimizer}")
    command_list.append(f"--logging.profiler pytorch")
    command_list.append(f"--pipeline.model.background_color black")
    command_list.append(f"--pipeline.model.near_plane {near_plane}")
    command_list.append(f"--pipeline.model.far_plane {far_plane}")
    command = " ".join(command_list)
    return command

if __name__ == '__main__':

    parser = config_parser()
    args = parser.parse_args()

    import os

    data_folder = args.datadir
    checkpoint = args.checkpoint
    downscale_factor = args.downscale_factor
    port = args.port
    num_sample = args.num_sample
    save_latest = args.save_latest
    pose_optimizer = args.pose_optimizer
    near_plane = args.near_plane
    far_plane = args.far_plane

    # Check if the data folder exists and is a directory
    if os.path.exists(data_folder) and os.path.isdir(data_folder):
        # Run the command using os.system
        command = construct_command(data_folder, checkpoint, downscale_factor, port, num_sample, save_latest, pose_optimizer, near_plane, far_plane)
        
        print(command)
        os.system(command)
        # # Check if the command gets killed due to reasons other than user interrupt
        # if os.system(command) != 0:
        #     # reduce the number of images to sample from
        #     num_sample = int(num_sample/2)
        #     # Run the command again
        #     command = construct_command(data_folder, checkpoint, downscale_factor, port, num_sample, save_latest, pose_optimizer, near_plane, far_plane)
        #     print(command)
        #     os.system(command)
    else:
        # Print an error message if the data folder is invalid
        print("Invalid data folder path.")