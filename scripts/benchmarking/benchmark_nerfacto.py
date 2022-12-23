"""
Process the Mipnerf360 dataset.
"""

from nerfstudio.utils.scripts import run_command

import GPUtil
import time
import threading


capture_names = [
    "Egypt",
    "person",
    "kitchen",
    "plane",
    "dozer",
    "floating-tree",
    "aspen",
    "stump",
    "sculpture",
    "Giannini-Hall",
]
table_rows = [
    ("nerfacto", "--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance True"),
    (
        "w/o-pose",
        "--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance True --pipeline.datamanager.camera-optimizer.mode off",
    ),
    (
        "w/o-app",
        "--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance False --pipeline.model.use-appearance-embedding False",
    ),
    (
        "w/o-pose-app",
        "--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance False --pipeline.datamanager.camera-optimizer.mode off --pipeline.model.use-appearance-embedding False",
    ),
    (
        "1-prop-network",
        '--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance True --pipeline.model.num-proposal-samples-per-ray "256" --pipeline.model.num_proposal_iterations 1',
    ),
    (
        "l2-contraction",
        "--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance True --pipeline.model.scene-contraction-norm l2",
    ),
    (
        "shared-prop-network",
        "--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance True --pipeline.model.use-same-proposal-network True",
    ),
    (
        "random-background-color",
        "--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance True --pipeline.model.background-color random",
    ),
]

# 30K iterations

# make a list of all the jobs that need to be fun
jobs = []
for capture_name in capture_names:
    for table_row_name, table_row_command in table_rows:
        command = " ".join(
            (
                f"ns-train nerfacto",
                "--vis wandb",
                f"--data data/nerfstudio/{capture_name}",
                "--output-dir outputs/nerfacto-ablations",
                "--trainer.steps-per-eval-batch 0 --trainer.steps-per-eval-image 0",
                "--trainer.steps-per-eval-all-images 5000 --trainer.max-num-iterations 30001",
                f"--wandb-name {capture_name}_{table_row_name}",
                f"--experiment-name {capture_name}_{table_row_name}",
                table_row_command,
            )
        )
        jobs.append(command)

while jobs:

    # check which GPUs have capacity to run these jobs
    """Returns the available GPUs."""
    gpu_devices_available = GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1)

    print("Available GPUs: ", gpu_devices_available)

    # thread list
    threads = []
    while gpu_devices_available and jobs:
        gpu = gpu_devices_available.pop(0)
        command = f"CUDA_VISIBLE_DEVICES={gpu} " + jobs.pop(0)

        def task():
            print("Starting command: ", command)
            out = run_command(command, verbose=False)
            # time.sleep(5)
            print("Finished command: ", command)

        threads.append(threading.Thread(target=task))
        threads[-1].start()

        # NOTE(ethan): need a delay otherwise the wandb/tensorboard naming is messed up
        # not sure why?
        time.sleep(5)

    # wait for all threads to finish
    for t in threads:
        t.join()

    print("Finished all threads")
