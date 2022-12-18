"""
Process the Mipnerf360 dataset.
"""

from nerfstudio.utils.scripts import run_command

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
    ("nerfacto", ""),
    ("w/o-pose", "--pipeline.datamanager.camera-optimizer.mode off"),
    ("w/o-app", "--pipeline.model.use-appearance-embedding False"),
    (
        "w/o-pose-app",
        "--pipeline.datamanager.camera-optimizer.mode off --pipeline.model.use-appearance-embedding False",
    ),
    (
        "1-prop-network",
        "--pipeline.model.num-proposal-samples-per-ray \"256\" --pipeline.model.num_proposal_iterations 1",
    ),
]

# 30K iterations

for capture_name in capture_names:
    for table_row_name, table_row_command in table_rows:

        # TODO: find an available GPU
        gpu = 0

        command = " ".join(
            (
                # f"CUDA_VISIBLE_DEVICES={gpu}",
                f"ns-train nerfacto",
                "--vis wandb",
                f"--data data/nerfstudio/{capture_name}",
                "--pipeline.eval_optimize_cameras True --pipeline.eval_optimize_appearance True",
                f"--wandb-name {capture_name}-{table_row_name} --experiment-name {capture_name}-{table_row_name}",
                table_row_command,
            )
        )
        print(command)
