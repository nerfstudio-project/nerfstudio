"""
Code to train model.
"""
import copy
import logging
import os
from pydoc import locate
from time import time
from typing import Dict

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from mattport.nerf.dataset.collate import CollateIterDataset, collate_batch_size_one
from mattport.nerf.dataset.image_dataset import ImageDataset, collate_batch
from mattport.nerf.dataset.utils import DatasetInputs, get_dataset_inputs
from mattport.nerf.optimizers import Optimizers
from mattport.utils import profiler, stats_tracker, writer
from mattport.utils.decorators import check_main_thread

logging.getLogger("PIL").setLevel(logging.WARNING)


class Trainer:
    """Training class"""

    def __init__(self, config: DictConfig, local_rank: int = 0, world_size: int = 1):
        """_summary_

        Args:
            config (DictConfig): _description_
            local_rank (int, optional): _description_. Defaults to 0.
            world_size (int, optional): _description_. Defaults to 1.
            cpu (bool, optional): Whether or not to use the CPU.
        """
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        # dataset variables
        self.train_image_dataset = None
        self.train_dataset = None
        self.train_dataloader = None
        self.val_image_dataset = None
        self.val_image_intrinsics = None
        self.val_image_camera_to_world = None
        # model variables
        self.graph = None
        self.optimizers = None
        self.start_step = 0
        # logging variables
        stats_tracker.setup_stats_tracker(config)
        writer.setup_event_writers(config.logging.writer)
        profiler.setup_profiler(config.logging)
        self.device = "cpu" if self.world_size == 0 else f"cuda:{self.local_rank}"

    def setup(self, test_mode=False):
        """Setup the Trainer by calling other setup functions."""
        dataset_inputs_train = get_dataset_inputs(**self.config.data.dataset, split="train")
        if test_mode:
            dataset_inputs_eval = get_dataset_inputs(**self.config.data.dataset, split="test")
        else:
            config_data_dataset_val = copy.deepcopy(self.config.data.dataset)
            config_data_dataset_val.downscale_factor = self.config.data.val_downscale_factor
            dataset_inputs_eval = get_dataset_inputs(**config_data_dataset_val, split="val")
        self.setup_dataset_train(dataset_inputs_train)
        self.setup_dataset_eval(dataset_inputs_eval)
        self.setup_graph(dataset_inputs_train)

    def collate_fn(self, batch_list):
        """TODO(ethan): I need to replace this.
        I'm only using this for multiprocess pickle issues for now.
        """
        return collate_batch(batch_list, self.config.data.dataloader.num_rays_per_batch, keep_full_image=False)

    @profiler.time_function
    def setup_dataset_eval(self, dataset_inputs: DatasetInputs):
        """Helper method to load test or val dataset based on test/train mode"""
        self.val_image_dataset = ImageDataset(
            image_filenames=dataset_inputs.image_filenames,
            downscale_factor=dataset_inputs.downscale_factor,
            alpha_color=dataset_inputs.alpha_color,
        )
        self.val_image_intrinsics = dataset_inputs.intrinsics
        self.val_image_camera_to_world = dataset_inputs.camera_to_world

    @profiler.time_function
    def setup_dataset_train(self, dataset_inputs: DatasetInputs):
        """_summary_"""
        self.train_image_dataset = ImageDataset(
            image_filenames=dataset_inputs.image_filenames,
            downscale_factor=dataset_inputs.downscale_factor,
            semantics=dataset_inputs.semantics,
            alpha_color=dataset_inputs.alpha_color,
        )
        self.train_dataset = CollateIterDataset(
            self.train_image_dataset,
            collate_fn=self.collate_fn,
            num_samples_to_collate=self.config.data.dataloader.num_images_to_sample_from,
            num_times_to_repeat=self.config.data.dataloader.num_times_to_repeat_images,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=1,
            num_workers=self.config.data.dataloader.num_workers,
            collate_fn=collate_batch_size_one,
            pin_memory=True,
        )

    @profiler.time_function
    def setup_graph(self, dataset_inputs: DatasetInputs):
        """Setup the graph. The dataset inputs should be set with the training data.

        Args:
            dataset_inputs (DatasetInputs): The inputs which will be used to define the camera parameters.
        """
        # hydra instantiate fails because of scene_bounds, so we are doing the following instead
        kwargs = {k: v for k, v in self.config.graph.network.items() if k != "_target_"}
        graph_class = locate(self.config.graph.network._target_)  # pylint: disable=protected-access
        self.graph = graph_class(
            **kwargs,
            intrinsics=dataset_inputs.intrinsics,
            camera_to_world=dataset_inputs.camera_to_world,
            scene_bounds=dataset_inputs.scene_bounds,
            stuff_classes=dataset_inputs.semantics.stuff_classes,
        )
        self.graph.to(self.device)

        self.setup_optimizers()  # NOTE(ethan): can this be before DDP?

        if self.config.graph.resume_train.load_dir:
            self.load_checkpoint(self.config.graph.resume_train)

        if self.world_size > 1:
            self.graph = DDP(self.graph, device_ids=[self.local_rank])
            dist.barrier(device_ids=[self.local_rank])

    def setup_optimizers(self):
        """_summary_"""
        self.optimizers = Optimizers(self.config.graph.param_groups, self.graph.get_param_groups())

    def load_checkpoint(self, load_config: DictConfig) -> int:
        """Load the checkpoint from the given path

        Args:
            load_path (str): path from which to load the model

        Returns:
            int: step iteration associated with the loaded checkpoint
        """
        load_path = os.path.join(load_config.load_dir, f"step-{load_config.load_step:09d}.ckpt")
        assert os.path.exists(load_path), f"Checkpoint {load_path} does not exist"
        loaded_state = torch.load(load_path, map_location="cpu")
        self.graph.load_state_dict({key.replace("module.", ""): value for key, value in loaded_state["model"].items()})
        for k, v in loaded_state["optimizers"].items():
            self.optimizers.optimizers[k].load_state_dict(v)
        self.start_step = loaded_state["step"] + 1
        logging.info("done loading checkpoint from %s", load_path)

    @check_main_thread
    def save_checkpoint(self, output_dir: str, step: int) -> None:
        """Save the model and optimizers

        Args:
            output_dir (str): directory to save the checkpoint
            step (int): number of steps in training for given checkpoint
            model (Graph): Graph model to be saved
            optimizers (Optimizers): Optimizers to be saved
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ckpt_path = os.path.join(output_dir, f"step-{step:09d}.ckpt")
        torch.save(
            {
                "step": step,
                "model": self.graph.module.state_dict() if hasattr(self.graph, "module") else self.graph.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
            },
            ckpt_path,
        )

    @classmethod
    def get_aggregated_loss(cls, loss_dict: Dict[str, torch.tensor]):
        """Returns the aggregated losses and the scalar for calling .backwards() on.
        # TODO: move this out to another file/class/etc.
        """
        loss_sum = 0.0
        for loss_name in loss_dict.keys():
            # TODO(ethan): add loss weightings here from a config
            loss_sum += loss_dict[loss_name]
        return loss_sum

    def train(self) -> None:
        """_summary_"""
        # self.graph.train()
        # assert self.graph.training, "Graph is not in train mode!"
        train_start = time()
        num_iterations = self.config.graph.max_num_iterations
        iter_dataset = iter(self.train_dataloader)
        for i, step in enumerate(range(self.start_step, self.start_step + num_iterations)):
            data_start = time()
            batch = next(iter_dataset)
            stats_tracker.update_stats(
                {"name": stats_tracker.Stats.ITER_LOAD_TIME, "start_time": data_start, "end_time": time(), "step": step}
            )

            iter_start = time()
            loss_dict = self.train_iteration(batch, step)
            stats_tracker.update_stats(
                {
                    "name": stats_tracker.Stats.RAYS_PER_SEC,
                    "start_time": iter_start,
                    "end_time": time(),
                    "step": step,
                    "batch_size": batch["indices"].shape[0],
                },
            )
            stats_tracker.update_stats(
                {
                    "name": stats_tracker.Stats.ITER_TRAIN_TIME,
                    "start_time": iter_start,
                    "end_time": time(),
                    "step": step,
                }
            )

            if step != 0 and step % self.config.logging.steps_per_log == 0:
                writer.write_scalar_dict(scalar_dict=loss_dict, step=step, group="Loss", prefix="train-")
                # TODO: add the learning rates to tensorboard/logging
            if step != 0 and self.config.graph.steps_per_save and step % self.config.graph.steps_per_save == 0:
                self.save_checkpoint(self.config.graph.model_dir, step)
            if step % self.config.graph.steps_per_test == 0:  # NOTE(ethan): we should still run this in dry-run mode!
                for image_idx in self.config.data.val_image_indices:
                    _ = self.test_image(image_idx=image_idx, step=step)
            stats_tracker.print_stats(i / num_iterations)

        stats_tracker.update_stats(
            {"name": stats_tracker.Stats.TOTAL_TRAIN_TIME, "start_time": train_start, "end_time": time()}
        )
        stats_tracker.print_stats(1.0)

    @profiler.time_function
    def train_iteration(self, batch: dict, step: int):
        """Run one iteration with a batch of inputs."""
        # move batch to correct device
        ray_indices = batch["indices"].to(self.device)
        _, loss_dict = self.graph(ray_indices, batch=batch)
        loss = loss_dict["aggregated_loss"]
        self.optimizers.zero_grad_all()
        loss.backward()
        self.optimizers.optimizer_step_all()
        self.optimizers.scheduler_step_all(step)
        return loss_dict

    @profiler.time_function
    def test_image(self, image_idx, step):
        """Test a specific image."""
        self.graph.eval()
        intrinsics = self.val_image_intrinsics[image_idx]
        camera_to_world = self.val_image_camera_to_world[image_idx]
        chunk_size = 1024
        training_camera_index = image_idx  # TODO(ethan): change this because training and test should not be the same
        outputs = self.graph.get_outputs_for_camera(
            intrinsics, camera_to_world, chunk_size=chunk_size, training_camera_index=training_camera_index
        )
        image = self.val_image_dataset[image_idx]["image"].to(self.device)
        psnr = self.graph.log_test_image_outputs(image_idx, step, image, outputs)
        self.graph.train()
        return psnr
