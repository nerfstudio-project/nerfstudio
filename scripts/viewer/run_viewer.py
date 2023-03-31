#!/usr/bin/env python
"""
Starts viewer in eval mode.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import tyro
from rich.console import Console

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.viewer.server import viewer_utils

CONSOLE = Console(width=120)


@dataclass
class ViewerConfigWithoutNumRays(ViewerConfig):
    """Configuration for viewer instantiation"""

    num_rays_per_chunk: tyro.conf.Suppress[int] = -1
    start_train: tyro.conf.Suppress[bool] = False

    def as_viewer_config(self):
        """Converts the instance to ViewerConfig"""
        return ViewerConfig(**{x.name: getattr(self, x.name) for x in fields(self)})


@dataclass
class RunViewer:
    """Load a checkpoint and start the viewer."""

    load_config: Path
    """Path to config YAML file."""
    viewer: ViewerConfigWithoutNumRays = field(default_factory=ViewerConfigWithoutNumRays)
    """Viewer configuration"""

    def main(self) -> None:
        """Main function."""
        config, pipeline, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=None,
            test_mode="test",
        )
        num_rays_per_chunk = config.viewer.num_rays_per_chunk
        assert self.viewer.num_rays_per_chunk == -1
        config.vis = "viewer"
        config.viewer = self.viewer.as_viewer_config()
        config.viewer.num_rays_per_chunk = num_rays_per_chunk

        self._start_viewer(config, pipeline)

    def _start_viewer(self, config: TrainerConfig, pipeline: Pipeline):
        base_dir = config.get_base_dir()
        viewer_log_path = base_dir / config.viewer.relative_log_filename
        viewer_state, banner_messages = viewer_utils.setup_viewer(
            config.viewer, log_filename=viewer_log_path, datapath=pipeline.datamanager.get_datapath()
        )

        # We don't need logging, but writer.GLOBAL_BUFFER needs to be populated
        config.logging.local_writer.enable = False
        writer.setup_local_writer(config.logging, max_iter=config.max_num_iterations, banner_messages=banner_messages)

        assert viewer_state and pipeline.datamanager.train_dataset
        viewer_state.init_scene(
            dataset=pipeline.datamanager.train_dataset,
            start_train=False,
        )
        while True:
            viewer_state.vis["renderingState/isTraining"].write(False)
            self._update_viewer_state(viewer_state, pipeline)

    def _update_viewer_state(self, viewer_state: viewer_utils.ViewerState, pipeline: Pipeline):
        """Updates the viewer state by rendering out scene with current pipeline
        Returns the time taken to render scene.

        """
        # NOTE: step must be > 0 otherwise the rendering would not happen
        step = 1
        num_rays_per_batch = pipeline.datamanager.get_train_rays_per_batch()
        with TimeWriter(writer, EventName.ITER_VIS_TIME) as _:
            try:
                viewer_state.update_scene(self, step, pipeline.model, num_rays_per_batch)
            except RuntimeError:
                time.sleep(0.03)  # sleep to allow buffer to reset
                assert viewer_state.vis is not None
                viewer_state.vis["renderingState/log_errors"].write(
                    "Error: GPU out of memory. Reduce resolution to prevent viewer from crashing."
                )

    def save_checkpoint(self, *args, **kwargs):
        """
        Mock method because we pass this instance to viewer_state.update_scene
        """


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RunViewer).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RunViewer)  # noqa
