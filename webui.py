import multiprocessing
import os
import random
import subprocess
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog
from typing import Literal

import gradio as gr
import numpy as np
import tyro
import yaml
from torch import manual_seed

from nerfstudio.configs import dataparser_configs as dc, method_configs as mc
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.scripts import train
from nerfstudio.scripts.process_data import (
    ImagesToNerfstudioDataset,
    ProcessMetashape,
    ProcessODM,
    ProcessPolycam,
    ProcessRealityCapture,
    ProcessRecord3D,
    VideoToNerfstudioDataset,
)
from nerfstudio.utils.rich_utils import CONSOLE


def run_cmd(cmd):
    if os.name == "nt":  # Windows
        # For Windows, 'start' launches a new command prompt window
        # '/K' keeps the window open, and 'cmd.exe /c' ensures the command is executed
        subprocess.Popen(["start", "cmd.exe", "/K", cmd], shell=True)
    else:
        # For POSIX systems (Linux, macOS, etc.)
        # We try to detect the available terminal emulator and then run the command within it
        # This part might need adjustments based on the specific terminal emulator you have

        # Common terminal emulators
        terminals = [
            "x-terminal-emulator",  # Generic command for the default terminal in some Linux distributions
            "gnome-terminal",  # GNOME
            "konsole",  # KDE
            "xfce4-terminal",  # XFCE
            "xterm",  # X Window System
        ]

        terminal_found = False

        for terminal in terminals:
            if subprocess.call(["which", terminal], stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0:
                if terminal in ["gnome-terminal", "konsole", "xfce4-terminal"]:
                    # These terminals require the '-e' argument to execute the command
                    subprocess.Popen([terminal, "-e", 'bash -c "{}; exec bash"'.format(cmd)])
                else:
                    # For 'x-terminal-emulator' and 'xterm', the command can be passed directly
                    subprocess.Popen([terminal, "-e", cmd])
                terminal_found = True
                break

        if not terminal_found:
            print(
                "No suitable terminal emulator found. Please install one of the supported terminals or update the script."
            )


def generate_args(config, visible=True):
    config_dict = asdict(config)
    config_inputs = []
    config_labels = []
    # print(config_dict)
    for key, value in config_dict.items():
        # if type is float, then add a textbox
        config_labels.append(key)
        if isinstance(value, float):
            config_inputs.append(gr.Number(label=key, value=value, visible=visible, interactive=True, step=0.01))
        # if type is bool, then add a checkbox
        elif isinstance(value, bool):
            config_inputs.append(gr.Checkbox(label=key, value=value, visible=visible, interactive=True))
        # if type is int, then add a number
        elif isinstance(value, int):
            config_inputs.append(gr.Number(label=key, value=value, visible=visible, interactive=True, precision=0))
        # if type is Literal, then add a radio
        # TODO: fix this
        elif hasattr(value, "__origin__") and value.__origin__ is Literal:
            print(value.__args__)
            config_inputs.append(gr.Radio(choices=value.__args__, label=key, visible=visible, interactive=True))
        # if type is str, then add a textbox
        elif isinstance(value, str):
            config_inputs.append(gr.Textbox(label=key, lines=1, value=value, visible=visible, interactive=True))
        else:
            # erase the last one
            config_labels.pop()
            continue
    # print(config_inputs)
    return config_inputs, config_labels


def get_folder_path(x):
    if len(x) > 0:
        x = x[0]
    return str(x)


def browse_folder():
    root = tk.Tk()
    root.wm_attributes("-topmost", 1)
    root.withdraw()  # Hide the main window
    root.lift()  # Move to the top of all windows
    folder_path = filedialog.askdirectory(title="Select Folder")
    root.destroy()
    return folder_path


def browse_cfg():
    # select a file ending with .yml
    root = tk.Tk()
    root.wm_attributes("-topmost", 1)
    root.withdraw()  # Hide the main window
    root.lift()  # Move to the top of all windows
    path = filedialog.askopenfilename(title="Select Config", filetypes=[("YAML files", "*.yml")])
    root.destroy()
    return path


def browse_video():
    root = tk.Tk()
    root.wm_attributes("-topmost", 1)
    root.withdraw()
    root.lift()
    path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video files", "*.mp4")])
    root.destroy()
    return path


class WebUITrainer:
    def __init__(self):
        self.trainer = None
        self.config = None

    def train_loop(self, local_rank: int, world_size: int, config: TrainerConfig, global_rank: int = 0):
        def _set_random_seed(seed) -> None:
            """Set randomness seed in torch and numpy"""
            random.seed(seed)
            np.random.seed(seed)
            manual_seed(seed)

        _set_random_seed(config.machine.seed + global_rank)
        self.trainer = config.setup(local_rank=local_rank, world_size=world_size)
        self.trainer.setup()
        self.trainer.train()

    def main(self):
        assert self.config is not None, "Config is not set"
        if self.config.data:
            CONSOLE.log("Using --data alias for --data.pipeline.datamanager.data")
            self.config.pipeline.datamanager.data = self.config.data

        if self.config.prompt:
            CONSOLE.log("Using --prompt alias for --data.pipeline.model.prompt")
            self.config.pipeline.model.prompt = self.config.prompt

        if self.config.load_config:
            CONSOLE.log(f"Loading pre-set config from: {self.config.load_config}")
            self.config = yaml.load(self.config.load_config.read_text(), Loader=yaml.Loader)

        # quit the viewer when training is done to avoid blocking
        self.config.viewer.quit_on_train_completion = True

        self.config.set_timestamp()

        # print and save config
        self.config.print_to_terminal()
        self.config.save_config()
        train.launch(
            main_func=self.train_loop,
            num_devices_per_machine=self.config.machine.num_devices,
            device_type=self.config.machine.device_type,
            num_machines=self.config.machine.num_machines,
            machine_rank=self.config.machine.machine_rank,
            dist_url=self.config.machine.dist_url,
            config=self.config,
        )


class TrainTab(WebUITrainer):
    def __init__(self, **kwargs):
        super().__init__()
        self.root_dir = kwargs.get("root_dir", "./")  # root directory
        self.run_in_new_terminal = kwargs.get("run_in_new_terminal", False)  # run in new terminal

        self.model_args_cmd = ""
        self.dataparser_args_cmd = ""
        self.model_args = {}
        self.dataparser_args = {}

        self.dataparser_groups = []  # keep track of the dataparser groups
        self.dataparser_group_idx = {}  # keep track of the dataparser group index
        self.dataparser_arg_list = []  # gr components for the dataparser args
        self.dataparser_arg_names = []  # keep track of the dataparser args names
        self.dataparser_arg_idx = {}  # record the start and end index of the dataparser args

        self.model_groups = []  # keep track of the model groups
        self.model_group_idx = {}  # keep track of the model group index
        self.model_arg_list = []  # gr components for the model args
        self.model_arg_names = []  # keep track of the model args names
        self.model_arg_idx = {}  # record the start and end index of the model args

    def setup_ui(self):
        with gr.Tab(label="Train"):
            status = gr.Textbox(label="Status", lines=1, placeholder="Waiting")
            with gr.Row():
                run_button = gr.Button(value="Train", variant="primary")
                stop_button = gr.Button(value="Stop", variant="stop")
                pause_button = gr.Button(value="Pause", variant="secondary")
                cmd_button = gr.Button(value="Show Command")
                gr.Button(value="Open Viser", link="http://localhost:7007/")

            with gr.Row():
                max_num_iterations = gr.Slider(
                    minimum=0, maximum=50000, step=100, label="Max Num Iterations", value=30000
                )
                steps_per_save = gr.Slider(minimum=0, maximum=10000, step=100, label="Steps Per Save", value=2000)

            if os.name == "nt":
                with gr.Row():
                    data_path = gr.Textbox(label="Data Path", lines=1, placeholder="Path to the data folder", scale=4)
                    browse_button = gr.Button(value="Browse", scale=1)
                    browse_button.click(browse_folder, None, outputs=data_path)
                    gr.ClearButton(components=[data_path], scale=1)
            else:
                with gr.Row():
                    data_path = gr.Textbox(label="Data Path", lines=1, placeholder="Path to the data folder", scale=5)
                    choose_button = gr.Button(value="Submit", scale=1)
                with gr.Row():
                    file_explorer = gr.FileExplorer(
                        label="Browse", scale=1, root_dir=self.root_dir, file_count="multiple", height=300
                    )
                    file_explorer.change(get_folder_path, inputs=file_explorer, outputs=data_path)
                    choose_button.click(lambda x: str(x), inputs=data_path, outputs=data_path)

            with gr.Row():
                with gr.Column():
                    method = gr.Radio(choices=list(mc.descriptions.keys()), label="Method")
                    description = gr.Textbox(label="Description", visible=True)
                    method.change(self.get_model_description, inputs=method, outputs=description)
                with gr.Column():
                    dataparser = gr.Radio(choices=list(dc.dataparsers.keys()), label="Data Parser")
                    visualizer = gr.Radio(
                        choices=[
                            "viewer",
                            "wandb",
                            "tensorboard",
                            "comet",
                            "viewer+wandb",
                            "viewer+tensorboard",
                            "viewer+comet",
                            "viewer_legacy",
                        ],
                        label="Visualizer",
                        value="viewer",
                    )

            with gr.Accordion("Model Config", open=False):
                for key, value in mc.descriptions.items():
                    with gr.Group(visible=False) as group:
                        if key in mc.method_configs:
                            model_config = mc.method_configs[key].pipeline.model  # type: ignore
                            generated_args, labels = generate_args(model_config, visible=True)
                            self.model_arg_list += generated_args
                            self.model_arg_names += labels
                            self.model_arg_idx[key] = [
                                len(self.model_arg_list) - len(generated_args),
                                len(self.model_arg_list),
                            ]
                            self.model_groups.append(group)
                            self.model_group_idx[key] = len(self.model_groups) - 1
                method.change(self.update_model_args_visibility, inputs=method, outputs=self.model_groups)

            with gr.Accordion("Data Parser Config", open=False):
                for key, parser_config in dc.dataparsers.items():
                    with gr.Group(visible=False) as group:
                        generated_args, labels = generate_args(parser_config, visible=True)
                        self.dataparser_arg_list += generated_args
                        self.dataparser_arg_names += labels
                        self.dataparser_arg_idx[key] = [
                            len(self.dataparser_arg_list) - len(generated_args),
                            len(self.dataparser_arg_list),
                        ]
                        self.dataparser_groups.append(group)
                        self.dataparser_group_idx[key] = len(self.dataparser_groups) - 1
                dataparser.change(
                    self.update_dataparser_args_visibility, inputs=dataparser, outputs=self.dataparser_groups
                )

            run_button.click(
                self.get_model_args,
                inputs=[method] + self.model_arg_list,
                outputs=None,
            ).then(
                self.get_data_parser_args,
                inputs=[dataparser] + self.dataparser_arg_list,  # type: ignore
                outputs=None,
            ).then(
                self.run_train,
                inputs=[data_path, method, max_num_iterations, steps_per_save, dataparser, visualizer],
                outputs=status,
            )

            update_event = run_button.click(
                self.update_status,
                inputs=[data_path, method, dataparser, visualizer],
                outputs=status,
                every=1,
            )

            pause_button.click(self.pause, inputs=None, outputs=pause_button)

            stop_button.click(self.stop, inputs=None, outputs=status, cancels=[update_event])

            cmd_button.click(
                self.get_model_args,
                inputs=[method] + self.model_arg_list,
                outputs=None,
            ).then(
                self.get_data_parser_args,
                inputs=[dataparser] + self.dataparser_arg_list,  # type: ignore
                outputs=None,
            ).then(
                self.generate_cmd,
                inputs=[data_path, method, max_num_iterations, steps_per_save, dataparser, visualizer],
                outputs=status,
            )

    def update_status(self, data_path, method, data_parser, visualizer):
        if self.trainer is not None and self.trainer.step != 0:
            return "Step: " + str(self.trainer.step)
        else:
            check = self.check(data_path, method, data_parser, visualizer)
            if check is not None:
                return check
            return "Initializing..."

    def pause(self):
        if self.trainer is not None:
            if self.trainer.training_state == "paused":
                self.trainer.training_state = "training"
                return "Pause"
            else:
                self.trainer.training_state = "paused"
                return "Resume"
        else:
            raise gr.Error("Please run the training first")

    def stop(self):
        # stop the training
        # FIXME: this will not release the resources
        if self.trainer is not None:
            config_path = self.config.get_base_dir() / "config.yml"
            ckpt_path = self.trainer.checkpoint_dir
            self.trainer.early_stop = True
            print("Early Stopped. Config and checkpoint saved at " + str(config_path) + " and " + str(ckpt_path))
            return "Early Stopped. Config and checkpoint saved at " + str(config_path) + " and " + str(ckpt_path)
        else:
            raise gr.Error("Please run the training first")

    def run_train(self, data_path, method, max_num_iterations, steps_per_save, data_parser, visualizer):
        cmd = self.generate_cmd(data_path, method, max_num_iterations, steps_per_save, data_parser, visualizer)
        print(cmd)
        # run the command
        if self.run_in_new_terminal:
            run_cmd(cmd)
        else:
            config = mc.all_methods[method]
            config.data = Path(data_path)
            config.max_num_iterations = max_num_iterations
            config.steps_per_save = steps_per_save
            config.vis = visualizer
            config.pipeline.datamanager.dataparser = dc.all_dataparsers[data_parser]
            for key, value in self.dataparser_args.items():
                setattr(config.pipeline.datamanager.dataparser, key, value)
            for key, value in self.model_args.items():
                setattr(config.pipeline.model, key, value)
            # from nerfstudio.scripts import train
            self.config = config
            self.main()
            config_path = self.config.get_base_dir() / "config.yml"
            assert self.trainer is not None
            ckpt_path = self.trainer.checkpoint_dir

            return "Training finished. Config and checkpoint saved at " + str(config_path) + " and " + str(ckpt_path)

    def generate_cmd(self, data_path, method, max_num_iterations, steps_per_save, data_parser, visualizer):
        # generate the command
        if data_parser == "":
            raise gr.Error("Please select a data parser")
        if method == "":
            raise gr.Error("Please select a method")
        if data_path == "":
            raise gr.Error("Please select a data path")
        if visualizer == "":
            raise gr.Error("Please select a visualizer")
        cmd = f"ns-train {method} {self.model_args_cmd} --vis {visualizer} --max-num-iterations {max_num_iterations} \
        --steps-per-save {steps_per_save} --data {data_path} {data_parser} {self.dataparser_args_cmd}"
        check = self.check(data_path, method, data_parser, visualizer)
        if check is not None:
            return check
        return cmd

    def check(self, data_path, method, data_parser, visualizer):
        if data_path == "":
            return "Please select a data path"
        elif method == "":
            return "Please select a method"
        elif data_parser == "":
            return "Please select a data parser"
        elif visualizer == "":
            return "Please select a visualizer"
        else:
            return None

    def get_model_args(self, method, *args):
        temp_args = {}
        args = list(args)
        cmd = ""
        values = args[self.model_arg_idx[method][0] : self.model_arg_idx[method][1]]
        names = self.model_arg_names[self.model_arg_idx[method][0] : self.model_arg_idx[method][1]]
        for key, value in zip(names, values):
            cmd += f"--pipeline.model.{key} {value} "
            temp_args[key] = value
        # remove the last space
        self.model_args_cmd = cmd[:-1]
        self.model_args = temp_args

    def get_data_parser_args(self, dataparser, *args):
        temp_args = {}
        args = list(args)
        cmd = ""
        names = self.dataparser_arg_names[
            self.dataparser_arg_idx[dataparser][0] : self.dataparser_arg_idx[dataparser][1]
        ]
        values = args[self.dataparser_arg_idx[dataparser][0] : self.dataparser_arg_idx[dataparser][1]]
        for key, value in zip(names, values):
            # change key to --{key}
            cmd += f"--{key} {value} "
            temp_args[key] = value
        # remove the last space
        self.dataparser_args_cmd = cmd[:-1]
        self.dataparser_args = temp_args

    def get_model_description(self, method):
        return mc.descriptions[method]

    def update_dataparser_args_visibility(self, dataparser):
        # print(group_keys)
        # print(dataparser_args)
        idx = self.dataparser_group_idx[dataparser]
        # if the dataparser is not the current one, then hide the dataparser args
        update_info = [gr.update(visible=False)] * len(self.dataparser_groups)
        update_info[idx] = gr.update(visible=True)
        return update_info

    def update_model_args_visibility(self, method):
        if method not in self.model_group_idx.keys():
            return [gr.update(visible=False)] * len(self.model_groups)

        idx = self.model_group_idx[method]
        # if the method is not the current one, then hide the model args
        update_info = [gr.update(visible=False)] * len(self.model_groups)
        update_info[idx] = gr.update(visible=True)
        return update_info


dataprocessor_configs = {
    "ImagesToNerfstudioDataset": ImagesToNerfstudioDataset,
    "VideoToNerfstudioDataset": VideoToNerfstudioDataset,
    "ProcessPolycam": ProcessPolycam,
    "ProcessMetashape": ProcessMetashape,
    "ProcessRealityCapture": ProcessRealityCapture,
    "ProcessRecord3D": ProcessRecord3D,
    "ProcessODM": ProcessODM,
}


class DataProcessorTab:
    def __init__(self, **kwargs):
        super().__init__()
        self.root_dir = kwargs.get("root_dir", "./")  # root directory
        self.run_in_new_terminal = kwargs.get("run_in_new_terminal", False)  # run in new terminal

        self.dataprocessor_args = {}

        self.dataprocessor_groups = []  # keep track of the dataprocessor groups
        self.dataprocessor_group_idx = {}  # keep track of the dataprocessor group index
        self.dataprocessor_arg_list = []  # gr components for the dataprocessor args
        self.dataprocessor_arg_names = []  # keep track of the dataprocessor args names
        self.dataprocessor_arg_idx = {}  # record the start and end index of the dataprocessor args

    def setup_ui(self):
        with gr.Tab(label="Process Data "):
            status = gr.Textbox(label="Status", lines=1, placeholder="Waiting")
            with gr.Row():
                dataprocessor = gr.Radio(choices=list(dataprocessor_configs.keys()), label="Method", scale=5)
                run_button = gr.Button(value="Process", variant="primary", scale=1)

            if os.name == "nt":
                with gr.Row():
                    data_path = gr.Textbox(label="Data Path", lines=1, placeholder="Path to the data folder", scale=4)
                    browse_button = gr.Button(value="Browse Image", scale=1)
                    browse_button.click(browse_folder, None, outputs=data_path)
                    browse_video_button = gr.Button(value="Browse Video", scale=1)
                    browse_video_button.click(browse_video, None, outputs=data_path)
                    gr.ClearButton(components=[data_path], scale=1)
            else:
                with gr.Row():
                    data_path = gr.Textbox(label="Data Path", lines=1, placeholder="Path to the data folder", scale=5)
                    input_button = gr.Button(value="Submit", scale=1)
                with gr.Row():
                    file_explorer = gr.FileExplorer(
                        label="Browse", scale=1, root_dir=self.root_dir, file_count="multiple", height=300
                    )
                    file_explorer.change(get_folder_path, inputs=file_explorer, outputs=data_path)
                    input_button.click(lambda x: str(x), inputs=data_path, outputs=data_path)
                with gr.Row():
                    output_dir = gr.Textbox(
                        label="Output Path", lines=1, placeholder="Path to the output folder", scale=5
                    )
                    out_button = gr.Button(value="Submit", scale=1)
                with gr.Row():
                    file_explorer = gr.FileExplorer(
                        label="Browse", scale=1, root_dir=self.root_dir, file_count="multiple", height=300
                    )
                    file_explorer.change(get_folder_path, inputs=file_explorer, outputs=output_dir)
                    out_button.click(lambda x: str(x), inputs=output_dir, outputs=output_dir)

            # with gr.Accordion("Data Processor Config", open=False):
            #     for key, config in dataprocessor_configs.items():
            #         with gr.Group(visible=False) as group:
            #             print(config)
            #             generated_args, labels = generate_args(config, visible=True)
            #             self.dataprocessor_arg_list += generated_args
            #             self.dataprocessor_arg_names += labels
            #             self.dataprocessor_arg_idx[key] = [
            #                 len(self.dataprocessor_arg_list) - len(generated_args),
            #                 len(self.dataprocessor_arg_list),
            #             ]
            #             self.dataprocessor_groups.append(group)
            #             self.dataprocessor_group_idx[key] = len(self.dataprocessor_groups) - 1
            #     dataprocessor.change(
            #         self.update_dataprocessor_args_visibility, inputs=dataprocessor, outputs=self.dataprocessor_groups
            #     )
            run_button.click(
                self.run_dataprocessor,
                inputs=[dataprocessor, data_path, output_dir],
                outputs=status,
            )

    def update_status(self, data_path, method, data_parser, visualizer):
        if self.trainer is not None and self.trainer.step != 0:
            return "Step: " + str(self.trainer.step)
        else:
            check = self.check(data_path, method, data_parser, visualizer)
            if check is not None:
                return check
            return "Initializing..."

    def run_dataprocessor(self, datapocessor, data_path, output_dir):
        data_path = Path(data_path)
        output_dir = Path(output_dir)
        if datapocessor == "ImagesToNerfstudioDataset":
            processor = ImagesToNerfstudioDataset(data_path, output_dir)
        elif datapocessor == "VideoToNerfstudioDataset":
            processor = VideoToNerfstudioDataset(data_path, output_dir)
        elif datapocessor == "ProcessPolycam":
            processor = ProcessPolycam(data_path, output_dir)
        elif datapocessor == "ProcessMetashape":
            processor = ProcessMetashape(data_path, output_dir)
        elif datapocessor == "ProcessRealityCapture":
            processor = ProcessRealityCapture(data_path, output_dir)
        elif datapocessor == "ProcessRecord3D":
            processor = ProcessRecord3D(data_path, output_dir)
        elif datapocessor == "ProcessODM":
            processor = ProcessODM(data_path, output_dir)
        else:
            raise gr.Error("Please select a data processor")

        p = multiprocessing.Process(target=processor.main)
        p.start()
        p.join()

        return "Processing finished"

    def update_dataprocessor_args_visibility(self, dataprocessor):
        idx = self.dataprocessor_group_idx[dataprocessor]
        update_info = [gr.update(visible=False)] * len(self.dataprocessor_groups)
        update_info[idx] = gr.update(visible=True)
        return update_info


class VisualizerTab:
    def __init__(self, **kwargs):
        self.root_dir = kwargs.get("root_dir", "./")  # root directory
        self.run_in_new_terminal = kwargs.get("run_in_new_terminal", False)  # run in new terminal

    def setup_ui(self):
        with gr.Tab(label="Visualize"):
            status = gr.Textbox(label="Status", lines=1, placeholder="Waiting")
            with gr.Row():
                vis_button = gr.Button(value="Run Viser", variant="primary")
                vis_cmd_button = gr.Button(value="Show Command")
                gr.Button(value="Open Viser", link="http://localhost:7007/")

            if os.name == "nt":
                with gr.Row():
                    config_path = gr.Textbox(label="Config Path", lines=1, placeholder="Path to the config", scale=4)
                    cfg_browse_button = gr.Button(value="Browse", scale=1)
                    cfg_browse_button.click(browse_cfg, None, outputs=config_path)
                    gr.ClearButton(components=[config_path], scale=1)
            else:
                with gr.Row():
                    config_path = gr.Textbox(label="Config Path", lines=1, placeholder="Path to the config", scale=5)
                    cfg_choose_button = gr.Button(value="Submit", scale=1)
                with gr.Row():
                    cfg_file_explorer = gr.FileExplorer(
                        label="Browse",
                        scale=1,
                        root_dir=self.root_dir,
                        file_count="single",
                        height=300,
                        glob="*.yml",
                    )
                    cfg_file_explorer.change(lambda x: str(x), inputs=cfg_file_explorer, outputs=config_path)
                    cfg_choose_button.click(lambda x: str(x), inputs=config_path, outputs=config_path)

            vis_button.click(self.run_vis, inputs=[config_path], outputs=status)
            vis_cmd_button.click(self.generate_vis_cmd, inputs=[config_path], outputs=status)

    def run_vis(self, config_path):
        cmd = self.generate_vis_cmd(config_path)
        # run the command
        if self.run_in_new_terminal:
            run_cmd(cmd)
        else:
            from nerfstudio.scripts.viewer.run_viewer import RunViewer

            viewer = RunViewer()
            viewer.load_config = Path(config_path)
            tyro.extras.set_accent_color("bright_yellow")
            tyro.cli(viewer).main()
        return cmd

    def generate_vis_cmd(self, config_path):
        # generate the command
        if config_path == "":
            raise gr.Error("Please select a config path")
        # this only works on windows
        cmd = f"ns-viewer --load-config {config_path}"
        # run the command
        # result = run_ns_train_realtime(cmd)
        # print(cmd)
        return cmd

    def check(self, data_path, method, data_parser, visualizer):
        if data_path == "":
            return "Please select a data path"
        elif method == "":
            return "Please select a method"
        elif data_parser == "":
            return "Please select a data parser"
        elif visualizer == "":
            return "Please select a visualizer"
        else:
            return None


class WebUI:
    def __init__(self, **kwargs):
        super().__init__()
        self.root_dir = kwargs.get("root_dir", "./")  # root directory
        self.run_in_new_terminal = kwargs.get("run_in_new_terminal", False)  # run in new terminal
        self.demo = gr.Blocks()
        self.train_tab = TrainTab(**kwargs)
        self.visualizer_tab = VisualizerTab(**kwargs)
        self.data_processor_tab = DataProcessorTab(**kwargs)

        self.setup_ui()

    def setup_ui(self):
        with self.demo:
            self.train_tab.setup_ui()
            self.visualizer_tab.setup_ui()
            self.data_processor_tab.setup_ui()

    def launch(self, *args, **kwargs):
        self.demo.launch(*args, **kwargs)


if __name__ == "__main__":
    app = WebUI(root_dir="./", run_in_new_terminal=False)
    app.launch()
