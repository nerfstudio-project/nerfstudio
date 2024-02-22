import os
import subprocess
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog
from typing import Literal

import gradio as gr

from nerfstudio.configs import dataparser_configs as dc, method_configs as mc


# from nerfstudio.scripts import train
def get_shell_command():
    if os.name == "nt":  # Windows
        return ["cmd.exe"]
    else:  # POSIX (Linux, Unix, etc.)
        return ["/bin/bash"]


def run(data_path, method, max_num_iterations, steps_per_save, data_parser, visualizer):
    # generate the command
    cmd = generate_cmd(data_path, method, max_num_iterations, steps_per_save, data_parser, visualizer)
    # run the command
    process = subprocess.Popen(get_shell_command(), stdin=subprocess.PIPE, text=True)
    process.stdin.write(cmd + "\n")
    process.stdin.flush()
    import time

    time.sleep(1)
    process.stdin.close()
    process.wait()
    # generate the cofig, useless for now
    # config = mc.all_methods[method]
    # config.data = Path(data_path)
    # config.max_num_iterations = max_num_iterations
    # config.steps_per_save = steps_per_save
    # config.vis = visualizer
    # config.pipeline.datamanager.dataparser = dc.all_dataparsers[data_parser]
    # for key, value in dataparser_args.items():
    #     setattr(config.pipeline.datamanager.dataparser, key, value)
    # for key, value in model_args.items():
    #     setattr(config.pipeline.model, key, value)
    # train.main(config)
    return cmd


def run_vis(config_path):
    cmd = generate_vis_cmd(config_path)
    # run the command
    process = subprocess.Popen(get_shell_command(), stdin=subprocess.PIPE, text=True)
    process.stdin.write(cmd + "\n")
    process.stdin.flush()
    import time

    time.sleep(1)
    process.stdin.close()
    process.wait()
    return cmd


def generate_cmd(data_path, method, max_num_iterations, steps_per_save, data_parser, visualizer):
    # generate the command
    if data_parser == "":
        raise gr.Error("Please select a data parser")
    if method == "":
        raise gr.Error("Please select a method")
    if data_path == "":
        raise gr.Error("Please select a data path")
    if visualizer == "":
        raise gr.Error("Please select a visualizer")
    # this only works on windows
    cmd = f"ns-train {method} {model_args_cmd} --vis {visualizer} --max-num-iterations {max_num_iterations} \
    --steps-per-save {steps_per_save} --data {data_path} {data_parser} {dataparser_args_cmd}"
    # run the command
    # result = run_ns_train_realtime(cmd)
    print(cmd)
    return cmd


def generate_vis_cmd(config_path):
    # generate the command
    if config_path == "":
        raise gr.Error("Please select a config path")
    # this only works on windows
    cmd = f"ns-viewer --load-config {config_path}"
    # run the command
    # result = run_ns_train_realtime(cmd)
    print(cmd)
    return cmd


def check(data_path, method, data_parser, visualizer):
    if data_path == "":
        return "Please select a data path"
    elif method == "":
        return "Please select a method"
    elif data_parser == "":
        return "Please select a data parser"
    elif visualizer == "":
        return "Please select a visualizer"
    else:
        return "Running..."


def get_model_args(method, *args):
    global model_args_cmd
    global model_args
    temp_args = {}
    args = list(args)
    cmd = ""
    values = args[model_arg_idx[method][0] : model_arg_idx[method][1]]
    names = model_arg_names[model_arg_idx[method][0] : model_arg_idx[method][1]]
    for key, value in zip(names, values):
        cmd += f"--pipeline.model.{key} {value} "
        temp_args[key] = value
    # remove the last space
    model_args_cmd = cmd[:-1]
    model_args = temp_args


def get_data_parser_args(dataparser, *args):
    global dataparser_args_cmd
    global dataparser_args
    temp_args = {}
    args = list(args)
    cmd = ""
    names = dataparser_arg_names[dataparser_arg_idx[dataparser][0] : dataparser_arg_idx[dataparser][1]]
    values = args[dataparser_arg_idx[dataparser][0] : dataparser_arg_idx[dataparser][1]]
    for key, value in zip(names, values):
        # change key to --{key}
        cmd += f"--{key} {value} "
        temp_args[key] = value
    # remove the last space
    dataparser_args_cmd = cmd[:-1]
    dataparser_args = temp_args


def get_model_description(method):
    return mc.all_descriptions[method]


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


def vis_data_parser_args(dataparser):
    # print(group_keys)
    # print(dataparser_args)
    idx = dataparser_group_idx[dataparser]
    # if the dataparser is not the current one, then hide the dataparser args
    update_info = [gr.update(visible=False)] * len(dataparser_groups)
    update_info[idx] = gr.update(visible=True)
    return update_info


def vis_model_args(method):
    if method not in model_group_idx.keys():
        return [gr.update(visible=False)] * len(model_groups)

    idx = model_group_idx[method]
    # if the method is not the current one, then hide the model args
    update_info = [gr.update(visible=False)] * len(model_groups)
    update_info[idx] = gr.update(visible=True)
    return update_info


def browse():
    if os.name == "nt":
        root = tk.Tk()
        root.wm_attributes("-topmost", 1)
        root.withdraw()  # Hide the main window
        root.lift()  # Move to the top of all windows
        folder_path = filedialog.askdirectory(title="Select Folder")
        root.destroy()
    else:
        # not supported on linux
        folder_path = ""
        raise gr.Error("Not supported on linux,please input the path manually")
    return folder_path


def browse_cfg():
    if os.name == "nt":
        # select a file ending with .yml
        root = tk.Tk()
        root.wm_attributes("-topmost", 1)
        root.withdraw()  # Hide the main window
        root.lift()  # Move to the top of all windows
        folder_path = filedialog.askopenfilename(title="Select Config", filetypes=[("YAML files", "*.yml")])
        root.destroy()
    else:
        # not supported on linux
        folder_path = ""
        raise gr.Error("Not supported on linux,please input the path manually")
    return folder_path


model_args_cmd = ""
dataparser_args_cmd = ""
model_args = {}
dataparser_args = {}

dataparser_groups = []  # keep track of the dataparser groups
dataparser_group_idx = {}  # keep track of the dataparser group index
dataparser_arg_list = []  # gr components for the dataparser args
dataparser_arg_names = []  # keep track of the dataparser args names
dataparser_arg_idx = {}  # record the start and end index of the dataparser args

model_groups = []  # keep track of the model groups
model_group_idx = {}  # keep track of the model group index
model_arg_list = []  # gr components for the model args
model_arg_names = []  # keep track of the model args names
model_arg_idx = {}  # record the start and end index of the model args


with gr.Blocks() as demo:
    with gr.Tab(label="Train"):
        status = gr.Textbox(label="Status", lines=1, placeholder="Waiting for input")
        with gr.Row():
            run_button = gr.Button(value="Run Train", variant="primary")
            cmd_button = gr.Button(value="Show Command")
            viser_button = gr.Button(value="Open Viser", link="http://localhost:7007/")

        # TODO: Add a progress bar
        # TODO: Make the run button disabled when the process is running

        # input data path
        with gr.Row():
            max_num_iterations = gr.Slider(minimum=0, maximum=50000, step=100, label="Max Num Iterations", value=30000)
            steps_per_save = gr.Slider(minimum=0, maximum=10000, step=100, label="Steps Per Save", value=2000)
        if os.name == "nt":
            with gr.Row():
                data_path = gr.Textbox(label="Data Path", lines=1, placeholder="Path to the data folder", scale=4)
                browse_button = gr.Button(value="Browse", scale=1)
                browse_button.click(browse, None, outputs=data_path)
                gr.ClearButton(components=[data_path], scale=1)

        else:
            with gr.Row():
                data_path = gr.Textbox(label="Data Path", lines=1, placeholder="Path to the data folder", scale=5)
                choose_button = gr.Button(value="Submit", scale=1)

            def get_folder_path(x):
                if len(x) > 0:
                    x = x[0]
                x = Path(x)
                return str(x)

            file_explorer = gr.FileExplorer(label="Browse", scale=1, root_dir="../", file_count="multiple", height=300)
            choose_button.click(get_folder_path, inputs=file_explorer, outputs=data_path)

        # file_explorer.change(fn=get_folder_path, inputs=file_explorer, outputs=data_path)

        with gr.Row():
            with gr.Column():
                method = gr.Radio(choices=mc.all_descriptions.keys(), label="Method")
                description = gr.Textbox(label="Description", visible=True)
                method.change(fn=get_model_description, inputs=method, outputs=description)
            with gr.Column():
                dataparser = gr.Radio(choices=dc.all_dataparsers.keys(), label="Data Parser")
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
            for key, value in mc.all_descriptions.items():
                with gr.Group(visible=False) as group:
                    if key in mc.method_configs:
                        model_config = mc.method_configs[key].pipeline.model
                        generated_args, labels = generate_args(model_config, visible=True)
                        model_arg_list += generated_args
                        model_arg_names += labels
                        model_arg_idx[key] = [len(model_arg_list) - len(generated_args), len(model_arg_list)]

                        model_groups.append(group)
                        model_group_idx[key] = len(model_groups) - 1
            # when the model changes, make the corresponding model args visible
            method.change(fn=vis_model_args, inputs=method, outputs=model_groups)

        with gr.Accordion("Data Parser Config", open=False):
            for key, parser_config in dc.all_dataparsers.items():
                with gr.Group(visible=False) as group:
                    generated_args, labels = generate_args(parser_config, visible=True)
                    dataparser_arg_list += generated_args
                    dataparser_arg_names += labels
                    dataparser_arg_idx[key] = [len(dataparser_arg_list) - len(generated_args), len(dataparser_arg_list)]

                    dataparser_groups.append(group)
                    dataparser_group_idx[key] = len(dataparser_groups) - 1

            # when the dataparser changes, make the corresponding dataparser args visible
            dataparser.change(fn=vis_data_parser_args, inputs=dataparser, outputs=dataparser_groups)

        run_button.click(
            get_model_args,
            inputs=[method] + model_arg_list,
            outputs=None,
        ).then(
            get_data_parser_args,
            inputs=[dataparser] + dataparser_arg_list,
            outputs=None,
        ).then(
            check,
            inputs=[data_path, method, dataparser, visualizer],
            outputs=status,
        ).then(
            run,
            inputs=[data_path, method, max_num_iterations, steps_per_save, dataparser, visualizer],
            outputs=None,
        )

        cmd_button.click(
            get_model_args,
            inputs=[method] + model_arg_list,
            outputs=None,
        ).then(
            get_data_parser_args,
            inputs=[dataparser] + dataparser_arg_list,
            outputs=None,
        ).then(
            generate_cmd,
            inputs=[data_path, method, max_num_iterations, steps_per_save, dataparser, visualizer],
            outputs=status,
        )

    with gr.Tab(label="Visualize"):
        status = gr.Textbox(label="Status", lines=1, placeholder="Waiting for input")
        with gr.Row():
            vis_button = gr.Button(
                value="Run Viser",
                variant="primary",
            )
            vis_cmd_button = gr.Button(value="Show Command")
            viser_button = gr.Button(value="Open Viser", link="http://localhost:7007/")
        if os.name == "nt":
            with gr.Row():
                config_path = gr.Textbox(label="Config Path", lines=1, placeholder="Path to the config", scale=4)
                cfg_browse_button = gr.Button(value="Browse", scale=1)
                cfg_browse_button.click(browse_cfg, None, outputs=config_path)
                gr.ClearButton(components=[data_path], scale=1)
        else:
            with gr.Row():
                config_path = gr.Textbox(label="Config Path", lines=1, placeholder="Path to the config", scale=5)
                cfg_choose_button = gr.Button(value="Submit", scale=1)

            cfg_file_explorer = gr.FileExplorer(
                label="Browse", scale=1, root_dir="../", file_count="single", height=300, glob=".yml"
            )
            cfg_choose_button.click(lambda x: str(x), inputs=cfg_file_explorer, outputs=config_path)

        vis_button.click(
            run_vis,
            inputs=[config_path],
            outputs=status,
        )
        vis_cmd_button.click(
            generate_vis_cmd,
            inputs=[config_path],
            outputs=status,
        )


demo.launch(inbrowser=True)
