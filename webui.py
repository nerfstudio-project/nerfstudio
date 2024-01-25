import subprocess
import tkinter as tk
from dataclasses import asdict
from tkinter import filedialog
from typing import Literal

import gradio as gr

from nerfstudio.configs import dataparser_configs as dc, method_configs as mc


def run_ns_train_realtime(cmd):
    try:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        for line in iter(process.stdout.readline, ""):
            print(line, end="")

        return_code = process.wait()

        if return_code != 0:
            return f"Error: ns-train returned non-zero exit code {return_code}"

        return "ns-train completed successfully"
    except Exception as e:
        return f"Error: {e}"


def run(data_path, method, max_num_iterations, steps_per_save, data_parser):
    # generate the command
    model_args = get_model_args(method) + " --steps_per_save {steps_per_save} --max_num_iterations {max_num_iterations}"
    data_parser_args = get_data_parser_args(data_parser)
    cmd = f"ns-train {method} {model_args} --data {data_path} {data_parser} {data_parser_args}"
    # run the command
    result = run_ns_train_realtime(cmd)
    return result


def get_model_args(method):
    cmd = ""
    args = model_args[method]
    for arg in args:
        key, value = arg.label, arg.value
        print(key, value)
        # change key to --pipeline.model.{key}
        key = key.replace("_", "-")
        cmd += f"--pipeline.model.{key} {value} "
    # remove the last space
    cmd = cmd[:-1]
    return cmd


def get_data_parser_args(dataparser):
    cmd = ""
    args = dataparser_args[dataparser]
    for arg in args:
        key, value = arg.label, arg.value
        print(key, value)
        # change key to --pipeline.model.{key}
        key = key.replace("_", "-")
        cmd += f"--pipeline.datamanager.{key} {value} "
    # remove the last space
    cmd = cmd[:-1]
    return cmd


def get_model_description(method):
    return mc.all_descriptions[method]


def generate_args(config, visible=True):
    config_dict = asdict(config)
    config_inputs = []
    # print(config_dict)
    for key, value in config_dict.items():
        # if type is float, then add a textbox
        if isinstance(value, float):
            config_inputs.append(gr.Textbox(label=key, lines=1, value=value, visible=visible, interactive=True))
        # if type is bool, then add a checkbox
        elif isinstance(value, bool):
            config_inputs.append(gr.Checkbox(label=key, value=value, visible=visible, interactive=True))
        # if type is int, then add a number
        elif isinstance(value, int):
            config_inputs.append(gr.Textbox(label=key, lines=1, value=value, visible=visible, interactive=True))
        # if type is Literal, then add a radio
        elif hasattr(value, "__origin__") and value.__origin__ is Literal:
            print(value.__args__)
            config_inputs.append(gr.Radio(choices=value.__args__, label=key, visible=visible, interactive=True))
        # if type is str, then add a textbox
        elif isinstance(value, str):
            config_inputs.append(gr.Textbox(label=key, lines=1, value=value, visible=visible, interactive=True))
        else:
            continue
    # print(config_inputs)
    return config_inputs


def vis_data_parser_args(dataparser):
    # print(group_keys)
    # print(dataparser_args)
    idx = dp_group_idx[dataparser]
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
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title="Select Folder")
    return folder_path


dataparser_groups = []  # keep track of the dataparser groups
dp_group_idx = {}  # keep track of the dataparser group index
dataparser_args = {}  # keep track of the dataparser args

model_groups = []  # keep track of the model groups
model_group_idx = {}  # keep track of the model group index
model_args = {}

with gr.Blocks() as demo:
    run_button = gr.Button(
        value="Run",
    )

    # input data path
    with gr.Row():
        max_num_iterations = gr.Slider(minimum=0, maximum=50000, step=100, label="Max Num Iterations", value=30000)
        steps_per_save = gr.Slider(minimum=0, maximum=10000, step=100, label="Steps Per Save", value=2000)
    with gr.Row():
        data_path = gr.Textbox(label="Data Path", lines=1, placeholder="Path to the data folder", scale=4)
        browse_button = gr.Button(value="Browse", scale=1)
        browse_button.click(browse, None, outputs=data_path)
        gr.ClearButton(components=[data_path], scale=1)

    with gr.Row():
        with gr.Column():
            method = gr.Radio(choices=mc.all_descriptions.keys(), label="Method")
            description = gr.Textbox(label="Description", visible=True)
            method.change(fn=get_model_description, inputs=method, outputs=description)
        dataparser = gr.Radio(choices=dc.all_dataparsers.keys(), label="Data Parser")

    with gr.Accordion("Data Parser Args"):
        for key, value in dc.all_dataparsers.items():
            with gr.Group(visible=False) as group:
                dataparser_args[key] = generate_args(value, visible=True)
                dataparser_groups.append(group)
                dp_group_idx[key] = len(dataparser_groups) - 1

        # when the dataparser changes, make the corresponding dataparser args visible
        dataparser.change(fn=vis_data_parser_args, inputs=dataparser, outputs=dataparser_groups)

    with gr.Accordion("Model Args"):
        for key, value in mc.all_descriptions.items():
            with gr.Group(visible=False) as group:
                if key in mc.method_configs:
                    model_config = mc.method_configs[key].pipeline.model
                    model_args[key] = generate_args(model_config, visible=True)
                    model_groups.append(group)
                    model_group_idx[key] = len(model_groups) - 1
        # when the model changes, make the corresponding model args visible
        method.change(fn=vis_model_args, inputs=method, outputs=model_groups)

    run_button.click(run, inputs=[data_path, method, dataparser])

demo.launch()
