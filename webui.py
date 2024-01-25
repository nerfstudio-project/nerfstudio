import subprocess
import tkinter as tk
from dataclasses import asdict
from tkinter import filedialog
from typing import Literal

import gradio as gr

from nerfstudio.configs import dataparser_configs as dc, method_configs as mc


def run_ns_train_realtime(cmd):
    # TODO: add direct CLI support
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
    # model_args = (
    #     get_model_args(method) + f" --steps-per-save {steps_per_save} --max-num-iterations {max_num_iterations}"
    # )
    # TODO: add data parser args, idk why it doesn't work
    # data_parser_args = get_data_parser_args(data_parser)
    # print(model_args)
    # print(dataparser_args)
    cmd = f"ns-train {method} {model_args} --max-num-iterations {max_num_iterations} --steps-per-save {steps_per_save}  --data {data_path} {data_parser}"
    # run the command
    result = run_ns_train_realtime(cmd)
    return result


def get_model_args(method, *args):
    global model_args
    args = list(args)
    cmd = ""
    values = args[model_arg_idx[method][0] : model_arg_idx[method][1]]
    names = model_arg_names[model_arg_idx[method][0] : model_arg_idx[method][1]]
    for key, value in zip(names, values):
        cmd += f"--pipeline.model.{key} {value} "
    # remove the last space
    model_args = cmd[:-1]


def get_data_parser_args(dataparser, *args):
    global dataparser_args
    args = list(args)
    cmd = ""
    names = dataparser_arg_names[dataparser_arg_idx[dataparser][0] : dataparser_arg_idx[dataparser][1]]
    values = args[dataparser_arg_idx[dataparser][0] : dataparser_arg_idx[dataparser][1]]
    for key, value in zip(names, values):
        # change key to --{key}
        cmd += f"--{key} {value} "
    # remove the last space
    dataparser_args = cmd[:-1]


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
            config_inputs.append(gr.Textbox(label=key, lines=1, value=value, visible=visible, interactive=True))
        # if type is bool, then add a checkbox
        elif isinstance(value, bool):
            config_inputs.append(gr.Checkbox(label=key, value=value, visible=visible, interactive=True))
        # if type is int, then add a number
        elif isinstance(value, int):
            config_inputs.append(gr.Number(label=key, value=value, visible=visible, interactive=True))
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
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title="Select Folder")
    return folder_path


model_args = ""
dataparser_args = ""

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
    with gr.Row():
        run_button = gr.Button(
            value="Run",
        )
        viser_button = gr.Button(value="Open Viser", link="http://127.0.0.1:7007")
    # TODO: Add a progress bar
    # TODO: Make the run button disabled when the process is running

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

    with gr.Accordion("Model Args", open=False):
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

    with gr.Accordion("Data Parser Args", open=False):
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
        run,
        inputs=[data_path, method, max_num_iterations, steps_per_save, dataparser],
        outputs=None,
    )


demo.launch()
