# CLI

We provide a command line interface for training your own NeRFs (no coding necessary). You can learn more about each command by using the `--help` argument.

## Commands

Here are the popular commands that we offer. If you've cloned the repo, you can also look at the [pyproject.toml file](https://github.com/nerfstudio-project/nerfstudio/blob/main/pyproject.toml) at the `[project.scripts]` section for details.

| Command                              | Description                            | Filename                           |
| ------------------------------------ | -------------------------------------- | ---------------------------------- |
| [ns-install-cli](ns_install_cli)     | Install tab completion for all scripts | scripts/completions/install.py     |
| [ns-process-data](ns_process_data)   | Generate a dataset from your own data  | scripts/process_data.py            |
| [ns-download-data](ns_download_data) | Download existing captures             | scripts/downloads/download_data.py |
| [ns-train](ns_train)                 | Generate a NeRF                        | scripts/train.py                   |
| [ns-viewer](ns_viewer)               | View a trained NeRF                    | scripts/viewer/run_viewer.py       |
| [ns-eval](ns_eval)                   | Run evaluation metrics for your Model  | scripts/eval.py                    |
| [ns-render](ns_render)               | Render out a video of your NeRF        | scripts/render.py                  |
| [ns-export](ns_export)               | Export a NeRF into other formats       | scripts/exporter.py                |

```{toctree}
:maxdepth: 1
:hidden:

ns_install_cli
ns_process_data
ns_download_data
ns_train
ns_render
ns_viewer
ns_export
ns_eval
```
