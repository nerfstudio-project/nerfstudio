# Local writer

The `LocalWriter` simply outputs numerical stats to the terminal.
You can specify additional parameters to customize your logging experience.
A skeleton of the local writer config is defined below.

```python
"""nerfstudio/configs/base_config.py""""

@dataclass
class LocalWriterConfig(InstantiateConfig):
    """Local Writer config"""

    _target: Type = writer.LocalWriter
    enable: bool = False
    stats_to_track: Tuple[writer.EventName, ...] = (
        writer.EventName.ITER_TRAIN_TIME,
        ...
    )
    max_log_size: int = 10

```

You can customize the local writer by editing the attributes:
- `enable`: enable/disable the logger.
- `stats_to_track`: all the stats that you want to print to the terminal (see list under `EventName` in `utils/writer.py`). You can add or remove any of the defined enums.
- `max_log_size`: how much content to print onto the screen (By default, only print 10 lines onto the screen at a time). If 0, will print everything without deleting any previous lines.

:::{admonition} Tip
:class: info

If you want to create a new stat to track, simply add the stat name to the `EventName` enum.
- Remember to call some put event (e.g. `put_scalar` from `utils/writer.py` to place the value in the `EVENT_STORAGE`. 
- Remember to add the new enum to the `stats_to_track` list
  :::

The local writer is easily configurable via CLI.
A few common commands to use:

- Disable local writer
    ```bash
    ns-train {METHOD_NAME} --logging.local-writer.no-enable
    ```

- Disable line wrapping
    ```bash
    ns-train {METHOD_NAME} --logging.local-writer.max-log-size=0
    ```