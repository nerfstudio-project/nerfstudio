# Local writer

The `LocalWriter` simply outputs numerical stats to the terminal.
You can specify additional parameters to customize your logging experience.

```python
@dataclass
class LocalWriterConfig(InstantiateConfig):
    """Local Writer config"""

    _target: Type = writer.LocalWriter
    """target class to instantiate"""
    enable: bool = False
    """if True enables local logging, else disables"""
    stats_to_track: Tuple[writer.EventName, ...] = (
        writer.EventName.ITER_TRAIN_TIME,
        writer.EventName.TRAIN_RAYS_PER_SEC,
        writer.EventName.CURR_TEST_PSNR,
        writer.EventName.VIS_RAYS_PER_SEC,
        writer.EventName.TEST_RAYS_PER_SEC,
    )
    """specifies which stats will be logged/printed to terminal"""
    max_log_size: int = 10
    """maximum number of rows to print before wrapping. if 0, will print everything."""
```

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

```bash
# disable local writer
ns-train {METHOD_NAME} --logging.local-writer.no-enable

# disable line wrapping
ns-train {METHOD_NAME} --logging.local-writer.max-log-size=0
```