# Logging Features

We provide integration with multiple logging interfaces to log images and statistics during training. 
All of these loggers are configurable via the config:

```yaml
# e.g. configs/graphs_default.yaml

logging:
    steps_per_log: 10       # how often to log losses/images
    max_buffer_size: 20     # defines how many steps to average over (if reporting averages)
    writer:                 # definition of all writers you want to use
        TensorboardWriter:  
            log_dir: "./"  # must specify the output path for all writers
        LocalWriter:
            log_dir: "./"
            stats_to_track:
                [ITER_LOAD_TIME, ITER_TRAIN_TIME, RAYS_PER_SEC, CURR_TEST_PSNR]
            max_log_size: 10 
```

Currently, we provide support for the following loggers:

1. [Tensorboard](https://www.tensorflow.org/tensorboard): `TensorBoardWriter`
2. [Weights and Biases](https://wandb.ai/site): `WandbWriter`
3. Local writer: `LocalWriter`

The `LocalWriter` simply outputs to the terminal. However, you can specify additional parameters to customize the logging.
* stats_to_track: all the stats that you want to print to the terminal (see list under `EventName` in `utils/writer.py`). You can add or remove any of the defined enums. 
* max_log_size: how much content to print onto the screen (e.g. only print 10 lines on the screen at a time). If 0, will print everything without deleting any previous lines.
