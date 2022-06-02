# Logging and profiling features

We provide many logging functionalities for timing and/or tracking losses during training. All of these loggers are configurable via `configs/logging.yml`

1. **Writer**: Logs losses and generated images during training to a specified output stream. Specify the type of writer (Tensorboard, Local Writer, Weights and Biases), and how often to log in the config.

2. **Profiler**: Computes the average total time of execution for any function with the `@profiler.time_function` decorator. Prints out the full profile at the termination of training or the program.