"""
Input/output utils.
"""
import datetime
import enum
import json
import os
from typing import Optional

from omegaconf import DictConfig

from mattport.utils.decorators import check_main_thread, check_print_stats_step


class Stats(enum.Enum):
    """Possible Stats values for StatsTracker

    Args:
        enum (_type_): _description_
    """

    ITER_LOAD_TIME = "Data Load (time)"
    ITER_TRAIN_TIME = "Train Iter (time)"
    TOTAL_TRAIN_TIME = "Train Total (time)"
    RAYS_PER_SEC = "Rays Per Sec (1/s)"
    CURR_TEST_PSNR = "Test PSNR"
    ETA = "ETA (time)"


class StatsTracker:
    """Stats Tracker class"""

    def __init__(self, config: DictConfig, is_main_thread: bool, max_history: Optional[int] = 20):
        self.config = config
        self.is_main_thread = is_main_thread
        self.max_history = max_history
        self.step = 0
        self.stats_dict = {}
        self.past_stats = []
        self.new_key = False

    def update_value(self, name: str, value: float, step: int):
        """update stats dictionary with key value pair

        Args:
            name (str): name of statistic we are logging
            value (float): value to update.
            step (int): number of total iteration steps.
        """
        self.step = step
        self.new_key = not name in self.stats_dict or self.new_key
        self.stats_dict[name] = value

    def update_time(self, name: str, start_time: float, end_time: float, step: int, batch_size: int = None):
        """update the stats dictionary with running averages/cumulative durations

        Args:
            name (str): name of statistic we are logging
            start_time (float): start time for the call in seconds
            end_time (float): end time when the call finished executing in seconds
            step (int): number of total iteration steps.
            batch_size (int, optional): total number of rays in a batch;
                if None, reports duration instead of batch per second. Defaults to None.
        """
        self.step = step
        self.new_key = not name in self.stats_dict or self.new_key
        val = end_time - start_time
        if batch_size:
            # calculate the batch per second stat
            val = batch_size / val

        if step == -1:
            # logging total time instead of average
            self.stats_dict[name] = val
        else:
            # calculate updated average
            self.stats_dict[name] = (self.stats_dict.get(name, 0) * step + val) / (step + 1)

        if name == Stats.ITER_TRAIN_TIME:
            # update ETA if logging iteration train time
            remain_iter = self.config.max_num_iterations - step
            self.stats_dict[Stats.ETA] = remain_iter * self.stats_dict[name]

    @check_print_stats_step
    @check_main_thread
    def print_stats(self, fraction_done: float):
        """helper to print out the stats dictionary.

        Args:
            fraction_done (float): fraction of steps executed in training iterations
        """
        # print a new header line if there is a new key added
        if self.step == 0 or self.new_key:
            mssg = f"{'Step (% Done)':<20}"
            for k in self.stats_dict:
                mssg += f"{k.value:<20} "
            if self.step > 0:
                cursorup(len(self.past_stats) + 2)
            print(mssg)
            print("-" * len(mssg))
            if self.step > 0:
                for mssg in self.past_stats:
                    print(mssg)

        # generate a new stats reporting message
        if self.step > 0:
            curr_mssg = f"{self.step} ({fraction_done*100:.02f}%)"
            curr_mssg = f"{curr_mssg:<20}"
            for k, v in self.stats_dict.items():
                if "(time)" in k.value:
                    v = str(datetime.timedelta(seconds=v))
                else:
                    v = f"{v:0.4f}"
                curr_mssg += f"{v:<20} "
            # update the history buffer
            if len(self.past_stats) >= self.max_history:
                self.past_stats.pop(0)
                cursorup(len(self.past_stats) + 1)
                for mssg in self.past_stats:
                    print(mssg)
            print(curr_mssg)
            self.past_stats.append(curr_mssg)

    @check_main_thread
    def dump_stats(self):
        """Dump stats locally to a json file"""
        raise NotImplementedError


def cursorup(x: int):
    """utility tool to move the cursor up on the terminal

    Args:
        x (int): amount of lines to move cursor upward
    """
    print(f"\r\033[{x}A", end="")


def load_from_json(filename: str):
    """_summary_

    Args:
        filename (str): _description_

    Returns:
        _type_: _description_
    """
    assert filename.endswith(".json")
    with open(filename, "r", encoding="UTF-8") as file:
        return json.load(file)


def write_to_json(filename: str, content: dict):
    """_summary_

    Args:
        filename (str): _description_
        content (dict): _description_
    """
    assert filename.endswith(".json")
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(content, file)


def get_git_root(path, dirs=(".git",), default=None):
    """_summary_
    "https://stackoverflow.com/questions/22081209/find-the-root-of-the-git-repository-where-the-file-lives

    Args:
        path (_type_): _description_
        dirs (tuple, optional): _description_. Defaults to (".git",).
        default (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    prev, test = None, os.path.abspath(path)
    while prev != test:
        if any(os.path.isdir(os.path.join(test, directory)) for directory in dirs):
            return test
        prev, test = test, os.path.abspath(os.path.join(test, os.pardir))
    return default


def get_absolute_path(path, proj_root_func=get_git_root):
    """
    Returns the full, absolute path.
    Relative paths are assumed to start at the repo directory.
    """
    if path == "":
        return ""
    absolute_path = path
    if absolute_path[0] != "/":
        absolute_path = os.path.join(proj_root_func(path), absolute_path)
    return absolute_path


def make_dir(filename_or_folder):
    """Make the directory for either the filename or folder.
    Note that filename_or_folder currently needs to end in / for it to be recognized as a folder.
    """
    if filename_or_folder[-1] != "/" and filename_or_folder.find(".") < 0:
        folder = os.path.dirname(filename_or_folder + "/")
    else:
        folder = os.path.dirname(filename_or_folder)
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except Exception as e:  # pylint: disable=broad-except
            print(f"Couldn't create folder: {folder}. Maybe due to a parallel process?")
            print(e)
    return filename_or_folder
