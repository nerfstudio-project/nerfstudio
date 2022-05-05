"""
Default test to make sure train runs
"""
import pytest
from hydra import compose, initialize

from scripts.run_train import main


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_run_train():
    """test run train script works properly"""
    initialize(config_path="../configs/test/")
    conf = compose("test_dryrun.yaml")
    main(conf)
