from tty import CFLAG
import pytest
import runpy

from mattport.nerf.trainer import Trainer
from hydra import compose, initialize

CONFIG_DIR = "configs"

def test_multiprocess_train():
    """Test for trainer to validate distributed data parallel working."""
    with initialize(config_path=CONFIG_DIR):
        cfg = compose(config_name='test_basic_graph')
    trainer = Trainer(cfg)
    trainer.run_multiprocess_train()