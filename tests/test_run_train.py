"""
Default test to make sure train runs
"""
import pytest
import omegaconf
from mattport.utils.io import get_absolute_path

from scripts.run_train import main


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_run_train():
    """test run train script works properly"""
    test_config = get_absolute_path("./tests/configs/test_default.yml")
    cfg = omegaconf.OmegaConf.load(test_config)
    main(cfg)
