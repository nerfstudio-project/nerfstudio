"""
run_train_nerf.py
"""
import logging

import hydra
from omegaconf import DictConfig

from mattport.nerf.trainer import Trainer

logging.basicConfig(format="[%(filename)s:%(lineno)d] %(message)s", level=logging.DEBUG)


@hydra.main(config_path="../configs", config_name="config.yaml")
def main(config: DictConfig):
    """Main function."""
    trainer = Trainer(config)
    trainer.run_multiprocess_train()


if __name__ == "__main__":
    main()
