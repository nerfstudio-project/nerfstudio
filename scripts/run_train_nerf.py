import os
from typing import Dict
from mattport.nerf.trainer import Trainer
import logging
import hydra
from omegaconf import DictConfig

logging.basicConfig(format="[%(filename)s:%(lineno)d] %(message)s", level=logging.DEBUG)


@hydra.main(config_path="../configs", config_name="config.yaml")
def main(config: DictConfig):
    trainer = Trainer(config)
    print("trainer!")

    trainer.setup_dataset()
    # trainer.setup_graph()
    # trainer.setup_optimizer()


if __name__ == "__main__":
    main()
