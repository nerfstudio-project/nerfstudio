import os
from typing import Dict
from mattport.nerf.trainer import Trainer
import logging
import hydra
from omegaconf import DictConfig

logging.basicConfig(format="[%(filename)s:%(lineno)d] %(message)s", level=logging.DEBUG)


@hydra.main(config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    print(cfg)

    print("hello mattport")

    # TODO(ethan): add some sort of config script

    config = {}
    trainer = Trainer(config)
    print("trainer!")
    trainer.train()


if __name__ == "__main__":
    main()
