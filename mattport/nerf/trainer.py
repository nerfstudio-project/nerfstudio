import logging
import torch


class Trainer(object):
    def __init__(self, config: None):
        self.config = config
        self.optimizer = None

    def load_checkpoint(self):
        """_summary_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def save_checkpoint(self):
        """_summary_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def load_dataset(self):
        """_summary_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def train(self):
        """_summary_"""
        # TODO(ethan): replace print statements with something more proper
        logging.info("train function")
