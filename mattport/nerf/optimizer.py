"""
Optimizer class.
"""

from torch import optim


class Optimizer:
    """_summary_"""

    def __init__(self, params: dict, lr: float):
        """
        Args:
            params (dict): _description_
            lr (float): Learning rate.
        """
        self.optimizer = optim.Adam(params)
        self.lr = lr

    def test(self):
        """_summary_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
