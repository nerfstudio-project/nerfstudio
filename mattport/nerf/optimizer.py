from torch import optim


class Optimizer:
    """_summary_"""

    def __init__(
        self, 
        params: dict,
        lr: float) -> None:
        """
        Args:
            params (dict): _description_
            lr (float): Learning rate.
        """
        self.optimizer = optim.Adam(params)

    def test(self):
        raise NotImplementedError

        