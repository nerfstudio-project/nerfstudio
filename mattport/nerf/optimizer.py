import torch.optim as optim


class Optimizer:
    """_summary_"""

    def __init__(self, params: dict) -> None:
        self.optimizer = optim.Adam(params)

    def test(self):
        raise NotImplementedError

        