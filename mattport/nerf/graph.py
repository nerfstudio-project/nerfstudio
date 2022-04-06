from torch import nn


class Graph(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    