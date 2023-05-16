# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytorch fields implementations.
"""

import torch

from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.mlp import MLP


class NetworkWithInputEncoding(torch.nn.Module):
    """
    Input encoding, followed by a neural network.

    This module is more efficient than invoking individual `Encoding`
    and `Network` modules in sequence.

    Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
    it to a tensor of shape `[:, n_output_dims]`.

    Parameters
    ----------
    n_input_dims : `int`
            Determines the shape of input tensors as `[:, n_input_dims]`
    n_output_dims : `int`
            Determines the shape of output tensors as `[:, n_output_dims]`
    encoding_config: `dict`
            Configures the encoding. Possible configurations are documented at
            https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
    network_config: `dict`
            Configures the neural network. Possible configurations are documented at
            https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
    """

    def __init__(self, n_input_dims, n_output_dims, encoding_config, network_config):
        super().__init__()

        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.encoding_config = encoding_config
        self.network_config = network_config

        self.encoder = Encoding(n_input_dims, encoding_config)
        self.network = Network(self.encoder.n_output_dims, n_output_dims, network_config)

    def forward(self, x):
        x = self.encoder(x)
        x = self.network(x)
        return x


class Network(torch.nn.Module):
    """
    Neural network.

    Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
    it to a tensor of shape `[:, n_output_dims]`.

    Parameters
    ----------
    n_input_dims : `int`
            Determines the shape of input tensors as `[:, n_input_dims]`
    n_output_dims : `int`
            Determines the shape of output tensors as `[:, n_output_dims]`
    network_config: `dict`
            Configures the neural network. Possible configurations are documented at
            https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
    """

    def __init__(self, n_input_dims, n_output_dims, network_config):
        super().__init__()

        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.network_config = network_config

        otype = network_config["otype"]
        assert otype == "FullyFusedMLP"
        activations_dict = {"None": None, "ReLU": torch.nn.ReLU(), "Sigmoid": torch.nn.Sigmoid()}
        self.network = MLP(
            in_dim=n_input_dims,
            num_layers=network_config["n_hidden_layers"],
            layer_width=network_config["n_neurons"],
            activation=activations_dict[network_config["activation"]],
            out_activation=activations_dict[network_config["output_activation"]],
        )
        self.n_output_dims = self.network.get_out_dim()

    def forward(self, x):
        x = self.network(x)
        return x


class Encoding(torch.nn.Module):
    """
    Input encoding to a neural network.

    Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
    it to a `dtype` tensor of shape `[:, self.n_output_dims]`, where
    `self.n_output_dims` depends on `n_input_dims` and the configuration
    `encoding_config`.

    Parameters
    ----------
    n_input_dims : `int`
            Determines the shape of input tensors as `[:, n_input_dims]`
    encoding_config: `dict`
            Configures the encoding. Possible configurations are documented at
            https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
    """

    def __init__(self, n_input_dims, encoding_config):
        super().__init__()

        self.n_input_dims = n_input_dims
        self.encoding_config = encoding_config

        otype = encoding_config["otype"]
        if otype == "SphericalHarmonics":
            self.encoder = SHEncoding(levels=encoding_config["degree"])
        elif otype == "Frequency":
            self.encoder = NeRFEncoding(
                n_input_dims, encoding_config["n_frequencies"], min_freq_exp=0.0, max_freq_exp=9.0
            )
        elif otype == "HashGrid":
            self.encoder = HashEncoding(
                num_levels=encoding_config["n_levels"],
                # TODO:
                implementation="torch",
            )

        self.n_output_dims = self.encoder.get_out_dim()

    def forward(self, x):
        x = self.encoder(x)
        return x
