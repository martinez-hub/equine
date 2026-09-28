# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
Embedding architectures that ship with EQUINE (D-191 spike).

These are registered on import, so a model file that names them can be loaded
anywhere EQUINE is installed without the user defining a class.
"""

from __future__ import annotations

import torch

from .registry import embedding_architecture

_ACTIVATIONS = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
}


@embedding_architecture("equine.mlp")
class MLP(torch.nn.Module):
    """
    A plain multilayer perceptron for tabular features.

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_sizes : list[int]
        Width of each hidden layer, in order. An empty list gives a linear map.
    out_features : int
        Embedding dimension (``emb_out_dim`` for the EQUINE model).
    activation : str
        One of ``"relu"``, ``"gelu"``, ``"tanh"``, ``"sigmoid"``.
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: list[int],
        out_features: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {activation!r}; choose from {sorted(_ACTIVATIONS)}."
            )
        layers: list[torch.nn.Module] = []
        width = in_features
        for hidden in hidden_sizes:
            layers += [torch.nn.Linear(width, hidden), _ACTIVATIONS[activation]()]
            width = hidden
        layers.append(torch.nn.Linear(width, out_features))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
