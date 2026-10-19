"""Tanh/Xavier MLP; preserve v1 state keys and registered normalization buffers."""
import torch
from torch import nn


class Layer(nn.Module):
    def __init__(self, n_in: int, n_out: int, activation):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, value):
        value = self.layer(value)
        return self.activation(value) if self.activation is not None else value


class PortableDNN(nn.Module):
    """Same Tanh/Xavier architecture, but normalization tensors are portable buffers."""
    def __init__(self, layers, ub, lb, dtype):
        super().__init__()
        self.net = nn.ModuleList(
            Layer(layers[i], layers[i + 1],
                  nn.Tanh() if i < len(layers) - 2 else None)
            for i in range(len(layers) - 1))
        self.register_buffer("ub", torch.as_tensor(ub, dtype=dtype))
        self.register_buffer("lb", torch.as_tensor(lb, dtype=dtype))
        self.to(dtype=dtype)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, value):
        value = (value - self.lb) / (self.ub - self.lb)
        for layer in self.net:
            value = layer(value)
        return value
