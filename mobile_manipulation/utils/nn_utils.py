import math
from typing import Sequence

import torch
from torch import nn


class Flatten(nn.Module):
    r"""Copied from torch 1.9."""
    __constants__ = ["start_dim", "end_dim"]
    start_dim: int
    end_dim: int

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.flatten(self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        return "start_dim={}, end_dim={}".format(self.start_dim, self.end_dim)


class MLP(nn.ModuleList):
    def __init__(self, c_in: int, c_outs: Sequence[int]):
        super().__init__()
        for c_out in c_outs:
            self.append(nn.Linear(c_in, c_out))
            self.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.output_size = c_in

    def forward(self, x):
        for m in self:
            x = m(x)
        return x

    def orthogonal_(self, gain=math.sqrt(2)):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=gain)
                torch.nn.init.zeros_(m.bias)
        return self
