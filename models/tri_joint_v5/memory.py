from typing import Optional

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv_z = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding, bias=True)
        self.conv_r = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding, bias=True)
        self.conv_h = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding, bias=True)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor]) -> torch.Tensor:
        if h_prev is None:
            h_prev = x.new_zeros(x.shape[0], self.hidden_dim, x.shape[2], x.shape[3])
        stacked = torch.cat([x, h_prev], dim=1)
        z = torch.sigmoid(self.conv_z(stacked))
        r = torch.sigmoid(self.conv_r(stacked))
        candidate = torch.tanh(self.conv_h(torch.cat([x, r * h_prev], dim=1)))
        return (1.0 - z) * h_prev + z * candidate
