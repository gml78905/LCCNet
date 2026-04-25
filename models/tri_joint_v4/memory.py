from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class _ConvGRUCell(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(in_ch + hidden_ch, hidden_ch * 2, kernel_size=3, padding=1)
        self.candidate = nn.Conv2d(in_ch + hidden_ch, hidden_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gates = self.gates(torch.cat([x, h], dim=1))
        z, r = torch.chunk(gates, chunks=2, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        cand = torch.tanh(self.candidate(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * cand


class DenseJointTemporalMemory(nn.Module):
    """
    Dense spatiotemporal memory over the joint calibration field.
    """

    def __init__(self, d_joint: int = 384, d_hidden: int = 256):
        super().__init__()
        self.d_hidden = d_hidden
        self.pre = nn.Sequential(
            nn.Conv2d(d_joint, d_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_hidden),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.cell = _ConvGRUCell(d_hidden, d_hidden)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def init_state(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"m_joint": torch.zeros((x.shape[0], self.d_hidden, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)}

    def forward(
        self,
        J1: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if state is None:
            state = self.init_state(J1)
        x = self.pre(J1)
        m_joint = self.cell(x, state["m_joint"])
        z_mem = self.pool(m_joint).flatten(1)
        return m_joint, {"m_joint": m_joint}, z_mem
