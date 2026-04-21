from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class SharedMemoryCell(nn.Module):
    """
    Shared temporal memory cell.

    Input:
      z_shared_coarse: [B, 256]
      h_prev: [B, 256]
    Output:
      h_shared: [B, 256]
    """

    def __init__(self, d_in: int = 256, d_hidden: int = 256):
        super().__init__()
        self.cell = nn.GRUCell(input_size=d_in, hidden_size=d_hidden)

    def forward(self, z_shared_coarse: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return self.cell(z_shared_coarse, h_prev)


class PairMemoryCell(nn.Module):
    """
    Pair-specific temporal memory cell.

    Input:
      r_pair_coarse: [B, 384]
      h_pair_prev: [B, 256]
    Output:
      h_pair: [B, 256]
    """

    def __init__(self, d_in: int = 384, d_hidden: int = 256):
        super().__init__()
        self.cell = nn.GRUCell(input_size=d_in, hidden_size=d_hidden)

    def forward(self, r_pair_coarse: torch.Tensor, h_pair_prev: torch.Tensor) -> torch.Tensor:
        return self.cell(r_pair_coarse, h_pair_prev)


class TriTemporalMemory(nn.Module):
    """
    Temporal memory manager.
    - shared memory: [B, 256]
    - pair memories: [B, 256] each
    """

    def __init__(self, d_shared: int = 256, d_pair_in: int = 384, d_pair_hidden: int = 256):
        super().__init__()
        # TODO(v3): evaluate transformer/state-space memory alternatives for longer horizons.
        self.d_shared = d_shared
        self.d_pair_hidden = d_pair_hidden
        self.shared = SharedMemoryCell(d_in=d_shared, d_hidden=d_shared)
        self.pair_cl = PairMemoryCell(d_in=d_pair_in, d_hidden=d_pair_hidden)
        self.pair_cr = PairMemoryCell(d_in=d_pair_in, d_hidden=d_pair_hidden)
        self.pair_lr = PairMemoryCell(d_in=d_pair_in, d_hidden=d_pair_hidden)

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        return {
            "h_shared": torch.zeros((batch_size, self.d_shared), device=device, dtype=dtype),
            "h_cl": torch.zeros((batch_size, self.d_pair_hidden), device=device, dtype=dtype),
            "h_cr": torch.zeros((batch_size, self.d_pair_hidden), device=device, dtype=dtype),
            "h_lr": torch.zeros((batch_size, self.d_pair_hidden), device=device, dtype=dtype),
        }

    def forward(
        self,
        z_shared_coarse: torch.Tensor,
        r_cl_coarse: torch.Tensor,
        r_cr_coarse: torch.Tensor,
        r_lr_coarse: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        bsz = z_shared_coarse.shape[0]
        if state is None:
            state = self.init_state(bsz, z_shared_coarse.device, z_shared_coarse.dtype)

        h_shared = self.shared(z_shared_coarse, state["h_shared"])
        h_cl = self.pair_cl(r_cl_coarse, state["h_cl"])
        h_cr = self.pair_cr(r_cr_coarse, state["h_cr"])
        h_lr = self.pair_lr(r_lr_coarse, state["h_lr"])

        new_state = {
            "h_shared": h_shared,
            "h_cl": h_cl,
            "h_cr": h_cr,
            "h_lr": h_lr,
        }
        return h_shared, h_cl, h_cr, h_lr, new_state
