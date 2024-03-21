import torch
from torch import nn


class MaskedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """Same as parent class, but nan values in the target are treated as masked"""

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(target)
        input = input[mask]
        target = target[mask]
        return super().forward(input, target)
