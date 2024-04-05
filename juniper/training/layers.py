import pandas as pd
import torch
from torch import nn

from juniper.training import nan_ops


class SummaryPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return torch.cat(
                [
                    torch.nanmean(x, dim=-1),
                    nan_ops.nanstd(x, dim=-1),
                    nan_ops.nanmax(x, dim=-1)[0],
                    nan_ops.nanmin(x, dim=-1)[0],
                ],
                dim=-1,
            )
        else:
            return torch.cat(
                [
                    torch.mean(x, dim=-1),
                    torch.std(x, dim=-1),
                    torch.max(x, dim=-1)[0],
                    torch.min(x, dim=-1)[0],
                ],
                dim=-1,
            )


class Unify(nn.Module):
    def __init__(
        self,
        modules: dict[str, nn.Module],
        padding_value=0.0,
    ):
        super().__init__()
        self.modules = modules
        self.padding_value = padding_value

    def forward(self, x: pd.DataFrame) -> torch.Tensor:
        ret = torch.Tensor()
        for col, module in self.modules.items():
            tensors = x[col].apply(torch.tensor).apply(lambda x: x.T)
            if self.padding_value is not None:
                y = torch.nn.utils.rnn.pad_sequence(
                    tensors.values.tolist(), batch_first=True, padding_value=self.padding_value
                )  # BxSxV
                y = y.permute(0, 2, 1)  # BxVxS
            else:
                # Untested as torch support for nested (jagged) tensors is still experimental
                y = torch.nested.nested_tensor(tensors.values.tolist())  # BxVxS
            y = module(y)  # BxV
            ret = torch.cat([ret, y], dim=-1)
            x = x.drop(columns=[col])
        ret = torch.cat([ret, torch.tensor(x.values, dtype=torch.float32)], dim=-1)
        return ret
