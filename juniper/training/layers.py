import numpy as np
import pandas as pd
import torch
from torch import nn

from juniper.training import nan_ops


class SummaryPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.shape[0] > 1:
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
        padding_value=np.nan,
    ):
        super().__init__()
        self.modules = modules
        self.padding_value = padding_value

    def forward(self, x: pd.DataFrame | dict) -> torch.Tensor:
        if isinstance(x, pd.DataFrame):
            return self.forward_df(x)
        elif isinstance(x, dict):
            return self.forward_dict(x)
        else:
            raise ValueError(f"Unsupported input type {type(x)}")

    # TODO: ensure order is always the same
    def forward_df(self, x: pd.DataFrame) -> torch.Tensor:
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
        ret = torch.cat(
            [ret, torch.tensor(x.drop(columns=list(self.modules.keys())).values, dtype=torch.float32)], dim=-1
        )
        return ret

    def forward_dict(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """This method should only be used during ONNX export"""
        ret = torch.Tensor()
        for col, module in self.modules.items():
            y = x[col].T  # VxS
            y = y.unsqueeze(0)  # 1xVxS
            y = module(y)
            ret = torch.cat([ret, y], dim=-1)
        ret = torch.cat([ret, x["features"]], dim=-1)
        return ret
