import abc

import torch


class Model(abc.ABC):
    @abc.abstractmethod
    def __init__(self, inputs: dict[str, list[str]], outputs: list[str], hyperparameters: dict):
        """
        :param inputs: Inverse mapping of preprocessor outputs to preprocessor inputs
        :param outputs: List of output column names
        """
        pass

    @abc.abstractmethod
    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        pass
