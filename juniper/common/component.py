import abc
from typing import Any

import onnx

from juniper.modeling.metrics import EvalMetrics


class ModelComponent(abc.ABC):
    model: Any = None

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def to_onnx(self, name: str | None = None, metrics: list[EvalMetrics] | None = None) -> onnx.ModelProto:
        pass

    @abc.abstractmethod
    def validate(self, model: onnx.ModelProto):
        pass

    def save(self, path: str, metrics: list[EvalMetrics] | None = None):
        model_onnx = self.to_onnx(metrics=metrics)
        self.validate(model_onnx)
        onnx.save_model(model_onnx, path)
