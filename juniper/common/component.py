import abc

import onnx

from juniper.training.metrics import EvalMetrics


class ModelComponent(abc.ABC):
    model = None

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def to_onnx(self, metrics: list[EvalMetrics] | None) -> onnx.ModelProto:
        pass

    @abc.abstractmethod
    def validate(self, model: onnx.ModelProto):
        pass

    def save(self, model: onnx.ModelProto, path: str):
        self.validate(model)
        onnx.save_model(model, path)
