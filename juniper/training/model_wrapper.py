import abc
import io
import logging
import time
from typing import Type, Callable

import numpy as np
import onnx
import pandas as pd
import torch

from juniper.common.component import ModelComponent
from juniper.common.export import merge_models, add_default_metadata, add_metrics
from juniper.common.schema_tools import get_input_mapping
from juniper.preprocessor.preprocessor import ColumnTransformer
from juniper.training.metrics import evaluate_model, EvalMetrics
from juniper.training.utils import batches, dummy_inference


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


class TorchModel(ModelComponent):
    def __init__(self, model_cls: Type, loss_fn: Callable, preprocessor: ColumnTransformer):
        self.preprocessor = preprocessor
        self.preprocessor_onnx = self.preprocessor.to_onnx()
        self.preprocessor_inputs = get_input_mapping(self.preprocessor)
        self.model_cls = model_cls
        self.loss_fn = loss_fn

    def _loss(self, x: pd.DataFrame, y: pd.DataFrame):
        yhat = self.model.forward(x)
        return self.loss_fn(yhat, torch.tensor(y.values, dtype=torch.float32))

    def partial_fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> float:
        self.optimizer.zero_grad()
        loss = self._loss(x_train, y_train)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame = None,
        y_test: pd.DataFrame = None,
        hyperparameters: dict = None,
    ) -> list[EvalMetrics]:
        assert (
            x_train.shape[0] == y_train.shape[0]
        ), f"x_train and y_train must have the same number of rows, got {x_train.shape[0]} and {y_train.shape[0]}"
        if x_test is not None:
            assert y_test is not None, "If x_test is provided, y_test must also be provided"
            assert (
                x_test.shape[0] == y_test.shape[0]
            ), f"x_test and y_test must have the same number of rows, got {x_test.shape[0]} and {y_test.shape[0]}"
            assert (
                x_train.shape[1] == x_test.shape[1]
            ), f"x_train and x_test must have the same number of columns, got {x_train.shape[1]} and {x_test.shape[1]}"
            assert (
                y_train.shape[1] == y_test.shape[1]
            ), f"y_train and y_test must have the same number of columns, got {y_train.shape[1]} and {y_test.shape[1]}"

        self.model_outputs = y_train.columns.tolist()
        self.model = self.model_cls(
            inputs=self.preprocessor_inputs, outputs=self.model_outputs, hyperparameters=hyperparameters
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hyperparameters["learning_rate"])

        metrics = []
        for epoch in range(1, hyperparameters["epochs"] + 1):
            train_loss, train_n = 0.0, 0
            t0 = time.monotonic()
            for batch_x, batch_y in batches(x_train, y_train, hyperparameters["batch_size"]):
                tb = time.monotonic()
                train_loss += self.partial_fit(batch_x, batch_y)
                train_n += batch_x.shape[0]
                logging.debug(f"Batch time: {time.monotonic() - tb:.2f}s")
            avg_train_loss = train_loss / train_n
            t1 = time.monotonic()
            if x_test is not None and y_test is not None:
                tb = time.monotonic()
                with torch.no_grad():
                    metrics_ = evaluate_model(self.model, x_test, y_test, epoch)
                logging.debug(f"Validation time: {time.monotonic() - tb:.2f}s")
                metrics.append(metrics_)
            logging.info(f"Epoch {epoch} ({t1-t0:.2f}s): train loss {avg_train_loss:.4f}")
        return metrics

    def validate(self, model: onnx.ModelProto):
        out = dummy_inference(model)
        for val in out.values():
            assert not np.isnan(val).any(), "Model produced NaN values"

    def to_onnx(self, name: str | None = None, metrics: list[EvalMetrics] | None = None) -> onnx.ModelProto:
        dummy_input = dummy_inference(self.preprocessor_onnx)
        dummy_input = {k: torch.tensor(v) for k, v in dummy_input.items()}

        with io.BytesIO() as f:
            torch.onnx.export(
                self.model,
                args=(dummy_input, {}),  # Trailing empty dict is for kwargs, see docstring for this function
                f=f,
                input_names=list(dummy_input.keys()),
                output_names=self.model_outputs,
                dynamic_axes={k: {1: "sequence"} for k in dummy_input.keys() if k != "features"},
            )
            f.seek(0)
            model = onnx.load(f)
        io_map = [(k, k) for k in dummy_input.keys()]
        merged = merge_models(self.preprocessor_onnx, model, io_map=io_map)
        onnx.checker.check_model(model, full_check=True)

        self.validate(merged)
        add_default_metadata(merged)
        if metrics is not None:
            add_metrics(merged, metrics)
        return merged
