import io
import logging
import time
from typing import Type, Callable

import onnx
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer

from juniper.common.export import merge_models, to_onnx, add_default_metadata, add_metrics
from juniper.common.schema_tools import get_input_mapping
from juniper.training.metrics import evaluate_model, EvalMetrics
from juniper.training.utils import _to_tensor, batches


class Model:
    def __init__(self, model_cls: Type, loss_fn: Callable, preprocessor: ColumnTransformer):
        self.preprocessor = preprocessor
        self.preprocessor_onnx = to_onnx(self.preprocessor)
        self.preprocessor_inputs = get_input_mapping(self.preprocessor)
        self.model_cls = model_cls
        self.loss_fn = loss_fn

    def _loss(self, x: pd.DataFrame, y: pd.DataFrame):
        x_ = _to_tensor(self.model, x)
        yhat = self.model.forward(x_)
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
        epochs: int = 100,
        batch_size: int = 1024,
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

        self.model_outputs = y_train.columns
        self.model = self.model_cls(inputs=self.preprocessor_inputs, outputs=self.model_outputs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        metrics = []
        for epoch in range(1, epochs + 1):
            train_loss, train_n = 0.0, 0
            t0 = time.monotonic()
            for batch_x, batch_y in batches(x_train, y_train, batch_size):
                train_loss += self.partial_fit(batch_x, batch_y)
                train_n += batch_x.shape[0]
            avg_train_loss = train_loss / train_n
            t1 = time.monotonic()
            if x_test is not None and y_test is not None:
                with torch.no_grad():
                    metrics_ = evaluate_model(self.model, x_test, y_test, epoch)
                metrics.append(metrics_)
            logging.info(f"Epoch {epoch} ({t1-t0:.2f}s): train loss {avg_train_loss:.4f}")
        return metrics

    def save(self, path: str, metrics: list[EvalMetrics]):
        dummy_input = {"features": torch.zeros((1, len(self.preprocessor_inputs["features"])), dtype=torch.float32)}
        dummy_input.update(
            {
                name.replace(".", "_"): torch.zeros((1, len(cols), 1), dtype=torch.float32)
                for name, cols in self.preprocessor_inputs.items()
                if name != "features"
            }
        )
        with io.BytesIO() as f:
            torch.onnx.export(
                self.model,
                args=(dummy_input, {}),
                f=f,
                input_names=list(dummy_input.keys()),
                output_names=list(self.model_outputs),
                dynamic_axes={k: {2: "seq"} for k in self.preprocessor_inputs.keys() if k != "features"},
            )
            f.seek(0)
            model = onnx.load(f)
            # onnx.checker.check_model(model, full_check=True)
        merged = merge_models(
            self.preprocessor_onnx, model, [(node.name.replace(".", "_"), node.name) for node in model.graph.input]
        )
        add_default_metadata(merged)
        add_metrics(merged, metrics)
        onnx.save_model(merged, path)
