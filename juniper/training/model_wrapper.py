import time
from typing import Type, Callable

import onnxruntime as ort
import pandas as pd
import torch

from juniper.training.utils import _to_tensor, batches


class Model:
    def __init__(self, model_cls: Type, loss_fn: Callable, preprocessor_path: str):
        assert preprocessor_path.endswith(".onnx"), "Only ONNX preprocessors are supported"
        preprocessor = ort.InferenceSession(preprocessor_path)
        self.model_inputs = {node.name: node.shape[1] for node in preprocessor.get_outputs()}
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
    ) -> float | None:
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

        self.model = self.model_cls(self.model_inputs, y_train.shape[1])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        avg_test_loss = None
        for epoch in range(1, epochs + 1):
            train_loss, train_n = 0.0, 0
            t0 = time.monotonic()
            for batch_x, batch_y in batches(x_train, y_train, batch_size):
                train_loss += self.partial_fit(batch_x, batch_y)
                train_n += batch_x.shape[0]
            avg_train_loss = train_loss / train_n
            t1 = time.monotonic()
            if x_test is not None and y_test is not None:
                test_loss, test_n = 0.0, 0
                with torch.no_grad():
                    for batch_x, batch_y in batches(x_test, y_test, batch_size):
                        test_loss += self._loss(batch_x, batch_y)
                        test_n += batch_x.shape[0]
                avg_test_loss = test_loss / test_n
                print(f"Epoch {epoch} ({t1-t0:.2f}s): train loss {avg_train_loss:.4f}, test loss {avg_test_loss:.4f}")
            else:
                print(f"Epoch {epoch} ({t1-t0:.2f}s): train loss {avg_train_loss:.4f}")
        return avg_test_loss

    def save(self, path: str):
        raise NotImplementedError
