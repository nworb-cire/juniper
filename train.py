import importlib
import os

import pandas as pd

from juniper.common.setup import load_config
from juniper.training.losses import MaskedBCEWithLogitsLoss
from juniper.training.model_wrapper import Model


def get_model_class():
    config = load_config()
    model_module = importlib.import_module(config["model"]["module"])
    model_class = getattr(model_module, config["model"]["class"])
    importlib.import_module(config["model"]["module"])
    return model_class


def get_training_dates():
    dates = []
    for file in os.listdir("data/processed"):
        if file.startswith("preprocessor_"):
            date = file.split("_")[1].split(".")[0]
            dates.append(date)
    dates = sorted(dates)
    return dates


def load_training_data(date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_train = pd.read_parquet(f"data/processed/x_train_{date}.parquet")
    y_train = pd.read_parquet(f"data/processed/y_train_{date}.parquet")
    x_test = pd.read_parquet(f"data/processed/x_test_{date}.parquet")
    y_test = pd.read_parquet(f"data/processed/y_test_{date}.parquet")
    y_train = y_train.drop(columns=["submit_ts"], errors="ignore")
    y_test = y_test.drop(columns=["submit_ts"], errors="ignore")

    return x_train, y_train, x_test, y_test


def train(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    preprocessor_path: str,
) -> tuple[float | None, Model]:
    model_cls = get_model_class()
    model = Model(
        model_cls=model_cls,
        loss_fn=MaskedBCEWithLogitsLoss(),
        preprocessor_path=preprocessor_path,
    )
    loss = model.fit(x_train, y_train, x_test, y_test, epochs=0, batch_size=4096)
    return loss, model


if __name__ == "__main__":
    model = None
    for date in get_training_dates()[:1]:
        x_train, y_train, x_test, y_test = load_training_data(date)
        score, model = train(x_train, y_train, x_test, y_test, f"data/processed/preprocessor_{date}.onnx")
        print(f"Score for {date}: {score}")
    assert model is not None
    model.save("model.onnx")
