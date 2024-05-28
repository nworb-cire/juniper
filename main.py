import logging
import time

import pandas as pd

from juniper.common.setup import init_services, load_config
from juniper.data_loading.feature_store import LocalParquetFeatureStore
from juniper.data_loading.outcomes import LocalStandardOutcomes
from juniper.preprocessor.preprocessor import ColumnTransformer
from juniper.modeling.losses import MaskedBCEWithLogitsLoss
from juniper.modeling.torch import TorchModel
from juniper.validation.time_series_split import TimeSeriesSplit
from juniper.modeling.utils import get_model_class

if __name__ == "__main__":
    init_services()

    feature_store = LocalParquetFeatureStore()
    outcomes = LocalStandardOutcomes()

    config = load_config()
    cv_split = TimeSeriesSplit(
        pd.Timedelta(days=config["model"]["training"]["cv_split"]["days"]),
        n_splits=config["model"]["training"]["cv_split"]["n_splits"],
    )
    for train_idx, test_idx, train_time_end in cv_split.split(feature_store, outcomes):
        # TODO: Move this to a separate function
        train, test = feature_store.load_train_test(train_idx, test_idx)
        assert test is not None
        y_train, y_test = outcomes.load_train_test(train.index, test.index, train_time_end)
        assert y_test is not None
        train = train.reindex(y_train.index)
        test = test.reindex(y_test.index)

        t = time.monotonic()
        preprocessor = ColumnTransformer(feature_store=feature_store)
        x_train = preprocessor.fit_transform(train)
        logging.info(f"Preprocessor fitted in {time.monotonic() - t:.2f} seconds")
        t = time.monotonic()
        x_test = preprocessor.transform(test)
        logging.info(f"Validation data preprocessed in {time.monotonic() - t:.2f} seconds")

        model = TorchModel(
            model_cls=get_model_class(),
            loss_fn=MaskedBCEWithLogitsLoss(),
            preprocessor=preprocessor,
        )
        metrics = model.fit(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            hyperparameters=config["model"]["hyperparameters"],
        )
        # logging.info(f"Training metrics: {metrics[-1]}")
        onnx = model.to_onnx(metrics=metrics)
        model.save(path=f"models/model_{train_time_end.date()}.onnx")
