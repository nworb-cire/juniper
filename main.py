import logging
import time

import pandas as pd

from juniper.common.setup import init_services
from juniper.data_loading.feature_store import LocalParquetFeatureStore
from juniper.data_loading.outcomes import LocalStandardOutcomes
from juniper.preprocessor.preprocessor import get_preprocessor
from juniper.training.losses import MaskedBCEWithLogitsLoss
from juniper.training.model_wrapper import Model
from juniper.validation.time_series_split import TimeSeriesSplit
from juniper.training.utils import get_model_class

if __name__ == "__main__":
    init_services()

    feature_store = LocalParquetFeatureStore()
    outcomes = LocalStandardOutcomes()

    cv_split = TimeSeriesSplit(pd.Timedelta(days=30), n_splits=3)
    for train_idx, test_idx, train_time_end in cv_split.split(outcomes):
        # TODO: Move this to a separate function
        train, test = feature_store.load_train_test(train_idx, test_idx)
        y_train, y_test = outcomes.load_train_test(train.index, test.index, train_time_end)
        train = train.reindex(y_train.index)
        test = test.reindex(y_test.index)

        t = time.monotonic()
        preprocessor = get_preprocessor(feature_store=feature_store)
        x_train = preprocessor.fit_transform(train)
        x_train.columns = [c.replace(".", "_") for c in x_train.columns]
        logging.info(f"Preprocessor fitted in {time.monotonic() - t:.2f} seconds")
        t = time.monotonic()
        x_test = preprocessor.transform(test)
        x_test.columns = [c.replace(".", "_") for c in x_test.columns]
        logging.info(f"Validation data preprocessed in {time.monotonic() - t:.2f} seconds")

        model = Model(
            model_cls=get_model_class(),
            loss_fn=MaskedBCEWithLogitsLoss(),
            preprocessor=preprocessor,
        )
        metrics = model.fit(x_train, y_train, x_test, y_test, epochs=15, batch_size=1024)
        model.save(path=f"models/model_{train_time_end}.onnx", metrics=metrics)
