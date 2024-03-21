import random

import pandas as pd

from juniper.common.export import to_onnx
from juniper.common.setup import init_services
from juniper.data_loading.feature_store import LocalParquetFeatureStore
from juniper.data_loading.outcomes import LocalStandardOutcomes
from juniper.preprocessor.preprocessor import get_preprocessor
from juniper.validation.time_series_split import TimeSeriesSplit

if __name__ == "__main__":
    init_services()

    feature_store = LocalParquetFeatureStore()
    outcomes = LocalStandardOutcomes()

    cv_split = TimeSeriesSplit(pd.Timedelta(days=30), n_splits=3)
    for train_idx, test_idx, train_time_end in cv_split.split(outcomes):
        random.seed(0)
        train_idx = pd.Index(random.sample(train_idx.tolist(), 100_000), name=train_idx.name)
        test_idx = pd.Index(random.sample(test_idx.tolist(), 10_000), name=test_idx.name)
        train, test = feature_store.load_train_test(train_idx, test_idx)
        y_train, y_test = outcomes.load_train_test(train.index, test.index, train_time_end)

        preprocessor = get_preprocessor(feature_store=feature_store)
        x_train = preprocessor.fit_transform(train)
        x_test = preprocessor.transform(test)

        x_train.to_parquet(f"data/processed/x_train_{train_time_end}.parquet")
        x_test.to_parquet(f"data/processed/x_test_{train_time_end}.parquet")
        y_train.to_parquet(f"data/processed/y_train_{train_time_end}.parquet")
        y_test.to_parquet(f"data/processed/y_test_{train_time_end}.parquet")

        onnx = to_onnx(preprocessor)
        with open(f"data/processed/preprocessor_{train_time_end}.onnx", "wb") as f:
            f.write(onnx.SerializeToString())

        # model = Model()
        # model.fit(x_train, y_train)
        # preds = model.predict(x_test)
        # score = model.score(preds, y_test)
        # cv_split.add_score(score)
