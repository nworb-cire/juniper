import pandas as pd

from juniper.common.setup import init_services
from juniper.data_loading.feature_store import FeatureStore
from juniper.data_loading.outcomes import StandardOutcomes
from juniper.preprocessor.preprocessor import get_preprocessor
from juniper.validation.time_series_split import TimeSeriesSplit

if __name__ == "__main__":
    init_services()

    feature_store = FeatureStore()
    outcomes = StandardOutcomes()

    cv_split = TimeSeriesSplit(pd.Timedelta(days=30), n_splits=3)
    for train_idx, test_idx, train_time_end in cv_split.split(outcomes):
        train, test = feature_store.load_train_test(train_idx, test_idx)
        y_train, y_test = outcomes.load_train_test(train_idx, test_idx, train_time_end)

        preprocessor = get_preprocessor(schema=feature_store.schema)
        x_train = preprocessor.fit_transform(train)
        x_test = preprocessor.transform(test)

        # model = Model()
        # model.fit(x_train, y_train)
        # preds = model.predict(x_test)
        # score = model.score(preds, y_test)
        # cv_split.add_score(score)
