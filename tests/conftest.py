import pytest

from juniper.data_loading.outcomes import LocalStandardOutcomes


@pytest.fixture
def feature_store():
    return LocalParquetFeatureStore()


class TestOutcomes(LocalStandardOutcomes):
    def _get_columns(self, columns: list[str] | None = None) -> list[str]:
        return list(self.binary_outcomes_list)


@pytest.fixture
def outcomes(config):
    return TestOutcomes()
