import os

import pytest


@pytest.fixture
def config():
    os.environ["CONFIG_LOCATION"] = "tests/test_config.toml"
    yield
    del os.environ["CONFIG_LOCATION"]
