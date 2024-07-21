import functools
import logging
import os
import tomllib
from pathlib import Path

import boto3
from s3path import register_configuration_parameter, PureS3Path, S3Path


def project_root() -> Path:
    return Path(__file__).parent.parent.parent.absolute()


@functools.lru_cache
def load_config() -> dict:
    config_location = os.environ.get("CONFIG_LOCATION", "config.toml")
    config_file = project_root() / config_location
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    for source in ["feature_store", "outcomes"]:
        SQL_PREFIXES = ("sqlite://", "postgresql://", "mysql://", "trino://")  # TODO
        if config["data_sources"][source]["location"].startswith("s3://"):
            config["data_sources"][source]["location"] = S3Path(config["data_sources"][source]["location"][4:])
        elif config["data_sources"][source]["location"].startswith(SQL_PREFIXES):
            pass
        else:
            config["data_sources"][source]["location"] = project_root() / config["data_sources"][source]["location"]
    return config
