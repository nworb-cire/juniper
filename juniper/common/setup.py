import logging
import tomllib
from pathlib import Path

import boto3
from s3path import register_configuration_parameter, PureS3Path, S3Path


def load_config() -> dict:
    project_root = Path(__file__).parent.parent.parent
    config_file = project_root / "config.toml"
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    for source in ["feature_store", "outcomes"]:
        config["data_sources"][source]["location"] = S3Path(config["data_sources"][source]["location"][4:])
    return config


def init_logging(config):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def init_minio(config):
    minio_resource = boto3.resource(
        "s3",
        endpoint_url=config["minio"]["endpoint_url"],
        aws_access_key_id=config["minio"]["aws_access_key_id"],
        aws_secret_access_key=config["minio"]["aws_secret_access_key"],
    )
    register_configuration_parameter(PureS3Path("/"), resource=minio_resource)


def init_services():
    config = load_config()

    init_logging(config)
    init_minio(config)
