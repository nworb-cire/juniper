import logging
import os
import tomllib
from pathlib import Path

import boto3
from dask_kubernetes.operator import KubeCluster
from s3path import register_configuration_parameter, PureS3Path, S3Path


def load_config() -> dict:
    project_root = Path(__file__).parent.parent.parent
    config_file = project_root / "config.toml"
    with open(config_file, "rb") as f:
        return tomllib.load(f)


def init_logging(config):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def init_dask(config):
    cluster = KubeCluster(
        name=config["dask"]["cluster_name"],
        image=config["dask"]["image"],
        resources=config["dask"]["resources"],
        shutdown_on_close=False,
    )
    cluster.adapt(
        minimum=config["dask"]["min_workers"],
        maximum=config["dask"]["max_workers"],
    )
    client = cluster.get_client()
    logging.info(f"Dask client dashboard: {client.dashboard_link}")    
    client.wait_for_workers(config["dask"]["min_workers"])
    # benchmark = client.benchmark_hardware()
    # logging.info(f"Dask benchmark: {benchmark}")


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
    init_dask(config)
    init_minio(config)


# TODO: Move this to config
def s3_location_from_env() -> tuple[S3Path, S3Path]:
    assert os.environ['FEATURE_STORE_LOCATION'].startswith('s3://')
    feature_store_location = S3Path(os.environ['FEATURE_STORE_LOCATION'][4:])
    assert feature_store_location.exists()

    assert os.environ['OUTCOMES_LOCATION'].startswith('s3://')
    outcomes_location = S3Path(os.environ['OUTCOMES_LOCATION'][4:])
    assert outcomes_location.exists()
    
    return feature_store_location, outcomes_location
