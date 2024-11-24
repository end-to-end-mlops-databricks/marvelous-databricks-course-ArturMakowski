"""Evaluate the model and register it if it performs better than the previous model."""

import argparse
import sys

import mlflow
import mlflow.sklearn
from databricks import feature_engineering
from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession
from sklearn.metrics import f1_score

from mlops_with_databricks.data_preprocessing.dataclasses import (
    DatabricksConfig,
    ModelServingConfig,
    ProcessedAdClickDataConfig,
)

logger.remove()

logger.add(sink=sys.stderr, level="DEBUG")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--new_model_uri",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--git_branch",
    action="store",
    default=None,
    type=str,
    required=True,
)


args = parser.parse_args()
new_model_uri = args.new_model_uri
job_run_id = args.job_run_id
git_sha = args.git_sha
git_branch = args.git_branch


spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

num_features = ProcessedAdClickDataConfig.num_features
cat_features = ProcessedAdClickDataConfig.cat_features
target = ProcessedAdClickDataConfig.target
catalog_name = DatabricksConfig.catalog_name
schema_name = DatabricksConfig.schema_name

serving_endpoint_name = ModelServingConfig.serving_endpoint_name
serving_endpoint = workspace.serving_endpoints.get(serving_endpoint_name)
model_name = serving_endpoint.config.served_models[0].model_name
model_version = serving_endpoint.config.served_models[0].model_version
previous_model_uri = f"models:/{model_name}/{model_version}"

test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

X_test = test_set[list(num_features) + list(cat_features)]
y_test = test_set[target]

logger.debug(f"New Model URI: {new_model_uri}")
logger.debug(f"Previous Model URI: {previous_model_uri}")

model_new = mlflow.sklearn.load_model(new_model_uri)
predictions_new = model_new.predict(X_test)

model_previous = mlflow.sklearn.load_model(previous_model_uri)
predictions_previous = model_previous.predict(X_test)

logger.info(f"Predictions for New Model: {predictions_new}")
logger.info(f"Previous for Old Model: {predictions_previous}")


# Calculate F1 scores
f1_new = f1_score(y_test, predictions_new)
f1_previous = f1_score(y_test, predictions_previous)

logger.info(f"F1 Score for New Model: {f1_new}")
logger.info(f"F1 Score for Old Model: {f1_previous}")

if f1_new > f1_previous:
    logger.info("New model performs better. Registering...")
    model_version = mlflow.register_model(
        model_uri=new_model_uri,
        name=f"{catalog_name}.{schema_name}.ad_click_model_basic",
        tags={"branch": git_branch, "git_sha": f"{git_sha}", "job_run_id": job_run_id},
    )
    workspace.dbutils.jobs.taskValues.set(key="model_version", value=model_version.version)
    workspace.dbutils.jobs.taskValues.set(key="model_update", value=1)
    logger.info(f"New model registered with version: {model_version.version}")
else:
    logger.info("Previous model performs better. No update needed.")
    workspace.dbutils.jobs.taskValues.set(key="model_update", value=0)
