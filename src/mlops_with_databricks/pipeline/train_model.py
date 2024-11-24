"""Train a LightGBM model with preprocessing and log the model to MLflow."""

import argparse

import mlflow
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mlops_with_databricks.data_preprocessing.dataclasses import (
    DatabricksConfig,
    ProcessedAdClickDataConfig,
    light_gbm_config,
)

parser = argparse.ArgumentParser()
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
parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
git_sha = args.git_sha
git_branch = args.git_branch
job_run_id = args.job_run_id


spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")


num_features = ProcessedAdClickDataConfig.num_features
cat_features = ProcessedAdClickDataConfig.cat_features
target = ProcessedAdClickDataConfig.target
catalog_name = DatabricksConfig.catalog_name
schema_name = DatabricksConfig.schema_name

train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

X_train = train_set[list(num_features) + list(cat_features)]
y_train = train_set[target]

X_test = test_set[list(num_features) + list(cat_features)]
y_test = test_set[target]

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
)
pipeline = Pipeline(steps=[("onehot", preprocessor), ("classifier", LGBMClassifier(**light_gbm_config))])

mlflow.set_experiment(experiment_name="/Shared/ad-click")

with mlflow.start_run(tags={"branch": git_branch, "git_sha": f"{git_sha}", "job_run_id": job_run_id}) as run:
    run_id = run.info.run_id

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_param("model_type", "LightGBM with preprocessing")

    parameters = {
        "classifier__learning_rate": light_gbm_config["learning_rate"],
        "classifier__n_estimators": light_gbm_config["n_estimators"],
        "classifier__max_depth": light_gbm_config["max_depth"],
    }

    mlflow.log_params(parameters)
    mlflow.log_metrics({"f1": f1, "accuracy": accuracy, "precision": precision, "recall": recall, "roc_auc": roc_auc})
    signature = infer_signature(model_input=X_test, model_output=y_pred)

    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")

    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)


model_uri = f"runs:/{run_id}/lightgbm-pipeline-model"
workspace.dbutils.jobs.taskValues.set(key="new_model_uri", value=model_uri)
