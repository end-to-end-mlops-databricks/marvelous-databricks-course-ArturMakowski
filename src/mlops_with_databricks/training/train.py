# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_students/armak58/packages/mlops_with_databricks-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import subprocess

import mlflow
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mlops_with_databricks.data_preprocessing.dataclasses import (
    DatabricksConfig,
    ProcessedAdClickDataConfig,
    light_gbm_config,
)


def get_git_info():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("ascii").strip()
        return {"git_sha": sha, "branch": branch}
    except Exception:
        return {"git_sha": "unknown", "branch": "unknown"}


mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")  # It must be -uc for registering models to Unity Catalog

# COMMAND ----------

# Extract configuration details
num_features = ProcessedAdClickDataConfig.num_features
cat_features = ProcessedAdClickDataConfig.cat_features
target = ProcessedAdClickDataConfig.target
catalog_name = DatabricksConfig.catalog_name
schema_name = DatabricksConfig.schema_name
parameters = light_gbm_config

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

# Load training and testing sets from Databricks tables
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
print(train_set.head())
X_train = train_set[list(num_features) + list(cat_features)]
y_train = train_set[target]

X_test = test_set[list(num_features) + list(cat_features)]
y_test = test_set[target]

# COMMAND ----------
# Define the preprocessor for categorical features
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
)

# Create the pipeline with preprocessing and the LightGBM regressor
pipeline = Pipeline(steps=[("onehot", preprocessor), ("classifier", LGBMClassifier(**parameters))])

# Define parameter grid for hyperparameter tuning
param_grid = {
    "classifier__learning_rate": [0.001, 0.01, 0.1],
    "classifier__n_estimators": [100, 200, 300, 400],
    "classifier__max_depth": [5, 10, 15, 20],
}

# Perform hyperparameter tuning with GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=4, scoring="f1", n_jobs=-1)

# COMMAND ----------
mlflow.set_experiment(experiment_name="/Shared/ad-click")

# Start an MLflow run to track the training process
with mlflow.start_run(
    tags=get_git_info(),
) as run:
    run_id = run.info.run_id

    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    y_pred = best_pipeline.predict(X_test)

    # Evaluate the model performance
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(best_params)
    mlflow.log_metrics({"f1": f1, "accuracy": accuracy, "precision": precision, "recall": recall, "roc_auc": roc_auc})
    signature = infer_signature(model_input=X_test, model_output=y_pred)

    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")

    mlflow.sklearn.log_model(sk_model=best_pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)


# COMMAND ----------
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model",
    name=f"{catalog_name}.{schema_name}.ad_click_model_basic",
    tags=get_git_info(),
)
