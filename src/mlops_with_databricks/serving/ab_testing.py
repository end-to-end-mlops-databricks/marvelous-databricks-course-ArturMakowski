# Databricks notebook source
# MAGIC %pip install --force-reinstall "/Volumes/mlops_students/armak58/packages/mlops_with_databricks-0.0.1-py3-none-any.whl[dev]"

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
import hashlib
import time

import mlflow
import pandas as pd
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from lightgbm import LGBMClassifier, LGBMRegressor
from mlflow import MlflowClient
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
    ABTestConfig,
    DatabricksConfig,
    ProcessedAdClickDataConfig,
)
from mlops_with_databricks.training.train import get_git_info

# Set up MLflow for tracking and model registry
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Initialize the MLflow client for model management
client = MlflowClient()

# Load configuration
num_features = ProcessedAdClickDataConfig.num_features
cat_features = ProcessedAdClickDataConfig.cat_features
target = ProcessedAdClickDataConfig.target
catalog_name = DatabricksConfig.catalog_name
schema_name = DatabricksConfig.schema_name
ab_test_params = ABTestConfig

# COMMAND ----------

# Set up specific parameters for model A and model B as part of the A/B test
parameters_a = {
    "classifier__learning_rate": ab_test_params.learning_rate_a,
    "classifier__n_estimators": ab_test_params.n_estimators,
    "classifier__max_depth": ab_test_params.max_depth_a,
}

parameters_b = {
    "classifier__learning_rate": ab_test_params.learning_rate_b,
    "classifier__n_estimators": ab_test_params.n_estimators,
    "classifier__max_depth": ab_test_params.max_depth_b,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Prepare Training and Testing Datasets

# COMMAND ----------

# Initialize a Databricks session for Spark operations
spark = SparkSession.builder.getOrCreate()

# Load the training and testing sets from Databricks tables
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = train_set_spark.toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# Define features and target variables
X_train = train_set[list(num_features) + list(cat_features)]
y_train = train_set[target]

X_test = test_set[list(num_features) + list(cat_features)]
y_test = test_set[target]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Model A and Log with MLflow

# COMMAND ----------

# Define a preprocessor for categorical features, which will one-hot encode categorical variables
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
)

# Build a pipeline combining preprocessing and model training steps
pipeline = Pipeline(steps=[("onehot", preprocessor), ("classifier", LGBMClassifier(**parameters_a))])

# Set the MLflow experiment to track this A/B testing project
mlflow.set_experiment(experiment_name="/Shared/ad-click-ab")
model_name = f"{catalog_name}.{schema_name}.ad_click_model_basic_ab"

# Git commit hash for tracking model version
git = get_git_info()

# Start MLflow run to track training of Model A
with mlflow.start_run(tags={"model_class": "A", "git_sha": git["git_sha"]}) as run:
    run_id = run.info.run_id

    # Train the model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters_a)
    mlflow.log_metrics({"f1": f1, "accuracy": accuracy, "precision": precision, "recall": recall, "roc_auc": roc_auc})
    signature = infer_signature(model_input=X_test, model_output=y_pred)

    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")

    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)


# COMMAND ----------
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model",
    name=f"{catalog_name}.{schema_name}.ad_click_model_basic",
    tags=git,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Model A and Assign Alias

# COMMAND ----------

# Assign alias for easy reference in future A/B tests
model_version_alias = "model_A"

client.set_registered_model_alias(model_name, model_version_alias, f"{model_version.version}")
model_uri = f"models:/{model_name}@{model_version_alias}"
model_A = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Model B and Log with MLflow

# COMMAND ----------

# Repeat the training and logging steps for Model B using parameters for B
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LGBMRegressor(**parameters_b))])

# Start MLflow run to track training of Model A
with mlflow.start_run(tags={"model_class": "B", "git_sha": git["git_sha"]}) as run:
    run_id = run.info.run_id

    # Train the model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters_a)
    mlflow.log_metrics({"f1": f1, "accuracy": accuracy, "precision": precision, "recall": recall, "roc_auc": roc_auc})
    signature = infer_signature(model_input=X_test, model_output=y_pred)

    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")

    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)


# COMMAND ----------
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model",
    name=f"{catalog_name}.{schema_name}.ad_click_model_basic",
    tags=git,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Model A and Assign Alias

# COMMAND ----------

# Assign alias for easy reference in future A/B tests
model_version_alias = "model_B"

client.set_registered_model_alias(model_name, model_version_alias, f"{model_version.version}")
model_uri = f"models:/{model_name}@{model_version_alias}"
model_B = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Custom A/B Test Model

# COMMAND ----------


class AdClickModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, models):
        self.models = models
        self.model_a = models[0]
        self.model_b = models[1]

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            house_id = str(model_input["Id"].values[0])
            hashed_id = hashlib.md5(house_id.encode(encoding="UTF-8")).hexdigest()
            # convert a hexadecimal (base-16) string into an integer
            if int(hashed_id, 16) % 2:
                predictions = self.model_a.predict(model_input.drop(["Id"], axis=1))
                return {"Prediction": predictions[0], "model": "Model A"}
            else:
                predictions = self.model_b.predict(model_input.drop(["Id"], axis=1))
                return {"Prediction": predictions[0], "model": "Model B"}
        else:
            raise ValueError("Input must be a pandas DataFrame.")


# COMMAND ----------
X_train = train_set[num_features + cat_features]
X_test = test_set[num_features + cat_features]


# COMMAND ----------
models = [model_A, model_B]
wrapped_model = AdClickModelWrapper(models)  # we pass the loaded models to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
print("Example Prediction:", example_prediction)

# COMMAND ----------
mlflow.set_experiment(experiment_name="/Shared/ad-click-ab-testing")
model_name = f"{catalog_name}.{schema_name}.ad_click_model_pyfunc_ab_test"

with mlflow.start_run() as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={"Prediction": 1234.5, "model": "Model B"})
    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")
    mlflow.pyfunc.log_model(python_model=wrapped_model, artifact_path="pyfunc-ad-click-model-ab", signature=signature)
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc-ad-click-model-ab", name=model_name, tags={"git_sha": f"{git["git_sha"]}"}
)

# COMMAND ----------
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version.version}")

# Run prediction
predictions = model.predict(X_test.iloc[0:1])

# Display predictions
print(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create serving endpoint

# COMMAND ----------

workspace = WorkspaceClient()

workspace.serving_endpoints.create(
    name="ad-click-model-serving-ab-test",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.ad_click_model_pyfunc_ab_test",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=model_version.version,
            )
        ]
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Call the endpoint

# COMMAND ----------

token = workspace.dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

required_columns = num_features + cat_features
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

start_time = time.time()

model_serving_endpoint = f"https://{host}/serving-endpoints/ad-click-model-serving-ab-test/invocations"

response = requests.post(
    f"{model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[0]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")
