# Databricks notebook source
# MAGIC %pip install --force-reinstall "/Volumes/mlops_students/armak58/packages/mlops_with_databricks-0.0.1-py3-none-any.whl[dev]"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

"""
Create feature table in unity catalog, it will be a delta table
Create online table which uses the feature delta table created in the previous step
Create a feature spec. When you create a feature spec,
you specify the source Delta table.
This allows the feature spec to be used in both offline and online scenarios.
For online lookups, the serving endpoint automatically uses the online table to perform low-latency feature lookups.
The source Delta table and the online table must use the same primary key.
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
import pandas as pd
import requests
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.sql import SparkSession

from mlops_with_databricks.data_preprocessing.dataclasses import (
    DatabricksConfig,
    FeatureServingConfig,
    FeatureTableConfig,
    ModelConfig,
    ProcessedAdClickDataConfig,
)

spark = SparkSession.builder.getOrCreate()

# Initialize Databricks clients
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# Set the MLflow registry URI
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load config, train and test tables

# COMMAND ----------

# Load config
num_features = list(ProcessedAdClickDataConfig.num_features)
cat_features = list(ProcessedAdClickDataConfig.cat_features)
target = ProcessedAdClickDataConfig.target
catalog_name = DatabricksConfig.catalog_name
schema_name = DatabricksConfig.schema_name
feature_table_stem = FeatureTableConfig.feature_table_name
online_table_stem = FeatureTableConfig.online_table_name
model_name = ModelConfig.model_name
model_version = ModelConfig.model_version
serving_endpoint_name = FeatureServingConfig.serving_endpoint_name


# Define table names
feature_table_name = f"{catalog_name}.{schema_name}.{feature_table_stem}"
online_table_name = f"{catalog_name}.{schema_name}.{online_table_stem}"

# Load training and test sets from Catalog
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

df = pd.concat([train_set, test_set])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load a registered model

# COMMAND ----------

# Load the MLflow model for predictions
pipeline = mlflow.sklearn.load_model(f"models:/{catalog_name}.{schema_name}.{model_name}/{model_version}")

# COMMAND ----------

# Prepare the DataFrame for predictions and feature table creation - these features are the ones we want to serve.
preds_df = pd.DataFrame({"Id": range(len(df))})
preds_df["Predicted_AdClick"] = pipeline.predict(df[cat_features + num_features])

preds_df = spark.createDataFrame(preds_df)

# 1. Create the feature table in Databricks

fe.create_table(
    name=feature_table_name, primary_keys=["Id"], df=preds_df, description="Ad click predictions feature table"
)

# Enable Change Data Feed
spark.sql(f"""
    ALTER TABLE {feature_table_name}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

# 2. Create the online table using feature table

spec = OnlineTableSpec(
    primary_key_columns=["Id"],
    source_table_full_name=feature_table_name,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False,
)

# Create the online table in Databricks
online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)

# COMMAND ----------

# 3. Create feture look up and feature spec table feature table

# Define features to look up from the feature table
features = [FeatureLookup(table_name=feature_table_name, lookup_key="Id", feature_names=["Predicted_AdClick"])]

# Create the feature spec for serving
feature_spec_name = f"{catalog_name}.{schema_name}.return_predictions"

fe.create_feature_spec(name=feature_spec_name, features=features, exclude_columns=None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Feature Serving Endpoint

# COMMAND ----------

# 4. Create endpoing using feature spec

# Create a serving endpoint for the house prices predictions
workspace.serving_endpoints.create(
    name=serving_endpoint_name,
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=feature_spec_name,  # feature spec name defined in the previous step
                scale_to_zero_enabled=True,
                workload_size="Small",  # Define the workload size (Small, Medium, Large)
            )
        ]
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call The Endpoint
# COMMAND ----------

token = workspace.dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")
# COMMAND ----------

start_time = time.time()
serving_endpoint = f"https://{host}/serving-endpoints/{serving_endpoint_name}/invocations"
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": [{"Id": "182"}]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")


# COMMAND ----------

# another way to call the endpoint

response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_split": {"columns": ["Id"], "data": [["182"]]}},
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Test

# COMMAND ----------

# Initialize variables
serving_endpoint = f"https://{host}/serving-endpoints/{serving_endpoint_name}/invocations"
id_list = preds_df.select("Id").rdd.flatMap(lambda x: x).collect()
headers = {"Authorization": f"Bearer {token}"}
num_requests = 10


# Function to make a request and record latency
def send_request():
    random_id = random.choice(id_list)
    start_time = time.time()
    response = requests.post(
        serving_endpoint,
        headers=headers,
        json={"dataframe_records": [{"Id": random_id}]},
    )
    end_time = time.time()
    latency = end_time - start_time  # Calculate latency for this request
    return response.status_code, latency


# Measure total execution time
total_start_time = time.time()
latencies = []

# Send requests concurrently
with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(send_request) for _ in range(num_requests)]

    for future in as_completed(futures):
        status_code, latency = future.result()
        latencies.append(latency)

total_end_time = time.time()
total_execution_time = total_end_time - total_start_time

# Calculate the average latency
average_latency = sum(latencies) / len(latencies)

print("\nTotal execution time:", total_execution_time, "seconds")
print("Average latency per request:", average_latency, "seconds")
