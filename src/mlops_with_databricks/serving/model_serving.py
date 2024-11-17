# Databricks notebook source
# MAGIC %pip install ../housing_price-0.0.1-py3-none-any.whl

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    Route,
    ServedEntityInput,
    TrafficConfig,
)
from pyspark.sql import SparkSession

from mlops_with_databricks.data_preprocessing.dataclasses import (
    DatabricksConfig,
    FeatureTableConfig,
    ModelConfig,
    ModelServingConfig,
    ProcessedAdClickDataConfig,
)

workspace = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

num_features = list(ProcessedAdClickDataConfig.num_features)
cat_features = list(ProcessedAdClickDataConfig.cat_features)
target = ProcessedAdClickDataConfig.target
catalog_name = DatabricksConfig.catalog_name
schema_name = DatabricksConfig.schema_name
feature_table_stem = FeatureTableConfig.feature_table_name
online_table_stem = FeatureTableConfig.online_table_name
model_name = ModelConfig.model_name
model_version = ModelConfig.model_version
serving_endpoint_name = ModelServingConfig.serving_endpoint_name

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()

workspace.serving_endpoints.create(
    name=serving_endpoint_name,
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.{model_name}",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=model_version,
            )
        ],
        # Optional if only 1 entity is served
        traffic_config=TrafficConfig(
            routes=[Route(served_model_name=f"{model_name}-{model_version}", traffic_percentage=100)]
        ),
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call the endpoint

# COMMAND ----------

token = workspace.dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create sample request body

# COMMAND ----------

required_columns = num_features + cat_features

sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

"""
Each body should be list of json with columns

[{'LotFrontage': 78.0,
  'LotArea': 9317,
  'OverallQual': 6,
  'OverallCond': 5,
  'YearBuilt': 2006,
  'Exterior1st': 'VinylSd',
  'Exterior2nd': 'VinylSd',
  'MasVnrType': 'None',
  'Foundation': 'PConc',
  'Heating': 'GasA',
  'CentralAir': 'Y',
  'SaleType': 'WD',
  'SaleCondition': 'Normal'}]
"""

# COMMAND ----------
start_time = time.time()

model_serving_endpoint = f"https://{host}/serving-endpoints/{serving_endpoint_name}/invocations"
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Test

# COMMAND ----------

# Initialize variables
model_serving_endpoint = f"https://{host}/serving-endpoints/{serving_endpoint_name}/invocations"

headers = {"Authorization": f"Bearer {token}"}
num_requests = 1000


# Function to make a request and record latency
def send_request():
    random_record = random.choice(dataframe_records)
    start_time = time.time()
    response = requests.post(
        model_serving_endpoint,
        headers=headers,
        json={"dataframe_records": random_record},
    )
    end_time = time.time()
    latency = end_time - start_time
    return response.status_code, latency


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
