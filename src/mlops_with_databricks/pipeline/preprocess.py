"""Preprocess data and update train and test sets."""

import argparse

from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import max as spark_max

from mlops_with_databricks.data_preprocessing.dataclasses import DatabricksConfig

workspace = WorkspaceClient()


parser = argparse.ArgumentParser()

args = parser.parse_args()

spark = SparkSession.builder.getOrCreate()

catalog_name = DatabricksConfig.catalog_name
schema_name = DatabricksConfig.schema_name


source_data = spark.table(f"{catalog_name}.{schema_name}.source_data")

max_train_timestamp = (
    spark.table(f"{catalog_name}.{schema_name}.train_set")
    .select(spark_max("update_timestamp_utc").alias("max_update_timestamp"))
    .collect()[0]["max_update_timestamp"]
)

max_test_timestamp = (
    spark.table(f"{catalog_name}.{schema_name}.test_set")
    .select(spark_max("update_timestamp_utc").alias("max_update_timestamp"))
    .collect()[0]["max_update_timestamp"]
)

latest_timestamp = max(max_train_timestamp, max_test_timestamp)

new_data = source_data.filter(col("update_timestamp_utc") > latest_timestamp)

new_data_train, new_data_test = new_data.randomSplit([0.8, 0.2], seed=42)

new_data_train.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.train_set")
new_data_test.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.test_set")

affected_rows_train = new_data_train.count()
affected_rows_test = new_data_test.count()

if affected_rows_train > 0 or affected_rows_test > 0:
    refreshed = 1
else:
    refreshed = 0

workspace.dbutils.jobs.taskValues.set(key="refreshed", value=refreshed)
