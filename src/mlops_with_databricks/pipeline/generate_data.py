"""Generate synthetic data and save it to the source_data table."""

import numpy as np
import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp

from mlops_with_databricks.data_preprocessing.dataclasses import (
    DatabricksConfig,
)

catalog_name = DatabricksConfig.catalog_name
schema_name = DatabricksConfig.schema_name

spark = SparkSession.builder.getOrCreate()

train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
combined_set = pd.concat([train_set, test_set], ignore_index=True)


def create_synthetic_data(df: pd.DataFrame, num_rows=100) -> pd.DataFrame:
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        logger.info(f"Creating synthetic data for column: {column}")
        if column == "click":
            synthetic_data[column] = np.random.choice([0, 1], num_rows, p=[0.5, 0.5])
        else:
            if pd.api.types.is_numeric_dtype(df[column]):
                max, min = df[column].max(), df[column].min()
                synthetic_data[column] = np.random.randint(min, max, num_rows)

            elif pd.api.types.is_object_dtype(df[column]):
                synthetic_data[column] = np.random.choice(
                    df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
                )

            elif isinstance(df[column].dtype, pd.CategoricalDtype) or isinstance(df[column].dtype, pd.StringDtype):
                synthetic_data[column] = np.random.choice(
                    df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
                )

            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                min_date, max_date = df[column].min(), df[column].max()
                if min_date < max_date:
                    synthetic_data[column] = pd.to_datetime(np.random.randint(min_date.value, max_date.value, num_rows))
                else:
                    synthetic_data[column] = [min_date] * num_rows

            else:
                synthetic_data[column] = np.random.choice(df[column], num_rows)

    return synthetic_data


synthetic_df = create_synthetic_data(combined_set)

existing_schema = spark.table(f"{catalog_name}.{schema_name}.train_set").schema

synthetic_spark_df = spark.createDataFrame(synthetic_df, schema=existing_schema)

train_set_with_timestamp = synthetic_spark_df.withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

train_set_with_timestamp.show(5)
train_set_with_timestamp.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.source_data")

spark.sql(
    f"ALTER TABLE {DatabricksConfig.catalog_name}.{DatabricksConfig.schema_name}.source_data "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)
