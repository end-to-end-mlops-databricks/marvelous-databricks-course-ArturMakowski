# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_students/armak58/packages/mlops_with_databricks-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from mlops_with_databricks.data_preprocessing.preprocess import DataProcessor

data = DataProcessor("/Volumes/mlops_students/armak58/data/ad_click_dataset.csv").preprocess_data()
data[0].head()
