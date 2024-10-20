# Databricks notebook source

# COMMAND ----------

from mlops_with_databricks.data_preprocessing.preprocess import DataProcessor

data = DataProcessor("../data/ad_click_dataset.csv").preprocess_data()
data.head()
