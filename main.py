"""Main script to run the data preprocessing pipeline."""

from mlops_with_databricks.data_preprocessing.preprocess import DataProcessor

X, y = DataProcessor("data/ad_click_dataset.csv").preprocess_data()
X.head()
