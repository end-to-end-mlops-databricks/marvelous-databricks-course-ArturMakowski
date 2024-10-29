"""Data Preprocessing module."""

from pathlib import Path

import pandas as pd
from databricks.connect import DatabricksSession
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mlops_with_databricks.data_preprocessing.dataclasses import AdClickDataColumns, AdClickDataConfig, DatabricksConfig


class DataProcessor:
    """Data Preprocessor for the Ad Click Data."""

    def __init__(self) -> None:
        self.df = None
        self.X = None
        self.y = None
        self.preprocessor = None

    def load_data(self, filepath: str | Path) -> None:
        """Load the data from the given filepath."""
        self.df = pd.read_csv(filepath)

    @classmethod
    def from_pandas(cls, pandas_df: pd.DataFrame) -> "DataProcessor":
        """Create a DataProcessor object from a pandas DataFrame."""
        instance = cls()
        instance.df = pandas_df
        return instance

    def preprocess_data(self) -> None:
        """Preprocess the data. Fill missing values, cast types, and split features and target."""
        self.df = self.df.drop(columns=[AdClickDataColumns.id, AdClickDataColumns.full_name])
        self.df = self.fill_missing_values(self.df)
        self.df[AdClickDataColumns.browsing_history] = self.df[AdClickDataColumns.browsing_history].str.replace(
            " ", "_"
        )
        self.df = self.cast_types(self.df)

        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                # ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, list(AdClickDataConfig.num_features)),
                ("cat", categorical_transformer, list(AdClickDataConfig.cat_features)),
            ]
        ).set_output(transform="pandas")
        preprocessed_features = self.preprocessor.fit_transform(self.df)
        preprocessed_features["click"] = self.df[AdClickDataColumns.click].astype("int64")
        self.df = preprocessed_features

    @staticmethod
    def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in the given DataFrame."""
        df[AdClickDataColumns.age] = df[AdClickDataColumns.age].fillna(df[AdClickDataColumns.age].mean())
        df[AdClickDataColumns.gender] = df[AdClickDataColumns.gender].fillna(df[AdClickDataColumns.gender].mode()[0])
        df[AdClickDataColumns.device_type] = df[AdClickDataColumns.device_type].fillna(
            df[AdClickDataColumns.device_type].mode()[0]
        )
        df[AdClickDataColumns.ad_position] = df[AdClickDataColumns.ad_position].fillna(
            df[AdClickDataColumns.ad_position].mode()[0]
        )
        df[AdClickDataColumns.browsing_history] = df[AdClickDataColumns.browsing_history].fillna(
            df[AdClickDataColumns.browsing_history].mode()[0]
        )
        df[AdClickDataColumns.time_of_day] = df[AdClickDataColumns.time_of_day].fillna(
            df[AdClickDataColumns.time_of_day].mode()[0]
        )
        return df

    @staticmethod
    def cast_types(df: pd.DataFrame) -> pd.DataFrame:
        """Cast the types of the columns in the given DataFrame."""
        df[AdClickDataColumns.gender] = df[AdClickDataColumns.gender].astype("category")
        df[AdClickDataColumns.device_type] = df[AdClickDataColumns.device_type].astype("category")
        df[AdClickDataColumns.ad_position] = df[AdClickDataColumns.ad_position].astype("category")
        df[AdClickDataColumns.browsing_history] = df[AdClickDataColumns.browsing_history].astype("category")
        df[AdClickDataColumns.time_of_day] = df[AdClickDataColumns.time_of_day].astype("category")
        df[AdClickDataColumns.age] = df[AdClickDataColumns.age].astype("int64")
        return df

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> list:
        """Split the data into training and testing sets."""
        return train_test_split(self.df, test_size=test_size, random_state=random_state)

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.option("overwriteSchema", "true").mode("overwrite").saveAsTable(
            f"{DatabricksConfig.catalog_name}.{DatabricksConfig.schema_name}.train_set"
        )

        test_set_with_timestamp.write.option("overwriteSchema", "true").mode("overwrite").saveAsTable(
            f"{DatabricksConfig.catalog_name}.{DatabricksConfig.schema_name}.test_set"
        )

        spark.sql(
            f"ALTER TABLE {DatabricksConfig.catalog_name}.{DatabricksConfig.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {DatabricksConfig.catalog_name}.{DatabricksConfig.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )


if __name__ == "__main__":
    spark = DatabricksSession.builder.profile("dbc-643c4c2b-d6c9").getOrCreate()

    df = spark.read.csv(
        "/Volumes/mlops_students/armak58/data/ad_click_dataset.csv", header=True, inferSchema=True
    ).toPandas()

    data_processor = DataProcessor.from_pandas(df)
    data_processor.preprocess_data()
    train_set, test_set = data_processor.split_data()
    data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
