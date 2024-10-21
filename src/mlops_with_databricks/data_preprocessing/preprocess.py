"""Data Preprocessing module."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mlops_with_databricks.data_preprocessing.dataclasses import AdClickDataColumns, AdClickDataConfig


class DataProcessor:
    """Data Preprocessor for the Ad Click Data."""

    def __init__(self, filepath: str | Path) -> None:
        self.df = self.load_data(filepath)
        self.X = None
        self.y = None
        self.preprocessor = None

    def load_data(self, filepath: str | Path) -> pd.DataFrame:
        """Load the data from the given filepath."""
        return pd.read_csv(filepath)

    def preprocess_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Preprocess the data. Fill missing values, cast types, and split features and target.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Preprocessed features and target.
        """
        self.df = self.df.drop(columns=[AdClickDataColumns.id, AdClickDataColumns.full_name])
        self.df = self.fill_missing_values(self.df)
        self.df = self.cast_types(self.df)

        self.X = self.df[list(AdClickDataConfig.num_features) + list(AdClickDataConfig.cat_features)]
        self.y = self.df[AdClickDataConfig.target]

        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, list(AdClickDataConfig.num_features)),
                ("cat", categorical_transformer, list(AdClickDataConfig.cat_features)),
            ]
        ).set_output(transform="pandas")
        return self.preprocessor.fit_transform(self.X), self.y

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
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
