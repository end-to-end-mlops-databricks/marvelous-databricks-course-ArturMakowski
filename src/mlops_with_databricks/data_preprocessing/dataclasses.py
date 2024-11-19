"""Dataclasses for the Ad Click Data."""

from dataclasses import dataclass
from typing import TypedDict


@dataclass
class AdClickDataColumns:
    """Dataclass for the Ad Click Data columns."""

    id: str = "id"
    full_name: str = "full_name"
    age: str = "age"
    gender: str = "gender"
    device_type: str = "device_type"
    ad_position: str = "ad_position"
    browsing_history: str = "browsing_history"
    time_of_day: str = "time_of_day"
    click: str = "click"


@dataclass
class AdClickDataConfig:
    """Dataclass for the Ad Click Data configuration."""

    target: str = AdClickDataColumns.click
    num_features: tuple[str] = (AdClickDataColumns.age,)
    cat_features: tuple[str] = (
        AdClickDataColumns.gender,
        AdClickDataColumns.device_type,
        AdClickDataColumns.ad_position,
        AdClickDataColumns.browsing_history,
        AdClickDataColumns.time_of_day,
    )


@dataclass
class ProcessedAdClickDataConfig:
    """Dataclass for the Processed Ad Click Data configuration."""

    target: str = "click"
    num_features: tuple[str] = ("num__age",)
    cat_features: tuple[str] = (
        "cat__gender",
        "cat__device_type",
        "cat__ad_position",
        "cat__browsing_history",
        "cat__time_of_day",
    )


@dataclass
class DatabricksConfig:
    """Dataclass for the Databricks configuration."""

    workspace_url: str = "https://dbc-643c4c2b-d6c9.cloud.databricks.com/"
    catalog_name: str = "mlops_students"
    schema_name: str = "armak58"


@dataclass
class FeatureTableConfig:
    """Dataclass for the Feature Table configuration."""

    feature_table_name: str = "adclick_preds"
    online_table_name: str = "adclick_preds_online"


@dataclass
class ModelConfig:
    """Dataclass for the Model configuration."""

    model_name: str = "ad_click_model_basic"
    model_version: int = 7


@dataclass
class FeatureServingConfig:
    """Dataclass for the Serving configuration."""

    serving_endpoint_name: str = "ad-click-feature-serving"


@dataclass
class ModelServingConfig:
    """Dataclass for the Model Serving configuration."""

    serving_endpoint_name: str = "ad-click-model-serving"


@dataclass
class ABTestConfig:
    """Dataclass for the A/B Test configuration."""

    learning_rate_a: float = 0.01
    learning_rate_b: float = 0.001
    n_estimators: int = 1000
    max_depth_a: int = 10
    max_depth_b: int = 100


class LightGBMConfig(TypedDict):
    learning_rate: float
    n_estimators: int
    max_depth: int


light_gbm_config = LightGBMConfig(learning_rate=0.001, n_estimators=200, max_depth=10)
