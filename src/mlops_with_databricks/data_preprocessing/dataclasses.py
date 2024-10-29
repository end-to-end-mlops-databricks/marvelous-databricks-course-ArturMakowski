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

    target: str = "cat__click_0"
    num_features: tuple[str] = ("num__age",)
    cat_features: tuple[str] = (
        "cat__gender_Female",
        "cat__gender_Male",
        "cat__gender_Non-Binary",
        "cat__device_type_Desktop",
        "cat__device_type_Mobile",
        "cat__device_type_Tablet",
        "cat__ad_position_Bottom",
        "cat__ad_position_Side",
        "cat__ad_position_Top",
        "cat__browsing_history_Education",
        "cat__browsing_history_Entertainment",
        "cat__browsing_history_News",
        "cat__browsing_history_Shopping",
        "cat__browsing_history_Social_Media",
        "cat__time_of_day_Afternoon",
        "cat__time_of_day_Evening",
        "cat__time_of_day_Morning",
        "cat__time_of_day_Night",
    )


@dataclass
class DatabricksConfig:
    """Dataclass for the Databricks configuration."""

    workspace_url: str = "https://dbc-643c4c2b-d6c9.cloud.databricks.com/"
    catalog_name: str = "mlops_students"
    schema_name: str = "armak58"


class LightGBMConfig(TypedDict):
    learning_rate: str = 0.001
    n_estimators: str = 200
    max_depth: str = 10


light_gbm_config = LightGBMConfig(learning_rate=0.001, n_estimators=200, max_depth=10)
