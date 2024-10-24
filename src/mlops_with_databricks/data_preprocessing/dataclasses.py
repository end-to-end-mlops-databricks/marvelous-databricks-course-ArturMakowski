"""Dataclasses for the Ad Click Data."""

from dataclasses import dataclass


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
