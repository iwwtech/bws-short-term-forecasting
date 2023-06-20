import holidays
import pandas as pd
from typing import Union
from datetime import datetime

from darts import TimeSeries

from app.utils.constants import COUNTRY_CODE, SUBDIVISION_CODE


def get_time_features_darts(df: pd.DataFrame, date_col="index"):
    """
    Adds time features to the dataframe, e.g. day of the week, month, year, etc.
    :param df: The dataframe to add the time features to (requires a date column or index).
    :param date_col: The name of the column containing the dates or "index".
    :return: The dataframe with the added time features and a list of labels of the added features.
    """
    holiday_dates = holidays.country_holidays(COUNTRY_CODE, subdiv=SUBDIVISION_CODE)

    date_series = df.index.to_series() if date_col == "index" else df[date_col]

    df["day"] = date_series.dt.day
    df["month"] = date_series.dt.month
    df["year"] = date_series.dt.year
    df["is_weekend"] = (date_series.dt.dayofweek >= 5).astype(int)
    df["is_holiday"] = date_series.isin(holiday_dates).astype(int)
    feature_labels = ["day", "month", "year", "is_weekend", "is_holiday"]

    return df[feature_labels]


def add_consumption_features(df: pd.DataFrame, target_col="consumption"):
    """
    Adds features to the dataframe like the mean of the mean of the last 24 hours
    """
    df["consumption_mean_24h"] = df[target_col].rolling(24).mean()
    df["consumption_mean_7d"] = df[target_col].rolling(7 * 24).mean()
    return df


def ts_to_list(ts: TimeSeries, what="values"):
    """
        Converts a Darts TimeSeries into a serializable list of floats and dates
        :param ts: Darts TimeSeries
        :param what: "values" or "dates"
        :return: Tuple of list of float values and list of dates
        """
    if what == "values":
        return ts.pd_series().tolist()
    elif what == "dates":
        return [
            t.isoformat(timespec="seconds")
            for t in ts.time_index.to_pydatetime().tolist()
        ]
    else:
        raise ValueError("ts_to_list() requires \"what\" to be 'values' or 'dates'")


def get_start_of_day(date: Union[datetime, str]):
    if isinstance(date, str):
        date = datetime.fromisoformat(date)
    return date.replace(hour=0, minute=0, second=0, microsecond=0)

def to_camel_case(snake_str):
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])

def is_str_true(value):
    return str(value).lower() in ("yes", "true", "t", "1")
