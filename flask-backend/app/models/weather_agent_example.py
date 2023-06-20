import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Union
from app.controller.data_controller import get_physical_meter_ids_of


class WeatherAgent:
    def __init__(self):
        self._csv_path = os.path.join(
            os.getcwd(), "app", "data", "weather", "example_precipitation.csv"
        )

    def get_available_features(
        self,
        meter_ids: Union[str, List[str]] = None,
        coords: Union[Dict[str, float], List[Dict[str, float]]] = None,
    ):
        """
        Returns a list of available feature names. The features may depend on the meter_ids or coordinates data is requested for,
        and either represent aggregations of weather feature of all meters or the raw weather features of each meter.
        Example 1 return value: ["temperature (°C)", "precipitation (mm)", "precipitation (yes/no)"]
        Example 2 return value: ["temperature (°C) - meter_1", "temperature (°C) - meter_2", "precipitation (mm) - meter_1", ...]
        :param meter_ids: String or list of meter ids.
        :param coords: Dict or list of dicts of the form {"lat": lat, "lon": lon}
        :return: List of string feature names.
        """
        # pm_ids = self._get_physical_meter_ids_of(meter_ids)
        df = self._read_dataframe()

        # In this example, we assume that the same features are available for all meters and that
        # want to return a feature only once (in aggregated form), even if it is available for multiple meters.
        # The first level of the column index is the feature name, the second level is the meter id
        # Here, we return a list of unique weather features that exist.
        # In case different meters have different features, we may want to return the features
        # that all meters have in common if we only want to return a single time series per feature.
        # However, we could also return a time series / feature for every meter. In such a case,
        # we would e.g. not return ["precipitation (mm)"], but ["precipitation (mm) - meter_1", "precipitation (mm) - meter_2", ...]
        available_features = df.columns.get_level_values(0).unique().tolist()
        return available_features

    def get_data(
        self,
        meter_ids: Union[str, List[str]],
        start_date: datetime,
        end_date: datetime,
        coords: Union[Dict[str, float], List[Dict[str, float]]] = None,
    ):
        """
        Returns a DataFrame of weather data (or other features) given either
        a meter_id(s) or lat/lon coordinates or both.        
        :param meter_ids: String or list of meter ids.
        :param start_date: The start date of the requested weather data (inclusive).
        :param end_date: Optionally, the end date of the requested weather data (inclusive).
        :param coords: Dict or list of dicts of the form {"lat": lat, "lon": lon}
        :return: Pandas DataFrame with hourly datetime index where each column is named after a feature from get_available_features().
        """
        pm_ids = self._get_physical_meter_ids_of(meter_ids)
        df = self._read_dataframe()
        df = self._aggregate_for_meters(df, pm_ids)
        df = self._handle_missing_data(df, start_date, end_date)
        return df

    def _get_physical_meter_ids_of(self, meter_id: Union[str, List[str]]):
        """
        Returns a list of physical meter ids that are associated with the given meter_id(s).
        :param meter_id: String or list of virtual and/or physical meter ids.
        :return: List of physical meter ids.
        """
        if isinstance(meter_id, str):
            return get_physical_meter_ids_of(meter_id)
        elif isinstance(meter_id, list):
            meter_ids = []
            for m_id in meter_id:
                meter_ids.extend(get_physical_meter_ids_of(m_id))
            return meter_ids
        else:
            raise TypeError(
                f"meter_id must be of type str or list, but was {type(meter_id)}"
            )

    def _read_dataframe(self):
        """
        Reads in the weather data from the pickle file.
        In this example, the weather data is stored as a pandas DataFrame that has
        a multiindex for columns where the first level indicates the data type and
        the second level represents the meter id.
        :return: Pandas DataFrame with weather data.
        """
        return pd.read_csv(self._csv_path, header=[0, 1], index_col=0, parse_dates=True)

    def _aggregate_for_meters(self, df, pm_ids):
        """
        Selects and aggregates columns of a DataFrame that contain the given meter ids.
        :param df: Pandas DataFrame.
        :param pm_ids: List of physical meter ids.
        :return: Pandas DataFrame with selected columns.
        """
        # Select columns of all features (first level of multiindex)
        # and the relevant meters (second level of multiindex)
        df = df.loc[:, pd.IndexSlice[:, pm_ids]]

        # Aggregate all meters to a single time series per feature
        # Note: This is just a very basic example. In practice, there are
        # more reasonable approaches. For example,
        df = df.groupby(level=0, axis=1).mean()

        return df

    def _handle_missing_data(
        self,
        df: pd.DataFrame,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ):
        """
        Fills in zeros for missing data in the given DataFrame,
        given the start and end date of the final DataFrame.
        :param df: Pandas DataFrame.
        :param start_date: Start date of the requested data.
        :param end_date: End date of the requested data.
        :return: Pandas DataFrame with data.
        """
        if start_date is None and end_date is None:
            raise ValueError("Either start_date or end_date must be given.")
        start_date = (
            pd.to_datetime(start_date) if start_date is not None else df.index[0]
        )
        end_date = pd.to_datetime(end_date) if end_date is not None else df.index[-1]

        # Note that we could repeat whole days or weeks in case there is data missing,
        # however, this example dataset only contains precipitation data which does not
        # follow a daily or weekly pattern. Therefore, we simply fill in zeros.
        if start_date < df.index[0]:
            missing_hours = (df.index[0] - start_date).total_seconds() // 3600
            idx = pd.date_range(start=start_date, periods=missing_hours, freq="H")
            df_start = pd.DataFrame(0, index=idx, columns=df.columns)
            df = pd.concat([df_start, df], axis=0)
        if end_date > df.index[-1]:
            missing_hours = (end_date - df.index[-1]).total_seconds() // 3600
            idx = pd.date_range(
                start=df.index[-1], periods=missing_hours + 1, freq="H"
            )[1:]
            df_end = pd.DataFrame(0, index=idx, columns=df.columns)
            df = pd.concat([df, df_end], axis=0)

        return df
