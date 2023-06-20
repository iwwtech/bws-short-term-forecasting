from ray import tune
from app.models.algorithms.darts_base import DartsForecaster
from app.utils.constants import COUNTRY_CODE
from darts.models import Prophet


class ProphetForecaster(DartsForecaster):
    def __init__(self, meter_id: str, weather_agent: object = None):
        super().__init__(
            meter_id,
            Prophet,
            name="Prophet",
            description="Prophet is an algorithm developed by Meta that works best for data with strong seasonal effects and several seasons of historical data. It is based on an additive model where the historical and future development is modeled as a sum of several seasonal components on different scales (daily, weekly, yearly, etc...).",
            monitors_val=False,
            weather_agent=weather_agent,
            add_time_features=True,
        )

    def get_hyperparam_search_space(self, **kwargs):
        search_space = {
            "country_holidays": tune.choice([COUNTRY_CODE, "None"]),
            "daily_seasonality": tune.randint(5, 12),
            "weekly_seasonality": tune.randint(2, 5),
            "yearly_seasonality": tune.randint(8, 12),
        }
        if kwargs.get("static_params"):
            return self.keep_tunable(search_space, kwargs["static_params"])
        return search_space

    def get_default_tunable_params(self, **kwargs):
        return {
            "country_holidays": COUNTRY_CODE,
            "daily_seasonality": 8,
            "weekly_seasonality": 3,
            "yearly_seasonality": 10,
        }

    def get_static_params(self, **kwargs):
        return {}
        # return {
        #    "country_holidays": COUNTRY_CODE,
        # }

    def get_param_descriptions(self, **kwargs):
        return {
            "country_holidays": "Country code for the country holidays to be considered. If no code is given, no country holidays are considered.",
            "daily_seasonality": "Fourier order of the daily seasonality component. Higher values allow for more complex daily seasonality patterns but are more prone to capture noise.",
            "weekly_seasonality": "Fourier order of the weekly seasonality component. Higher values allow for more complex weekly seasonality patterns but are more prone to capture noise.",
            "yearly_seasonality": "Fourier order of the yearly seasonality component. Higher values allow for more complex yearly seasonality patterns but are more prone to capture noise.",
        }

    def convert_format(self, tunable_args, static_args):
        """
        Converts the format in which the model parameters are specified
        to the format which the algorithm implementation expects.
        """
        args = {**tunable_args, **static_args}
        is_hol_defined = not args["country_holidays"] and not (
            isinstance(args["country_holidays"], str)
            and args["country_holidays"].lower() in ["none", "null"]
        )
        args["country_holidays"] = args["country_holidays"] if is_hol_defined else None
        return args
