from time import time

from darts.models import ExponentialSmoothing
from darts import TimeSeries

from app.models.algorithms.darts_base import DartsForecaster

ts_from_df = TimeSeries.from_dataframe


class TESForecaster(DartsForecaster):
    def __init__(self, meter_id: str, weather_agent: object = None):
        super().__init__(
            meter_id,
            ExponentialSmoothing,
            name="TripleExponentialSmoothing",
            description="Triple Exponential Smoothing (TES), also called Holt-Winters method, is a forecasting method for seasonal data that uses a weighted average of past observations to predict future values. It takes three factors into account, namely the trend of the series, the seasonality of the series and the remaining part that is not explained by the trend nor seasonality. Note that these parameters are always optimized and thus they don't have to be specified.",
            monitors_val=False,
            add_time_features=False,
        )

    def get_hyperparam_search_space(self, **kwargs):
        return {}

    def get_static_params(self, **kwargs):
        return {}

    def get_default_tunable_params(self, **kwargs):
        # Note: By default TES optimizes these values to maximize the likelihood of the training data
        return {}

    def get_param_descriptions(self, **kwargs):
        return {}

    # NOTE: Overrides the DartsForecaster.fit method, as hyperparameter optimization is always performed implicitly
    def fit(self, train_size: float = 0.8, hyperparameters: dict = {}, **kwargs):
        """
        Fits the model to the data and optionally performs hyperparameter optimization.
        :param train_size: Fraction of the data to use for training.
        """
        df_measurements = self.get_measurements(new_col_label="consumption")

        datasets = self.train_test_split(
            df_measurements,
            df_covariates=None,
            train_size=train_size,
            with_validation=False,
        )
        datasets = self.scale(datasets)

        # Model evaluation on train-test split
        start_time = time()
        train_datasets = self.get_train_datasets(datasets)
        self.model = self.Algorithm(**hyperparameters)
        self.model.fit(**train_datasets)
        self.eval_results = self.evaluate(
            self.model, datasets, include_predictions=True
        )

        # Final training run on full dataset
        target_series = ts_from_df(df_measurements, value_cols=self.target_label)
        target_series = self.target_scaler.fit_transform(target_series)
        self.model.fit(series=target_series)
        self.hyperparameters = hyperparameters
        self.train_time = time() - start_time
        self.train_end_date = target_series.time_index[-1]

        return self
