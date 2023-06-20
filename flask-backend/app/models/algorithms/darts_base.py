import shutil
import numpy as np
import pandas as pd
from time import time
from warnings import warn
from typing import Union
from datetime import datetime, timedelta
from copy import deepcopy

from app.models.algorithms import ForecasterBase
from app.controller import data_controller
from app.utils.exception_handling import BadRequestException
from app.utils.feature_processing import (
    get_time_features_darts,
    ts_to_list,
    get_start_of_day,
)
from app.utils.constants import (
    RAY_NUM_CPUS,
    RAY_NUM_GPUS,
    RAY_N_MODELS,
    RAY_MAX_CONCURRENT,
    RAY_NUM_GPUS_PER_TRIAL,
    RAY_REDUCTION_FACTOR,
    RAY_MAX_T,
)

from ray import tune, shutdown as ray_shutdown, init as ray_init
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.air import session

from darts import TimeSeries
from darts.metrics import mse, rmse, smape, mape
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.forecasting_model import (
    GlobalForecastingModel,
    FutureCovariatesLocalForecastingModel,
    TransferableFutureCovariatesLocalForecastingModel,
)
from darts.models.forecasting.torch_forecasting_model import (
    TorchForecastingModel,
    PastCovariatesTorchModel,
    DualCovariatesTorchModel,
    MixedCovariatesTorchModel,
)
from darts.models import XGBModel

from torchmetrics import MeanSquaredError
from pytorch_lightning.callbacks import EarlyStopping

ts_from_df = TimeSeries.from_dataframe


class DartsForecaster(ForecasterBase):
    def __init__(
        self,
        meter_id: str,
        algorithm: object,
        name: str,
        description: str,
        monitors_val: bool = False,
        weather_agent: object = None,
        add_time_features: bool = True,
    ):
        """
        Base class for all Darts forecasting models.
        :param meter_id: The ID of the meter to be forecasted.
        :param algorithm: The Darts forecasting model to be used.
        :param name: The name of the algorithm.
        :param description: A description of the algorithm.
        :param monitors_val: Whether the algorithm monitors the validation set
                             during training (e.g. for early stopping).
        :param weather_agent: An object that provides weather (or other additional) 
                              features. If none, no external features are used.
        :param add_time_features: Whether to incorporate time features. This is usually
                                  recommended as long as the model supports additional
                                  features (covariates).
        """
        super().__init__(name, meter_id, description)
        self.Algorithm = algorithm
        self.add_time_features = add_time_features
        self.weather_agent = weather_agent
        if weather_agent is not None:
            self.feature_labels = {
                "weather": self.weather_agent.get_available_features()
            }
        else:
            self.feature_labels = {}

        # Darts models accept either future_covariates, past_covariates, or none at all as input
        # Depending on what the model supports, we need to pass them as different arguments
        # Note that past_covariates are always a subset of future_covariates but not vice versa.
        # Find more info here: https://unit8co.github.io/darts/userguide/covariates.html
        if (
            issubclass(algorithm, FutureCovariatesLocalForecastingModel)
            or issubclass(algorithm, TransferableFutureCovariatesLocalForecastingModel)
            or issubclass(algorithm, MixedCovariatesTorchModel)
            or issubclass(algorithm, DualCovariatesTorchModel)
            or issubclass(algorithm, XGBModel)
        ):
            self.cov_label = "future_covariates"
            # print(f"Debug: algorithm {name} supports future covariates")
        elif issubclass(algorithm, PastCovariatesTorchModel):
            self.cov_label = "past_covariates"
            # print(f"Debug: algorithm {name} supports past covariates")
        else:
            self.cov_label = None
            # print(f"DEBUG: algorithm {name} does not support covariates")

        should_use_covs = weather_agent is not None or self.add_time_features
        can_use_covs = self.cov_label is not None
        self.has_covariates = should_use_covs and can_use_covs

        self.is_global = issubclass(self.Algorithm, GlobalForecastingModel)
        # print(f"Debug: {name} is global: {self.is_global}")

        # Whether the model uses a torch backend and can make use of features
        # like early stopping during hyperparameter search to speed up training
        self.uses_torch = issubclass(algorithm, TorchForecastingModel)
        # print(f"Debug: {name} uses torch: {self.uses_torch}")

        self.target_label = "consumption"

        # Whether the fit() function of the model requires a validation dataset argument
        # to monitor the validation loss during training (e.g. for early stopping)
        self.monitors_val = monitors_val

        # How many hours of water demand to predict in one go with the model
        self.forecast_horizon = 24

    def get_measurements(self, new_col_label: str = "consumption"):
        """
        Loads the meter measurements, labels the column correctly and returns the dataframe.
        """
        df_measurements = data_controller.get_measurements(
            self.meter_id, aggregate=True, diff=False
        )
        col_label = df_measurements.columns[0]
        df_measurements = df_measurements.rename(columns={col_label: new_col_label})
        return df_measurements

    def get_covariates(self, df_measurements: pd.DataFrame, future_steps=0):
        """
        Get covariates (additional features) for the given measurements.
        :param df_measurements: Dataframe with meter measurements to extend with covariates.
                                The dataframe may contain all historical measurements of the
                                associated meters even though the covariates are generated for
                                the prediction of future values. In that case, only covariates
                                for future time stamps are needed.
        :param future_steps: Number of hours from the last measurement for which covariates
                             should be generated. This is useful when for example more than
                             24 hours have to be predicted or the model uses covariates that
                             refer to the future (e.g. "it is going to rain in 5 hours").
        """
        is_prediction_time = future_steps > 0
        if is_prediction_time:
            ## Create covariates at prediction time
            # "future covariates", as defined by Darts, refer both to the past and the future.
            # In the current implementation, no model uses more than one week full of historical
            # values to predict the next value. Therefore, it is sufficient to return covariates
            # for the last week of historical values + the future steps.
            start_date = df_measurements.index[-1] - pd.Timedelta(hours=7 * 24)
            end_date = df_measurements.index[-1] + pd.Timedelta(hours=future_steps)
        else:
            ## Create covariates at train time for the whole dataset
            start_date = df_measurements.index[0]
            end_date = df_measurements.index[-1]

        date_index = pd.date_range(start=start_date, end=end_date, freq="H")
        df_covariates = pd.DataFrame(index=date_index, columns=[])

        if self.add_time_features:
            df_time_features = get_time_features_darts(df_covariates, date_col="index")
            self.feature_labels["time"] = df_time_features.columns
            df_covariates = df_time_features
        if self.weather_agent is not None:
            df_weather_feature = self.weather_agent.get_data(
                self.meter_id, start_date, end_date
            )
            df_covariates = pd.concat([df_covariates, df_weather_feature], axis=1)
        return df_covariates

    def train_test_split(
        self,
        df_measurements: pd.DataFrame,
        df_covariates: pd.DataFrame,
        train_size: float,
        with_validation: bool = False,
    ):
        """
        Creates a train-test split of the data, distinguishing between covariates and measurements.
        :param df_measurements: Dataframe with meter measurements.
        :param df_covariates: Dataframe with covariates.
        :param train_size: Fraction of the data to use for training.
        :param with_validation: Whether to return a validation dataset in addition to the train and test set.
        """
        split_date = df_measurements.index[int(len(df_measurements) * train_size)]
        target_series = ts_from_df(df_measurements, value_cols=self.target_label)
        train_target, test_target = target_series.split_before(split_date)
        datasets = {
            "series": train_target,
            "test_series": test_target,
        }

        if self.has_covariates:
            covariate_series = ts_from_df(df_covariates, value_cols=self.covariates_in)
            train_covariates, test_covariates = covariate_series.split_before(
                split_date
            )
            datasets[self.cov_label] = train_covariates
            datasets[f"test_{self.cov_label}"] = test_covariates

        if with_validation:
            # Split test set in half to get validation and new test set
            validation_size = (1 - train_size) / 2
            split_date = df_measurements.index[
                int(len(df_measurements) * (train_size + validation_size))
            ]
            val_target, test_target = test_target.split_before(split_date)
            datasets["val_series"] = val_target
            datasets["test_series"] = test_target

            if self.has_covariates:
                val_covariates, test_covariates = test_covariates.split_before(
                    split_date
                )
                datasets[f"val_{self.cov_label}"] = val_covariates
                datasets[f"test_{self.cov_label}"] = test_covariates

        return datasets

    def get_num_hours_to_predict(
        self,
        last_measurement_date: datetime,
        forecast_date: Union[datetime, str] = None,
    ):
        """
        Returns the number of hours from the last measurement that the model should predict.
        This number corresponds to the forecast horizon plus the number of hours between the
        last measurement and the day for which to create the forecast.
        :param last_measurement_date: Date of the last measurement.
        :param forecast_date: Date / day for which to create the forecast.If None, the forecast
                              is created for the next day (from now's perspective).
        """
        if forecast_date is None:
            forecast_date = datetime.utcnow() + timedelta(days=1)
        forecast_day_start = get_start_of_day(forecast_date)
        forecast_day_end = forecast_day_start + timedelta(days=1)

        if forecast_day_end < last_measurement_date:
            forecast_date = (
                forecast_date
                if isinstance(forecast_date, str)
                else forecast_date.isoformat()
            )
            raise BadRequestException(
                f'Forecast date "{forecast_date}" must be in the future'
                + f" but meter measurements end at {last_measurement_date.isoformat()}"
            )

        full_days = (forecast_day_end - last_measurement_date).days
        remaining_hours = (forecast_day_end - last_measurement_date).seconds // 3600
        return full_days * 24 + remaining_hours

    def find_best_hyperparams(
        self, datasets, static_model_params={}, n_configs=RAY_N_MODELS
    ):
        """
        Performs hyperparameter optimization using BOHB.
        :param datasets: Datasets to use for training and validation specified as a
                         dictionary with the keys "series", "val_series", "future_covariates"
                         and "val_future_covariates" (or past covariates).
                        Values are darts TimeSeries objects.
        :param static_model_params: Dictionary with model parameters that have static values
                                    and thus won't be tuned.
        """
        # Start local ray cluster
        # Note: log_to_driver disables logging from the individual trials to the console
        ray_init(num_cpus=RAY_NUM_CPUS, num_gpus=RAY_NUM_GPUS, log_to_driver=False)

        search_space = self.get_hyperparam_search_space(
            static_params=static_model_params
        )

        search_alg = TuneBOHB()
        search_alg = tune.search.ConcurrencyLimiter(
            search_alg, max_concurrent=RAY_MAX_CONCURRENT
        )

        # Note: Bad trials may be terminated which requires the time attribute
        # This is only possible for algorithms that can use early stopping (like neural networks)
        scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=RAY_MAX_T,
            reduction_factor=RAY_REDUCTION_FACTOR,
            stop_last_trials=False,
        )

        # If the model runs on torch, add a callback for reporting the validation loss
        # on every epoch so the scheduler can terminate bad runs early. In addition to that,
        # add an EarlyStopping callback to increase efficiency even more.
        if self.uses_torch:
            stopper = EarlyStopping(
                monitor="val_MeanSquaredError", patience=8, min_delta=0, mode="min",
            )

            # Reports the loss from torch to ray tune, e.g. for stopping bad runs early
            tune_callback = TuneReportCallback(
                {"loss": "val_Loss", "val_MeanSquaredError": "val_MeanSquaredError",},
                on="validation_end",
            )

            # torch_metrics = MetricCollection([MeanAbsolutePercentageError(), MeanAbsoluteError(), MeanSquaredError()])
            static_model_params["torch_metrics"] = MeanSquaredError()
            static_model_params["pl_trainer_kwargs"] = {
                "callbacks": [tune_callback, stopper],
            }

        trainable_fn = tune.with_parameters(
            self.train_with_tune_parameters,
            datasets=datasets,
            static_model_params=static_model_params,
        )

        # Allocate resources to each trial if a GPU is used
        if RAY_NUM_GPUS > 0:
            trainable_fn = tune.with_resources(
                trainable_fn, {"gpu": RAY_NUM_GPUS_PER_TRIAL}
            )

        tuner = tune.Tuner(
            trainable_fn,
            param_space=search_space,
            # Where to log the output of the individual trials
            # run_config=air.RunConfig(log_to_file=("/core-tool/ray_stdout.log", "/core-tool/ray_stderr.log")),
            tune_config=tune.TuneConfig(
                metric="val_MeanSquaredError",
                mode="min",
                scheduler=scheduler,
                search_alg=search_alg,
                num_samples=n_configs,
            ),
        )
        results = tuner.fit()
        best_parameters = results.get_best_result(
            metric="val_MeanSquaredError", mode="min"
        ).metrics["config"]

        # Clean up /root/ray_results to avoid filling up disk
        def delete_print_err(func, path, exc_info):
            warn(
                f"Error deleting {path} when cleaning up ray results. Error: {exc_info}"
            )

        shutil.rmtree("/root/ray_results", onerror=delete_print_err)

        # Note: This is only needed if we are using a temporary local ray cluster
        # instead of a permanent one that we submit jobs to
        ray_shutdown()

        return best_parameters

    def train_with_tune_parameters(self, model_args, datasets, static_model_params):
        """
        Creates, trains and evaluates a combination of hyperparameters that are passed down
        from Ray Tune and reports back the results to Ray Tune.
        """
        model = self.Algorithm(**self.convert_format(model_args, static_model_params))

        # Some models, like Prophet, do not accept validation sets as arguments, therefore we need
        # to remove it before passing the data to fit() and evaluate separately on it afterwards.
        if not self.name in ["Prophet"]:
            model.fit(**datasets)
        else:
            model.fit(**self.get_train_datasets(datasets))

        # If using a torch model, the validation loss is reported after every training epoch.
        # For all other models, we have to manually evaluate the model after training finished.
        if not self.uses_torch:
            results = self.evaluate(model, datasets)
            session.report(
                {"val_MeanSquaredError": results["metrics"]["mse"], "done": True}
            )

    def concat_train_val_test(self, datasets: dict):
        """
        Concatenates the train, validation and test sets for the series and covariates.
        It might be the case that both or only one of val and test sets are present
        as evaluate() might be called during final evaluation (val+test) or during training (val)
        :return: A dictionary with the concatenated datasets for series and covariates, as well as
                 the date when the validation set (or test set if present) starts.
        """
        full_datasets = {}

        has_val = "val_series" in datasets
        has_test = "test_series" in datasets

        full_series = datasets["series"]
        if has_val:
            val_series = datasets["val_series"]
            full_series = full_series.concatenate(val_series)
        if has_test:
            test_series = datasets["test_series"]
            full_series = full_series.concatenate(test_series)
        full_datasets["series"] = full_series

        if self.has_covariates:
            full_cov = datasets[self.cov_label]
            if has_val:
                val_fut_cov = datasets[f"val_{self.cov_label}"]
                full_cov = full_cov.concatenate(val_fut_cov)
            if has_test:
                test_fut_cov = datasets[f"test_{self.cov_label}"]
                full_cov = full_cov.concatenate(test_fut_cov)
            full_datasets[self.cov_label] = full_cov

        if has_test:
            start_eval = datasets["test_series"].start_time()
        else:
            start_eval = datasets["val_series"].start_time()

        return full_datasets, start_eval

    def backtest(
        self,
        model,
        series: TimeSeries,
        start_eval: datetime,
        metrics: dict,
        covariates: TimeSeries = None,
        include_predictions: bool = False,
    ):
        """
        Given a model and a dataset, applies the model to all full days (24 hours) starting
        from the date start_eval. Returns the mean error on those days and returns the raw
        predictions and actual values, if requested.
        :param model: The model to evaluate (one of Darts' forecasting models)
        :param series: The TimeSeries object containing both the data to evaluate on and
                       historical data that may has to be passed to predict() too.
        :param start_eval: The date from which to start evaluating forecasts.
        :param metrics: A dictionary that maps strings to callables. The callables should take
                        two TimeSeries objects (true, pred) as input and return a float.
        :param covariates: The TimeSeries object containing the covariates for the future.
                           Only required if the model supports and was trained with covariates.
        :param include_predictions: If True, the raw predictions and actual values are returned, too.
                                    Useful for visualizations.
        """
        assert (
            series.time_index.freqstr == "H"
        ), "Data must have hourly freq for backtest()"

        results = {"metrics": {}}
        errors = {name: [] for name in metrics.keys()}
        actual_series = series.split_before(start_eval)[1]

        pred_args = {}
        if self.has_covariates:
            pred_args[self.cov_label] = covariates

        # local forecasting models cannot make use of past observations that
        # were not seen during training, thus we have to predict all values until the
        # desired date, starting from the last observation seen during training.
        # Instead of predicting one day at a time, we predict all days at once,
        # compute the errors 24 hours at a time and then average the days.
        if not self.is_global:
            pred_args["n"] = len(actual_series)
            forecasted_series = model.predict(**pred_args)

            start = start_eval
            end = start + pd.Timedelta(days=1)
            while end in series.time_index:
                y_true = actual_series.slice_n_points_after(start, 24)
                y_pred = forecasted_series.slice_n_points_after(start, 24)
                for name, metric_fn in metrics.items():
                    try:
                        errors[name].append(metric_fn(y_true, y_pred))
                    except:
                        # Metrics like MAPE can be undefined if the true value is 0.
                        # If you see messages like "ValueError: The actual series must be strictly positive to compute the MAPE.",
                        # it is printed by the above line and you can ignore it.
                        errors[name].append(np.nan)
                start = end
                end = start + pd.Timedelta(days=1)

        # Global models make use of the most recent observations at inference time
        # and thus we can move in 24h steps through the dataset and create a forecast for each day.
        else:
            forecasted_series = None
            start = start_eval
            end = start + pd.Timedelta(days=1)
            while end in series.time_index:
                hist_series, eval_series = series.split_before(start)
                pred_args["n"] = 24
                pred_args["series"] = hist_series
                y_pred = model.predict(**pred_args)
                y_true = eval_series.slice_n_points_after(start, 24)

                if forecasted_series is None:
                    forecasted_series = y_pred
                else:
                    forecasted_series = forecasted_series.append(y_pred)

                for name, metric_fn in metrics.items():
                    errors[name].append(metric_fn(y_true, y_pred))

                # Move to next day
                start = end
                end = start + pd.Timedelta(days=1)

        for name, error_list in errors.items():
            results["metrics"][name] = np.mean(error_list)
            if np.isnan(results["metrics"][name]):
                results["metrics"][name] = None

        if include_predictions:
            actual_series = self.target_scaler.inverse_transform(actual_series)
            forecasted_series = self.target_scaler.inverse_transform(forecasted_series)

            results["actualTestConsumption"] = ts_to_list(actual_series, what="values")
            results["predictedTestConsumption"] = ts_to_list(
                forecasted_series, what="values"
            )
            results["testTimestamps"] = ts_to_list(actual_series, what="dates")

            if self.has_covariates:
                # Undo scaling and select the covariates for the test period only
                covariates = self.covariate_scaler.inverse_transform(covariates)
                covariates = covariates.slice_intersect(forecasted_series)

                # Add every covariate / feature individually
                results["testCovariates"] = {}
                for cov_name in covariates.components:
                    results["testCovariates"][cov_name] = ts_to_list(
                        covariates.univariate_component(cov_name), what="values"
                    )

        return results

    def evaluate(self, model, datasets, include_predictions=False):
        """
        Applies the model to all full days in the validation set and evaluates it.
        :param model: The model to evaluate (one of Darts' forecasting models)
        :param datasets: Dictionary that contains the training, val and/or test datasets.
        :param include_predictions: If True, the raw predictions and actual values are returned, too.
        :return: A dictionary containing the evaluation results. The key "metrics" returns a dictionary
                 of metrics that describe the mean error per day. The keys "actualTestConsumption",
                 "predictedTestConsumption" and "testTimestamps" refer to the actual and predicted values
                 as well as the timestamps of the test set (only if `include_predictions` is True).
        """

        full_datasets, start_eval = self.concat_train_val_test(datasets)

        # Make sure that the start time is at the first midnight in the dataset
        # so that every 24h forecast, covers exactly a single day. First forecast
        # should be for 1am, last should be 12 pm.
        if start_eval.hour != 1:
            start_eval = pd.Timestamp(
                year=start_eval.year, month=start_eval.month, day=start_eval.day, hour=1
            ) + pd.Timedelta(days=1)

        metrics = {
            "mse": mse,
            "rmse": rmse,
            "smape": smape,
            "mape": mape,
        }

        backtest_args = {
            "model": model,
            "series": full_datasets["series"],
            "start_eval": start_eval,
            "metrics": metrics,
            "include_predictions": include_predictions,
        }
        if self.has_covariates:
            backtest_args["covariates"] = full_datasets[self.cov_label]

        return self.backtest(**backtest_args)

    def predict(
        self,
        forecast_date: Union[datetime, str] = None,
        include_intermediary: bool = False,
    ):
        """
        Given a forecast date in the future, predicts the consumption for the 24 hours
        of that day, as well as all hours between the last measurement and the forecast date if wanted.
        :param forecast_date: The date for which to predict the consumption. If None, the
                              the next day from when this function is called is used.
        :param include_intermediary: If True, the predictions for all hours between the last
                                     measurement and the forecast date are included in the result.
                                     Otherwise, only the 24h forecast for the forecast date is returned.
                                     This is only important for "local" models that need to predict
                                     all hours between the last measurement and the forecast date.
        """
        if forecast_date is None:
            forecast_date = datetime.utcnow() + timedelta(days=1)

        df_measurements = self.get_measurements(new_col_label="consumption")

        # If this is a local forecasting model, we need to predict all hours starting from the
        # date of the last training sample. A global model however can use the additional observations
        # since the last training and does not have to predict them.
        last_obs = df_measurements.index[-1] if self.is_global else self.train_end_date
        n = self.get_num_hours_to_predict(last_obs, forecast_date)
        pred_args = {"n": n}

        if self.is_global:
            series = ts_from_df(df_measurements, value_cols=self.target_label)
            series = self.target_scaler.transform(series)
            pred_args["series"] = series

        if self.has_covariates:
            df_covariates = self.get_covariates(df_measurements, future_steps=n)
            covariates = ts_from_df(df_covariates, value_cols=self.covariates_in)
            covariates = self.covariate_scaler.transform(covariates)
            pred_args[self.cov_label] = covariates

        # print(f"Debug pred_arg keys: {[k for k in pred_args.keys()]}")
        # print(f"Debug model: {self.model}")
        forecast = self.model.predict(**pred_args)
        forecast = self.target_scaler.inverse_transform(forecast)

        if not include_intermediary:
            # Note that start and end are inclusive, therefore we need to add 23 hours
            # the first prediction has to be at 1am, the last at 12pm
            midnight = pd.Timestamp(get_start_of_day(forecast_date))
            start = midnight + pd.Timedelta(hours=1)
            end = start + timedelta(hours=22)
            forecast = forecast.slice(start, end)

        results = {
            "forecastValues": ts_to_list(forecast, what="values"),
            "forecastTimestamps": ts_to_list(forecast, what="dates"),
        }

        if self.has_covariates:
            covariates = self.covariate_scaler.inverse_transform(covariates)
            covariates = covariates.slice_intersect(forecast)
            results["forecastCovariates"] = {}
            for cov_name in covariates.components:
                results["forecastCovariates"][cov_name] = ts_to_list(
                    covariates.univariate_component(cov_name), what="values"
                )

        return results

    def scale(self, datasets):
        """Applies min-max scaling to the datasets."""
        # NOTE: We apply min-max scaling, but it could be worth exploring standardization instead
        self.target_scaler = Scaler()
        datasets["series"] = self.target_scaler.fit_transform(datasets["series"])
        datasets["test_series"] = self.target_scaler.transform(datasets["test_series"])

        has_val_set = "val_series" in datasets
        if has_val_set:
            datasets["val_series"] = self.target_scaler.transform(
                datasets["val_series"]
            )

        if self.has_covariates:
            self.covariate_scaler = Scaler()
            datasets[self.cov_label] = self.covariate_scaler.fit_transform(
                datasets[self.cov_label]
            )
            datasets[f"test_{self.cov_label}"] = self.covariate_scaler.transform(
                datasets[f"test_{self.cov_label}"]
            )
            if has_val_set:
                datasets[f"val_{self.cov_label}"] = self.covariate_scaler.transform(
                    datasets[f"val_{self.cov_label}"]
                )

        return datasets

    def get_train_datasets(self, datasets, with_val=False, merge_val_into_train=False):
        """
        Constructs the training datasets to pass to the model's fit() method
        :param datasets: Dictionary that contains the observed series and covariates.
        :param with_val: If True, the validation set is returned, too.
                         Used for example during hyperparameter optimization to stop bad runs early.
        :param merge_val_into_train: If True, the validation set is merged into the training set.
                                     This can e.g. be necessary for final training after the validation set
                                     was used during hyperparameter optimization.
        """
        assert not (
            with_val and merge_val_into_train
        ), "Cannot use both options together"

        exists_val = "val_series" in datasets

        train_datasets = {"series": datasets["series"]}

        needs_val_set = self.monitors_val or with_val
        if needs_val_set and exists_val:
            train_datasets["val_series"] = datasets["val_series"]
        elif merge_val_into_train and exists_val:
            train_datasets["series"] = train_datasets["series"].concatenate(
                datasets["val_series"]
            )

        if self.has_covariates:
            train_datasets[self.cov_label] = datasets[self.cov_label]
        if self.has_covariates and needs_val_set and exists_val:
            train_datasets[f"val_{self.cov_label}"] = datasets[f"val_{self.cov_label}"]
        elif self.has_covariates and merge_val_into_train and exists_val:
            train_datasets[self.cov_label] = train_datasets[self.cov_label].concatenate(
                datasets[f"val_{self.cov_label}"]
            )

        return train_datasets

    def fit(
        self,
        train_size: float = 0.8,
        hyper_opt: bool = False,
        hyperparameters: dict = {},
        n_configs: int = RAY_N_MODELS,
    ):
        """
        Fits the model to the data and optionally performs hyperparameter optimization.
        :param train_size: Fraction of the data to use for training.
        :param hyper_opt: Whether to perform hyperparameter optimization. If false, default
                          hyperparameters are used.
        """
        df_measurements = self.get_measurements(new_col_label="consumption")

        if self.has_covariates:
            df_covariates = self.get_covariates(df_measurements)
            self.covariates_in = df_covariates.columns
        else:
            df_covariates = None

        datasets = self.train_test_split(
            df_measurements,
            df_covariates,
            train_size=train_size,
            with_validation=(self.monitors_val or hyper_opt),
        )
        datasets = self.scale(datasets)

        # Hyperparameter optimization
        tunable_params = self.get_default_tunable_params()
        static_params = self.get_static_params()

        # Make parameters non-tunable if they are given a value by the user
        for param, value in hyperparameters.items():
            static_params[param] = value
            if param in tunable_params:
                tunable_params.pop(param)

        if hyper_opt:
            hyperopt_datasets = self.get_train_datasets(datasets, with_val=True)
            best_hyperparams = self.find_best_hyperparams(
                hyperopt_datasets,
                static_model_params=deepcopy(static_params),
                n_configs=n_configs,
            )
            tunable_params.update(best_hyperparams)

        start_time = time()

        ## Model evaluation on train-test split
        # If the model does not require a validation set, we use it for training
        # Note that if the model is not a global one, it can only start predicting
        # from the last training timestamp. If at the same time, we use a validation set,
        # we will have a large gap between training and test set which we want to avoid,
        # as it would by design inflate the error.
        train_datasets = self.get_train_datasets(
            datasets, merge_val_into_train=not self.is_global
        )
        self.model = self.Algorithm(
            **self.convert_format(tunable_params, static_params)
        )
        self.model.fit(**train_datasets)
        self.eval_results = self.evaluate(
            self.model, datasets, include_predictions=True
        )

        # Final training run on full dataset
        target_series = ts_from_df(df_measurements, value_cols=self.target_label)
        target_series = self.target_scaler.fit_transform(target_series)
        fit_args = {"series": target_series}

        if self.has_covariates:
            covariates = ts_from_df(df_covariates, value_cols=self.covariates_in)
            covariates = self.covariate_scaler.fit_transform(covariates)
            fit_args[self.cov_label] = covariates

        self.model.fit(**fit_args)

        self.hyperparameters = {**tunable_params, **static_params}

        self.train_end_date = target_series.time_index[-1]

        self.train_time = time() - start_time

        return self

    def keep_tunable(self, search_space: dict, static_params: dict):
        """
        Helper method for get_hyperparam_search_space to remove all
        static parameters from the tunable search space.
        """
        for key in static_params:
            if key in search_space:
                del search_space[key]
        return search_space

    def convert_format(self, tunable_args, static_args):
        """
        Converts the format in which the model parameters are specified
        to the format which the algorithm implementation expects.
        Note that this method is overwritten by some subclasses.
        """
        return {**tunable_args, **static_args}

    def get_evaluation_results(self):
        """Returns the evaluation results of the last call to fit()"""
        return self.eval_results

    def get_hyperparam_search_space(self):
        """Returns a dictionary of tunable hyperparameters of the model and their search space"""
        raise NotImplementedError(
            "Subclasses must implement the method get_hyperparam_search_space()"
        )

    def get_default_tunable_params(self):
        """
        Returns a dictionary of default hyperparameters of the model in case no
        hyperparameter optimization is performed
        """
        raise NotImplementedError(
            "Subclasses must implement the method get_default_tunable_params()"
        )

    def get_static_params(self):
        """Returns a dictionary of hyperparameters of the model that should not be tuned."""
        raise NotImplementedError(
            "Subclasses must implement the method get_static_params()"
        )

    def get_param_descriptions(self):
        """Returns a dictionary of hyperparameters of the model and their descriptions."""
        raise NotImplementedError(
            "Subclasses must implement the method get_param_descriptions()"
        )
