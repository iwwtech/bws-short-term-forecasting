from ray import tune
from app.models.algorithms.darts_base import DartsForecaster
from darts.models import NBEATSModel
from app.utils.constants import N_EPOCHS_TORCH


class NBEATSForecaster(DartsForecaster):
    def __init__(self, meter_id: str, weather_agent: object = None):
        super().__init__(
            meter_id,
            NBEATSModel,
            name="NBEATS",
            description="NBEATS (Neural Basis Expansion Analysis for Time Series) is a deep neural network architecture designed for time series forecasting. Note that this algorithm does not support future weather predictions.",
            monitors_val=True,
            weather_agent=weather_agent,
            add_time_features=True,
        )

    def get_hyperparam_search_space(self, **kwargs):
        search_space = {
            "input_chunk_length": tune.choice([24, 2 * 24, 3 * 24, 7 * 24]),
            "num_blocks": tune.choice([1, 2, 3]),
            "num_layers": tune.choice([2, 4]),
            "layer_widths": tune.choice([256, 512, 1024, 2048]),
            "batch_size": tune.choice([16, 32, 64, 128]),
            "lr": tune.loguniform(0.0001, 0.01),
        }
        if kwargs.get("static_params"):
            return self.keep_tunable(search_space, kwargs["static_params"])
        return search_space

    def get_static_params(self, **kwargs):
        return {
            "n_epochs": N_EPOCHS_TORCH,
            "output_chunk_length": 24,
        }

    def get_default_tunable_params(self, **kwargs):
        return {
            "input_chunk_length": 72,
            "num_blocks": 1,
            "num_layers": 4,
            "layer_widths": 512,
            "batch_size": 32,
            "lr": 0.005,
        }

    def get_param_descriptions(self, **kwargs):
        return {
            "input_chunk_length": "Number of past hours to use for prediction. Given a value of 48, the model will use the past 48 hours to predict the next 24 hours.",
            "num_blocks": "The number of blocks per stack.",
            "num_layers": "Number of fully connected layers with ReLu activation per block",
            "layer_widths": "Number of neurons of the fully connected layers with ReLu activation in the blocks.",
            "batch_size": "Number of samples to process per update step.",
            "lr": "Learning rate for the optimizer. A larger learning rate can lead to faster training but also to worse results or instable training.",
        }

    def convert_format(self, tunable_args, static_args):
        # For torch models we need to pass the learning rate as a specific keyword argument
        args = {
            **tunable_args,
            **static_args,
            "optimizer_kwargs": {"lr": tunable_args["lr"]},
        }
        del args["lr"]
        return args
