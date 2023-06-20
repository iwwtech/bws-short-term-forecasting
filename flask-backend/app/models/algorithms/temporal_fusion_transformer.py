from ray import tune
from app.models.algorithms.darts_base import DartsForecaster
from darts.models import TFTModel
from app.utils.constants import N_EPOCHS_TORCH


class TFTForecaster(DartsForecaster):
    def __init__(self, meter_id: str, weather_agent: object = None):
        super().__init__(
            meter_id,
            TFTModel,
            name="TemporalFusionTransformer",
            description="Temporal Fusion Transformer (TFT) is a transformer-based neural network architecture developed by Google that combines the strengths of recurrent neural networks (RNNs) for local processing and self-attention layers for capturing long-term dependencies.",
            monitors_val=True,
            weather_agent=weather_agent,
            add_time_features=True,
        )

    def get_hyperparam_search_space(self, **kwargs):
        # Search space derived from the original publication: https://www.sciencedirect.com/science/article/pii/S0169207021000637
        search_space = {
            "input_chunk_length": tune.choice([24, 2 * 24, 3 * 24, 7 * 24]),
            "hidden_size": tune.choice([10, 20, 40, 80, 160]),
            # "lstm_layers": tune.choice([1, 2]),
            "num_attention_heads": tune.choice([1, 4]),
            "dropout": tune.uniform(0.1, 0.5),
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
            "lstm_layers": 1,
        }

    def get_default_tunable_params(self, **kwargs):
        return {
            "input_chunk_length": 72,
            "hidden_size": 40,
            "num_attention_heads": 4,
            "dropout": 0.1,
            "batch_size": 32,
            "lr": 0.005,
        }

    def get_param_descriptions(self, **kwargs):
        return {
            "input_chunk_length": "Number of past hours to use for prediction. Given a value of 48, the model will use the past 48 hours to predict the next 24 hours.",
            "dropout": "Dropout rate to in the model. Dropout is a regularization technique that randomly sets a fraction of the input units for a layer to 0 at each update during training time, which helps generalize the model and reduce overfitting.",
            "hidden_size": "Number of hidden units to store the state of the lstm layer. The larger the state, the more information can be stored after seeing the last t timesteps.",
            "num_attention_heads": "Number of attention heads to in the model. Attention heads are used to capture long-term dependencies in the data.",
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
