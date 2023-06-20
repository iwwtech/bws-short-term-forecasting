from app.models.algorithms.base import ForecasterBase
from app.models.algorithms.darts_base import DartsForecaster
from app.models.algorithms.xgb import XGBForecaster
from app.models.algorithms.triple_exponential_smoothing import TESForecaster
from app.models.algorithms.auto_arima import AutoArimaForecaster
from app.models.algorithms.nbeats import NBEATSForecaster
from app.models.algorithms.temporal_fusion_transformer import TFTForecaster
from app.models.algorithms.prophet import ProphetForecaster

from app.utils.exception_handling import BadRequestException
from app.utils.feature_processing import to_camel_case

from app.controller.data_controller import get_train_time

from app.models.weather_agent_example import WeatherAgent

from app.utils.constants import USE_WEATHER

import ray


VALID_ALGORITHMS = [
    "prophet",
    "xgboost",
    "tripleexponentialsmoothing",
    "autoarima",
    "nbeats",
    "temporalfusiontransformer",
]


def get_model(meter_id, algorithm: str):
    algorithm = algorithm.strip().lower()
    wa = WeatherAgent() if USE_WEATHER else None
    if algorithm == "prophet":
        return ProphetForecaster(meter_id, weather_agent=wa)
    if algorithm == "xgboost":
        return XGBForecaster(meter_id, weather_agent=wa)
    elif algorithm == "tripleexponentialsmoothing":
        return TESForecaster(meter_id, weather_agent=wa)
    elif algorithm == "autoarima":
        return AutoArimaForecaster(meter_id, weather_agent=wa)
    elif algorithm == "nbeats":
        return NBEATSForecaster(meter_id, weather_agent=wa)
    elif algorithm == "temporalfusiontransformer":
        return TFTForecaster(meter_id, weather_agent=wa)
    else:
        raise BadRequestException(
            f'No implementation found for algorithm "{algorithm}"'
        )


def get_all_algorithms(include_params: bool = False):
    """
    Generates specifications of all algorithms.
    :param include_params: If True, the specifications will include the
        parameters of each algorithm.
    :return: A list of dictionaries, each containing the name, description
        and parameters of an algorithm.
    """
    algs = [
        ProphetForecaster,
        XGBForecaster,
        TESForecaster,
        AutoArimaForecaster,
        NBEATSForecaster,
        TFTForecaster,
    ]
    return [get_algorithm_spec(alg(None), include_params) for alg in algs]


def get_algorithm_spec(algorithm: ForecasterBase, include_params: bool = False):
    alg_spec = {
        "name": algorithm.name,
        "description": algorithm.description,
        "estimatedTrainingTime": get_train_time(algorithm.name),
    }
    if include_params:
        alg_spec["parameters"] = get_param_specs(algorithm)
    return alg_spec


def get_param_specs(model):
    """
    Accepts a model and generates a specification of its parameters.
    
    If this is a Darts based model, the search space, default values and
    descriptions of all the parameters are inferred from the model's functions
    get_hyperparam_search_space, get_static_params and get_hyperparam_description.
    Example output of get_hyperparam_search_space:
        {"colsample_bytree": tune.uniform(0.2, 0.8), "gamma": tune.uniform(0, 5)}
    Example output of get_default_tunable_params:
        {"colsample_bytree": 0.3, "gamma": 1}
    Example output of get_param_descriptions:
        {"colsample_bytree": "Subsample ratio of columns when constructing each tree.",
         "gamma": "Minimum loss reduction required to make a further partition on a leaf node of the tree."}

    :return: A dictionary that maps parameter names to their well defined specification.
        Example output:
        [{
            "name": "colsample_bytree",
            "description": "Subsample ratio of columns when constructing each tree.",
            "type": "float",
            "default": 0.3,
            "options": {"minValue": 0.2, "maxValue": 0.8}
        },
        {
            "name": "gamma",
            ...
        }
        ]
    """
    if isinstance(model, DartsForecaster):
        search_space = model.get_hyperparam_search_space()
        default_params = model.get_default_tunable_params()
        descriptions = model.get_param_descriptions()

        params_specs = []
        for param_name, param in search_space.items():
            param_in_defaults = param_name in default_params
            param_in_descriptions = param_name in descriptions
            if not param_in_defaults or not param_in_descriptions:
                raise ValueError(
                    f"Lacking default value or description for parameter {param_name}."
                )

            param_spec = {
                "name": to_camel_case(param_name),
                "description": descriptions[param_name],
                "default": default_params[param_name],
            }

            is_float = isinstance(param, ray.tune.search.sample.Float)
            is_int = isinstance(param, ray.tune.search.sample.Integer)
            is_cat = isinstance(param, ray.tune.search.sample.Categorical)
            is_bool = is_cat and isinstance(param.categories[0], bool)

            if is_float:
                param_spec["type"] = "float"
            elif is_int:
                param_spec["type"] = "integer"
            elif is_bool:
                param_spec["type"] = "boolean"
            elif is_cat:
                param_spec["type"] = "categorical"
            else:
                raise ValueError(
                    f"Parameter {param_name} has an invalid type {type(param)}."
                )

            if is_float or is_int:
                param_spec["options"] = {
                    "minValue": param.lower,
                    "maxValue": param.upper,
                }
            elif is_cat or is_bool:
                param_spec["options"] = {"categories": param.categories}

            params_specs.append(param_spec)

        return params_specs

    else:
        raise ValueError(
            f"Generating parameter specification for model {model} not implemented."
        )
