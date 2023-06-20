from app.utils.exception_handling import BadRequestException

from app.schemas import (
    ProphetHyperparameters,
    TripleExponentialSmoothingHyperparameters,
    AutoArimaHyperparameters,
    XGBoostHyperparameters,
    NbeatsHyperparameters,
    TemporalFusionTransformerHyperparameters,
)

from marshmallow import EXCLUDE


def parse_hyperparameters(hyperparameters, algorithm):
    """Validates the hyperparameters for the specified algorithm."""
    if algorithm == "prophet":
        return ProphetHyperparameters().load(hyperparameters, unknown=EXCLUDE)
    elif algorithm == "tripleexponentialsmoothing":
        return TripleExponentialSmoothingHyperparameters().load(
            hyperparameters, unknown=EXCLUDE
        )
    elif algorithm == "autoarima":
        return AutoArimaHyperparameters().load(hyperparameters, unknown=EXCLUDE)
    elif algorithm == "xgboost":
        return XGBoostHyperparameters().load(hyperparameters, unknown=EXCLUDE)
    elif algorithm == "nbeats":
        return NbeatsHyperparameters().load(hyperparameters, unknown=EXCLUDE)
    elif algorithm == "temporalfusiontransformer":
        return TemporalFusionTransformerHyperparameters().load(
            hyperparameters, unknown=EXCLUDE
        )
    else:
        raise BadRequestException("Invalid algorithm.")
