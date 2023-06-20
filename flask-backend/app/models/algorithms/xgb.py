from ray import tune
from app.models.algorithms.darts_base import DartsForecaster
from darts.models import XGBModel


class XGBForecaster(DartsForecaster):
    def __init__(self, meter_id: str, weather_agent: object = None):
        super().__init__(
            meter_id,
            XGBModel,
            name="XGBoost",
            description="XGBoost (Extreme Gradient Boosting) is an ensemble learning method that combines multiple weak models (decision trees) to create a strong predictive model.",
            monitors_val=False,
            weather_agent=weather_agent,
            add_time_features=True,
        )

        # Number of past hours of all features / covariates to use for predicting the next hour
        self.past_covariate_lags = 24

        # Number of past hours of all features / covariates to use for predicting the next hour
        self.future_covariate_lags = 0

        # Number of hours of past water consumption to use for predicting the next hour
        self.past_target_lags = 24 * 7

    # For xgboost, here are good explanations of the hyperparameters:
    # https://towardsdatascience.com/mastering-xgboost-2eb6bce6bc76
    def get_hyperparam_search_space(self, **kwargs):
        search_space = {
            "colsample_bytree": tune.uniform(0.2, 0.8),
            "gamma": tune.uniform(0, 5),
            "max_depth": tune.choice([3, 4, 5]),
            "min_child_weight": tune.uniform(1, 8),
            "learning_rate": tune.loguniform(0.005, 0.3),
        }
        if kwargs.get("static_params"):
            return self.keep_tunable(search_space, kwargs["static_params"])
        return search_space

    def get_static_params(self, **kwargs):
        # A note on covariate lags: We need to specify the lags of the covariates for Darts implementations. For example, depending on the context, a lag of k can mean that the covariate value at time t-k will be used to predict the target value at time t.
        # See also: https://unit8.com/resources/time-series-forecasting-using-past-and-future-external-data-with-darts/
        # A tuple (past=24, future=12) means that the 24 past and future 12 hours will be used for prediction
        return {
            "lags": self.past_target_lags,
            "lags_future_covariates": (
                self.past_covariate_lags,
                self.future_covariate_lags,
            ),
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "verbosity": 0,
            "verbose": 0,
        }

    def get_default_tunable_params(self, **kwargs):
        return {
            "colsample_bytree": 0.3,
            "gamma": 1,
            "max_depth": 3,
            "min_child_weight": 1,
            "learning_rate": 0.01,
        }

    def get_param_descriptions(self, **kwargs):
        return {
            "colsample_bytree": "Subsample ratio of columns when constructing each tree.",
            "gamma": "Minimum loss reduction required to make a further partition on a leaf node of the tree.",
            "max_depth": "Maximum depth of a tree.",
            "min_child_weight": "Minimum sum of instance weight (hessian) needed in a child.",
            "learning_rate": "Boosting learning rate (xgb's 'eta').",
        }
