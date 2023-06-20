from datetime import datetime
from typing import Union


class ForecasterBase:
    def __init__(self, name: str, meter_id: str, description: str):
        """
        :param name: The name of the algorithm.
        :param meter_id: The ID of the meter for which the algorithm is used.
        :param description: A short description of the algorithm.
        """
        self.name = name
        self.meter_id = meter_id
        self.description = description

    def fit(self, train_size: float = 0.8, hyper_opt: bool = False, **kwargs):
        """
        Fits the model to the data.
        :param train_size: The fraction of the data to use for training.
                           The rest may be used only for testing or be
                           split again into a test and validation set.
        :param hyper_opt: Whether to perform hyperparameter optimization.
        """
        raise NotImplementedError

    def predict(
        self,
        forecast_date: Union[datetime, str] = None,
        include_intermediary: bool = False,
        **kwargs
    ):
        """
        Given a forecast date in the future, predicts the consumption for the 24 hours
        of that day, as well as all hours between the last measurement and the forecast date if wanted.
        The results contain the covariates (e.g. weather features) used for the prediction, too.
        :param forecast_date: The date for which to predict the consumption. If None, the
                              the next day from when this function is called is used.
        :param include_intermediary: If True, the predictions for all hours between the last
                                     measurement and the forecast date are included in the result, too.
                                     Note: Most algorithms compute these intermediary steps anyway
                                     which is why this option is included here. It is not an essential
                                     feature of the tool.
        Example output:
        {
            "forecastValues": [0.1, 0.2, 0.3, 0.6, 0.8, ...],
            "forecastTimestamps": [pd.Timestamp("2022-09-04T01:00:00"), ...)],
            "forecastCovariates": {
                "temperature": [12.3, 12.4, 12.5, 12.6, 12.7, ...],
                "precipitation": [0.0, 0.0, 0.0, 0.0, 0.0, ...],
            },
        }
        }
        """
        raise NotImplementedError

    def get_evaluation_results(self, **kwargs):
        """
        Returns the evaluation results of the last call to fit().
        :return: Dictionary with the following keys:
        - "metrics": dictionary that maps metric names to their values
        - "predictedTestConsumption": List of predicted water consumption of the test set
        - "actualTestConsumption": List of actual water consumption of the test set
        - "testTimestamps": List of timestamps of the test set
        - "testCovariates": Dictionary that maps feature / covariate names to lists of values for the test set.
                            Covariates are additional, optional features that may be used by the algorithm.
        Example return value:
        {
            "actualTestConsumption": [0.1, 0.2, 0.3, 0.6, 0.8, ...],
            "predictedTestConsumption": [0.08, 0.16, 0.22, 0.75, 0.93, ...],
            "testTimestamps": [
                "2022-09-04T01:00:00",
                "2022-09-04T02:00:00",
                "2022-09-04T03:00:00",
                "2022-09-04T04:00:00",
                "2022-09-04T05:00:00",
                ...
            ]
            "metrics": {
                "mape": 11.45,
                "rmse": 0.0706,
                "smape": 7.863
            },
            "testCovariates": {
                "temperature": [12.3, 12.4, 12.5, 12.6, 12.7, ...],
                "precipitation": [0.0, 0.0, 0.0, 0.0, 0.0, ...],
                "is_weekend": [0, 0, 0, 0, 0, ...],
            },
        }
        """
        raise NotImplementedError

    def get_param_descriptions(self, **kwargs):
        """
        Returns a dictionary that maps names of tunable parameters to their descriptions.
        This method is called to get information on available parameters for the frontend.
        Example return value:
            {
                "colsample_bytree": "Subsample ratio of columns when constructing each tree.",
                "gamma": "Minimum loss reduction required to make a further partition on a leaf node of the tree.",
                "max_depth": "Maximum depth of a tree.",
                "min_child_weight": "Minimum sum of instance weight (hessian) needed in a child.",
                "learning_rate": "Boosting learning rate (xgb's 'eta').",
            }
        """
        raise NotImplementedError

    def get_hyperparam_search_space(self, **kwargs):
        """
        Returns a dictionary that maps names of tunable parameters to their search space.
        The search space is defined in terms of ray.tune parameter ranges. This method
        is called to get information on available parameters for the frontend and to
        perform hyperparameter optimization with ray.
        Example return value:
            {
                "colsample_bytree": tune.uniform(0.2, 0.8),
                "gamma": tune.uniform(0, 5),
                "max_depth": tune.choice([3, 4, 5]),
                "min_child_weight": tune.uniform(1, 8),
                "learning_rate": tune.loguniform(0.005, 0.3),
            }
        """
        raise NotImplementedError

    def get_default_tunable_params(**kwargs):
        """
        Returns a dictionary that maps names of tunable parameters to their default values.
        This method is called to get information on available parameters for the frontend
        and as model parameters if no values are provided by the user and no hyperparameter
        optimization is performed.
        Example return value:
            {
                "colsample_bytree": 0.3,
                "gamma": 1,
                "max_depth": 3,
                "min_child_weight": 1,
                "learning_rate": 0.01,
            }
        """
        raise NotImplementedError
