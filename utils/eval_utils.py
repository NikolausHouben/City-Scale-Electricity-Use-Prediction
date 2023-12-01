# eval_utils.py

"""Some utility functions for evaluation."""

import sys
import os
from typing import Callable, Optional, Sequence, Tuple, Union, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import wandb
from darts.metrics import rmse
from darts import TimeSeries


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .data_utils import (
    make_index_same,
    get_df_diffs,
    get_df_compares_list,
    ts_list_concat,
    shorten_historics_to_n_ahead,
)

from .pipeline import Config
from .paths import CLEAN_DATA_DIR


def predict_testset(model, ts, ts_covs, n_lags, n_ahead, eval_stride, pipeline):
    """
    This function predicts the test set using a model and returns the predictions as a dataframe. Used in hyperparameter tuning.
    """

    print("Predicting test set...")

    historics = model.historical_forecasts(
        ts,
        future_covariates=ts_covs if model.supports_future_covariates else None,
        start=ts.get_index_at_point(n_lags),
        verbose=False,
        stride=eval_stride,
        forecast_horizon=n_ahead,
        retrain=False,
        last_points_only=False,  # leave this as False unless you want the output to be one series, the rest will not work with this however
    )

    historics_gt = [ts.slice_intersect(historic) for historic in historics]
    score = np.array(rmse(historics_gt, historics)).mean()

    n_ahead_historics = shorten_historics_to_n_ahead(historics, eval_stride)
    ts_predictions = ts_list_concat(
        n_ahead_historics, eval_stride
    )  # concatenating the batches into a single time series for plot 1, this keeps the n_ahead
    ts_predictions_inverse = pipeline.inverse_transform(
        ts_predictions
    )  # inverse transform the predictions, we need the original values for the evaluation

    return ts_predictions_inverse.pd_series().to_frame("prediction"), score


def backtesting(models_dict, pipeline, test_sets, config):
    """
    This function runs the backtesting used in the 'evaluate' function.
    Takes in models, runs them on on the test sets (summer & winter)
    and returns the predictions and the ground truth in a dictionary for each model, and season

    """
    dict_result_season = {}
    for season, (ts, ts_cov, gt) in test_sets.items():
        print(f"Testing on {season} data")
        # Generating Historical Forecasts for each model
        ts_predictions_per_model = {}
        historics_per_model = {}
        for model_name, model in models_dict.items():
            print(f"Generating historical forecasts with {model_name}")
            historics = model.historical_forecasts(
                ts,
                future_covariates=ts_cov if model.supports_future_covariates else None,
                start=ts.get_index_at_point(config.n_lags),
                verbose=True,
                stride=1,  # this allows us to later differentiate between the different horizons
                forecast_horizon=config.timesteps_per_hour
                * 48,  # 48 hours is our max horizon
                retrain=False,
                last_points_only=False,
            )

            historics_inverted = [
                pipeline.inverse_transform(historic) for historic in historics
            ][
                1:
            ]  # the first historic is partly nan, so we skip it
            historics_per_model[
                model_name
            ] = historics_inverted  # storing the forecasts in batches of the forecasting horizon, for plot 2

        dict_result_season[season] = historics_per_model, gt

    return dict_result_season


def extract_forecasts_per_horizon(config, dict_result_season):
    n_aheads = [
        i * config.timesteps_per_hour for i in [1, 4, 8, 24, 48]
    ]  # horizons in hours are then multiplied by the timesteps per hour to get the horizons in timesteps
    dict_result_n_ahead = {}

    for n_ahead in n_aheads:
        dict_result_season_update = {}
        for season, (historics_per_model, gt) in dict_result_season.items():
            ts_predictions_per_model = {}
            historics_per_model_update = {}
            for model_name, historics in historics_per_model.items():
                n_ahead_historics = shorten_historics_to_n_ahead(historics, n_ahead)
                ts_predictions = ts_list_concat(n_ahead_historics, n_ahead)
                ts_predictions_per_model[model_name] = ts_predictions
                historics_per_model_update[model_name] = n_ahead_historics

            ts_predictions_per_model["48-Hour Persistence"] = gt.shift(
                config.timesteps_per_hour * 48
            )  # adding the 48-hour persistence model as a benchmark
            dict_result_season_update[season] = (
                historics_per_model_update,
                ts_predictions_per_model,
                gt,
            )
        dict_result_n_ahead[n_ahead] = dict_result_season_update

    return dict_result_n_ahead
