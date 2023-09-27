# eval.py

import os
from functools import wraps
from inspect import signature
from typing import Callable, Optional, Sequence, Tuple, Union
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import wandb
from darts.logging import get_logger, raise_if_not, raise_log
from darts.utils import _build_tqdm_iterator, _parallel_apply
from darts.metrics import rmse, mse, mape, mae, r2_score, smape
from darts import TimeSeries
from darts.metrics.metrics import multivariate_support, multi_ts_support, _get_values, _get_values_or_raise

from utils import (
    make_index_same,
    ts_list_concat_new,
    get_df_diffs,
    get_df_compares_list,
    ts_list_concat,
    create_directory
)

from train import data_pipeline

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dir_path = os.path.join(root_path, "data", "clean_data")
evaluations_path = os.path.join(root_path, "data", "evaluations")


logger = get_logger(__name__)



@multi_ts_support
@multivariate_support
def max_peak_error(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    reduction: Callable[[np.ndarray], float] = np.mean,
    inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
    n_jobs: int = 1,
    verbose: bool = False,
) -> Union[float, np.ndarray]:
    """Mean Absolute Error (MAE).

    For two time series :math:`y^1` and :math:`y^2` of length :math:`T`, it is computed as

    .. math:: \\frac{1}{T}\\sum_{t=1}^T{(|y^1_t - y^2_t|)}.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    reduction
        Function taking as input a ``np.ndarray`` and returning a scalar value. This function is used to aggregate
        the metrics of different components in case of multivariate ``TimeSeries`` instances.
    inter_reduction
        Function taking as input a ``np.ndarray`` and returning either a scalar value or a ``np.ndarray``.
        This function can be used to aggregate the metrics of different series in case the metric is evaluated on a
        ``Sequence[TimeSeries]``. Defaults to the identity function, which returns the pairwise metrics for each pair
        of ``TimeSeries`` received in input. Example: ``inter_reduction=np.mean``, will return the average of the
        pairwise metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        The maximum peak error
    """

    y1, y2 = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=True # type: ignore
    )

    y1_max = np.max(y1)
    y1_idx_max = np.argmax(y1)
    y2_max = y2[y1_idx_max]

    return np.abs(y1_max - y2_max)


@multi_ts_support
@multivariate_support
def mean_n_peak_error(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    reduction: Callable[[np.ndarray], float] = np.mean,
    inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
    n_jobs: int = 1,
    verbose: bool = False,
    n: int = 5,
) -> Union[float, np.ndarray]:
    """Mean N Peak Error.

    For two time series :math:`y^1` and :math:`y^2` of length :math:`T`, it is computed as

    .. math::

    If any of the series is stochastic (containing several samples), the median sample value is considered.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    reduction
        Function taking as input a ``np.ndarray`` and returning a scalar value. This function is used to aggregate
        the metrics of different components in case of multivariate ``TimeSeries`` instances.
    inter_reduction
        Function taking as input a ``np.ndarray`` and returning either a scalar value or a ``np.ndarray``.
        This function can be used to aggregate the metrics of different series in case the metric is evaluated on a
        ``Sequence[TimeSeries]``. Defaults to the identity function, which returns the pairwise metrics for each pair
        of ``TimeSeries`` received in input. Example: ``inter_reduction=np.mean``, will return the average of the
        pairwise metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        The mean peak error for the top n peaks
    """

    y1, y2 = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=True # type: ignore
    )

    y1_sorted_indices = np.argsort(
        y1
    )  # Get indices that would sort array y1 in ascending order
    y1_top5_indices = y1_sorted_indices[
        -n:
    ]  # Get the indices of the top 5 highest values in y1

    y1_top5_max = np.max(
        y1[y1_top5_indices]
    )  # Get the maximum value among the top 5 values in y1
    y2_top5_max = np.max(y2[y1_top5_indices])  # Get the corresponding value in y2
    mean_difference = np.mean(np.abs(y1_top5_max - y2_top5_max))

    return mean_difference


def calc_error_scores(metrics, ts_predictions_inverse, trg_inversed):
    metrics_scores = {}
    for metric in metrics:
        score = metric(ts_predictions_inverse, trg_inversed)
        metrics_scores[metric.__name__] = score
    return metrics_scores


def get_error_metric_table(metrics, ts_predictions_per_model, trg_test_inversed):
    error_metric_table = {}
    for model_name, ts_predictions_inverse in ts_predictions_per_model.items():
        ts_predictions_inverse, trg_inversed = make_index_same(
            ts_predictions_inverse, trg_test_inversed
        )
        metrics_scores = calc_error_scores(
            metrics, ts_predictions_inverse, trg_inversed
        )
        error_metric_table[model_name] = metrics_scores

    df_metrics = pd.DataFrame(error_metric_table).T
    df_metrics.index.name = "model"
    return df_metrics


def calc_metrics(df_compare, metrics):
    "calculates metrics for a dataframe with a ground truth column and predictions, ground truth column must be the first column"
    metric_series_list = {}
    for metric in metrics:
        metric_name = metric.__name__
        metric_result = df_compare.apply(
            lambda x: metric(x, df_compare.iloc[:, 0]), axis=0
        )
        if metric.__name__ == "mean_squared_error":
            metric_result = np.sqrt(metric_result)
            metric_name = "root_mean_squared_error"
        elif metric.__name__ == "r2_score":
            metric_result = 1 - metric_result

        metric_series_list[metric_name] = metric_result

    df_metrics = pd.DataFrame(metric_series_list).iloc[1:, :]
    return df_metrics


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

    ts_predictions = ts_list_concat(
        historics, eval_stride
    )  # concatenating the batches into a single time series for plot 1, this keeps the n_ahead
    ts_predictions_inverse = pipeline.inverse_transform(
        ts_predictions
    )  # inverse transform the predictions, we need the original values for the evaluation

    return ts_predictions_inverse.pd_series().to_frame("prediction"), score





def evaluate(config, models_dict):

    '''
        Loads existing run results (from wandb, TODO) if they exist, 
        otherwise runs a backtest for each model on the val and test set, and then formats it into the various horizons
        
    '''
    location = config.location
    scale = config.spatial_scale

    evaluation_dict_path = os.path.join(evaluations_path, scale)
    create_directory(evaluation_dict_path)

    try:
        with open(os.path.join(evaluation_dict_path, f'{location}.pkl'), 'rb') as f:
            dict_result_n_ahead = pickle.load(f)
            print("Evaluation dictionary found, loading")

    except:
        print("No evaluation dictionary found, running backtest")
        
        (
            pipeline,
            ts_train_piped,
            ts_val_piped,
            ts_test_piped,
            ts_train_weather_piped,
            ts_val_weather_piped,
            ts_test_weather_piped,
            trg_train_inversed,
            trg_val_inversed,
            trg_test_inversed,
        ) = data_pipeline(config)

        test_sets = {  # see data_prep.ipynb for the split
            "Winter": (
                ts_val_piped[config.longest_ts_val_idx],
                None
                if not config.weather
                else ts_val_weather_piped[config.longest_ts_val_idx], # type: ignore
                trg_val_inversed,
            ),
            "Summer": (
                ts_test_piped[config.longest_ts_test_idx],
                None
                if not config.weather
                else ts_test_weather_piped[config.longest_ts_test_idx], # type: ignore
                trg_test_inversed,
            ),
        }

        dict_result_season = backtesting(models_dict, pipeline, test_sets, config)

        dict_result_n_ahead = extract_forecasts_per_horizon(config, dict_result_season)

        with open(os.path.join(evaluation_dict_path, f'{location}.pkl'), 'wb') as f:
            pickle.dump(dict_result_n_ahead, f)

    return dict_result_n_ahead


def backtesting(models_dict, pipeline, test_sets, config):
    '''
    This function runs the backtesting used in the 'evaluate' function.
    Takes in models, runs them on on the test sets (summer & winter) 
    and returns the predictions and the ground truth in a dictionary for each model, and season
    
    '''
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
                ts_predictions = ts_list_concat_new(historics, n_ahead)
                ts_predictions_per_model[model_name] = ts_predictions
                historics_per_model_update[model_name] = historics

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


def get_run_results(dict_result_n_ahead, config):
    
    df_metrics = error_metrics_table(dict_result_n_ahead, config)

    side_by_side(dict_result_n_ahead, config)

    error_metric_trajectory(dict_result_n_ahead, config)

    error_distribution(dict_result_n_ahead, config)

    daily_sum(dict_result_n_ahead, config)

    return df_metrics


def error_metrics_table(dict_result_n_ahead, config):
    print("Calculating error metrics")

    list_metrics = [
        rmse,
        r2_score,
        mae,
        smape,
        mape,
        max_peak_error,
        mean_n_peak_error,
    ]  # evaluation metrics

    metrics_tables = []

    for n_ahead, dict_result_season in dict_result_n_ahead.items():
        for season, (_, preds_per_model, gt) in dict_result_season.items():
            df_metrics = get_error_metric_table(list_metrics, preds_per_model, gt)
            rmse_persistence = df_metrics.iloc[[-1], :]['rmse'].values[0] # the last row is the X-hour persistence model specified in the training
            df_metrics.drop(labels=[config.model_names[-1]], axis=0, inplace=True)
            df_metrics.reset_index(inplace=True)
            df_metrics["season"] = season
            df_metrics.set_index("season", inplace=True)
            df_metrics.reset_index(inplace=True)
            df_metrics["horizon_in_hours"] = n_ahead // config.timesteps_per_hour
            df_metrics.set_index("horizon_in_hours", inplace=True)
            df_metrics.reset_index(inplace=True)
            df_metrics["rmse_skill_score"] = 1 - df_metrics["rmse"] / rmse_persistence
            metrics_tables.append(df_metrics)

    df_metrics = pd.concat(metrics_tables, axis=0, ignore_index=True).sort_values(
        by=["season", "horizon_in_hours"]
    )
    try:
        wandb.log({f"Error metrics": wandb.Table(dataframe=df_metrics)})
    except:
        print("Wandb is not initialized, skipping logging")


    return df_metrics


def side_by_side(dict_result_n_ahead, config):
    print("Plotting side-by-side comparison of predictions and the ground truth")

    df_cov_train = pd.read_hdf(
        os.path.join(dir_path, f"{config.spatial_scale}.h5"),
        key=f"{config.location}/{config.temp_resolution}min/train_cov",
    )
    df_cov_val = pd.read_hdf(
        os.path.join(dir_path, f"{config.spatial_scale}.h5"),
        key=f"{config.location}/{config.temp_resolution}min/val_cov",
    )
    df_cov_test = pd.read_hdf(
        os.path.join(dir_path, f"{config.spatial_scale}.h5"),
        key=f"{config.location}/{config.temp_resolution}min/test_cov",
    )

    temp_data = {"Summer": df_cov_test.iloc[:, 0], "Winter": df_cov_val.iloc[:, 0]} # type: ignore

    for n_ahead, dict_result_season in dict_result_n_ahead.items():
        for season, (_, preds_per_model, gt) in dict_result_season.items():
            fig = go.Figure()

            # Add the ground truth data to the left axis
            fig.add_trace(
                go.Scatter(
                    x=gt.pd_series().index,
                    y=gt.pd_series().values,
                    name="Ground Truth",
                    yaxis="y1",
                )
            )

            for model_name in config.model_names:
                preds = preds_per_model[model_name]
                fig.add_trace(
                    go.Scatter(
                        x=preds.pd_series().index,
                        y=preds.pd_series().values,
                        name=model_name,
                        yaxis="y1",
                    )
                )

            # Add the df_cov_test data to the right axis

            series_weather = temp_data[season]
            fig.add_trace(
                go.Scatter(
                    x=series_weather.index,
                    y=series_weather.values,
                    name="temperature",
                    yaxis="y2",
                    line=dict(dash="dot", color="grey"),  # Set the line style to dotted
                )
            )

            fig.update_layout(
                title=f"{season} - Horizon: {n_ahead// config.timesteps_per_hour} Hours",
                xaxis=dict(title="Time"),
                yaxis=dict(title=f"Power [{config.unit}]", side="left"),
                yaxis2=dict(title="Temperature [°C]", overlaying="y", side="right"),
            )

            try:
                wandb.log(
                    {
                        f"{season} - Side-by-side comparison of predictions and the ground truth": fig
                    }
                )
            except:
                print("Wandb is not initialized, skipping logging")
                fig.show()


def error_metric_trajectory(dict_result_n_ahead, config):
    print("Plotting error metric trajectory")

    n_ahead, dict_result_season = list(dict_result_n_ahead.items())[-1]

    dict_result_season = dict_result_n_ahead[n_ahead]
    df_smapes_per_season = {}
    df_nrmse_per_season = {}

    for season, (historics_per_model, _, gt) in dict_result_season.items():
        df_smapes_per_model = []
        df_rmse_per_model = []
        for model_name, historics in historics_per_model.items():
            df_list = get_df_compares_list(historics, gt)
            diffs = get_df_diffs(df_list)
            df_smapes = abs(diffs).mean(axis=1) # type: ignore
            df_smapes.columns = [model_name]
            df_rmse = np.square(diffs).mean(axis=1)
            df_rmse.columns = [model_name]

            df_smapes_per_model.append(df_smapes)
            df_rmse_per_model.append(df_rmse)

        df_smapes_per_model = (
            pd.concat(df_smapes_per_model, axis=1).ewm(alpha=0.1).mean()
        )
        df_smapes_per_model.columns = config.model_names
        df_nrmse_per_model = pd.concat(df_rmse_per_model, axis=1).ewm(alpha=0.1).mean()
        df_nrmse_per_model.columns = config.model_names
        df_smapes_per_season[season] = df_smapes_per_model
        df_nrmse_per_season[season] = df_nrmse_per_model

    for season in dict_result_season.keys():
        fig = df_smapes_per_season[season].plot(figsize=(10, 5))
        plt.xlabel("Horizon")
        plt.ylabel("MAPE [%]")
        plt.legend(loc="upper left", ncol=2)
        plt.xticks(np.arange(0, n_ahead, 2))
        plt.title(
            f"Mean Absolute Percentage Error of the Historical Forecasts in {season}"
        )
        try:
            wandb.log({f"MAPE of the Historical Forecasts in {season}": wandb.Image(fig)})
        except:
            print("Wandb is not initialized, skipping logging")
            plt.show()

    for season in dict_result_season.keys():
        fig = df_nrmse_per_season[season].plot(figsize=(10, 5))
        plt.xlabel("Horizon")
        plt.ylabel(f"RMSE [{config.unit}]")
        plt.xticks(np.arange(0, n_ahead, 2))
        plt.legend(loc="upper left", ncol=2)
        plt.title(f"Root Mean Squared Error of the Historical Forecasts in {season}")
        try:
            wandb.log({f"RMSE of the Historical Forecasts in {season}": wandb.Image(fig)})
        except:
            print("Wandb is not initialized, skipping logging")
            plt.show()




def error_distribution(dict_result_n_ahead, config):
    print("Plotting error distribution")

    n_ahead, dict_result_season = list(dict_result_n_ahead.items())[-1]
    for season, (historics_per_model, _, gt) in dict_result_season.items():
        df_smapes_per_model = []
        df_nrmse_per_model = []
        fig, ax = plt.subplots(
            ncols=len(config.model_names), figsize=(5 * len(config.model_names), 5)
        )
        fig.suptitle(f"Error Distribution of the Historical Forecasts in {season}")
        for i, (model_name, historics) in enumerate(historics_per_model.items()):
            df_list = get_df_compares_list(historics, gt)
            diffs = get_df_diffs(df_list)
            diffs_flat = pd.Series(
                diffs.values.reshape(
                    -1,
                )
            )
            ax[i].hist(diffs_flat, bins=100)
            ax[i].set_title(model_name)

        try:
            wandb.log(
                {
                    f"Error Distribution of the Historical Forecasts in {season}": wandb.Image(
                        fig
                    )
                }
            )
        except:
            print("Wandb is not initialized, skipping logging")
            plt.show()


def daily_sum(dict_result_n_ahead, config):
    print("Plotting daily sum of the predictions and the ground truth")

    dict_result_season = dict_result_n_ahead[list(dict_result_n_ahead.keys())[-1]]
    for season, (_, preds_per_model, gt) in dict_result_season.items():
        dfs_daily_sums = []
        for model_name, preds in preds_per_model.items():
            df_preds = preds.pd_series().to_frame(model_name + "_preds")
            z = df_preds.groupby(df_preds.index.date).sum() / config.timesteps_per_hour
            dfs_daily_sums.append(z)

        df_gt = gt.pd_series().to_frame("ground_truth")
        z = df_gt.groupby(df_gt.index.date).sum() / config.timesteps_per_hour
        dfs_daily_sums.append(z)
        df_compare = pd.concat(dfs_daily_sums, axis=1).dropna()
        fig = df_compare[:10].plot(kind="bar", figsize=(20, 10))
        plt.legend(loc="upper right", ncol=2)
        plt.ylabel(f"Energy [{config.unit}h]")
        plt.title(f"Daily Sum of the Predictions and the Ground Truth in {season}")
        
        try:
            wandb.log(
                {
                    f"Daily Sum of the Predictions and the Ground Truth in {season}": wandb.Image(
                        fig
                    )
                }
            )
        except:
            print("Wandb is not initialized, skipping logging")
            plt.show()
