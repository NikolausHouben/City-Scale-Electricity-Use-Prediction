# eval.py

import os
from functools import wraps
from inspect import signature
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import wandb
from darts.logging import get_logger, raise_if_not, raise_log
from darts.utils import _build_tqdm_iterator, _parallel_apply
from darts.metrics import rmse, mse, mape, mae, r2_score, smape
from darts import TimeSeries

from utils import (
    make_index_same,
    ts_list_concat_new,
    get_df_diffs,
    get_df_compares_list,
)


dir_path = os.path.join(os.path.dirname(os.getcwd()), "data", "clean_data")


logger = get_logger(__name__)


def multi_ts_support(func):
    """
    This decorator further adapts the metrics that took as input two univariate/multivariate ``TimeSeries`` instances,
    adding support for equally-sized sequences of ``TimeSeries`` instances. The decorator computes the pairwise metric
    for ``TimeSeries`` with the same indices, and returns a float value that is computed as a function of all the
    pairwise metrics using a `inter_reduction` subroutine passed as argument to the metric function.

    If a 'Sequence[TimeSeries]' is passed as input, this decorator provides also parallelisation of the metric
    evaluation regarding different ``TimeSeries`` (if the `n_jobs` parameter is not set 1).
    """

    @wraps(func)
    def wrapper_multi_ts_support(*args, **kwargs):
        actual_series = (
            kwargs["actual_series"] if "actual_series" in kwargs else args[0]
        )
        pred_series = (
            kwargs["pred_series"]
            if "pred_series" in kwargs
            else args[0]
            if "actual_series" in kwargs
            else args[1]
        )

        n_jobs = kwargs.pop("n_jobs", signature(func).parameters["n_jobs"].default)
        verbose = kwargs.pop("verbose", signature(func).parameters["verbose"].default)

        raise_if_not(isinstance(n_jobs, int), "n_jobs must be an integer")
        raise_if_not(isinstance(verbose, bool), "verbose must be a bool")

        actual_series = (
            [actual_series]
            if not isinstance(actual_series, Sequence)
            else actual_series
        )
        pred_series = (
            [pred_series] if not isinstance(pred_series, Sequence) else pred_series
        )

        raise_if_not(
            len(actual_series) == len(pred_series),
            "The two TimeSeries sequences must have the same length.",
            logger,
        )

        num_series_in_args = int("actual_series" not in kwargs) + int(
            "pred_series" not in kwargs
        )
        kwargs.pop("actual_series", 0)
        kwargs.pop("pred_series", 0)

        iterator = _build_tqdm_iterator(
            iterable=zip(actual_series, pred_series),
            verbose=verbose,
            total=len(actual_series),
        )

        value_list = _parallel_apply(
            iterator=iterator,
            fn=func,
            n_jobs=n_jobs,
            fn_args=args[num_series_in_args:],
            fn_kwargs=kwargs,
        )

        # in case the reduction is not reducing the metrics sequence to a single value, e.g., if returning the
        # np.ndarray of values with the identity function, we must handle the single TS case, where we should
        # return a single value instead of a np.array of len 1

        if len(value_list) == 1:
            value_list = value_list[0]

        if "inter_reduction" in kwargs:
            return kwargs["inter_reduction"](value_list)
        else:
            return signature(func).parameters["inter_reduction"].default(value_list)

    return wrapper_multi_ts_support


def multivariate_support(func):
    """
    This decorator transforms a metric function that takes as input two univariate TimeSeries instances
    into a function that takes two equally-sized multivariate TimeSeries instances, computes the pairwise univariate
    metrics for components with the same indices, and returns a float value that is computed as a function of all the
    univariate metrics using a `reduction` subroutine passed as argument to the metric function.
    """

    @wraps(func)
    def wrapper_multivariate_support(*args, **kwargs):
        # we can avoid checks about args and kwargs since the input is adjusted by the previous decorator
        actual_series = args[0]
        pred_series = args[1]

        raise_if_not(
            actual_series.width == pred_series.width,
            "The two TimeSeries instances must have the same width.",
            logger,
        )

        value_list = []
        for i in range(actual_series.width):
            value_list.append(
                func(
                    actual_series.univariate_component(i),
                    pred_series.univariate_component(i),
                    *args[2:],
                    **kwargs,
                )
            )  # [2:] since we already know the first two arguments are the series
        if "reduction" in kwargs:
            return kwargs["reduction"](value_list)
        else:
            return signature(func).parameters["reduction"].default(value_list)

    return wrapper_multivariate_support


def _get_values(
    series: TimeSeries, stochastic_quantile: Optional[float] = 0.5
) -> np.ndarray:
    """
    Returns the numpy values of a time series.
    For stochastic series, return either all sample values with (stochastic_quantile=None) or the quantile sample value
    with (stochastic_quantile {>=0,<=1})
    """
    if series.is_deterministic:
        series_values = series.univariate_values()
    else:  # stochastic
        if stochastic_quantile is None:
            series_values = series.all_values(copy=False)
        else:
            series_values = series.quantile_timeseries(
                quantile=stochastic_quantile
            ).univariate_values()
    return series_values


def _get_values_or_raise(
    series_a: TimeSeries,
    series_b: TimeSeries,
    intersect: bool,
    stochastic_quantile: Optional[float] = 0.5,
    remove_nan_union: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the processed numpy values of two time series. Processing can be customized with arguments
    `intersect, stochastic_quantile, remove_nan_union`.

    Raises a ValueError if the two time series (or their intersection) do not have the same time index.

    Parameters
    ----------
    series_a
        A univariate deterministic ``TimeSeries`` instance (the actual series).
    series_b
        A univariate (deterministic or stochastic) ``TimeSeries`` instance (the predicted series).
    intersect
        A boolean for whether or not to only consider the time intersection between `series_a` and `series_b`
    stochastic_quantile
        Optionally, for stochastic predicted series, return either all sample values with (`stochastic_quantile=None`)
        or any deterministic quantile sample values by setting `stochastic_quantile=quantile` {>=0,<=1}.
    remove_nan_union
        By setting `remove_non_union` to True, remove all indices from `series_a` and `series_b` which have a NaN value
        in either of the two input series.
    """

    raise_if_not(
        series_a.width == series_b.width,
        "The two time series must have the same number of components",
        logger,
    )

    raise_if_not(isinstance(intersect, bool), "The intersect parameter must be a bool")

    series_a_common = series_a.slice_intersect(series_b) if intersect else series_a
    series_b_common = series_b.slice_intersect(series_a) if intersect else series_b

    raise_if_not(
        series_a_common.has_same_time_as(series_b_common),
        "The two time series (or their intersection) "
        "must have the same time index."
        "\nFirst series: {}\nSecond series: {}".format(
            series_a.time_index, series_b.time_index
        ),
        logger,
    )

    series_a_det = _get_values(series_a_common, stochastic_quantile=stochastic_quantile)
    series_b_det = _get_values(series_b_common, stochastic_quantile=stochastic_quantile)

    if not remove_nan_union:
        return series_a_det, series_b_det

    b_is_deterministic = bool(len(series_b_det.shape) == 1)
    if b_is_deterministic:
        isnan_mask = np.logical_or(np.isnan(series_a_det), np.isnan(series_b_det))
    else:
        isnan_mask = np.logical_or(
            np.isnan(series_a_det), np.isnan(series_b_det).any(axis=2).flatten()
        )
    return (
        np.delete(series_a_det, isnan_mask),
        np.delete(series_b_det, isnan_mask, axis=0),
    )


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
        actual_series, pred_series, intersect, remove_nan_union=True
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
        actual_series, pred_series, intersect, remove_nan_union=True
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


def evaluation(config, models_dict):
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
            else ts_val_weather_piped[config.longest_ts_val_idx],
            trg_val_inversed,
        ),
        "Summer": (
            ts_test_piped[config.longest_ts_test_idx],
            None
            if not config.weather
            else ts_test_weather_piped[config.longest_ts_test_idx],
            trg_test_inversed,
        ),
    }

    dict_result_season = _eval(models_dict, pipeline, test_sets, config)

    dict_result_n_ahead = extract_forecasts_per_horizon(config, dict_result_season)

    return dict_result_n_ahead


def _eval(models_dict, pipeline, test_sets, config):
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
    ]  # horizon in hours
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
            rmse_persistence = df_metrics.loc[
                df_metrics.index == "24-Hour Persistence", "rmse"
            ].values[0]
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
    wandb.log({f"Error metrics": wandb.Table(dataframe=df_metrics)})

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

    temp_data = {"Summer": df_cov_test.iloc[:, 0], "Winter": df_cov_val.iloc[:, 0]}

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

            wandb.log(
                {
                    f"{season} - Side-by-side comparison of predictions and the ground truth": fig
                }
            )


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
            df_smapes = abs(diffs).mean(axis=1)
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
        wandb.log({f"MAPE of the Historical Forecasts in {season}": wandb.Image(fig)})

    for season in dict_result_season.keys():
        fig = df_nrmse_per_season[season].plot(figsize=(10, 5))
        plt.xlabel("Horizon")
        plt.ylabel(f"RMSE [{config.unit}]")
        plt.xticks(np.arange(0, n_ahead, 2))
        plt.legend(loc="upper left", ncol=2)
        plt.title(f"Root Mean Squared Error of the Historical Forecasts in {season}")
        wandb.log({f"RMSE of the Historical Forecasts in {season}": wandb.Image(fig)})


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

        wandb.log(
            {
                f"Error Distribution of the Historical Forecasts in {season}": wandb.Image(
                    fig
                )
            }
        )


def daily_sum(dict_result_n_ahead, config):
    print("Plotting daily sum of the predictions and the ground truth")

    dict_result_season = dict_result_n_ahead[list(dict_result_n_ahead.keys())[-1]]
    for season, (_, preds_per_model, gt) in dict_result_season.items():
        dfs_daily_sums = []
        for model_name, preds in preds_per_model.items():
            df_preds = preds.pd_series().to_frame(model_name + "_preds")
            z = df_preds.groupby(df_preds.index.date).sum()
            dfs_daily_sums.append(z)

        df_gt = gt.pd_series().to_frame("ground_truth")
        z = df_gt.groupby(df_gt.index.date).sum() / config.timesteps_per_hour
        dfs_daily_sums.append(z)
        df_compare = pd.concat(dfs_daily_sums, axis=1).dropna()
        fig = df_compare[:10].plot(kind="bar", figsize=(20, 10))
        plt.legend(loc="upper right", ncol=2)
        plt.ylabel(f"Energy [{config.unit}h]")
        plt.title(f"Daily Sum of the Predictions and the Ground Truth in {season}")
        wandb.log(
            {
                f"Daily Sum of the Predictions and the Ground Truth in {season}": wandb.Image(
                    fig
                )
            }
        )
