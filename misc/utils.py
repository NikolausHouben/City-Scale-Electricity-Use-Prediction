import datetime
import os
import pickle
import random
import sys
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.stats import boxcox
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import MinMaxScaler
import holidays

from scipy.stats import boxcox
from scipy.signal import find_peaks_cwt


def timeseries_dataframe_pivot(df):
    df_ = df.copy()
    df_["date"] = df_.index.date
    df_["time"] = df_.index.time

    df_pivot = df_.pivot(index="date", columns="time")

    n_days, n_timesteps = df_pivot.shape

    df_pivot.dropna(thresh=n_timesteps // 5, inplace=True)

    df_pivot = df_pivot.fillna(method="ffill", axis=0)

    df_pivot = df_pivot.droplevel(0, axis=1)

    df_pivot.columns.name = None

    df_pivot.index = pd.DatetimeIndex(df_pivot.index)

    return df_pivot


def unpivot_timeseries_dataframe(df: pd.DataFrame, column_name: str = "Q"):
    df_unstack = df.T.unstack().to_frame().reset_index()  # type: ignore
    df_unstack.columns = ["date", "time", "{}".format(column_name)]
    df_unstack["date_str"] = df_unstack["date"].apply(
        lambda t: datetime.datetime.strftime(t, format="%Y-%m-%d")  # type: ignore
    )
    df_unstack["time_str"] = df_unstack["time"].apply(
        lambda t: " {}:{}:{}".format(t.hour, t.minute, t.second)
    )
    df_unstack["datetime_str"] = df_unstack["date_str"] + df_unstack["time_str"]
    df_unstack = df_unstack.set_index(
        pd.to_datetime(df_unstack["datetime_str"], format="%Y-%m-%d %H:%M:%S")
    )[[column_name]]
    df_unstack.index.name = "datetime"

    return df_unstack


def boxcox_transform(dataframe, lam=None):
    """
    Perform a Box-Cox transform on a pandas dataframe timeseries.

    Args:
    dataframe (pandas.DataFrame): Pandas dataframe containing the timeseries to transform.
    lam (float): The lambda value to use for the Box-Cox transformation.

    Returns:
    transformed_dataframe (pandas.DataFrame): Pandas dataframe containing the transformed timeseries.
    """
    transformed_dataframe = dataframe.copy()
    for column in transformed_dataframe.columns:
        transformed_dataframe[column], lam = boxcox(transformed_dataframe[column], lam)  # type: ignore
    return transformed_dataframe, lam


def inverse_boxcox_transform(dataframe, lam):
    """
    Inverse the Box-Cox transform on a pandas dataframe timeseries.

    Args:
    dataframe (pandas.DataFrame): Pandas dataframe containing the timeseries to transform.
    lam (float): The lambda value used for the original Box-Cox transformation.

    Returns:
    transformed_dataframe (pandas.DataFrame): Pandas dataframe containing the inverse-transformed timeseries.
    """
    transformed_dataframe = dataframe.copy()
    for column in transformed_dataframe.columns:
        if lam == 0:
            transformed_dataframe[column] = np.exp(transformed_dataframe[column])
        else:
            transformed_dataframe[column] = np.exp(
                np.log(lam * transformed_dataframe[column] + 1) / lam
            )
    return transformed_dataframe


def concat_and_scale(df_ap, similar_pair):
    """This function takes in the dataframe with all the apartments and the pair of similar apartments"""
    df_ap_1 = df_ap[similar_pair[0]].to_frame("apartment_demand_W")
    df_ap_2 = df_ap[similar_pair[1]].to_frame("apartment_demand_W")
    shifted_idx = df_ap_2.index + pd.Timedelta(
        weeks=52
    )  # shifting the second apartment by one year
    # flipping every other row to make the two profiles more similar
    df_side_by_side = pd.concat([df_ap_1, df_ap_2], axis=1)
    df_flipped = df_side_by_side.iloc[::2, ::-1]
    df_side_by_side.iloc[::2, :] = df_flipped.values
    df_ap_1 = df_side_by_side.iloc[:, 0].to_frame("apartment_demand_W")
    df_ap_2 = df_side_by_side.iloc[:, 1].to_frame("apartment_demand_W")
    df_ap_2.index = shifted_idx
    # scaling both between 0 and 1
    scaler_1 = MinMaxScaler()
    scaler_2 = MinMaxScaler()
    # scaling the two dataframes
    df_ap_1[df_ap_1.columns] = scaler_1.fit_transform(df_ap_1[df_ap_1.columns])
    df_ap_2[df_ap_2.columns] = scaler_2.fit_transform(df_ap_2[df_ap_2.columns])
    # appending the two dataframes
    df_ap_to_concat_scaled = pd.concat([df_ap_1, df_ap_2], axis=0)
    # using scaler 1 to scale the whole dataframe back to the original scale
    df_ap_to_concat = pd.DataFrame(
        scaler_1.inverse_transform(df_ap_to_concat_scaled),
        columns=df_ap_to_concat_scaled.columns,
        index=df_ap_to_concat_scaled.index,
    )

    return df_ap_to_concat


def post_process_xgb_predictions(predictions, boxcox_bool, scaler=None, lam=None):
    "Post-process the predictions of the Multi-Output XGBoost model"
    predictions_reshaped = predictions.reshape(-1, 1).flatten()
    # set negative predictions to 5th percentile of the training data
    predictions_reshaped[predictions_reshaped < 0] = np.quantile(
        predictions_reshaped, 0.05
    )
    # reverse the scaling and boxcox transformation of the predictions
    if scaler is not None:
        predictions_reshaped = scaler.inverse_transform(
            predictions_reshaped.reshape(-1, 1)
        ).flatten()
    if boxcox_bool:
        predictions_reshaped = inverse_boxcox_transform(
            pd.DataFrame(predictions_reshaped), lam
        ).values.flatten()
    return predictions_reshaped


# model evaluation


def peak_error(preds, dtest):
    """
    Peak error is the absolute difference between the predicted peak and the true peak.
    This metric ensures that the hyperparameters of the model are chosen so the model behaves less conservative.
    """
    labels = dtest.get_label()
    samples, timesteps = preds.shape
    labels = labels.reshape(samples, timesteps)
    distance = 0
    for i in range(preds.shape[0]):
        pred, label = preds[i].reshape(-1, 1), labels[i].reshape(-1, 1)
        error = np.abs(pred.max() - label.max())
        distance += error
    return "peak_error", distance / preds.shape[0]


rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)


def infer_frequency(df):
    """Infers the frequency of a timeseries dataframe and returns the value in minutes"""
    freq = df.index.to_series().diff().mode()[0].seconds / 60
    return freq


def timeseries_peak_feature_extractor(df):
    "Extracts peak count, maximum peak height, and time of two largest peaks for each day in a pandas dataframe time series"

    timesteplen = infer_frequency(df)
    timesteps_per_day = 24 * 60 // timesteplen

    # Find peaks
    peak_idx = find_peaks_cwt(
        df.values.flatten(),
        widths=3,
        max_distances=[timesteps_per_day // 2],
        window_size=timesteps_per_day,
    )

    # Convert peak indices to datetime indices
    peak_times = [df.index[i] for i in peak_idx]

    # Group peaks by day
    peak_days = pd.Series(peak_times).dt.date

    # Count peaks for each day
    daily_peak_count = peak_days.value_counts().sort_index()

    # Find maximum and second maximum peak height and time for each day
    daily_peak_height = []
    daily_peak_time = []
    daily_second_peak_height = []
    daily_second_peak_time = []

    for day in daily_peak_count.index:
        day_peaks = [
            peak_idx[i] for i in range(len(peak_idx)) if peak_times[i].date() == day
        ]
        day_peak_vals = [df.values[i] for i in day_peaks]

        max_idx = np.argmax(day_peak_vals)
        daily_peak_height.append(day_peak_vals[max_idx][0])
        daily_peak_time.append((day_peaks[max_idx] % timesteps_per_day))

        if len(day_peak_vals) > 1:
            day_peak_vals[max_idx] = -np.inf
            second_max_idx = np.argmax(day_peak_vals)
            daily_second_peak_height.append(day_peak_vals[second_max_idx][0])
            daily_second_peak_time.append(
                (day_peaks[second_max_idx] % timesteps_per_day)
            )
        else:
            daily_second_peak_height.append(0)
            daily_second_peak_time.append(0)

    # Combine results into output DataFrame
    output_df = pd.DataFrame(
        {
            "height_highest_peak": daily_peak_height,
            "time_highest_peak": daily_peak_time,
            "height_second_highest_peak": daily_second_peak_height,
            "time_second_highest_peak": daily_second_peak_time,
        },
        index=daily_peak_count.index,
    )

    return output_df


def calc_rolling_sum_of_load(df, n_days):
    df["rolling_sum"] = df.sum(axis=1).rolling(n_days).sum().shift(1)
    df = df.dropna()
    return df


def create_datetime_features(df):
    df["day_of_week"] = df.index.dayofweek
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df.drop("day_of_week", axis=1, inplace=True)
    df["month"] = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # is weekend
    df["is_weekend"] = df.index.dayofweek > 4
    df.drop("month", axis=1, inplace=True)
    return df


def create_holiday_features(df, df_holidays, df_holiday_periods=None):
    df_1 = days_until_next_holiday_encoder(df, df_holidays)
    df_2 = days_since_last_holiday_encoder(df, df_holidays)

    df_3 = pd.concat([df_1, df_2], axis=1)

    if df_holiday_periods is not None:
        df_3 = pd.concat([df_3, df_holiday_periods], axis=1)

    df_3 = df_3.loc[~df_3.index.duplicated(keep="first")]

    df_3 = df_3.reindex(df.index, fill_value=0)

    return df_3


def days_until_next_holiday_encoder(df, df_holidays):
    df_concat = pd.concat([df, df_holidays], axis=1)
    df_concat["days_until_next_holiday"] = 0
    for ind in df_concat.index:
        try:
            next_holiday = df_concat["holiday_dummy"].loc[ind:].first_valid_index()
            days_until_next_holiday = (next_holiday - ind).days
            df_concat.loc[ind, "days_until_next_holiday"] = days_until_next_holiday
        except:
            pass

    return df_concat[["days_until_next_holiday"]]


def days_since_last_holiday_encoder(df, df_holidays):
    df_concat = pd.concat([df, df_holidays], axis=1)
    df_concat["days_since_last_holiday"] = 0
    for ind in df_concat.index:
        next_holiday = df_concat["holiday_dummy"].loc[:ind].last_valid_index()
        days_since_last_holiday = (ind - next_holiday).days
        df_concat.loc[ind, "days_since_last_holiday"] = days_since_last_holiday

    return df_concat[["days_since_last_holiday"]]


def get_year_list(df):
    "Return the list of years in the historic data"
    years = df.index.year.unique()
    years = years.sort_values()
    return list(years)


def get_holidays(years, shortcut):
    country = getattr(holidays, shortcut)
    holidays_dict = country(years=years)
    df_holidays = pd.DataFrame(holidays_dict.values(), index=holidays_dict.keys())
    df_holidays[0] = 1
    df_holidays_dummies = df_holidays
    df_holidays_dummies.columns = ["holiday_dummy"]
    df_holidays_dummies.index = pd.DatetimeIndex(df_holidays.index)
    df_holidays_dummies = df_holidays_dummies.sort_index()

    return df_holidays_dummies


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
