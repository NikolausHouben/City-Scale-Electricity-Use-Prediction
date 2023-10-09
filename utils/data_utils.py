# utils.py

"""This file contains utility functions that are used in the notebooks."""

import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
import requests
from timezonefinder import TimezoneFinder
import time
from darts.utils.missing_values import fill_missing_values
import h5py
import wandb
import plotly.express as px
import plotly.graph_objects as go


units_dict = {"county": "GW", "town": "MW", "village": "kW", "neighborhood": "kW"}


def calculate_stats_and_plot_hist(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    std_ts = df_scaled.std()
    std_ts_diff = df.diff().std()
    df_scaled.hist(bins=100, layout=(1, df.shape[1]), figsize=(5 * df.shape[1], 5))

    return std_ts, std_ts_diff


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


def get_hdf_keys(dir_path):
    """

    Function to show the keys in the h5py file.

    """

    locations_per_file = {}
    temporal_resolutions_per_file = {}

    for file_name in os.listdir(dir_path):
        if file_name.endswith(".h5"):
            # open the file in read mode
            with h5py.File(os.path.join(dir_path, file_name), "r") as f:
                # print the keys in the file
                locations = list(f.keys())
                locations_per_file[file_name] = locations
                for location in locations:
                    temporal_resolutions = list(f[location].keys())  # type: ignore
                    temporal_resolutions_per_file[file_name] = temporal_resolutions

    return locations_per_file, temporal_resolutions_per_file


def load_from_model_artifact_checkpoint(model_class, base_path, checkpoint_path):
    model = model_class.load(base_path)
    model.model = model._load_from_checkpoint(checkpoint_path)
    return model


def get_weather_data(lat, lng, start_date, end_date, variables: list, keep_UTC=True):
    """
    This function fetches weather data from the Open Meteo API and returns a dataframe with the weather data.

    Parameters

    lat: float
        Latitude of the location
    lng: float
        Longitude of the location
    start_date: str
        Start date of the weather data in the format YYYY-MM-DD
    end_date: str
        End date of the weather data in the format YYYY-MM-DD
    variables: list
        List of variables to fetch from the API.
    keep_UTC: bool
        If True, the weather data will be returned in UTC. If False, the weather data will be returned in the local timezone of the location.

    Returns

    df_weather: pandas.DataFrame
        Dataframe with the weather data
    """

    if keep_UTC:
        tz = "UTC"
    else:
        print("Fetching timezone from coordinates")
        tf = TimezoneFinder()
        tz = tf.timezone_at(lng=lng, lat=lat)

    df_weather = pd.DataFrame()
    for variable in variables:
        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive?latitude={}&longitude={}&start_date={}&end_date={}&hourly={}".format(
                lat, lng, start_date, end_date, variable
            )
        )
        df = pd.DataFrame(response.json()["hourly"])
        df = df.set_index("time")
        df_weather = pd.concat([df_weather, df], axis=1)

    df_weather.index = pd.to_datetime(df_weather.index)
    df_weather = df_weather.tz_localize("UTC").tz_convert(tz)

    return df_weather


def drop_duplicate_index(df):
    "This function drops duplicate indices from a dataframe."
    df = df[~df.index.duplicated(keep="first")]
    return df


def infer_frequency(df):
    """Infers the frequency of a timeseries dataframe and returns the value in minutes"""
    freq = df.index.to_series().diff().mode()[0].seconds / 60
    return freq


def make_index_same(ts1, ts2):
    """This function makes the indices of two time series the same"""
    ts1 = ts1.slice_intersect(ts2)
    ts2 = ts2.slice_intersect(ts1)
    return ts1, ts2


def review_subseries(ts, min_len, ts_cov=None):
    """
    Reviews a time series and covariate time series to make sure they are long enough for the model
    """
    ts_reviewed = []
    ts_cov_reviewed = []
    for ts in ts:
        if len(ts) > min_len:
            ts = fill_missing_values(ts)
            ts_reviewed.append(ts)
            if ts_cov is not None:
                ts_cov = fill_missing_values(ts_cov)
                ts_cov_reviewed.append(ts_cov.slice_intersect(ts))

    return ts_reviewed, ts_cov_reviewed


def get_longest_subseries_idx(ts_list):
    """
    Returns the longest subseries from a list of darts TimeSeries objects and its index
    """
    longest_subseries_length = 0
    longest_subseries_idx = 0
    for idx, ts in enumerate(ts_list):
        if len(ts) > longest_subseries_length:
            longest_subseries_length = len(ts)
            longest_subseries_idx = idx
    return longest_subseries_idx


def ts_list_concat_new(ts_list, n_ahead):
    """
    This function concatenates a list of time series into one time series.
    The result is a time series that concatenates the subseries so that n_ahead is preserved.

    """
    ts = ts_list[0][:n_ahead]
    for i in range(n_ahead, len(ts_list) - n_ahead, n_ahead):
        ts_1 = ts_list[i][ts.end_time() :]
        timestamp_one_before = ts_1.start_time() - ts.freq
        ts = ts[:timestamp_one_before].append(ts_1[:n_ahead])
    return ts


def ts_list_concat(ts_list, eval_stride):
    """
    This function concatenates a list of time series into one time series.
    The result is a time series that concatenates the subseries so that n_ahead is preserved.

    """
    ts = ts_list[0]
    n_ahead = len(ts)
    skip = n_ahead // eval_stride
    for i in range(skip, len(ts_list) - skip, skip):
        print(ts.end_time(), ts_list[i].start_time())
        ts_1 = ts_list[i][ts.end_time() :]
        timestamp_one_before = ts_1.start_time() - ts.freq
        ts = ts[:timestamp_one_before].append(ts_1)
    return ts


def get_df_compares_list(historics, gt):
    """Returns a list of dataframes with the ground truth and the predictions next to each other"""
    df_gt = gt.pd_dataframe()
    df_compare_list = []
    for ts in historics:
        if ts.is_probabilistic:
            df = ts.quantile_df(0.5)
        else:
            df = ts.pd_dataframe()

        df["gt"] = df_gt

        df.reset_index(inplace=True)
        df = df.iloc[:, 1:]
        df_compare_list.append(df)

    return df_compare_list


def get_df_diffs(df_list):
    """Returns a dataframe with the differences between the first column and the rest of the columns"""

    df_diffs = pd.DataFrame(index=range(df_list[0].shape[0]))
    for df in df_list:
        df_diff = df.copy()
        diff = (df_diff.iloc[:, 0].values - df_diff.iloc[:, 1]).values
        df_diffs = pd.concat([df_diffs, pd.DataFrame(diff)], axis=1)
    return df_diffs


### Transformations & Cleaning


def standardize_format(
    df: pd.DataFrame, type: str, timestep: int, location: str, unit: str
):
    """

    This function standardizes the format of the dataframes. It resamples the dataframes to the specified timestep and interpolates the missing values.

    Parameters

    df: pandas.DataFrame
        Dataframe with the data
    type: str
        Type of the data, e.g. 'electricity' and name of the column
    timestep: int
        Timestep in minutes
    location: str
        Location of the data, e.g. 'apartment'
    unit: str
        Unit of the data, e.g. 'W' and name of the column

    Returns

    df: pandas.DataFrame
        Dataframe with the data in the standardized format

    """

    current_timestep = infer_frequency(df)  # output is in minutes
    df = df.sort_index()
    df = remove_duplicate_index(df)
    if current_timestep <= timestep:
        df = df.resample(f"{timestep}T").mean()
    else:
        df = (
            df.resample(f"{timestep}T")
            .interpolate(method="linear", axis=0)
            .ffill()
            .bfill()
        )

    df.index.name = "datetime"
    df.columns = [f"{location}_{type}_{unit}"]
    return df


# a function to do a train test split, the train should be a full year and the test should be a tuple of datasets, each one month long


def split_train_val_test_datasets(
    df, train_start, train_end, val_start, val_end, test_start, test_end
):
    train = df.loc[train_start:train_end]
    val = df.loc[val_start:val_end]
    test = df.loc[test_start:test_end]
    # Save the dataframes
    return train, val, test


def remove_non_positive_values(df, set_nan=False):
    "Removes all non-positive values from a dataframe, interpolates the missing values and sets zeros to a very small value (for boxcox))"
    if set_nan:
        df[df <= 0] = np.nan
    else:
        df[df <= 0] = 1e-6

    df = interpolate_and_dropna(df)

    return df


def interpolate_and_dropna(df):
    df = df.interpolate(method="linear", axis=0, limit=8)
    df.dropna(inplace=True)
    return df


def remove_days(df_raw, p=0.05):
    "Removes days with less than p of average total energy consumption of all days"
    df = df_raw.copy()
    days_to_remove = []
    days = list(set(df.index.date))
    threshold = df.groupby(df.index.date).sum().quantile(p).values[0]
    for day in days:
        if df.loc[df.index.date == day].sum().squeeze() < threshold:
            days_to_remove.append(day)

    mask = np.in1d(df.index.date, days_to_remove)
    df = df[~mask].dropna()

    return df


def remove_duplicate_index(df):
    df = df.loc[~df.index.duplicated(keep="first")]
    return df


# results analysis / wandb api interaction


api = wandb.Api()


def choose_more_recent(file1, file2):
    if file1.updatedAt > file2.updatedAt:
        return file1
    else:
        return file2


def check_if_same_horizon_plot(file1, file2):
    if file1._attrs["name"].split("_")[-2] == file2._attrs["name"].split("_")[-2]:
        return True


def get_file_names(project_name, name_id_dict, run_to_visualize, season):
    run = api.run(f"wattcast/{project_name}/{name_id_dict[run_to_visualize]}")
    files = []
    for file in run.files():
        if "Side" in str(file) and season in str(file):
            files.append(file)
    return files


def get_latest_plotly_plots(files):
    # get the most recent file for each horizon
    zip_files = zip(files[::2], files[1::2])
    files_to_plot = []
    for file1, file2 in zip_files:
        if check_if_same_horizon_plot(file1, file2):
            files_to_plot.append(choose_more_recent(file1, file2))
        else:
            files_to_plot.append(file1)
            files_to_plot.append(file2)

    return files_to_plot


def download_plotly_plots(files_to_plot):
    side_by_side_plots_dict = {}
    for file in files_to_plot:
        plot = file.download(replace=True)
        data = json.load(plot)
        plot_name = data["layout"]["title"]["text"]
        side_by_side_plots_dict[plot_name] = data

    return side_by_side_plots_dict


def make_df_from_plot(plot):
    data = plot["data"]
    df = pd.DataFrame()
    for i in range(len(data)):
        subdata = data[i]
        try:
            del subdata["line"]
        except:
            pass
        df_line = pd.DataFrame(subdata)
        df_line.set_index("x", inplace=True)
        df_line.index.name = "Datetime"
        col_name = data[i]["name"]
        df_line = df_line.rename(columns={"y": col_name})
        df_line = df_line[[col_name]]
        df = pd.concat([df, df_line], axis=1)
    return df


def side_by_side_df(side_by_side_plots_dict):
    df_all = pd.DataFrame()
    for name, plot in side_by_side_plots_dict.items():
        df = make_df_from_plot(plot)
        df.columns = [col + " " + name for col in df.columns]
        df_all = pd.concat([df_all, df], axis=1)

    df_all.index = pd.to_datetime(df_all.index)
    df_gt = df_all.filter(like="Ground").iloc[:, :1]
    df_gt.columns = ["Ground Truth"]
    df_all.dropna(inplace=True)
    # drop column if contains 'Ground' or 'temperature', we will add the ground truth back in below
    df_all = df_all.loc[:, ~df_all.columns.str.contains("Ground")]
    df_all = df_all.loc[:, ~df_all.columns.str.contains("temperature")]
    df_all = df_all.join(df_gt)  # add ground truth back in

    return df_all


def get_best_model_per_scale_and_horizon(df, metric):
    """
    Gets the model with the best performance for each scale and horizon based in a pd.DataFrame with the error scores of the models,
    based on the metric specified.
    """
    df_sorted = df.sort_values(
        by=["scale", metric], ascending=False if metric == "rmse_skill_score" else True
    )
    df_sorted = df_sorted.drop_duplicates(
        subset=["scale", "horizon_in_hours"], keep="first"
    )
    df_sorted = df_sorted.sort_values(by=["scale", "horizon_in_hours"])
    df_sorted = df_sorted.reset_index(drop=True).sort_values(
        by=["scale", "horizon_in_hours"]
    )
    return df_sorted


def get_run_name_id_dict(runs):
    """Loops through all runs and returns a dictionary of run names and ids"""
    name_id_dict = {}
    for run_ in runs:
        l = run_.name.split("_")[:-1]
        l.insert(2, "in")
        n = "_".join(l)
        # remove the _ around in
        n = n.replace("_in_", "in")
        name_id_dict[n] = run_.id
    return name_id_dict


def remove_outliers(df, column, lower_percentile=0, upper_percentile=100):
    # Calculate the thresholds
    lower_threshold = df[column].quantile(lower_percentile / 100)
    upper_threshold = df[column].quantile(upper_percentile / 100)

    # Clip the values
    df[column] = df[column].clip(lower_threshold, upper_threshold)

    return df


def plot_location_splits(dir_path, scale_idx, location_idx, show_covariates=False):
    locations, temps = get_hdf_keys(dir_path)

    spatial_scale = list(locations.keys())[scale_idx]
    location = list(locations.values())[scale_idx][location_idx]
    temp_resolution = list(temps.values())[0][0]

    df_train = pd.read_hdf(
        os.path.join(dir_path, f"{spatial_scale}"),
        key=f"{location}/{temp_resolution}/train_target",
    )
    df_val = pd.read_hdf(
        os.path.join(dir_path, f"{spatial_scale}"),
        key=f"{location}/{temp_resolution}/val_target",
    )
    df_test = pd.read_hdf(
        os.path.join(dir_path, f"{spatial_scale}"),
        key=f"{location}/{temp_resolution}/test_target",
    )

    df_cov_train = pd.read_hdf(
        os.path.join(dir_path, f"{spatial_scale}"),
        key=f"{location}/{temp_resolution}/train_cov",
    )
    df_cov_val = pd.read_hdf(
        os.path.join(dir_path, f"{spatial_scale}"),
        key=f"{location}/{temp_resolution}/val_cov",
    )
    df_cov_test = pd.read_hdf(
        os.path.join(dir_path, f"{spatial_scale}"),
        key=f"{location}/{temp_resolution}/test_cov",
    )

    if not show_covariates:
        dfs = {
            "df_train": df_train,
            "df_val": df_val,
            "df_test": df_test,
        }
    else:
        dfs = {
            "df_train": df_train,
            "df_val": df_val,
            "df_test": df_test,
            "df_cov_train": df_cov_train,
            "df_cov_val": df_cov_val,
            "df_cov_test": df_cov_test,
        }

    # Create an empty figure
    fig = go.Figure()

    # For each dataframe in the dictionary, create a line plot trace
    for name, df in dfs.items():
        fig.add_trace(
            go.Scatter(x=df.index, y=df[df.columns[0]], mode="lines", name=name)
        )

    # Extract the location from the column and set the title and y-axis label
    loc_from_col = df_train.columns[0].split(".")[0]
    fig.update_layout(title=f"Load Data for {loc_from_col}", yaxis_title="Load [MW]")
    fig.show()
    return fig
