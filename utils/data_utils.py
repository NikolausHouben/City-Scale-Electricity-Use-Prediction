# utils.py

"""This file contains utility functions that are used in the notebooks."""


from itertools import combinations

import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from darts import TimeSeries
import requests
from timezonefinder import TimezoneFinder
from darts.utils.missing_values import fill_missing_values
import h5py
import wandb
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
                locations_per_file[file_name.split(".")[0]] = locations
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


def shorten_historics_to_n_ahead(historics, n_ahead):
    """
    For a n_ahead (forecasting horizon) of e.g., 4 this function returns the first, fifth, ninth, etc. element of a list of time series and shortens each to the horizon.
    """
    ts_list_shortened_skipped = [ts[:n_ahead] for ts in historics]

    return ts_list_shortened_skipped


def ts_list_concat(ts_list, n_ahead):
    """
    This function concatenates a list of time series into one time series.
    """
    df_forecast = pd.concat([ts.pd_dataframe() for ts in ts_list][::n_ahead], axis=0)
    ts = TimeSeries.from_dataframe(df_forecast)

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


def select_first_week_of_each_month(df, config):
    # Ensure that the DataFrame has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame must have a datetime index.")

    # Create an empty DataFrame to store the selected data
    selected_data = pd.DataFrame()

    # Iterate through each month
    for year_month, group in df.groupby(df.index.to_period("M")):
        # Check if there are enough days in the month for a full week
        if len(group) >= 7 * config.timesteps_per_hour * 24:
            # Select one week of data from the month
            selected_week = group.head(7 * config.timesteps_per_hour * 24)
            # Append the selected week to the result
            selected_data = selected_data.append(selected_week)  # type: ignore

    return selected_data


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


def get_file_names(project_name, name_id_dict, spatial_scale, location, season):
    run_to_visualize = f"{spatial_scale}_{location}"
    run = api.run(f"wattcast/{project_name}/{name_id_dict[run_to_visualize]}")
    files = []
    for file in run.files():
        if "Side" in str(file) and season in str(file):
            files.append(file)
    return files


def get_latest_plotly_plots(files):
    # get the most recent file for each horizon
    zip_files = combinations(files, 2)
    files_to_plot = {}
    for file1, file2 in zip_files:
        if check_if_same_horizon_plot(file1, file2):
            file_recent = choose_more_recent(file1, file2)
            files_to_plot[file_recent._attrs["name"]] = file_recent
        else:
            files_to_plot[file1._attrs["name"]] = file1
            files_to_plot[file2._attrs["name"]] = file2

    files_list = list(files_to_plot.values())
    return files_list


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


def select_horizon(df_all, horizon):
    df_fc = df_all.filter(regex=f"Horizon: {horizon} Hours")  # filtering by horizon
    df_fc = pd.concat(
        [df_fc, df_all["Ground Truth"]], axis=1
    )  # adding ground truth back to the dataframe
    columns_to_drop = df_fc.filter(like="Persistence").columns
    df_fc = df_fc.drop(columns_to_drop, axis=1)
    return df_fc


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
        scale = run_.name.split("_")[0] + "_" + run_.name.split("_")[1]
        location = run_.name.split("_")[2] + "_" + run_.name.split("_")[3]
        name_id_dict[f"{scale}_{location}"] = run_.id
    return name_id_dict


def remove_outliers(df, column, lower_percentile=0, upper_percentile=100):
    # Calculate the thresholds
    lower_threshold = df[column].quantile(lower_percentile / 100)
    upper_threshold = df[column].quantile(upper_percentile / 100)

    # Clip the values
    df[column] = df[column].clip(lower_threshold, upper_threshold)

    return df


def plot_location_splits(dir_path, scale_idx, location_idx, show="trg"):
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

    if show == "trg":
        dfs = {
            "df_train": df_train,
            "df_val": df_val,
            "df_test": df_test,
        }
    elif show == "cov":
        dfs = {
            "df_cov_train": df_cov_train,
            "df_cov_val": df_cov_val,
            "df_cov_test": df_cov_test,
        }
    else:  # assuming show="both"
        dfs = {
            "df_train": df_train,
            "df_val": df_val,
            "df_test": df_test,
            "df_cov_train": df_cov_train,
            "df_cov_val": df_cov_val,
            "df_cov_test": df_cov_test,
        }

    fig = go.Figure()

    for name, df in dfs.items():
        if "cov" in name and show == "both":
            yaxis = "y2"
        else:
            yaxis = "y1"

        fig.add_trace(
            go.Scatter(
                x=df.index, y=df[df.columns[0]], mode="lines", name=name, yaxis=yaxis
            )
        )

    # Setting up the layout based on the 'show' parameter
    loc_from_col = df_train.columns[0].split(".")[0]
    if show == "trg":
        fig.update_layout(
            title=f"Target Data for {loc_from_col}", yaxis_title="Load [MW]"
        )
    elif show == "cov":
        fig.update_layout(
            title=f"Covariate Data for {loc_from_col}", yaxis_title="Temperature [°C]"
        )
    else:
        fig.update_layout(
            title=f"Data for {loc_from_col}",
            yaxis=dict(title="Load [MW]"),
            yaxis2=dict(title="Temperature [°C]", overlaying="y", side="right"),
        )
    fig.show()
    return fig


def generate_ep_profile(df, hour_shift=3, mu=0.0, sigma=0.3):
    """Generate electricity price profiles based on the ground truth of the load"""

    timesteps_per_hour = int(infer_frequency(df) // 60)
    shift_in_timesteps = hour_shift * timesteps_per_hour
    # step 1: shift the ground truth by n hours
    series = df.iloc[:, 0]
    ep1 = series.shift(shift_in_timesteps)
    # step 2: add a random noise to it
    noise = np.random.normal(mu, sigma, len(ep1))
    ep2 = ep1 + noise
    # step 3: smooth it
    ep3 = ep2.ewm(span=timesteps_per_hour * 6).mean().fillna(method="bfill")
    ep4 = ep3.to_frame("ep")
    return ep4
