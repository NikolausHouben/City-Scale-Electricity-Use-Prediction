# evals.py

import pandas as pd
import numpy as np
import os
import sys
import json
import pickle
import wandb
import plotly.graph_objects as go
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import ROOT_DIR, EVAL_DIR, CLEAN_DATA_DIR, EXPERIMENT_WANDB

from utils.pipeline import Config
from utils.data_utils import get_hdf_keys, get_df_diffs, get_df_compares_list
from nle import run_nle

# metrics
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


def rmse(df):
    df_clean = df.dropna()
    try:
        return mean_squared_error(
            df_clean.iloc[:, 1], df_clean.iloc[:, 0], squared=False
        )
    except Exception:
        return np.nan


def mae(df):
    df_clean = df.dropna()
    try:
        return mean_absolute_error(df_clean.iloc[:, 1], df_clean.iloc[:, 0])
    except Exception:
        return np.nan


def mape(df):
    df_clean = df.dropna()
    try:
        return mean_absolute_percentage_error(df_clean.iloc[:, 1], df_clean.iloc[:, 0])
    except Exception:
        return np.nan


def r2(df):
    df_clean = df.dropna()
    try:
        return r2_score(df_clean.iloc[:, 1], df_clean.iloc[:, 0])
    except Exception:
        return np.nan


def smape(df):
    df_clean = df.dropna()
    try:
        return 200.0 * np.mean(
            np.abs(df_clean.iloc[:, 1] - df_clean.iloc[:, 0])
            / (np.abs(df_clean.iloc[:, 1]) + np.abs(df_clean.iloc[:, 0]))
        )
    except Exception:
        return np.nan


def max_peak_error(df):
    df_clean = df.dropna()
    try:
        return np.max(np.abs(df_clean.iloc[:, 1] - df_clean.iloc[:, 0]))
    except Exception:
        return np.nan


def mean_n_peak_error(df):
    df_clean = df.dropna()
    try:
        return np.mean(np.sort(np.abs(df_clean.iloc[:, 1] - df_clean.iloc[:, 0]))[-5:])
    except Exception:
        return np.nan


metrics_dict = {
    "rmse": rmse,
    "mape": mape,
    "mae": mae,
    "r2_score": r2,
    "smape": smape,
    "max_peak_error": max_peak_error,
    "mean_n_peak_error": mean_n_peak_error,
    "nle": run_nle,
}


def load_eval_dict(scale, location):
    with open(os.path.join(EVAL_DIR, f"{scale}/{location}.pkl"), "rb") as f:
        eval_dict = pickle.load(f)
    return eval_dict


def get_eval_df(eval_dict, horizon, season, model):
    gt = eval_dict[horizon][season][2].pd_dataframe()
    gt.columns = ["gt_" + col for col in gt.columns]
    df_preds = pd.concat(
        [
            pred_batch.pd_dataframe()
            for pred_batch in eval_dict[horizon][season][0][model]
        ],
        axis=0,
    )
    df_ = df_preds.join(gt, how="left")

    return df_


def get_metrics_table(eval_dict, metrics_dict, scale, location):
    """Loops through all horizons, seasons, and models and calculates metrics for each."""

    print("Calculating metrics table")
    keys = ["horizon_in_hours", "season", "model"]  #  nested dict keys of eval_dict
    df_results = pd.DataFrame(columns=keys + list(metrics_dict.keys()))
    for horizon in eval_dict.keys():
        for season in eval_dict[horizon].keys():
            if season == "Summer":
                for model in eval_dict[horizon][season][0].keys():
                    df_ = get_eval_df(eval_dict, horizon, season, model)
                    results_dict = dict(zip(keys, [horizon, season, model]))
                    for metric_str, metric_fn in metrics_dict.items():
                        if metric_str == "nle":
                            if horizon == 1:  # nle only works for horizon > 1
                                results_dict[metric_str] = np.nan
                            else:
                                results_dict[metric_str] = metric_fn(
                                    eval_dict, scale, location, horizon, season, model
                                )[0]
                        else:
                            results_dict[metric_str] = metric_fn(df_)

                    df_result = pd.DataFrame(results_dict, index=[0])
                    df_results = pd.concat([df_results, df_result], axis=0)

    df_results = df_results.reset_index(drop=True)

    # implementing rmse-skill score compared to linear regression for each row
    for metric in metrics_dict.keys():
        vals = df_results[metric]
        ref_vals = df_results.loc[
            df_results["model"] == "LinearRegressionModel", metric
        ]

        skill_scores = 1 - (vals / ref_vals.reindex(vals.index).bfill())

        df_results[f"{metric}_skill"] = skill_scores

    wandb.log({"Metrics_Table": wandb.Table(dataframe=df_results)})
    return df_results


def side_by_side(dict_result_n_ahead, config):
    print("Plotting side-by-side comparison of predictions and the ground truth")

    config = Config().from_dict(config)

    df_cov_train = pd.read_hdf(
        os.path.join(CLEAN_DATA_DIR, f"{config.spatial_scale}.h5"),
        key=f"{config.location}/{config.temp_resolution}min/train_cov",
    )
    df_cov_val = pd.read_hdf(
        os.path.join(CLEAN_DATA_DIR, f"{config.spatial_scale}.h5"),
        key=f"{config.location}/{config.temp_resolution}min/val_cov",
    )
    df_cov_test = pd.read_hdf(
        os.path.join(CLEAN_DATA_DIR, f"{config.spatial_scale}.h5"),
        key=f"{config.location}/{config.temp_resolution}min/test_cov",
    )

    model_names = list(dict_result_n_ahead[1]["Summer"][1].keys())
    temp_data = {"Summer": df_cov_test.iloc[:, 0], "Winter": df_cov_val.iloc[:, 0]}  # type: ignore

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

            for model_name in model_names:
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
    return None


def error_distribution(dict_result_n_ahead):
    print("Plotting error distribution")

    n_ahead, dict_result_season = list(dict_result_n_ahead.items())[-1]
    model_names = list(dict_result_n_ahead[1]["Summer"][1].keys())[:-2]
    for season, (historics_per_model, _, gt) in dict_result_season.items():
        fig, ax = plt.subplots(
            ncols=len(model_names), figsize=(5 * len(model_names), 5)
        )
        fig.suptitle(f"Absolute Error Distribution in {season} for Horizon {n_ahead}")
        for i, (model_name, historics) in enumerate(
            list(historics_per_model.items())[:-1]
        ):
            df_list = get_df_compares_list(historics, gt)
            diffs = get_df_diffs(df_list)
            diffs_flat = pd.Series(
                diffs.values.reshape(
                    -1,
                )
            )
            ax[i].hist(diffs_flat, bins=100)
            ax[i].set_title(model_name)

        wandb.log({f"Error Distribution in {season}": wandb.Image(fig)})
    return None


if __name__ == "__main__":
    # os.environ["WANDB_MODE"] = "dryrun"
    wandb.login()

    scale_locs = {
        "5_building": ["building_1", "building_2"],
        "4_neighborhood": ["neighborhood_0", "neighborhood_1", "neighborhood_2"],
        "2_town": ["town_0", "town_1", "town_2"],
        "1_county": ["Los_Angeles", "New_York", "Sacramento"],
    }

    for scale, locations in scale_locs.items():
        for location in locations:
            print(f"Running evaluation for {scale} - {location}")
            with open(os.path.join(ROOT_DIR, "init_config.json"), "r") as fp:
                init_config = json.load(fp)
                init_config["spatial_scale"] = scale
                init_config["location"] = location

            # Filter metrics
            metrics_dict = {
                metric_str: metrics_dict[metric_str]
                for metric_str in init_config["metrics"]
            }

            # starting wandb run
            name_id = scale + "_" + location

            wandb.init(project=EXPERIMENT_WANDB, name=name_id, config=init_config)

            eval_dict = load_eval_dict(scale, location)

            df_results = get_metrics_table(eval_dict, metrics_dict, scale, location)

            fig = side_by_side(eval_dict, init_config)

            dist = error_distribution(eval_dict)

            wandb.finish()
