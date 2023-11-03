# evals.py

import pandas as pd
import numpy as np
import os
import sys
import json
import pickle


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import ROOT_DIR, EVAL_DIR

from nle import run_nle

# metrics
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


rmse = lambda df: mean_squared_error(df.iloc[:, 1], df.iloc[:, 0], squared=False)
mae = lambda df: mean_absolute_error(df.iloc[:, 1], df.iloc[:, 0])
mape = lambda df: mean_absolute_percentage_error(df.iloc[:, 1], df.iloc[:, 0])
r2 = lambda df: r2_score(df.iloc[:, 1], df.iloc[:, 0])
smape = lambda df: 200.0 * np.mean(
    np.abs(df.iloc[:, 1] - df.iloc[:, 0])
    / (np.abs(df.iloc[:, 1]) + np.abs(df.iloc[:, 0]))
)
max_peak_error = lambda df: np.max(np.abs(df.iloc[:, 1] - df.iloc[:, 0]))
mean_n_peak_error = lambda df: np.mean(
    np.sort(np.abs(df.iloc[:, 1] - df.iloc[:, 0]))[-5:]
)
net_load_error = None


metrics_dict = {
    "rmse": rmse,
    "mape": mape,
    "mae": mae,
    "r2_score": r2,
    "smape": smape,
    "max_peak_error": max_peak_error,
    "mean_n_peak_error": max_peak_error,
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


if __name__ == "__main__":
    horizon = 48
    season = "Winter"
    model = "LightGBMModel"
    scale = "1_county"
    location = "Los_Angeles"

    with open(os.path.join(ROOT_DIR, "init_config.json"), "r") as fp:
        init_config = json.load(fp)

    metrics_dict = {
        metric_str: metrics_dict[metric_str] for metric_str in init_config["metrics"]
    }

    eval_dict = load_eval_dict(scale, location)

    df_ = get_eval_df(eval_dict, horizon, season, model)

    keys = ["horizon_in_hours", "season", "model"]

    results_dict = dict(zip(keys, [horizon, season, model]))

    for metric_str, metric_fn in metrics_dict.items():
        if metric_str != "nle":
            results_dict[metric_str] = metric_fn(df_)
        else:
            results_dict[metric_str] = metric_fn(
                eval_dict, scale, location, horizon, season, model
            )

    df_results = pd.DataFrame(results_dict, index=[0])
    print(df_results)
