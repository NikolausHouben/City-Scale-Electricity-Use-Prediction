# tuning.py

import os
import json
import sys

import argparse
import pandas as pd
import plotly.express as px

import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import ROOT_DIR, SWEEP_DIR, TUNING_WANDB

from utils.pipeline import load_data, derive_config_params, data_pipeline

from utils.data_utils import get_longest_subseries_idx

from utils.model_utils import get_model, train_models

from evaluation import predict_testset


def train_eval_tuning():
    wandb.init(project=TUNING_WANDB)
    wandb.config.update(init_config)
    config = wandb.config

    print("Getting data...")

    data = load_data(config)
    config = derive_config_params(config)
    piped_data, pipeline = data_pipeline(config, data)

    _, _, ts_test_piped, _, _, ts_test_weather_piped = piped_data
    longest_ts_test_idx = get_longest_subseries_idx(ts_test_piped)
    trg_test_inversed = pipeline.inverse_transform(ts_test_piped)[longest_ts_test_idx]

    print("Getting model instance...")
    model_instance = get_model(config)
    model_instance, _ = train_models(
        config, {config.model_abbr: model_instance}, {config.model_abbr: config}
    )  # need to pass in a dict of models and configs

    print("Evaluating model...")
    predictions, score = predict_testset(
        model_instance[0],
        ts_test_piped[longest_ts_test_idx],
        ts_test_weather_piped[longest_ts_test_idx],  # type: ignore
        config.n_lags,
        config.n_ahead,
        config.eval_stride,
        pipeline,
    )

    print("Plotting predictions...")
    df_compare = pd.concat(
        [trg_test_inversed.pd_dataframe(), predictions], axis=1
    ).dropna()
    df_compare.columns = ["target", "prediction"]
    fig = px.line(df_compare, title="Predictions vs. Test Set")

    wandb.log({"eval_loss": score})
    wandb.log({"predictions": fig})
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str)
    parser.add_argument("--location", type=str)
    parser.add_argument("--n_sweeps", type=int, default=1)
    parser.add_argument(
        "--models_to_train",
        nargs="+",
        type=str,
        default=["tft"],
    )
    args = parser.parse_args()

    for model in args.models_to_train:
        # placeholder initialization of config file (will be updated in train_eval_light())

        with open(os.path.join(ROOT_DIR, "init_config.json"), "r") as fp:
            init_config = json.load(fp)

        # sweep specific
        init_config["spatial_scale"] = args.scale
        init_config["location"] = args.location
        init_config["model_abbr"] = model

        with open(os.path.join(SWEEP_DIR, f"config_sweep_{model}.json"), "r") as fp:
            sweep_config = json.load(fp)

        sweep_config["name"] = (
            model
            + "sweep"
            + init_config["spatial_scale"]
            + "_"
            + init_config["location"]
            + "_"
            + str(init_config["temp_resolution"])
        )

        sweep_id = wandb.sweep(sweep_config, project=TUNING_WANDB)
        wandb.agent(sweep_id, train_eval_tuning, count=args.n_sweeps)
        wandb.finish()
