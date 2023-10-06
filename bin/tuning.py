# tuning.py

import os
import json

import argparse
import pandas as pd
import plotly.express as px

import wandb

from paths import SWEEP_DIR

from pipeline import load_data, derive_config_params, data_pipeline

from utils import get_longest_subseries_idx

from model_utils import get_model, train_models

from evaluation import predict_testset


def train_eval_tuning():
    wandb.init(project="Portland_AMI_tuning")
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
    parser.add_argument("--scale", type=str, default="GLENDOVEER")
    parser.add_argument("--location", type=str, default="13596.MWh")
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
        init_config = {
            "spatial_scale": args.scale,
            "temp_resolution": 60,
            "location": args.location,
            "model_abbr": model,
            "horizon_in_hours": 24,
            "lookback_in_hours": 24,
            "boxcox": True,
            "liklihood": None,
            "weather_available": True,
            "datetime_encodings": True,
            "heat_wave_binary": True,
            "datetime_attributes": ["dayofweek", "week"],
            "use_cov_as_past_cov": False,
            "use_auxilary_data": False,
        }

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

        sweep_id = wandb.sweep(sweep_config, project="Portland_AMI_tuning")
        wandb.agent(sweep_id, train_eval_tuning, count=args.n_sweeps)
        wandb.finish()
