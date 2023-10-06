# train_eval.py

import os
import wandb
import pandas as pd
import time
import json
import argparse
import inspect
from typing import List, Dict
import torch

from darts.models import (
    LinearRegressionModel,
    RandomForest,
    LightGBMModel,
    XGBModel,
    BlockRNNModel,
    NBEATSModel,
    TFTModel,
    TiDEModel,
)

import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluation import evaluate, get_run_results

from pipeline import (
    data_pipeline,
    Config,
    load_auxilary_training_data,
    pipeline_auxilary_data,
    get_best_run_config,
    load_data,
    derive_config_params,
)

from model_utils import (
    check_if_torch_model,
    load_trained_models,
    save_models_to_disk,
    get_model_instances,
)

from paths import ROOT_DIR


def train_models(config, untrained_models, config_per_model):
    """
    This function does the actual training and is used by 'training'.
    Takes in a list of models on the training data and validates them on the validation data if it is available.

    Returns the trained models and the runtimes (how long a model took to train).

    """

    run_times = {}

    data = load_data(config)

    aux_data = load_auxilary_training_data(config)

    models = []

    for model_abbr, model in untrained_models.items():
        start_time = time.time()
        print(f"Training {model.__class__.__name__}")

        model_config = config_per_model[model_abbr]

        piped_data, _ = data_pipeline(model_config, data)

        aux_trg, aux_cov = pipeline_auxilary_data(model_config, aux_data)

        (
            ts_train_piped,
            ts_val_piped,
            ts_test_piped,
            ts_train_weather_piped,
            ts_val_weather_piped,
            ts_test_weather_piped,
        ) = piped_data

        print("Extended training data with auxilary data")
        ts_train_piped.extend(aux_trg)  # type: ignore
        ts_train_weather_piped.extend(aux_cov)  # type: ignore

        if model.supports_future_covariates:
            try:
                model.fit(
                    ts_train_piped,
                    future_covariates=ts_train_weather_piped,
                    val_series=ts_val_piped,
                    val_future_covariates=ts_val_weather_piped,
                )
            except:
                model.fit(ts_train_piped, future_covariates=ts_train_weather_piped)
        elif model_config.use_cov_as_past_cov and not model.supports_future_covariates:
            try:
                model.fit(
                    ts_train_piped,
                    past_covariates=ts_train_weather_piped,
                    val_series=ts_val_piped,
                    val_past_covariates=ts_val_weather_piped,
                )
            except:
                model.fit(ts_train_piped, past_covariates=ts_train_weather_piped)
        else:
            try:
                model.fit(ts_train_piped, val_series=ts_val_piped)
            except:
                model.fit(ts_train_piped)

        models.append(model)

        end_time = time.time()
        run_times[model.__class__.__name__] = end_time - start_time
    return models, run_times


def training(init_config: Dict):
    """Loads existing models (from disk) if they exist, otherwise trains new models with optimial hyperparameters (from wandb) if they exist"""

    config = Config().from_dict(init_config)
    config = derive_config_params(config)
    models_to_train = config.models_to_train

    # Importing hyperparameters from wandb for models that have previously been tuned
    config_per_model = {}
    config_per_model.update(
        {"lr": config}
    )  # add the default config to the config_per_model dict for linear regression
    for model in models_to_train:
        model_config, _ = get_best_run_config(
            "Portland_AMI_tuning",
            "+eval_loss",
            model,
            config.spatial_scale,
            config.location,
        )
        # update model_config with basic config if they are not yet in the keys of the model config
        for key, value in config.data.items():
            if key not in model_config.data.keys():
                model_config[key] = value
        model_config.n_ahead = (
            config.n_ahead
        )  # the sweeps were done for 24h ahead, but we want to train for 48h ahead

        config_per_model[model] = model_config

    # getting the model instances for all models
    model_instances = get_model_instances(models_to_train, config_per_model)

    # loading the trained models from disk, which have been trained already
    trained_models, untrained_models = load_trained_models(config, model_instances)

    if len(untrained_models) > 0:
        print(untrained_models.keys())
        newly_trained_models, run_times = train_models(
            config, untrained_models, config_per_model
        )

        # dataframing and logging runtimes (how long each model took to train)
        df_runtimes = pd.DataFrame.from_dict(
            run_times, orient="index", columns=["runtime"]
        ).reset_index()
        wandb.log({"runtimes": wandb.Table(dataframe=df_runtimes)})
        save_models_to_disk(config, newly_trained_models)
        trained_models.extend(newly_trained_models)

    models_dict = {model.__class__.__name__: model for model in trained_models}
    wandb.config.update(config.data)

    return init_config, models_dict


if __name__ == "__main__":
    # argparse scale and location
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str, default="GLENDOVEER")
    parser.add_argument("--location", type=str, default="13596.MWh")
    parser.add_argument(
        "--models_to_train",
        nargs="+",
        type=str,
        default=["xgb", "gru"],
    )
    parser.add_argument("--evaluate", type=bool, default=False)
    args = parser.parse_args()

    init_config = {
        "spatial_scale": args.scale,
        "temp_resolution": 60,
        "location": args.location,
        "unit": "MWh",
        "models_to_train": args.models_to_train,
        "horizon_in_hours": 48,
        "lookback_in_hours": 24,
        "boxcox": True,
        "liklihood": None,
        "weather_available": True,
        "datetime_encodings": True,
        "heat_wave_binary": True,
        "datetime_attributes": ["dayofweek", "week"],
        "use_cov_as_past_cov": False,
        "use_auxilary_data": True,
    }

    wandb.login()
    # starting wandb run

    name_id = (
        init_config["spatial_scale"]
        + "_"
        + init_config["location"]
        + "_"
        + str(init_config["temp_resolution"])
        + "min"
        + "_"
        + "aux_data--"
        + str(init_config["use_auxilary_data"])
    )
    wandb.init(
        project="Portland_AMI_1", name=name_id, id=name_id
    )  # set id to continue existing runs
    config, models_dict = training(init_config)

    if args.evaluate:
        eval_dict = evaluate(config, models_dict)
        df_metrics = get_run_results(eval_dict, config)

    wandb.finish()
