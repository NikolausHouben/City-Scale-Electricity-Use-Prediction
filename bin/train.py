# train_eval.py

import os
import wandb
import pandas as pd
import time
import json
import argparse
import inspect
from typing import List, Dict

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
from wandb.xgboost import WandbCallback
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

from model_utils import check_if_torch_model, load_trained_models, save_models_to_disk

from paths import ROOT_DIR


def initialize_kwargs(config, model_class, additional_kwargs=None):
    """Initializes the kwargs for the model with the available wandb sweep config or with the default values."""
    model_name = config.model_abbr
    is_torch_model = check_if_torch_model(
        model_class
    )  # we will handle torch models a bit differently than sklearn-API type models

    sweep_path = os.path.join(
        ROOT_DIR, "sweep_configurations", f"config_sweep_{model_name}.json"
    )
    try:
        with open(sweep_path) as f:
            sweep_config = json.load(f)["parameters"]
    except:
        sweep_config = {}
        print(f"Could not find sweep config for model {model_name} saved locally")

    kwargs = config.data

    if additional_kwargs is not None:
        kwargs.update(additional_kwargs)

    if is_torch_model:
        valid_keywords = list(
            set(list(inspect.signature(model_class.__init__).parameters.keys())[1:])
            & set(kwargs.keys())
        )
        kwargs = {key: value for key, value in kwargs.items() if key in valid_keywords}

    else:
        print(f"Initializing kwargs for sklearn-API type model {model_name}...")
        m = model_class(
            lags=config.n_lags
        )  # need to initialize the model to get the available params, we will not use this model
        params = m.model.get_params()

        valid_keywords = list(set(params.keys()) & set(kwargs.keys()))

        kwargs = {key: value for key, value in kwargs.items() if key in valid_keywords}

    return kwargs


def get_model(config):
    """Returns model instance, based on the models specified in the config."""

    model_abbr = config.model_abbr

    # ----------------- #

    optimizer_kwargs = {}
    try:
        optimizer_kwargs["lr"] = config.learning_rate
    except:
        optimizer_kwargs["lr"] = 1e-3

    pl_trainer_kwargs = {
        "max_epochs": 20,
        "accelerator": "gpu",
        "devices": [0],
        "callbacks": [EarlyStopping(monitor="val_loss", patience=5, mode="min")],
        #'logger': WandbLogger(log_model='all'), #turn on in case you want to log the model itself to wandb
    }
    schedule_kwargs = {"patience": 2, "factor": 0.5, "min_lr": 1e-5, "verbose": True}
    # ----------------- #

    # ==== Tree-based models ====
    if model_abbr == "xgb":
        model_class = XGBModel
        xgb_kwargs = {
            "early_stopping_rounds": 5,
            "eval_metric": "rmse",
            "verbosity": 1,
        }
        kwargs = initialize_kwargs(config, model_class, additional_kwargs=xgb_kwargs)

        model = model_class(
            lags=config.n_lags,
            add_encoders=config.datetime_encoders,
            output_chunk_length=config.n_ahead,
            likelihood=config.liklihood,
            random_state=42,
            **kwargs,
        )

    elif model_abbr == "lgbm":
        model_class = LightGBMModel

        lightgbm_kwargs = {"early_stopping_round": 20, "eval_metric": "rmse"}
        kwargs = initialize_kwargs(
            config, model_class, additional_kwargs=lightgbm_kwargs
        )

        model = model_class(
            lags=config.n_lags,
            add_encoders=config.datetime_encoders,
            output_chunk_length=config.n_ahead,
            likelihood=config.liklihood,
            random_state=42,
            **kwargs,
        )

    elif model_abbr == "rf":
        model_class = RandomForest

        kwargs = initialize_kwargs(config, model_class)

        model = model_class(
            lags=config.n_lags,
            add_encoders=config.datetime_encoders,
            output_chunk_length=config.n_ahead,
            random_state=42,
            **kwargs,
        )

    # ==== Neural Network models ====

    elif model_abbr == "nbeats":
        model_class = NBEATSModel

        kwargs = initialize_kwargs(config, model_class)

        model = model_class(
            input_chunk_length=config.n_lags,
            output_chunk_length=config.n_ahead,
            add_encoders=config.datetime_encoders,
            likelihood=config.liklihood,
            pl_trainer_kwargs=pl_trainer_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=schedule_kwargs,
            random_state=42,
            **kwargs,
        )

    elif model_abbr == "gru":
        model_class = BlockRNNModel

        kwargs = initialize_kwargs(config, model_class)

        model = model_class(
            model="GRU",
            input_chunk_length=config.n_lags,
            output_chunk_length=config.n_ahead,
            add_encoders=config.datetime_encoders,
            likelihood=config.liklihood,
            pl_trainer_kwargs=pl_trainer_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=schedule_kwargs,
            random_state=42,
            **kwargs,
        )

    elif model_abbr == "tft":
        model_class = TFTModel

        kwargs = initialize_kwargs(config, model_class)

        model = model_class(
            input_chunk_length=config.n_lags,
            output_chunk_length=config.n_ahead,
            add_encoders=config.datetime_encoders,
            likelihood=config.liklihood,
            pl_trainer_kwargs=pl_trainer_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=schedule_kwargs,
            random_state=42,
        )

    elif model_abbr == "tide":
        model_class = TiDEModel

        kwargs = initialize_kwargs(config, model_class)

        model = model_class(
            input_chunk_length=config.n_lags,
            output_chunk_length=config.n_ahead,
            add_encoders=config.datetime_encoders,
            likelihood=config.liklihood,
            pl_trainer_kwargs=pl_trainer_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=schedule_kwargs,
            random_state=42,
        )

    else:
        model = None
        raise ValueError(f"Model {model_abbr} not supported.")

    return model


def get_model_instances(models: List, config_per_model: Dict) -> Dict:
    """Returns a list of model instances for the models that were tuned and appends a linear regression model."""

    model_instances = {}
    for model in models:
        print("Getting model instance for " + model + "...")
        model_config = Config().from_dict(config_per_model[model])
        model_instances[model] = get_model(model_config)

    # since we did not optimize the hyperparameters for the linear regression model, we need to create a new instance
    print("Getting model instance for linear regression...")
    lr_config = config_per_model[models[0]]
    lr_model = LinearRegressionModel(
        lags=lr_config.n_lags,
        lags_future_covariates=[0],
        output_chunk_length=lr_config.n_ahead,
        add_encoders=lr_config.datetime_encoders,
        random_state=42,
    )

    model_instances["lr"] = lr_model
    return model_instances


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


# experiments


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
        default=["xgb", "rf", "lgbm", "nbeats", "gru"],
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
        project="Portland_AMI", name=name_id, id=name_id
    )  # set id to continue existing runs
    config, models_dict = training(init_config)

    if args.evaluate:
        eval_dict = evaluate(config, models_dict)
        df_metrics = get_run_results(eval_dict, config)

    wandb.finish()
