# train_eval.py

import os
import wandb
import pandas as pd
import numpy as np
import time
import json
import argparse
import inspect
from typing import List, Dict


import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from utils import (
    review_subseries,
    load_trained_models,
    save_models_to_disk,
    check_if_torch_model,
    derive_config_params,
    load_data,
)

import darts
from darts.utils.missing_values import extract_subseries
from darts.dataprocessing.transformers.boxcox import BoxCox
from darts.dataprocessing.transformers.scaler import Scaler
from darts.dataprocessing import Pipeline
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


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dir_path = os.path.join(root_path, "data", "clean_data")


class Config:
    """
    Class to store config parameters, to circumvent the wandb.config when combining multiple models when debugging
    """

    def __init__(self):
        self.data = {}

    def __getattr__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == "data":
            # Allow normal assignment for the 'data' attribute
            super().__setattr__(key, value)
        else:
            self.data[key] = value

    def __delattr__(self, key):
        if key in self.data:
            del self.data[key]
        else:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    def __len__(self):
        return len(self.data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    @classmethod
    def from_dict(cls, data):
        config = cls()
        for key, value in data.items():
            config[key] = value  # Preserve nested dictionaries without converting
        return config


def initialize_kwargs(config, model_class, additional_kwargs=None):
    """Initializes the kwargs for the model with the available wandb sweep config or with the default values."""
    model_name = config.model
    is_torch_model = check_if_torch_model(
        model_class
    )  # we will handle torch models a bit differently than sklearn-API type models

    sweep_path = os.path.join(
        root_path, "sweep_configurations", f"config_sweep_{model_name}.json"
    )
    try:
        with open(sweep_path) as f:
            sweep_config = json.load(f)["parameters"]
    except:
        sweep_config = {}
        print(f"Could not find sweep config for model {model_name}")

    kwargs = {k: config.__getitem__(k) for k in sweep_config.keys()}

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

    model_abbr = config.model

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
        print(model_config.n_ahead)
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


def get_best_run_config(project_name, metric, model, scale, location: str):
    """

    Returns the config of the best run of a sweep for a given model and location.

    """

    api = wandb.Api()
    sweeps = []
    for project in api.projects():
        if project_name == project.name:
            sweeps = project.sweeps()

    config = None
    name = None
    transformer = location.split("-")[0]
    for sweep in sweeps:
        if model in sweep.name and scale in sweep.name and transformer in sweep.name:
            best_run = sweep.best_run(order=metric)
            config = best_run.config
            config = Config().from_dict(config)
            name = best_run.name
            print(f"Fetched sweep with name {name} for model {model}")

    if config == None:
        print(
            f"Could not find a sweep for model {model} and scale {scale} in project {project_name}."
        )
        config = Config()
        config.model = model

    return config, name


def data_pipeline(config, data):
    trg, cov = data["trg"], data["cov"]
    df_train, df_val, df_test = trg
    df_cov_train, df_cov_val, df_cov_test = cov

    # Heat wave covariate, categorical variable
    if config.heat_wave_binary:
        df_cov_train["heat_wave"] = df_cov_train[
            df_cov_train.columns[0]
        ] > df_cov_train[df_cov_train.columns[0]].quantile(0.95)
        df_cov_val["heat_wave"] = df_cov_val[df_cov_val.columns[0]] > df_cov_val[
            df_cov_val.columns[0]
        ].quantile(0.95)
        df_cov_test["heat_wave"] = df_cov_test[df_cov_test.columns[0]] > df_cov_test[
            df_cov_test.columns[0]
        ].quantile(0.95)

    # into darts format
    ts_train = darts.TimeSeries.from_dataframe(
        df_train, freq=str(config.temp_resolution) + "min"  # type: ignore
    )
    ts_train = extract_subseries(ts_train)
    ts_val = darts.TimeSeries.from_dataframe(
        df_val, freq=str(config.temp_resolution) + "min"  # type: ignore
    )
    ts_val = extract_subseries(ts_val)
    ts_test = darts.TimeSeries.from_dataframe(
        df_test, freq=str(config.temp_resolution) + "min"  # type: ignore
    )
    ts_test = extract_subseries(ts_test)

    # Covariates
    if config.weather_available:
        ts_cov_train = darts.TimeSeries.from_dataframe(
            df_cov_train, freq=str(config.temp_resolution) + "min"  # type: ignore
        )
        ts_cov_val = darts.TimeSeries.from_dataframe(
            df_cov_val, freq=str(config.temp_resolution) + "min"  # type: ignore
        )
        ts_cov_test = darts.TimeSeries.from_dataframe(
            df_cov_test, freq=str(config.temp_resolution) + "min"  # type: ignore
        )
    else:
        ts_cov_train = None
        ts_cov_val = None
        ts_cov_test = None

    # Reviewing subseries to make sure they are long enough
    ts_train, ts_cov_train = review_subseries(
        ts_train, config.n_lags + config.n_ahead, ts_cov_train
    )
    ts_val, ts_cov_val = review_subseries(
        ts_val, config.n_lags + config.n_ahead, ts_cov_val
    )
    ts_test, ts_cov_test = review_subseries(
        ts_test, config.n_lags + config.n_ahead, ts_cov_test
    )

    # Preprocessing Pipeline, global fit is important to turn on because we have split the entire ts into multiple timeseries at nans
    # but want to use the same params for all of them
    # Missing values have been filled in the 'data_prep.ipynb', so we don't need to do that here
    pipeline = Pipeline(
        [
            BoxCox(global_fit=True)
            if config.boxcox
            else Scaler(
                MinMaxScaler(), global_fit=True
            ),  # double scale in case boxcox is turned off
            Scaler(MinMaxScaler(), global_fit=True),
        ]
    )
    ts_train_piped = pipeline.fit_transform(ts_train)
    ts_val_piped = pipeline.transform(ts_val)
    ts_test_piped = pipeline.transform(ts_test)

    # Weather Pipeline
    if config.weather_available:
        pipeline_weather = Pipeline([Scaler(RobustScaler(), global_fit=True)])
        ts_train_weather_piped = pipeline_weather.fit_transform(ts_cov_train)
        ts_val_weather_piped = pipeline_weather.transform(ts_cov_val)
        ts_test_weather_piped = pipeline_weather.transform(ts_cov_test)
    else:
        ts_train_weather_piped = None
        ts_val_weather_piped = None
        ts_test_weather_piped = None

    piped_data = (
        ts_train_piped,
        ts_val_piped,
        ts_test_piped,
        ts_train_weather_piped,
        ts_val_weather_piped,
        ts_test_weather_piped,
    )

    return piped_data, pipeline


def train_models(config, untrained_models, config_per_model):
    """
    This function does the actual training and is used by 'training'.
    Takes in a list of models on the training data and validates them on the validation data if it is available.

    Returns the trained models and the runtimes (how long a model took to train).

    """

    run_times = {}

    data = load_data(config)

    models = []

    for model_abbr, model in untrained_models.items():
        start_time = time.time()
        print(f"Training {model.__class__.__name__}")

        model_config = config_per_model[model_abbr]

        piped_data, _ = data_pipeline(model_config, data)

        (
            ts_train_piped,
            ts_val_piped,
            ts_test_piped,
            ts_train_weather_piped,
            ts_val_weather_piped,
            ts_test_weather_piped,
        ) = piped_data

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
        print(model_config.n_ahead)

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
    parser.add_argument("--scale", type=str, default="2_town")
    parser.add_argument("--location", type=str, default="GLENDOVEER-13598")
    parser.add_argument("--models_to_train", nargs="+", type=str, default=["xgb"])
    args = parser.parse_args()

    init_config = {
        "spatial_scale": args.scale,
        "temp_resolution": 60,
        "location": args.location,
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
    )
    wandb.init(
        project="Portland_AMI", name=name_id, id=name_id
    )  # set id to continue existing runs
    config, models_dict = training(init_config)

    wandb.finish()
