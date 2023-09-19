# train_eval.py

import os
import wandb
import pandas as pd
import numpy as np
import time
import json
import argparse


import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from utils import (
    review_subseries,
    get_longest_subseries_idx,
    load_trained_models,
    save_models_to_disk,
)

import darts
from darts import TimeSeries
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
)

import wandb
from wandb.xgboost import WandbCallback


from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dir_path = os.path.join(root_path, "data", "clean_data")


class Config:
    """
    Class to store config parameters, to circumvent the wandb.config when combining multiple models.
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


def initialize_kwargs(config):
    """Initializes the kwargs for the model with the available wandb sweep config or with the default values."""
    model = config.model

    try:
        sweep_path = os.path.join(
            root_path, "sweep_configurations", f"config_sweep_{model}.json"
        )
        with open(sweep_path) as f:
            sweep_config = json.load(f)["parameters"]

        kwargs = {k: config.__getitem__(k) for k in sweep_config.keys()}
    except:
        kwargs = {}
        print("No sweep config found. Using default values.")
    return kwargs


def get_model_instances(tuned_models, config_per_model):
    """Returns a list of model instances for the models that were tuned and appends a linear regression model."""

    model_instances = {}
    for model in tuned_models:
        print("Getting model instance for " + model + "...")
        config = Config().from_dict(config_per_model[model][0])
        model_instances[model] = get_model(config)

    # since we did not optimize the hyperparameters for the linear regression model, we need to create a new instance
    print("Getting model instance for linear regression...")
    config = Config().from_dict(config_per_model[tuned_models[0]][0])
    lr_model = LinearRegressionModel(
        lags=config.n_lags,
        lags_future_covariates=[0],
        output_chunk_length=config.n_ahead,
        add_encoders=config.datetime_encoders,
        random_state=42,
    )

    model_instances["lr"] = lr_model
    return model_instances


def get_model(config):
    """Returns model instance, based on the models specified in the config."""

    model = config.model

    # ----------------- #

    optimizer_kwargs = {}
    try:
        optimizer_kwargs["lr"] = config.lr
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
    if model == "xgb":
        xgb_kwargs = initialize_kwargs(config)
        xgb_kwargs["early_stopping_rounds"] = 20
        # xgb_kwargs["callbacks"] = [WandbCallback()]
        xgb_kwargs["verbose"] = 10
        xgb_kwargs["eval_metric"] = "rmse"

        model = XGBModel(
            lags=config.n_lags,
            lags_future_covariates=[-1],
            add_encoders=config.datetime_encoders,
            output_chunk_length=config.n_ahead,
            likelihood=config.liklihood,
            random_state=42,
            **xgb_kwargs,
        )

    elif model == "lgbm":
        lightgbm_kwargs = initialize_kwargs(config)
        lightgbm_kwargs[
            "early_stopping_round"
        ] = 20  # note that 'early_stopping_rounds' is not supported by lightgbm only 'early_stopping_round'
        lightgbm_kwargs["metric"] = "rmse"

        model = LightGBMModel(
            lags=config.n_lags,
            lags_future_covariates=[-1],
            add_encoders=config.datetime_encoders,
            output_chunk_length=config.n_ahead,
            likelihood=config.liklihood,
            random_state=42,
            **lightgbm_kwargs,
        )

    elif model == "rf":
        rf_kwargs = initialize_kwargs(config)

        model = RandomForest(
            lags=config.n_lags,
            lags_future_covariates=[-1],
            add_encoders=config.datetime_encoders,
            output_chunk_length=config.n_ahead,
            random_state=42,
            **rf_kwargs,
        )

    # ==== Neural Network models ====

    elif model == "nbeats":
        nbeats_kwargs = initialize_kwargs(config)

        model = NBEATSModel(
            input_chunk_length=config.n_lags,
            output_chunk_length=config.n_ahead,
            add_encoders=config.datetime_encoders,
            likelihood=config.liklihood,
            pl_trainer_kwargs=pl_trainer_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=schedule_kwargs,
            random_state=42,
            **nbeats_kwargs,
        )

    elif model == "gru":
        rnn_kwargs = initialize_kwargs(config)

        model = BlockRNNModel(
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
            **rnn_kwargs,
        )

    elif model == "tft":
        tft_kwargs = initialize_kwargs(config)

        model = TFTModel(
            input_chunk_length=config.n_lags,
            output_chunk_length=config.n_ahead,
            add_encoders=config.datetime_encoders,
            likelihood=config.liklihood,
            pl_trainer_kwargs=pl_trainer_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=schedule_kwargs,
            random_state=42,
            **tft_kwargs,
        )

    else:
        raise ValueError(f"Model {model} not supported.")

    return model


def get_best_run_config(project_name, metric, model, scale, location):
    """

    Returns the config of the best run of a sweep for a given model and location.

    """

    sweeps = []
    config = None
    name = None

    api = wandb.Api()
    for project in api.projects():
        if project_name == project.name:
            sweeps = project.sweeps()

    for sweep in sweeps:
        if model in sweep.name and scale in sweep.name and location in sweep.name:
            best_run = sweep.best_run(order=metric)
            config = best_run.config
            name = best_run.name

    if config == None:
        print(
            f"Could not find a sweep for model {model} and scale {scale} in project {project_name}."
        )

    return config, name


def data_pipeline(config):
    if config.temp_resolution == 60:
        timestep_encoding = ["hour"]
    elif config.temp_resolution == 15:
        timestep_encoding = ["quarter"]
    else:
        timestep_encoding = ["hour", "minute"]

    datetime_encoders = {
        "cyclic": {"future": timestep_encoding},
        # "position": {
        #     "future": [
        #         "relative",
        #     ]
        # },
        "datetime_attribute": {"future": ["dayofweek", "week"]},
    }

    datetime_encoders = datetime_encoders if config.datetime_encodings else None

    config["datetime_encoders"] = datetime_encoders

    config.timesteps_per_hour = int(60 / config.temp_resolution)
    config.n_lags = config.lookback_in_hours * config.timesteps_per_hour
    config.n_ahead = config.horizon_in_hours * config.timesteps_per_hour
    config.eval_stride = int(
        np.sqrt(config.n_ahead)
    )  # evaluation stride, how often to evaluate the model, in this case we evaluate every n_ahead steps

    # Loading Data
    df_train = pd.read_hdf(
        os.path.join(dir_path, f"{config.spatial_scale}.h5"),
        key=f"{config.location}/{config.temp_resolution}min/train_target",
    )
    df_val = pd.read_hdf(
        os.path.join(dir_path, f"{config.spatial_scale}.h5"),
        key=f"{config.location}/{config.temp_resolution}min/val_target",
    )
    df_test = pd.read_hdf(
        os.path.join(dir_path, f"{config.spatial_scale}.h5"),
        key=f"{config.location}/{config.temp_resolution}min/test_target",
    )

    df_cov_train = pd.read_hdf(
        os.path.join(dir_path, f"{config.spatial_scale}.h5"),
        key=f"{config.location}/{config.temp_resolution}min/train_cov",
    )
    df_cov_val = pd.read_hdf(
        os.path.join(dir_path, f"{config.spatial_scale}.h5"),
        key=f"{config.location}/{config.temp_resolution}min/val_cov",
    )
    df_cov_test = pd.read_hdf(
        os.path.join(dir_path, f"{config.spatial_scale}.h5"),
        key=f"{config.location}/{config.temp_resolution}min/test_cov",
    )

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
    if config.weather:
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

    # getting the index of the longest subseries, to be used for evaluation later,
    # TODO: remove this so all of the data is used for evaluation
    config.longest_ts_val_idx = get_longest_subseries_idx(ts_val)
    config.longest_ts_test_idx = get_longest_subseries_idx(ts_test)

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
    if config.weather:
        pipeline_weather = Pipeline([Scaler(RobustScaler(), global_fit=True)])
        ts_train_weather_piped = pipeline_weather.fit_transform(ts_cov_train)
        ts_val_weather_piped = pipeline_weather.transform(ts_cov_val)
        ts_test_weather_piped = pipeline_weather.transform(ts_cov_test)
    else:
        ts_train_weather_piped = None
        ts_val_weather_piped = None
        ts_test_weather_piped = None

    trg_train_inversed = pipeline.inverse_transform(ts_train_piped, partial=True)
    trg_val_inversed = pipeline.inverse_transform(ts_val_piped, partial=True)[
        config.longest_ts_val_idx
    ]
    trg_test_inversed = pipeline.inverse_transform(ts_test_piped, partial=True)[
        config.longest_ts_test_idx
    ]

    return (
        pipeline,
        ts_train_piped,
        ts_val_piped,
        ts_test_piped,
        ts_train_weather_piped,
        ts_val_weather_piped,
        ts_test_weather_piped,
        trg_train_inversed,
        trg_val_inversed,
        trg_test_inversed,
    )


def train_models(
    models: list,
    ts_train_piped,
    ts_train_weather_piped=None,
    ts_val_piped=None,
    ts_val_weather_piped=None,
    use_cov_as_past=False,
):
    """
    This function does the actual training and is used by 'training'.
    Takes in a list of models on the training data and validates them on the validation data if it is available.

    Returns the trained models and the runtimes.

    """

    run_times = {}

    for model in models:
        start_time = time.time()
        print(f"Training {model.__class__.__name__}")
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
        elif use_cov_as_past and not model.supports_future_covariates:
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

        end_time = time.time()
        run_times[model.__class__.__name__] = end_time - start_time
    return models, run_times


# experiments


def training(scale, location, tuned_models):
    """Loads existing models (from disk) if they exist, otherwise trains new models with optimial hyperparameters (from wandb) if they exist"""

    units_dict = {"county": "GW", "town": "MW", "village": "kW", "neighborhood": "kW"}
    resolution = 60

    config_per_model = {}
    for model in tuned_models:
        config, name = get_best_run_config(
            "Wattcast_tuning", "-eval_loss", model, scale, location
        )
        print(f"Fetched sweep with name {name} for model {model}")
        config["horizon_in_hours"] = 48  # type: ignore
        config["location"] = location  # type: ignore
        config_per_model[model] = config, name

    name_id = scale + "_" + location + "_" + str(resolution) + "min"
    wandb.init(project="Wattcast", name=name_id, id=name_id)

    config = Config().from_dict(config_per_model[tuned_models[0]][0])

    (
        pipeline,
        ts_train_piped,
        ts_val_piped,
        ts_test_piped,
        ts_train_weather_piped,
        ts_val_weather_piped,
        ts_test_weather_piped,
        trg_train_inversed,
        trg_val_inversed,
        trg_test_inersed,
    ) = data_pipeline(config)

    model_instances = get_model_instances(tuned_models, config_per_model)

    trained_models, model_instances = load_trained_models(config, model_instances)

    if len(model_instances) > 0:
        just_trained_models, run_times = train_models(
            model_instances.values(),  # type: ignore
            ts_train_piped,
            ts_train_weather_piped if config.weather else None,
            ts_val_piped,
            ts_val_weather_piped if config.weather else None,
        )

        df_runtimes = pd.DataFrame.from_dict(
            run_times, orient="index", columns=["runtime"]
        ).reset_index()
        wandb.log({"runtimes": wandb.Table(dataframe=df_runtimes)})
        trained_models.extend(just_trained_models)

    models_dict = {model.__class__.__name__: model for model in trained_models}
    save_models_to_disk(config, models_dict)

    config.model_names = list(models_dict.keys())

    config.unit = units_dict[scale.split("_")[1]]

    wandb.config.update(config.data)

    return config, models_dict


if __name__ == "__main__":
    # argparse scale and location
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str, default="1_county")
    parser.add_argument("--location", type=str, default="Los_Angeles")
    parser.add_argument("--tuned_models", nargs="+", type=str, default=["xgb"])
    args = parser.parse_args()

    wandb.login()
    config, models_dict = training(args.scale, args.location, args.tuned_models)
    wandb.finish()
