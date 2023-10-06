from typing import List
import os
import json
from typing import List, Dict
import inspect
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

from paths import ROOT_DIR, MODEL_DIR
from utils import create_directory

from pipeline import Config


def save_models_to_disk(config, newly_trained_models: List):
    create_directory(MODEL_DIR)
    for model in newly_trained_models:
        model_path = os.path.join(
            MODEL_DIR, config.spatial_scale, config.location.split(".")[0]
        )
        create_directory(model_path)
        model.save(os.path.join(model_path, model.__class__.__name__ + ".joblib"))


def check_if_torch_model(obj):
    for cls in obj.mro():
        if "torch" in cls.__module__:
            return True
    return False


def load_trained_models(config, model_instances):
    """

    This function loads the trained models from the disk. If a model is not found, it is removed from the dictionary.

    Parameters

    config: Config
        Config object

    model_instances: dict
        Dictionary with the model instances

    Returns
    trained_models: list
    model_instances: dict

    """

    trained_models = []
    model_keys = list(model_instances.keys())  # Create a copy of the dictionary keys
    for model_abbr in model_keys:
        model = model_instances[model_abbr]
        try:
            model = model.load(
                os.path.join(
                    MODEL_DIR,
                    config.spatial_scale,
                    config.location.split(".")[0],
                    model.__class__.__name__ + ".joblib",
                )
            )
            trained_models.append(model)
            del model_instances[model_abbr]
        except:
            continue
    return trained_models, model_instances


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
    cuda = torch.cuda.is_available()

    if cuda:
        devices_value = [0]  # use GPU 0
    else:
        devices_value = 1  # use 1 CPU core

    pl_trainer_kwargs = {
        "max_epochs": 20,
        "accelerator": "gpu" if cuda else "cpu",
        "devices": devices_value,
        "callbacks": [EarlyStopping(monitor="val_loss", patience=5, mode="min")],
        # 'logger': WandbLogger(log_model='all'), #turn on in case you want to log the model itself to wandb
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
