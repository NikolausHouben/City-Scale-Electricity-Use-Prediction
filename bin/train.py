# train_eval.py
import wandb
import pandas as pd
import argparse
from typing import List, Dict
import os
import sys

import wandb

from evaluation import evaluate, get_run_results

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pipeline import (
    Config,
    get_best_run_config,
    derive_config_params,
)

from utils.model_utils import (
    load_trained_models,
    save_models_to_disk,
    get_model_instances,
    train_models,
)


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
    parser.add_argument("--location", type=str, default="13598.MWh")
    parser.add_argument(
        "--models_to_train",
        nargs="+",
        type=str,
        default=["rf"],
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
        project="Portland_AMI_2", name=name_id, id=name_id
    )  # set id to continue existing runs
    config, models_dict = training(init_config)

    if args.evaluate:
        eval_dict = evaluate(config, models_dict)
        df_metrics = get_run_results(eval_dict, config)

    wandb.finish()
