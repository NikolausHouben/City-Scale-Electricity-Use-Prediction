from typing import Dict
import os
import sys
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import get_longest_subseries_idx, create_directory
from utils.model_utils import Config, load_data
from utils.pipeline import data_pipeline
from utils.paths import EVAL_DIR
from utils.eval_utils import backtesting, extract_forecasts_per_horizon


def evaluate(init_config: Dict, models_dict: Dict):
    """
    Loads existing run results (from wandb, TODO) if they exist,
    otherwise runs a backtest for each model on the val and test set, and then formats it into the various horizons

    """

    config = Config().from_dict(init_config)
    evaluation_scale_path = os.path.join(EVAL_DIR, config.spatial_scale)
    create_directory(evaluation_scale_path)

    try:
        with open(
            os.path.join(evaluation_scale_path, f"{config.location.split('.')[0]}.pkl"),
            "rb",
        ) as f:
            dict_result_n_ahead = pickle.load(f)
        print(f"Existing evaluation for {config.location} found, loading...")

    except:
        print("No existing evaluation found, running evaluation...")

        data = load_data(config)

        piped_data, pipeline = data_pipeline(config, data)

        (
            _,
            ts_val_piped,
            ts_test_piped,
            _,
            ts_val_weather_piped,
            ts_test_weather_piped,
        ) = piped_data

        longest_ts_val_idx = get_longest_subseries_idx(ts_val_piped)
        longest_ts_test_idx = get_longest_subseries_idx(ts_test_piped)

        trg_val_inversed = pipeline.inverse_transform(ts_val_piped)[longest_ts_val_idx]
        trg_test_inversed = pipeline.inverse_transform(ts_test_piped)[
            longest_ts_test_idx
        ]

        test_sets = {  # see data_prep.ipynb for the split
            "Winter": (
                ts_val_piped[longest_ts_val_idx],
                None
                if not config.weather_available
                else ts_val_weather_piped[longest_ts_val_idx],  # type: ignore
                trg_val_inversed,
            ),
            "Summer": (
                ts_test_piped[longest_ts_test_idx],
                None
                if not config.weather_available
                else ts_test_weather_piped[longest_ts_test_idx],  # type: ignore
                trg_test_inversed,
            ),
        }

        test_sets = {k: v for k, v in test_sets.items() if k in config.eval_seasons}

        dict_result_season = backtesting(models_dict, pipeline, test_sets, config)
        dict_result_n_ahead = extract_forecasts_per_horizon(config, dict_result_season)

        with open(
            os.path.join(evaluation_scale_path, f"{config.location.split('.')[0]}.pkl"),
            "wb",
        ) as f:
            pickle.dump(dict_result_n_ahead, f)

    return dict_result_n_ahead
