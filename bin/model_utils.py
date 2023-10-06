from typing import List
import os

from paths import MODEL_DIR
from utils import create_directory


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
