"""Paths & WandB Projects used in the project."""

import os

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DATA_DIR = os.path.join(ROOT_DIR, "data", "clean_data")
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw_data")
EVAL_DIR = os.path.join(ROOT_DIR, "data", "evaluations")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
SWEEP_DIR = os.path.join(ROOT_DIR, "sweep_configurations")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


# wandb projects
TUNING_WANDB = "Multi_Scale_Paper_Tuning"
EXPERIMENT_WANDB = "Multi_Scale_Paper_Final_Runs_2"
SYNTHESIS_WANDB = "Multi_Scale_Paper_Synthesis"
