"""Paths used in the project."""

import os

# Path to the root of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DATA_DIR = os.path.join(ROOT_DIR, "data", "clean_data")
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw_data")
EVAL_DIR = os.path.join(ROOT_DIR, "data", "evaluations")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
SWEEP_DIR = os.path.join(ROOT_DIR, "sweep_configurations")
