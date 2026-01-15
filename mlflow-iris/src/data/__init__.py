"""Data loading module."""

from .load_data import (
    load_training_data,
    load_iris_local,
    log_data_info_to_mlflow,
)

__all__ = [
    "load_training_data",
    "load_iris_local",
    "log_data_info_to_mlflow",
]