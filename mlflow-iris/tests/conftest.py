"""
Pytest Configuration
====================

Shared fixtures and configuration for tests.
"""

import os
import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def iris_data():
    """Load Iris data for testing."""
    from src.data.load_data import load_iris_local
    return load_iris_local()


@pytest.fixture(scope="session")
def iris_splits(iris_data):
    """Get train/test splits."""
    from src.features.feature_builder import FeatureBuilder
    builder = FeatureBuilder(feature_group="base_v1")
    X_train, X_test, y_train, y_test = builder.fit_transform(
        iris_data, test_size=0.2, random_state=42
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": builder.feature_names,
        "builder": builder,
    }


@pytest.fixture(scope="session")
def trained_model(iris_splits):
    """Train a model for testing."""
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
    )
    model.fit(iris_splits["X_train"], iris_splits["y_train"])
    return model


@pytest.fixture
def mlflow_tracking_uri(tmp_path):
    """Create temporary MLflow tracking URI."""
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    os.environ["MLFLOW_TRACKING_URI"] = uri
    return uri


@pytest.fixture
def sample_input():
    """Sample input data for inference testing."""
    return {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }


@pytest.fixture
def sample_batch():
    """Sample batch data for inference testing."""
    return pd.DataFrame([
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        {"sepal_length": 6.2, "sepal_width": 2.9, "petal_length": 4.3, "petal_width": 1.3},
        {"sepal_length": 7.7, "sepal_width": 3.0, "petal_length": 6.1, "petal_width": 2.3},
    ])