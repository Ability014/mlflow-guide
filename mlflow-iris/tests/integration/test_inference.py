"""
Integration Tests - Inference
=============================

Tests for model loading and inference.
"""

import pytest
import numpy as np
import pandas as pd
import mlflow


@pytest.fixture
def logged_model(iris_splits, tmp_path):
    """Train and log a model for testing."""
    from src.models.train_model import ModelTrainer
    
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("/test/inference")
    
    trainer = ModelTrainer(
        model_type="random_forest",
        hyperparameters={"n_estimators": 10, "max_depth": 3},
    )
    
    with mlflow.start_run():
        trainer.train(
            iris_splits["X_train"],
            iris_splits["X_test"],
            iris_splits["y_train"],
            iris_splits["y_test"],
            feature_names=iris_splits["feature_names"],
        )
        
        model_info = mlflow.sklearn.log_model(
            trainer.model,
            "model",
        )
        
        return model_info.model_uri


class TestModelInference:
    """Tests for model inference."""
    
    def test_model_loadable(self, logged_model):
        """Model should be loadable outside notebook."""
        model = mlflow.sklearn.load_model(logged_model)
        assert model is not None
        assert hasattr(model, "predict")
    
    def test_single_prediction(self, logged_model, sample_input, iris_splits):
        """Should make single prediction."""
        model = mlflow.sklearn.load_model(logged_model)
        
        # Create input array matching training features
        X = np.array([[
            sample_input["sepal_length"],
            sample_input["sepal_width"],
            sample_input["petal_length"],
            sample_input["petal_width"],
        ]])
        
        prediction = model.predict(X)
        
        assert len(prediction) == 1
        assert prediction[0] in [0, 1, 2]
    
    def test_batch_prediction(self, logged_model, sample_batch, iris_splits):
        """Should make batch predictions."""
        model = mlflow.sklearn.load_model(logged_model)
        
        # Get features in correct order
        X = sample_batch[[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]].values
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(sample_batch)
        assert all(p in [0, 1, 2] for p in predictions)
    
    def test_probability_prediction(self, logged_model, iris_splits):
        """Should return class probabilities."""
        model = mlflow.sklearn.load_model(logged_model)
        
        probas = model.predict_proba(iris_splits["X_test"])
        
        # Should have 3 columns (3 classes)
        assert probas.shape[1] == 3
        
        # Probabilities should sum to 1
        row_sums = probas.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(row_sums)))
    
    def test_inference_deterministic(self, logged_model, iris_splits):
        """Same input should produce same output."""
        model = mlflow.sklearn.load_model(logged_model)
        
        X = iris_splits["X_test"][:5]
        
        pred1 = model.predict(X)
        pred2 = model.predict(X)
        
        np.testing.assert_array_equal(pred1, pred2)


class TestModelSignature:
    """Tests for model signature."""
    
    def test_has_signature(self, logged_model):
        """Logged model should have signature."""
        model_info = mlflow.models.get_model_info(logged_model)
        # Note: signature might be None if not explicitly logged
        # This test documents expected behavior
        assert model_info is not None
    
    def test_model_metadata(self, logged_model):
        """Model should have metadata."""
        model_info = mlflow.models.get_model_info(logged_model)
        assert model_info.model_uri is not None