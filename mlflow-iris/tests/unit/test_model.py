"""
Unit Tests - Model Training
===========================

Tests for model trainer and related functionality.
"""

import pytest
import numpy as np

from src.models.train_model import ModelTrainer, MODEL_CLASSES


class TestModelTrainer:
    """Tests for model trainer."""
    
    def test_initialization(self):
        """Should initialize with model type."""
        trainer = ModelTrainer(model_type="random_forest")
        assert trainer.model_type == "random_forest"
    
    def test_create_model(self):
        """Should create model instance."""
        trainer = ModelTrainer(model_type="random_forest")
        model = trainer.create_model()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
    
    def test_all_model_types(self):
        """All model types should be creatable."""
        for model_type in MODEL_CLASSES.keys():
            trainer = ModelTrainer(model_type=model_type)
            model = trainer.create_model()
            assert model is not None
    
    def test_unknown_model_raises(self):
        """Unknown model type should raise."""
        trainer = ModelTrainer(model_type="unknown_model")
        with pytest.raises(ValueError):
            trainer.create_model()
    
    def test_train(self, iris_splits):
        """Should train and return model."""
        trainer = ModelTrainer(
            model_type="random_forest",
            hyperparameters={"n_estimators": 10, "max_depth": 3},
        )
        
        model = trainer.train(
            iris_splits["X_train"],
            iris_splits["X_test"],
            iris_splits["y_train"],
            iris_splits["y_test"],
            feature_names=iris_splits["feature_names"],
        )
        
        assert model is not None
        assert trainer.model is not None
    
    def test_metrics_computed(self, iris_splits):
        """Should compute metrics after training."""
        trainer = ModelTrainer(
            model_type="random_forest",
            hyperparameters={"n_estimators": 10, "max_depth": 3},
        )
        
        trainer.train(
            iris_splits["X_train"],
            iris_splits["X_test"],
            iris_splits["y_train"],
            iris_splits["y_test"],
            feature_names=iris_splits["feature_names"],
        )
        
        # Should have train metrics
        assert "train_accuracy" in trainer.metrics
        assert "train_macro_f1" in trainer.metrics
        
        # Should have validation metrics
        assert "val_accuracy" in trainer.metrics
        assert "val_macro_f1" in trainer.metrics
        
        # Metrics should be valid
        assert 0 <= trainer.metrics["val_accuracy"] <= 1
        assert 0 <= trainer.metrics["val_macro_f1"] <= 1
    
    def test_artifacts_prepared(self, iris_splits):
        """Should prepare artifacts after training."""
        trainer = ModelTrainer(model_type="random_forest")
        
        trainer.train(
            iris_splits["X_train"],
            iris_splits["X_test"],
            iris_splits["y_train"],
            iris_splits["y_test"],
            feature_names=iris_splits["feature_names"],
        )
        
        assert "confusion_matrix" in trainer.artifacts
        assert "classification_report" in trainer.artifacts
        assert "feature_importance" in trainer.artifacts
    
    def test_cross_validation(self, iris_splits):
        """Should run cross-validation if enabled."""
        trainer = ModelTrainer(
            model_type="random_forest",
            hyperparameters={"n_estimators": 10, "max_depth": 3},
        )
        
        trainer.train(
            iris_splits["X_train"],
            iris_splits["X_test"],
            iris_splits["y_train"],
            iris_splits["y_test"],
            feature_names=iris_splits["feature_names"],
            cv_folds=3,
        )
        
        assert "cv_accuracy_mean" in trainer.metrics
        assert "cv_accuracy_std" in trainer.metrics


class TestComputeMetrics:
    """Tests for metric computation."""
    
    def test_metrics_range(self, iris_splits, trained_model):
        """All metrics should be in [0, 1]."""
        trainer = ModelTrainer(model_type="random_forest")
        
        y_pred = trained_model.predict(iris_splits["X_test"])
        metrics = trainer.compute_metrics(
            iris_splits["y_test"],
            y_pred,
            prefix="test_",
        )
        
        for name, value in metrics.items():
            assert 0 <= value <= 1, f"{name} out of range: {value}"
    
    def test_per_class_metrics(self, iris_splits, trained_model):
        """Should include per-class metrics."""
        trainer = ModelTrainer(model_type="random_forest")
        
        y_pred = trained_model.predict(iris_splits["X_test"])
        metrics = trainer.compute_metrics(
            iris_splits["y_test"],
            y_pred,
        )
        
        # Per-class F1
        assert "f1_setosa" in metrics
        assert "f1_versicolor" in metrics
        assert "f1_virginica" in metrics