"""
Integration Tests - Training Pipeline
=====================================

End-to-end tests for the training pipeline.
"""

import os
import pytest
import mlflow

from src.data.load_data import load_iris_local
from src.features.feature_builder import FeatureBuilder
from src.models.train_model import train_model, ModelTrainer
from src.models.model_validator import ModelValidator


class TestTrainingPipeline:
    """End-to-end training pipeline tests."""
    
    @pytest.fixture(autouse=True)
    def setup_mlflow(self, tmp_path):
        """Setup temporary MLflow tracking."""
        uri = f"sqlite:///{tmp_path}/mlflow.db"
        mlflow.set_tracking_uri(uri)
        os.environ["MLFLOW_TRACKING_URI"] = uri
        yield
        mlflow.set_tracking_uri(None)
    
    def test_full_training_pipeline(self):
        """Test complete training workflow."""
        # Load data
        df = load_iris_local()
        assert len(df) == 150
        
        # Build features
        builder = FeatureBuilder(feature_group="base_v1")
        X_train, X_test, y_train, y_test = builder.fit_transform(df)
        
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        
        # Train with MLflow tracking
        experiment_name = "/test/iris/classifier"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name="test_run") as run:
            trainer = ModelTrainer(
                model_type="random_forest",
                hyperparameters={"n_estimators": 10, "max_depth": 3},
            )
            
            model = trainer.train(
                X_train, X_test, y_train, y_test,
                feature_names=builder.feature_names,
                cv_folds=3,
            )
            
            # Log model
            model_uri = trainer.log_to_mlflow(
                feature_names=builder.feature_names,
                X_sample=X_test,
            )
            
            run_id = run.info.run_id
        
        # Verify run was logged
        client = mlflow.tracking.MlflowClient()
        run_data = client.get_run(run_id)
        
        assert run_data.data.metrics.get("val_accuracy") is not None
        assert run_data.data.metrics.get("val_macro_f1") is not None
        assert run_data.data.params.get("model_type") == "random_forest"
    
    def test_model_validation(self, iris_splits):
        """Test model validation."""
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
        
        # Create validator with low thresholds for test
        validator = ModelValidator(
            min_accuracy=0.5,
            min_f1=0.5,
            max_inference_time_ms=1000,
        )
        
        # Save model temporarily to validate
        mlflow.set_experiment("/test/validation")
        
        with mlflow.start_run():
            model_info = mlflow.sklearn.log_model(
                trainer.model,
                "model",
            )
            
            result = validator.validate(
                model_uri=model_info.model_uri,
                X_test=iris_splits["X_test"],
                y_test=iris_splits["y_test"],
            )
        
        # Should pass with low thresholds
        assert result.checks["model_loadable"] is True
        assert "accuracy" in result.metrics
    
    def test_different_model_types(self, iris_splits):
        """Test training with different model types."""
        model_types = ["random_forest", "logistic_regression"]
        
        for model_type in model_types:
            trainer = ModelTrainer(model_type=model_type)
            
            model = trainer.train(
                iris_splits["X_train"],
                iris_splits["X_test"],
                iris_splits["y_train"],
                iris_splits["y_test"],
                feature_names=iris_splits["feature_names"],
                cv_folds=0,  # Skip CV for speed
            )
            
            assert model is not None
            assert trainer.metrics["val_accuracy"] > 0.5


class TestFeatureGroups:
    """Test different feature groups."""
    
    def test_base_features(self, iris_data):
        """Test base feature group."""
        builder = FeatureBuilder(feature_group="base_v1")
        X_train, X_test, y_train, y_test = builder.fit_transform(iris_data)
        
        assert X_train.shape[1] == 4  # 4 base features
    
    def test_enhanced_features(self, iris_data):
        """Test enhanced feature group (with ratios)."""
        builder = FeatureBuilder(feature_group="enhanced_v1")
        X_train, X_test, y_train, y_test = builder.fit_transform(iris_data)
        
        assert X_train.shape[1] == 6  # 4 base + 2 ratios