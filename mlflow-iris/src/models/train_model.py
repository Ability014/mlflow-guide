"""
Model Training
==============

Training entrypoint for Iris species classifier.

IMPORTANT: This file MUST be named train_model.py per enterprise requirements.

Usage:
    python -m src.models.train_model \
        --experiment /data-science/iris/species_classifier \
        --model-type random_forest \
        --version 1.0.0 \
        --context full
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import cross_val_score

from ..utils.logging import get_logger, setup_logging
from ..utils.config import config
from ..utils.mlflow_utils import (
    MLflowRunContext,
    generate_run_name,
    sanitize_for_mlflow,
    log_model_with_metadata,
    get_git_info,
)

logger = get_logger(__name__)


# Supported model types
MODEL_CLASSES = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "gradient_boosting": GradientBoostingClassifier,
}

# Class names for Iris
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


class ModelTrainer:
    """
    Model trainer with comprehensive MLflow integration.
    
    Features:
    - Multiple model type support
    - Training and validation metrics
    - Cross-validation
    - Artifact generation (confusion matrix, feature importance)
    - MLflow logging with signatures and examples
    
    Example:
        >>> trainer = ModelTrainer(model_type="random_forest")
        >>> trainer.train(X_train, X_test, y_train, y_test, feature_names)
        >>> trainer.log_to_mlflow(feature_names, X_test)
    """
    
    def __init__(
        self,
        model_type: str = "random_forest",
        hyperparameters: Dict[str, Any] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model_type: Type of model (random_forest, logistic_regression, gradient_boosting)
            hyperparameters: Model hyperparameters
        """
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}
        self.model = None
        self.metrics: Dict[str, float] = {}
        self.artifacts: Dict[str, Any] = {}
        
        logger.info(f"ModelTrainer initialized: {model_type}")
    
    def create_model(self) -> Any:
        """
        Create model instance with configured hyperparameters.
        
        Returns:
            Configured model instance
        """
        if self.model_type not in MODEL_CLASSES:
            raise ValueError(
                f"Unknown model type: {self.model_type}. "
                f"Supported: {list(MODEL_CLASSES.keys())}"
            )
        
        model_class = MODEL_CLASSES[self.model_type]
        
        # Filter hyperparameters to only valid ones for the model
        valid_params = {}
        for key, value in self.hyperparameters.items():
            try:
                # Test if parameter is valid
                test_model = model_class(**{key: value})
                valid_params[key] = value
            except TypeError:
                logger.warning(f"Ignoring invalid parameter for {self.model_type}: {key}")
        
        return model_class(**valid_params)
    
    def train(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str] = None,
        cv_folds: int = 5,
    ) -> Any:
        """
        Train model and compute metrics.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            feature_names: Feature names for importance logging
            cv_folds: Number of CV folds (0 to disable)
            
        Returns:
            Trained model
        """
        logger.info("Starting model training...")
        
        # Create and train model
        self.model = self.create_model()
        self.model.fit(X_train, y_train)
        
        # Compute predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Compute training metrics
        train_metrics = self.compute_metrics(y_train, y_train_pred, prefix="train_")
        self.metrics.update(train_metrics)
        
        # Compute validation metrics
        val_metrics = self.compute_metrics(y_test, y_test_pred, prefix="val_")
        self.metrics.update(val_metrics)
        
        # Cross-validation
        if cv_folds > 0:
            cv_scores = cross_val_score(
                self.create_model(),
                X_train,
                y_train,
                cv=cv_folds,
                scoring="accuracy",
            )
            self.metrics["cv_accuracy_mean"] = float(cv_scores.mean())
            self.metrics["cv_accuracy_std"] = float(cv_scores.std())
            logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Prepare artifacts
        self._prepare_artifacts(y_test, y_test_pred, feature_names)
        
        logger.info(f"Training complete. Validation accuracy: {self.metrics['val_accuracy']:.4f}")
        
        return self.model
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = "",
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            prefix: Prefix for metric names (e.g., "train_" or "val_")
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
            f"{prefix}macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            f"{prefix}weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            f"{prefix}macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            f"{prefix}macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        
        # Per-class F1 scores
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, class_name in enumerate(CLASS_NAMES):
            if i < len(f1_per_class):
                metrics[f"{prefix}f1_{class_name}"] = float(f1_per_class[i])
        
        return metrics
    
    def _prepare_artifacts(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: List[str] = None,
    ) -> None:
        """Prepare artifacts for logging."""
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.artifacts["confusion_matrix"] = {
            "matrix": cm.tolist(),
            "labels": CLASS_NAMES,
        }
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
        self.artifacts["classification_report"] = report
        
        # Feature importance (if available)
        if hasattr(self.model, "feature_importances_") and feature_names:
            importance = self.model.feature_importances_
            self.artifacts["feature_importance"] = {
                "features": feature_names,
                "importance": importance.tolist(),
            }
        elif hasattr(self.model, "coef_") and feature_names:
            # For logistic regression, use absolute coefficient values
            importance = np.abs(self.model.coef_).mean(axis=0)
            self.artifacts["feature_importance"] = {
                "features": feature_names,
                "importance": importance.tolist(),
            }
    
    def log_to_mlflow(
        self,
        feature_names: List[str],
        X_sample: np.ndarray,
    ) -> str:
        """
        Log model and artifacts to MLflow.
        
        Args:
            feature_names: Feature names for signature
            X_sample: Sample data for input example
            
        Returns:
            Model URI
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Log parameters
        params = {
            "model_type": self.model_type,
            **self.hyperparameters,
        }
        mlflow.log_params(sanitize_for_mlflow(params))
        
        # Log metrics
        mlflow.log_metrics(self.metrics)
        
        # Log artifacts
        for name, content in self.artifacts.items():
            mlflow.log_dict(content, f"{name}.json")
        
        # Log model with signature and example
        y_sample = self.model.predict(X_sample)
        
        model_uri = log_model_with_metadata(
            model=self.model,
            artifact_path="model",
            feature_names=feature_names,
            X_sample=X_sample,
            y_sample=y_sample,
            pip_requirements=[
                "scikit-learn>=1.3.0",
                "pandas>=2.0.0",
                "numpy>=1.24.0",
            ],
        )
        
        logger.info(f"Model logged to MLflow: {model_uri}")
        
        return model_uri


def train_model(
    experiment_name: str,
    model_version: str,
    context: str = "full",
    model_type: str = "random_forest",
    feature_group: str = "base_v1",
    hyperparameters: Dict[str, Any] = None,
    build_id: str = None,
    use_local_data: bool = True,
) -> Dict[str, Any]:
    """
    Full training workflow with MLflow tracking.
    
    Args:
        experiment_name: MLflow experiment name
        model_version: Semantic version
        context: Training context (full/incremental)
        model_type: Model type
        feature_group: Feature group to use
        hyperparameters: Model hyperparameters
        build_id: CI/CD build ID
        use_local_data: Use sklearn data vs Unity Catalog
        
    Returns:
        Dictionary with training results
    """
    from ..data.load_data import load_iris_local
    from ..features.feature_builder import FeatureBuilder
    
    logger.info(f"Starting training: {experiment_name}")
    
    # Load data
    if use_local_data:
        df = load_iris_local()
    else:
        raise NotImplementedError("Unity Catalog loading not implemented in this example")
    
    # Build features
    builder = FeatureBuilder(feature_group=feature_group)
    X_train, X_test, y_train, y_test = builder.fit_transform(df)
    
    # Generate run name
    run_name = generate_run_name(model_version, context, build_id)
    
    # Train with MLflow tracking
    with MLflowRunContext(
        experiment_name=experiment_name,
        run_name=run_name,
        model_version=model_version,
        context=context,
        feature_version=builder.feature_version,
        config={"model_type": model_type, "feature_group": feature_group},
    ) as run:
        # Log feature info
        builder.log_to_mlflow()
        
        # Train model
        trainer = ModelTrainer(
            model_type=model_type,
            hyperparameters=hyperparameters or {},
        )
        
        trainer.train(
            X_train, X_test, y_train, y_test,
            feature_names=builder.feature_names,
        )
        
        # Log to MLflow
        model_uri = trainer.log_to_mlflow(
            feature_names=builder.feature_names,
            X_sample=X_test,
        )
        
        run_id = run.info.run_id
    
    return {
        "run_id": run_id,
        "model_uri": model_uri,
        "metrics": trainer.metrics,
        "model_version": model_version,
        "context": context,
    }


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Train Iris species classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="MLflow experiment name (uses config if not provided)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=list(MODEL_CLASSES.keys()),
        help="Model type to train",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Model version (semantic versioning)",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="full",
        choices=["full", "incremental"],
        help="Training context",
    )
    parser.add_argument(
        "--feature-group",
        type=str,
        default="base_v1",
        help="Feature group to use",
    )
    parser.add_argument(
        "--build-id",
        type=str,
        default=None,
        help="CI/CD build ID",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=True,
        help="Use local sklearn data",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Get experiment name
    experiment_name = args.experiment or config.get_experiment_name()
    
    # Get hyperparameters from config
    hyperparameters = config.model.get("hyperparameters", {}).get(args.model_type, {})
    
    # Run training
    result = train_model(
        experiment_name=experiment_name,
        model_version=args.version,
        context=args.context,
        model_type=args.model_type,
        feature_group=args.feature_group,
        hyperparameters=hyperparameters,
        build_id=args.build_id or os.environ.get("BUILD_ID"),
        use_local_data=args.local,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Run ID: {result['run_id']}")
    print(f"Model URI: {result['model_uri']}")
    print(f"Model Version: {result['model_version']}")
    print(f"Context: {result['context']}")
    print(f"\nMetrics:")
    for name, value in result['metrics'].items():
        if name.startswith("val_"):
            print(f"  {name}: {value:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()