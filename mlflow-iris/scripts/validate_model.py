#!/usr/bin/env python
"""
Model Validation Script
=======================

Validate a model before promotion.

Usage:
    python scripts/validate_model.py --local
    python scripts/validate_model.py --stage staging --version 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow

from src.data.load_data import load_iris_local
from src.features.feature_builder import FeatureBuilder
from src.models.train_model import train_model, ModelTrainer
from src.models.model_validator import ModelValidator


def main():
    parser = argparse.ArgumentParser(description="Validate model")
    parser.add_argument("--local", action="store_true", help="Train locally and validate")
    parser.add_argument("--stage", type=str, help="Stage to validate (staging/production)")
    parser.add_argument("--version", type=str, help="Model version to validate")
    parser.add_argument("--model-name", type=str, help="Model name in registry")
    
    args = parser.parse_args()
    
    # Load test data
    df = load_iris_local()
    builder = FeatureBuilder(feature_group="base_v1")
    X_train, X_test, y_train, y_test = builder.fit_transform(df)
    
    if args.local:
        # Train a model locally and validate
        print("Training model locally...")
        
        mlflow.set_experiment("/validation/test")
        
        with mlflow.start_run():
            trainer = ModelTrainer(
                model_type="random_forest",
                hyperparameters={"n_estimators": 50, "max_depth": 5},
            )
            
            trainer.train(
                X_train, X_test, y_train, y_test,
                feature_names=builder.feature_names,
            )
            
            model_info = mlflow.sklearn.log_model(
                trainer.model,
                "model",
            )
            
            model_uri = model_info.model_uri
    else:
        # Validate from registry
        if not args.model_name:
            print("Error: --model-name required when not using --local")
            sys.exit(1)
        
        if args.version:
            model_uri = f"models:/{args.model_name}/{args.version}"
        elif args.stage:
            model_uri = f"models:/{args.model_name}@{args.stage}"
        else:
            model_uri = f"models:/{args.model_name}/latest"
    
    # Validate
    print(f"\nValidating model: {model_uri}")
    
    validator = ModelValidator()
    result = validator.validate(model_uri, X_test, y_test)
    
    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Passed: {result.passed}")
    print(f"\nMetrics:")
    for name, value in result.metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"\nChecks:")
    for name, passed in result.checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    print(f"\nInference time: {result.inference_time_ms:.2f}ms")
    
    if result.errors:
        print(f"\nErrors:")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.warnings:
        print(f"\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    print("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()