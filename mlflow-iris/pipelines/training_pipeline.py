"""
Training Pipeline
=================

Orchestrated training workflow for production use.

Usage:
    python -m pipelines.training_pipeline --config configs/model_config.yaml
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow

from src.data.load_data import load_iris_local, load_training_data
from src.features.feature_builder import FeatureBuilder
from src.models.train_model import ModelTrainer
from src.models.model_registry import ModelRegistry, ModelStage
from src.models.model_validator import ModelValidator
from src.utils.config import config
from src.utils.logging import setup_logging, get_logger
from src.utils.mlflow_utils import generate_run_name, MLflowRunContext

logger = get_logger(__name__)


class TrainingPipeline:
    """
    Orchestrated training pipeline.
    
    Steps:
    1. Load data
    2. Build features
    3. Train model
    4. Validate model
    5. Register model
    6. Optionally promote to staging
    
    Example:
        >>> pipeline = TrainingPipeline()
        >>> result = pipeline.run(
        ...     model_version="1.0.0",
        ...     promote_to_staging=True
        ... )
    """
    
    def __init__(
        self,
        experiment_name: str = None,
        model_type: str = "random_forest",
        feature_group: str = "base_v1",
    ):
        self.experiment_name = experiment_name or config.get_experiment_name()
        self.model_type = model_type
        self.feature_group = feature_group
        
        self.model_config = config.model
        
        logger.info(f"TrainingPipeline initialized: {self.experiment_name}")
    
    def run(
        self,
        model_version: str,
        context: str = "full",
        use_local_data: bool = True,
        validate: bool = True,
        register: bool = True,
        promote_to_staging: bool = False,
        build_id: str = None,
    ) -> dict:
        """
        Run the full training pipeline.
        
        Args:
            model_version: Semantic version (e.g., "1.0.0")
            context: Training context ("full" or "incremental")
            use_local_data: Use sklearn data (vs Unity Catalog)
            validate: Run model validation
            register: Register to model registry
            promote_to_staging: Auto-promote to staging if validation passes
            build_id: CI/CD build ID
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE STARTED")
        logger.info("=" * 60)
        
        result = {
            "model_version": model_version,
            "context": context,
            "status": "running",
            "steps": {},
        }
        
        try:
            # Step 1: Load data
            logger.info("Step 1: Loading data...")
            if use_local_data:
                df = load_iris_local()
            else:
                data_config = self.model_config["data"]
                df = load_training_data(
                    catalog=data_config["catalog"],
                    schema=data_config["schema"],
                    table=data_config["table"],
                )
            result["steps"]["load_data"] = {"status": "success", "rows": len(df)}
            logger.info(f"  Loaded {len(df)} samples")
            
            # Step 2: Build features
            logger.info("Step 2: Building features...")
            builder = FeatureBuilder(feature_group=self.feature_group)
            X_train, X_test, y_train, y_test = builder.fit_transform(df)
            result["steps"]["build_features"] = {
                "status": "success",
                "feature_count": len(builder.feature_names),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }
            logger.info(f"  Features: {len(builder.feature_names)}")
            
            # Step 3: Train model
            logger.info("Step 3: Training model...")
            run_name = generate_run_name(model_version, context, build_id)
            
            hyperparameters = self.model_config.get("hyperparameters", {}).get(
                self.model_type, {}
            )
            
            with MLflowRunContext(
                experiment_name=self.experiment_name,
                run_name=run_name,
                model_version=model_version,
                context=context,
                feature_version=builder.feature_version,
                config={"model_type": self.model_type},
            ) as run:
                run_id = run.info.run_id
                
                builder.log_to_mlflow()
                
                trainer = ModelTrainer(
                    model_type=self.model_type,
                    hyperparameters=hyperparameters,
                )
                
                trainer.train(
                    X_train, X_test, y_train, y_test,
                    feature_names=builder.feature_names,
                )
                
                model_uri = trainer.log_to_mlflow(
                    feature_names=builder.feature_names,
                    X_sample=X_test,
                )
            
            result["steps"]["train_model"] = {
                "status": "success",
                "run_id": run_id,
                "model_uri": model_uri,
                "metrics": trainer.metrics,
            }
            result["model_uri"] = model_uri
            result["run_id"] = run_id
            logger.info(f"  Accuracy: {trainer.metrics['val_accuracy']:.4f}")
            
            # Step 4: Validate model
            if validate:
                logger.info("Step 4: Validating model...")
                validator = ModelValidator()
                validation_result = validator.validate(model_uri, X_test, y_test)
                
                result["steps"]["validate"] = {
                    "status": "success" if validation_result.passed else "failed",
                    "passed": validation_result.passed,
                    "checks": validation_result.checks,
                    "errors": validation_result.errors,
                }
                
                if not validation_result.passed:
                    logger.warning(f"  Validation FAILED: {validation_result.errors}")
                    result["status"] = "validation_failed"
                    return result
                
                logger.info("  Validation PASSED")
            
            # Step 5: Register model
            if register:
                logger.info("Step 5: Registering model...")
                registry = ModelRegistry()
                
                registered_version = registry.register_model(
                    model_uri=model_uri,
                    description=f"Iris classifier v{model_version}",
                    tags={
                        "model_version": model_version,
                        "context": context,
                        "build_id": build_id or "local",
                    },
                )
                
                result["steps"]["register"] = {
                    "status": "success",
                    "version": registered_version,
                }
                result["registered_version"] = registered_version
                logger.info(f"  Registered version: {registered_version}")
                
                # Step 6: Promote to staging
                if promote_to_staging:
                    logger.info("Step 6: Promoting to staging...")
                    registry.promote_to_staging(
                        version=registered_version,
                        validation_results=validation_result.to_dict() if validate else None,
                    )
                    result["steps"]["promote"] = {
                        "status": "success",
                        "stage": "staging",
                    }
                    logger.info("  Promoted to staging")
            
            result["status"] = "success"
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            raise
        
        finally:
            logger.info("=" * 60)
            logger.info(f"TRAINING PIPELINE {result['status'].upper()}")
            logger.info("=" * 60)
        
        return result


def main():
    parser = argparse.ArgumentParser(description="Run training pipeline")
    parser.add_argument("--version", type=str, default="1.0.0", help="Model version")
    parser.add_argument("--context", type=str, default="full", choices=["full", "incremental"])
    parser.add_argument("--model-type", type=str, default="random_forest")
    parser.add_argument("--feature-group", type=str, default="base_v1")
    parser.add_argument("--local", action="store_true", default=True)
    parser.add_argument("--validate", action="store_true", default=True)
    parser.add_argument("--register", action="store_true", default=False)
    parser.add_argument("--promote", action="store_true", default=False)
    parser.add_argument("--build-id", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    
    args = parser.parse_args()
    
    setup_logging(level=args.log_level)
    
    pipeline = TrainingPipeline(
        model_type=args.model_type,
        feature_group=args.feature_group,
    )
    
    result = pipeline.run(
        model_version=args.version,
        context=args.context,
        use_local_data=args.local,
        validate=args.validate,
        register=args.register,
        promote_to_staging=args.promote,
        build_id=args.build_id,
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE RESULT")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Model Version: {result['model_version']}")
    if "model_uri" in result:
        print(f"Model URI: {result['model_uri']}")
    if "registered_version" in result:
        print(f"Registry Version: {result['registered_version']}")
    print("=" * 60)


if __name__ == "__main__":
    main()