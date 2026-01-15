"""
MLflow Utilities
================

Helper functions for MLflow operations following enterprise best practices.
"""

import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import ColSpec, Schema
import pandas as pd
import numpy as np


def get_git_info() -> Dict[str, str]:
    """
    Get git repository information for tracking.
    
    Returns:
        Dictionary with git_sha, git_branch, git_dirty
    """
    info = {
        "git_sha": "unknown",
        "git_branch": "unknown",
        "git_dirty": "unknown",
    }
    
    try:
        # Get SHA
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["git_sha"] = result.stdout.strip()[:8]
        
        # Get branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["git_branch"] = result.stdout.strip()
        
        # Check if dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["git_dirty"] = "true" if result.stdout.strip() else "false"
            
    except Exception:
        pass
    
    return info


def generate_run_name(
    model_version: str,
    context: str = "full",
    build_id: Optional[str] = None,
) -> str:
    """
    Generate run name following naming convention.
    
    Pattern: {model_version}_{context}_{timestamp_or_build_id}
    
    Args:
        model_version: Semantic version (e.g., "1.0.0")
        context: Training context ("full" or "incremental")
        build_id: Optional CI/CD build ID
        
    Returns:
        Formatted run name
    """
    if build_id:
        identifier = f"build_{build_id}"
    else:
        identifier = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"v{model_version}_{context}_{identifier}"


def sanitize_for_mlflow(params: Dict[str, Any]) -> Dict[str, str]:
    """
    Sanitize parameters for MLflow logging.
    
    - Removes None values
    - Converts all values to strings
    - Truncates long strings
    """
    sanitized = {}
    for key, value in params.items():
        if value is None:
            continue
        str_value = str(value)
        # MLflow has a 500 char limit for param values
        if len(str_value) > 500:
            str_value = str_value[:497] + "..."
        sanitized[key] = str_value
    return sanitized


def log_training_context(
    model_version: str,
    context: str,
    feature_version: str,
    config: Dict[str, Any],
) -> None:
    """
    Log comprehensive training context to MLflow.
    
    Args:
        model_version: Model version being trained
        context: Training context (full/incremental)
        feature_version: Version of feature definitions
        config: Training configuration
    """
    # Get git info
    git_info = get_git_info()
    
    # Get build info from environment
    build_id = os.environ.get("BUILD_ID", "local")
    pipeline_url = os.environ.get("PIPELINE_URL", "")
    
    # Log tags
    tags = {
        "model_version": model_version,
        "training_context": context,
        "feature_version": feature_version,
        "build_id": build_id,
        **git_info,
    }
    
    if pipeline_url:
        tags["pipeline_url"] = pipeline_url
    
    mlflow.set_tags(tags)
    
    # Log config as params
    flat_config = _flatten_dict(config, sep=".")
    mlflow.log_params(sanitize_for_mlflow(flat_config))


def _flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_metrics_with_prefix(
    metrics: Dict[str, float],
    prefix: str,
) -> None:
    """
    Log metrics with a prefix (e.g., 'train_' or 'val_').
    
    Args:
        metrics: Dictionary of metric names to values
        prefix: Prefix to add to each metric name
    """
    prefixed = {f"{prefix}{k}": float(v) for k, v in metrics.items()}
    mlflow.log_metrics(prefixed)


def create_model_signature(
    feature_names: List[str],
    class_names: List[str],
) -> ModelSignature:
    """
    Create explicit model signature.
    
    Args:
        feature_names: List of input feature names
        class_names: List of output class names
        
    Returns:
        MLflow ModelSignature
    """
    input_schema = Schema([
        ColSpec("double", name) for name in feature_names
    ])
    
    output_schema = Schema([
        ColSpec("long", "prediction")
    ])
    
    return ModelSignature(inputs=input_schema, outputs=output_schema)


def log_model_with_metadata(
    model: Any,
    artifact_path: str,
    feature_names: List[str],
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    pip_requirements: List[str] = None,
    conda_env: Dict = None,
) -> str:
    """
    Log model with full metadata following best practices.
    
    Args:
        model: Trained model
        artifact_path: Path in MLflow artifacts
        feature_names: Input feature names
        X_sample: Sample input data
        y_sample: Sample predictions
        pip_requirements: Explicit pip dependencies
        conda_env: Conda environment specification
        
    Returns:
        Model URI
    """
    # Create input example
    input_example = pd.DataFrame(X_sample[:5], columns=feature_names)
    
    # Infer signature
    signature = infer_signature(X_sample, y_sample)
    
    # Default pip requirements
    if pip_requirements is None:
        pip_requirements = [
            "scikit-learn>=1.3.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
        ]
    
    # Log model
    model_info = mlflow.sklearn.log_model(
        model,
        artifact_path=artifact_path,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        conda_env=conda_env,
    )
    
    return model_info.model_uri


def log_artifacts_versioned(
    artifacts: Dict[str, Any],
    run_id: str = None,
) -> None:
    """
    Log artifacts with proper versioning.
    
    Each artifact is logged as a separate file, making them
    immutable per run.
    
    Args:
        artifacts: Dictionary of artifact_name -> content
        run_id: Optional specific run ID (uses active run if None)
    """
    for name, content in artifacts.items():
        if isinstance(content, pd.DataFrame):
            mlflow.log_dict(content.to_dict(), f"{name}.json")
        elif isinstance(content, dict):
            mlflow.log_dict(content, f"{name}.json")
        elif isinstance(content, (np.ndarray, list)):
            mlflow.log_dict({"data": list(content)}, f"{name}.json")
        else:
            mlflow.log_text(str(content), f"{name}.txt")


class MLflowRunContext:
    """
    Context manager for MLflow runs with automatic logging.
    
    Example:
        >>> with MLflowRunContext(
        ...     experiment_name="/team/project/model",
        ...     run_name="v1.0.0_full_20240115",
        ...     model_version="1.0.0",
        ...     context="full"
        ... ) as run:
        ...     # Training code here
        ...     mlflow.log_metrics({"accuracy": 0.95})
    """
    
    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        model_version: str,
        context: str,
        feature_version: str,
        config: Dict[str, Any] = None,
        nested: bool = False,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.model_version = model_version
        self.context = context
        self.feature_version = feature_version
        self.config = config or {}
        self.nested = nested
        self.run = None
    
    def __enter__(self):
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=self.run_name, nested=self.nested)
        
        # Log training context
        log_training_context(
            model_version=self.model_version,
            context=self.context,
            feature_version=self.feature_version,
            config=self.config,
        )
        
        return self.run
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Log exception info if training failed
        if exc_type is not None:
            mlflow.set_tag("training_status", "failed")
            mlflow.set_tag("error_type", exc_type.__name__)
        else:
            mlflow.set_tag("training_status", "success")
        
        mlflow.end_run()
        return False  # Don't suppress exceptions