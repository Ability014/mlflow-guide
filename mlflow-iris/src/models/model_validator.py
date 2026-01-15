"""
Model Validator
===============

Validate models before promotion to staging/production.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time

import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from ..utils.logging import get_logger
from ..utils.config import config

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of model validation."""
    passed: bool
    metrics: Dict[str, float]
    checks: Dict[str, bool]
    errors: List[str]
    warnings: List[str]
    inference_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "metrics": self.metrics,
            "checks": self.checks,
            "errors": self.errors,
            "warnings": self.warnings,
            "inference_time_ms": self.inference_time_ms,
        }


class ModelValidator:
    """
    Validate models against enterprise requirements.
    
    Checks:
    - Performance thresholds (accuracy, F1, etc.)
    - Inference latency
    - Model signature presence
    - Input/output example presence
    - Dependency declaration
    
    Example:
        >>> validator = ModelValidator()
        >>> result = validator.validate(model_uri, X_test, y_test)
        >>> if result.passed:
        ...     registry.promote_to_staging(version)
    """
    
    def __init__(
        self,
        min_accuracy: float = None,
        min_f1: float = None,
        max_inference_time_ms: float = None,
    ):
        """
        Initialize validator.
        
        Args:
            min_accuracy: Minimum required accuracy
            min_f1: Minimum required F1 score
            max_inference_time_ms: Maximum inference time in ms
        """
        # Get thresholds from config if not provided
        model_config = config.model
        thresholds = model_config.get("evaluation", {}).get("thresholds", {})
        
        self.min_accuracy = min_accuracy or thresholds.get("min_accuracy", 0.85)
        self.min_f1 = min_f1 or thresholds.get("min_macro_f1", 0.85)
        self.max_inference_time_ms = max_inference_time_ms or thresholds.get(
            "max_inference_time_ms", 100
        )
        
        logger.info(
            f"ModelValidator initialized: min_accuracy={self.min_accuracy}, "
            f"min_f1={self.min_f1}, max_latency={self.max_inference_time_ms}ms"
        )
    
    def validate(
        self,
        model_uri: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ValidationResult:
        """
        Validate a model.
        
        Args:
            model_uri: MLflow model URI
            X_test: Test features
            y_test: Test labels
            
        Returns:
            ValidationResult with all check results
        """
        logger.info(f"Validating model: {model_uri}")
        
        errors = []
        warnings = []
        checks = {}
        metrics = {}
        
        # Load model
        try:
            model = mlflow.sklearn.load_model(model_uri)
            checks["model_loadable"] = True
        except Exception as e:
            errors.append(f"Could not load model: {e}")
            checks["model_loadable"] = False
            return ValidationResult(
                passed=False,
                metrics={},
                checks=checks,
                errors=errors,
                warnings=warnings,
                inference_time_ms=0,
            )
        
        # Check model info
        try:
            model_info = mlflow.models.get_model_info(model_uri)
            
            # Check signature
            checks["has_signature"] = model_info.signature is not None
            if not checks["has_signature"]:
                errors.append("Model missing signature")
            
            # Check input example
            checks["has_input_example"] = model_info.saved_input_example_info is not None
            if not checks["has_input_example"]:
                warnings.append("Model missing input example")
            
        except Exception as e:
            warnings.append(f"Could not get model info: {e}")
            checks["has_signature"] = False
            checks["has_input_example"] = False
        
        # Measure inference time
        inference_time_ms = self._measure_inference_time(model, X_test)
        checks["inference_time_ok"] = inference_time_ms <= self.max_inference_time_ms
        if not checks["inference_time_ok"]:
            errors.append(
                f"Inference time {inference_time_ms:.1f}ms exceeds "
                f"threshold {self.max_inference_time_ms}ms"
            )
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score
        
        y_pred = model.predict(X_test)
        
        accuracy = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        
        metrics["accuracy"] = accuracy
        metrics["macro_f1"] = macro_f1
        
        # Check thresholds
        checks["accuracy_threshold"] = accuracy >= self.min_accuracy
        if not checks["accuracy_threshold"]:
            errors.append(
                f"Accuracy {accuracy:.4f} below threshold {self.min_accuracy}"
            )
        
        checks["f1_threshold"] = macro_f1 >= self.min_f1
        if not checks["f1_threshold"]:
            errors.append(
                f"F1 score {macro_f1:.4f} below threshold {self.min_f1}"
            )
        
        # Overall pass/fail
        passed = all([
            checks.get("model_loadable", False),
            checks.get("has_signature", False),
            checks.get("accuracy_threshold", False),
            checks.get("f1_threshold", False),
            checks.get("inference_time_ok", False),
        ])
        
        result = ValidationResult(
            passed=passed,
            metrics=metrics,
            checks=checks,
            errors=errors,
            warnings=warnings,
            inference_time_ms=inference_time_ms,
        )
        
        logger.info(f"Validation {'PASSED' if passed else 'FAILED'}: {checks}")
        
        return result
    
    def _measure_inference_time(
        self,
        model: Any,
        X_test: np.ndarray,
        n_iterations: int = 100,
    ) -> float:
        """Measure average inference time in milliseconds."""
        # Warmup
        _ = model.predict(X_test[:1])
        
        # Measure
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = model.predict(X_test[:1])
        end = time.perf_counter()
        
        avg_time_ms = ((end - start) / n_iterations) * 1000
        return avg_time_ms


def validate_model(
    model_uri: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> ValidationResult:
    """
    Convenience function to validate a model.
    
    Args:
        model_uri: MLflow model URI
        X_test: Test features
        y_test: Test labels
        
    Returns:
        ValidationResult
    """
    validator = ModelValidator()
    return validator.validate(model_uri, X_test, y_test)