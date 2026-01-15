"""
Scoring Component
=================

Stateless, reusable inference component for model serving.
Designed for extraction to Component Library.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import mlflow

from ..utils.logging import get_logger

logger = get_logger(__name__)


CLASS_NAMES = ["setosa", "versicolor", "virginica"]


class ScoringComponent:
    """
    Stateless scoring component for Iris classification.
    
    Features:
    - Load from registry by name/version/alias
    - Load from specific run
    - Structured output with predictions and probabilities
    - Ready for Component Library extraction
    
    Example:
        >>> scorer = ScoringComponent.from_registry(
        ...     model_name="catalog.schema.iris_classifier",
        ...     model_alias="production"
        ... )
        >>> result = scorer.score(df)
    """
    
    def __init__(
        self,
        model: Any,
        preprocessor: Any = None,
        model_name: str = None,
        model_version: str = None,
        model_alias: str = None,
    ):
        """
        Initialize scorer.
        
        Args:
            model: Loaded model instance
            preprocessor: Optional fitted preprocessor
            model_name: Model name for tracking
            model_version: Model version for tracking
            model_alias: Model alias (dev/staging/production)
        """
        self.model = model
        self.preprocessor = preprocessor
        self.model_name = model_name
        self.model_version = model_version
        self.model_alias = model_alias
        
        logger.info(
            f"ScoringComponent initialized: {model_name} "
            f"(version={model_version}, alias={model_alias})"
        )
    
    @classmethod
    def from_registry(
        cls,
        model_name: str,
        model_version: str = None,
        model_alias: str = None,
        preprocessor_path: str = None,
    ) -> "ScoringComponent":
        """
        Create scorer from MLflow Model Registry.
        
        Args:
            model_name: Registered model name
            model_version: Specific version (optional)
            model_alias: Alias like "production" (optional)
            preprocessor_path: Path to saved preprocessor (optional)
            
        Returns:
            Configured ScoringComponent
        """
        if model_version:
            model_uri = f"models:/{model_name}/{model_version}"
        elif model_alias:
            model_uri = f"models:/{model_name}@{model_alias}"
        else:
            model_uri = f"models:/{model_name}/latest"
        
        logger.info(f"Loading model from: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        
        preprocessor = None
        if preprocessor_path:
            from ..features.feature_builder import FeatureBuilder
            preprocessor = FeatureBuilder.load(preprocessor_path)
        
        return cls(
            model=model,
            preprocessor=preprocessor,
            model_name=model_name,
            model_version=model_version,
            model_alias=model_alias,
        )
    
    @classmethod
    def from_run(
        cls,
        run_id: str,
        artifact_path: str = "model",
    ) -> "ScoringComponent":
        """
        Create scorer from a specific MLflow run.
        
        Args:
            run_id: MLflow run ID
            artifact_path: Path to model artifact
            
        Returns:
            Configured ScoringComponent
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        logger.info(f"Loading model from run: {model_uri}")
        
        model = mlflow.sklearn.load_model(model_uri)
        
        return cls(
            model=model,
            model_name=f"run_{run_id[:8]}",
        )
    
    def preprocess(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess input data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Preprocessed numpy array
        """
        if self.preprocessor is not None:
            return self.preprocessor.transform(data)
        return data.values
    
    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Dict]],
    ) -> np.ndarray:
        """
        Generate class predictions.
        
        Args:
            data: Input data (DataFrame, array, dict, or list of dicts)
            
        Returns:
            Array of predicted class indices (0, 1, 2)
        """
        X = self._prepare_input(data)
        return self.model.predict(X)
    
    def predict_proba(
        self,
        data: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Dict]],
    ) -> np.ndarray:
        """
        Generate class probabilities.
        
        Args:
            data: Input data
            
        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        X = self._prepare_input(data)
        return self.model.predict_proba(X)
    
    def predict_species(
        self,
        data: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Dict]],
    ) -> List[str]:
        """
        Generate species name predictions.
        
        Args:
            data: Input data
            
        Returns:
            List of species names
        """
        predictions = self.predict(data)
        return [CLASS_NAMES[p] for p in predictions]
    
    def _prepare_input(
        self,
        data: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Dict]],
    ) -> np.ndarray:
        """Convert various input formats to numpy array."""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            return data
        
        if self.preprocessor is not None:
            return self.preprocessor.transform(data)
        
        return data.values
    
    def score(
        self,
        data: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Dict]],
        return_proba: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate predictions with full metadata.
        
        Args:
            data: Input data
            return_proba: Include class probabilities
            
        Returns:
            Dictionary with predictions, species, probabilities, and metadata
        """
        predictions = self.predict(data)
        species = [CLASS_NAMES[p] for p in predictions]
        
        result = {
            "predictions": predictions.tolist(),
            "species": species,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_alias": self.model_alias,
            "class_names": CLASS_NAMES,
        }
        
        if return_proba:
            probabilities = self.predict_proba(data)
            result["probabilities"] = probabilities.tolist()
            
            # Add confidence (max probability)
            result["confidence"] = [float(max(p)) for p in probabilities]
        
        return result
    
    def score_batch(
        self,
        df: pd.DataFrame,
        add_columns: bool = True,
    ) -> pd.DataFrame:
        """
        Score a batch and optionally add prediction columns.
        
        Args:
            df: Input DataFrame
            add_columns: Add prediction columns to DataFrame
            
        Returns:
            DataFrame with predictions (if add_columns=True)
        """
        result = df.copy()
        
        predictions = self.predict(df)
        probabilities = self.predict_proba(df)
        
        if add_columns:
            result["prediction"] = predictions
            result["species_predicted"] = [CLASS_NAMES[p] for p in predictions]
            result["confidence"] = [float(max(p)) for p in probabilities]
            
            for i, class_name in enumerate(CLASS_NAMES):
                result[f"prob_{class_name}"] = probabilities[:, i]
        
        return result


# Convenience functions

def score_batch(
    model_name: str,
    data: pd.DataFrame,
    model_version: str = None,
    model_alias: str = None,
) -> pd.DataFrame:
    """
    Score a batch using a registered model.
    
    Args:
        model_name: Registered model name
        data: Input DataFrame
        model_version: Optional model version
        model_alias: Optional model alias
        
    Returns:
        DataFrame with prediction columns added
    """
    scorer = ScoringComponent.from_registry(
        model_name=model_name,
        model_version=model_version,
        model_alias=model_alias,
    )
    return scorer.score_batch(data)


def score_single(
    model_name: str,
    record: Dict[str, Any],
    model_version: str = None,
    model_alias: str = None,
) -> Dict[str, Any]:
    """
    Score a single record using a registered model.
    
    Args:
        model_name: Registered model name
        record: Input record as dictionary
        model_version: Optional model version
        model_alias: Optional model alias
        
    Returns:
        Prediction result dictionary
    """
    scorer = ScoringComponent.from_registry(
        model_name=model_name,
        model_version=model_version,
        model_alias=model_alias,
    )
    
    result = scorer.score(record, return_proba=True)
    
    # Flatten for single record
    return {
        "prediction": result["predictions"][0],
        "species": result["species"][0],
        "confidence": result["confidence"][0],
        "probabilities": {
            CLASS_NAMES[i]: result["probabilities"][0][i]
            for i in range(len(CLASS_NAMES))
        },
        "model_name": result["model_name"],
        "model_version": result["model_version"],
        "model_alias": result["model_alias"],
    }