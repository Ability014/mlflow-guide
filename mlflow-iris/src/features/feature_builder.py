"""
Feature Builder
===============

Build features from raw data using versioned feature definitions.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from .feature_definitions import (
    FeatureRegistry,
    FeatureGroup,
    FeatureDefinition,
    get_feature_registry,
    compute_derived_feature,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


# Scaler classes
SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "none": None,
}


class FeatureBuilder:
    """
    Build features from raw data using versioned definitions.
    
    Features:
    - Versioned feature definitions from config
    - Feature groups for different model variants
    - Configurable scaling
    - Derived feature computation
    - MLflow artifact logging
    - Save/load for inference consistency
    
    Example:
        >>> builder = FeatureBuilder(feature_group="base_v1")
        >>> X_train, X_test, y_train, y_test = builder.fit_transform(df)
        >>> builder.log_to_mlflow()
    """
    
    def __init__(
        self,
        feature_group: str = "base_v1",
        scaling_method: str = "standard",
        registry: FeatureRegistry = None,
    ):
        """
        Initialize feature builder.
        
        Args:
            feature_group: Name of feature group to use
            scaling_method: Scaling method (standard, minmax, robust, none)
            registry: Optional feature registry (uses singleton if None)
        """
        self.feature_group_name = feature_group
        self.scaling_method = scaling_method
        
        # Get registry
        self._registry = registry or get_feature_registry()
        
        # Get feature group
        self._feature_group = self._registry.get_feature_group(feature_group)
        
        # Initialize scaler
        self._scaler = None
        if scaling_method != "none" and scaling_method in SCALERS:
            scaler_class = SCALERS[scaling_method]
            if scaler_class:
                self._scaler = scaler_class()
        
        # State
        self._is_fitted = False
        self._feature_names: List[str] = []
        self._source_col_mapping: Dict[str, str] = {}
        
        logger.info(
            f"FeatureBuilder initialized: group={feature_group}, "
            f"scaling={scaling_method}, features={len(self._feature_group.features)}"
        )
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names (only available after fit)."""
        if not self._is_fitted:
            raise RuntimeError("FeatureBuilder not fitted. Call fit_transform() first.")
        return self._feature_names.copy()
    
    @property
    def feature_version(self) -> str:
        """Get feature definition version."""
        return self._registry.version
    
    @property
    def feature_group(self) -> FeatureGroup:
        """Get feature group."""
        return self._feature_group
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: str = "species_encoded",
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the builder and transform data.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Fraction for test set
            random_state: Random seed
            stratify: Whether to stratify split
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Building features from {len(df)} samples")
        
        # Build source column mapping
        self._build_source_mapping()
        
        # Extract features
        X = self._extract_features(df)
        y = df[target_column].values
        
        # Split data
        stratify_col = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col,
        )
        
        # Fit and apply scaler
        if self._scaler is not None:
            X_train = self._scaler.fit_transform(X_train)
            X_test = self._scaler.transform(X_test)
        
        self._is_fitted = True
        
        logger.info(
            f"Features built: {X_train.shape[1]} features, "
            f"{X_train.shape[0]} train, {X_test.shape[0]} test"
        )
        
        return X_train, X_test, y_train, y_test
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted builder.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed feature array
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureBuilder not fitted. Call fit_transform() first.")
        
        X = self._extract_features(df)
        
        if self._scaler is not None:
            X = self._scaler.transform(X)
        
        return X
    
    def _build_source_mapping(self) -> None:
        """Build mapping from feature names to source columns."""
        self._source_col_mapping = {}
        
        for feat_name in self._feature_group.features:
            feat_def = self._registry.get_feature(feat_name)
            if feat_def.source_column:
                self._source_col_mapping[feat_name] = feat_def.source_column
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features from DataFrame."""
        feature_arrays = []
        self._feature_names = []
        
        for feat_name in self._feature_group.features:
            feat_def = self._registry.get_feature(feat_name)
            
            if feat_def.is_derived:
                # Compute derived feature
                values = compute_derived_feature(
                    df, feat_def, self._source_col_mapping
                )
            else:
                # Get raw feature
                source_col = feat_def.source_column
                if source_col not in df.columns:
                    raise KeyError(f"Source column not found: {source_col}")
                values = df[source_col].values
            
            feature_arrays.append(values.reshape(-1, 1))
            self._feature_names.append(feat_name)
        
        return np.hstack(feature_arrays)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get builder metadata for logging."""
        return {
            "feature_group": self.feature_group_name,
            "feature_version": self.feature_version,
            "feature_count": len(self._feature_group.features),
            "feature_names": self._feature_names,
            "scaling_method": self.scaling_method,
            "is_fitted": self._is_fitted,
        }
    
    def log_to_mlflow(self) -> None:
        """Log feature metadata to active MLflow run."""
        if mlflow.active_run() is None:
            logger.warning("No active MLflow run")
            return
        
        metadata = self.get_metadata()
        
        # Log as params
        mlflow.log_params({
            "feature_group": metadata["feature_group"],
            "feature_version": metadata["feature_version"],
            "feature_count": metadata["feature_count"],
            "scaling_method": metadata["scaling_method"],
        })
        
        # Log feature names as artifact
        mlflow.log_dict(
            {"feature_names": metadata["feature_names"]},
            "feature_names.json"
        )
        
        logger.info("Logged feature metadata to MLflow")
    
    def save(self, path: str) -> None:
        """
        Save fitted builder for inference.
        
        Args:
            path: Path to save builder
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted builder")
        
        state = {
            "feature_group_name": self.feature_group_name,
            "scaling_method": self.scaling_method,
            "feature_names": self._feature_names,
            "source_col_mapping": self._source_col_mapping,
            "scaler": self._scaler,
            "is_fitted": self._is_fitted,
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"FeatureBuilder saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "FeatureBuilder":
        """
        Load a saved builder.
        
        Args:
            path: Path to saved builder
            
        Returns:
            Loaded FeatureBuilder
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        builder = cls(
            feature_group=state["feature_group_name"],
            scaling_method=state["scaling_method"],
        )
        
        builder._feature_names = state["feature_names"]
        builder._source_col_mapping = state["source_col_mapping"]
        builder._scaler = state["scaler"]
        builder._is_fitted = state["is_fitted"]
        
        logger.info(f"FeatureBuilder loaded from {path}")
        
        return builder


def build_features(
    df: pd.DataFrame,
    feature_group: str = "base_v1",
    scaling_method: str = "standard",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, FeatureBuilder]:
    """
    Convenience function to build features.
    
    Args:
        df: Input DataFrame
        feature_group: Feature group name
        scaling_method: Scaling method
        test_size: Test set fraction
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, builder)
    """
    builder = FeatureBuilder(
        feature_group=feature_group,
        scaling_method=scaling_method,
    )
    
    X_train, X_test, y_train, y_test = builder.fit_transform(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    
    return X_train, X_test, y_train, y_test, builder