"""
Feature Definitions
===================

Versioned feature definitions following enterprise naming conventions.
Pattern: {domain}_{feature_name}_v{version}

Feature Lifecycle:
1. LOCAL - Defined in notebook/local code
2. VALIDATED - Tested for reuse
3. REFACTORED - Generalized for teams
4. REGISTERED - In Feature Store, governed via UC
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from ..utils.config import load_feature_config
from ..utils.logging import get_logger

logger = get_logger(__name__)


class FeatureStatus(Enum):
    """Feature lifecycle status."""
    LOCAL = "local"
    VALIDATED = "validated"
    REFACTORED = "refactored"
    REGISTERED = "registered"


@dataclass
class FeatureDefinition:
    """
    Definition of a single feature.
    
    Attributes:
        name: Full feature name ({domain}_{feature_name}_v{version})
        description: Human-readable description
        dtype: Data type (float64, int64, etc.)
        domain: Feature domain (e.g., "iris")
        version: Feature version number
        status: Lifecycle status
        source_column: Source column name (for raw features)
        derivation: Derivation logic (for derived features)
        feature_store_table: Feature store table (if registered)
    """
    name: str
    description: str
    dtype: str
    domain: str
    version: int
    status: FeatureStatus = FeatureStatus.LOCAL
    source_column: Optional[str] = None
    derivation: Optional[Dict[str, Any]] = None
    feature_store_table: Optional[str] = None
    
    @property
    def is_derived(self) -> bool:
        """Check if feature is derived from other features."""
        return self.derivation is not None
    
    @property
    def is_registered(self) -> bool:
        """Check if feature is registered in Feature Store."""
        return self.status == FeatureStatus.REGISTERED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "name": self.name,
            "description": self.description,
            "dtype": self.dtype,
            "domain": self.domain,
            "version": self.version,
            "status": self.status.value,
            "is_derived": self.is_derived,
            "is_registered": self.is_registered,
        }


@dataclass
class FeatureGroup:
    """
    A group of features used together for training.
    
    Attributes:
        name: Group name (e.g., "base_v1", "enhanced_v1")
        version: Group version
        features: List of feature names in this group
    """
    name: str
    version: int
    features: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "name": self.name,
            "version": self.version,
            "feature_count": len(self.features),
            "features": self.features,
        }


class FeatureRegistry:
    """
    Registry of all feature definitions.
    
    Loads feature definitions from configuration and provides
    methods to access and compute features.
    
    Example:
        >>> registry = FeatureRegistry()
        >>> feature = registry.get_feature("iris_sepal_length_cm_v1")
        >>> group = registry.get_feature_group("base_v1")
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize feature registry.
        
        Args:
            config_path: Path to feature config (uses default if None)
        """
        self._config = load_feature_config(config_path)
        self._features: Dict[str, FeatureDefinition] = {}
        self._groups: Dict[str, FeatureGroup] = {}
        self._load_definitions()
        
        logger.info(
            f"FeatureRegistry initialized with {len(self._features)} features "
            f"and {len(self._groups)} groups"
        )
    
    def _load_definitions(self) -> None:
        """Load feature definitions from config."""
        # Load individual features
        for name, spec in self._config.get("features", {}).items():
            status = FeatureStatus(spec.get("status", "local"))
            self._features[name] = FeatureDefinition(
                name=name,
                description=spec.get("description", ""),
                dtype=spec.get("dtype", "float64"),
                domain=spec.get("domain", ""),
                version=spec.get("version", 1),
                status=status,
                source_column=spec.get("source_column"),
                derivation=spec.get("derivation"),
                feature_store_table=spec.get("feature_store_table"),
            )
        
        # Load feature groups
        for name, spec in self._config.get("feature_groups", {}).items():
            self._groups[name] = FeatureGroup(
                name=name,
                version=spec.get("version", 1),
                features=spec.get("features", []),
            )
    
    def get_feature(self, name: str) -> FeatureDefinition:
        """Get a feature definition by name."""
        if name not in self._features:
            raise KeyError(f"Feature not found: {name}")
        return self._features[name]
    
    def get_feature_group(self, name: str) -> FeatureGroup:
        """Get a feature group by name."""
        if name not in self._groups:
            raise KeyError(f"Feature group not found: {name}")
        return self._groups[name]
    
    def list_features(self, status: FeatureStatus = None) -> List[str]:
        """List all feature names, optionally filtered by status."""
        if status is None:
            return list(self._features.keys())
        return [
            name for name, feat in self._features.items()
            if feat.status == status
        ]
    
    def list_feature_groups(self) -> List[str]:
        """List all feature group names."""
        return list(self._groups.keys())
    
    def get_target(self) -> Dict[str, Any]:
        """Get target definition."""
        return self._config.get("target", {})
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get feature configuration metadata."""
        return self._config.get("metadata", {})
    
    @property
    def version(self) -> str:
        """Get feature definition version."""
        return self.metadata.get("version", "unknown")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export registry for logging."""
        return {
            "metadata": self.metadata,
            "feature_count": len(self._features),
            "group_count": len(self._groups),
            "features": {name: f.to_dict() for name, f in self._features.items()},
            "groups": {name: g.to_dict() for name, g in self._groups.items()},
        }


# Singleton registry instance
_registry_instance: Optional[FeatureRegistry] = None


def get_feature_registry(config_path: str = None) -> FeatureRegistry:
    """
    Get or create the feature registry singleton.
    
    Args:
        config_path: Optional path to feature config
        
    Returns:
        FeatureRegistry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = FeatureRegistry(config_path)
    return _registry_instance


# Feature computation functions
def compute_ratio(
    df: pd.DataFrame,
    numerator_col: str,
    denominator_col: str,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Compute ratio feature with epsilon to avoid division by zero."""
    return df[numerator_col].values / (df[denominator_col].values + epsilon)


def compute_product(
    df: pd.DataFrame,
    factor_cols: List[str],
) -> np.ndarray:
    """Compute product of multiple columns."""
    result = np.ones(len(df))
    for col in factor_cols:
        result *= df[col].values
    return result


# Mapping of derivation types to functions
DERIVATION_FUNCTIONS = {
    "ratio": compute_ratio,
    "product": compute_product,
}


def compute_derived_feature(
    df: pd.DataFrame,
    feature_def: FeatureDefinition,
    source_col_mapping: Dict[str, str] = None,
) -> np.ndarray:
    """
    Compute a derived feature based on its definition.
    
    Args:
        df: Input DataFrame
        feature_def: Feature definition with derivation spec
        source_col_mapping: Optional mapping of feature names to column names
        
    Returns:
        Computed feature values
    """
    if not feature_def.is_derived:
        raise ValueError(f"Feature {feature_def.name} is not a derived feature")
    
    derivation = feature_def.derivation
    deriv_type = derivation.get("type")
    
    if deriv_type not in DERIVATION_FUNCTIONS:
        raise ValueError(f"Unknown derivation type: {deriv_type}")
    
    # Map feature names to column names if needed
    mapping = source_col_mapping or {}
    
    if deriv_type == "ratio":
        num_feat = derivation["numerator"]
        den_feat = derivation["denominator"]
        num_col = mapping.get(num_feat, num_feat)
        den_col = mapping.get(den_feat, den_feat)
        return compute_ratio(df, num_col, den_col)
    
    elif deriv_type == "product":
        factor_feats = derivation["factors"]
        factor_cols = [mapping.get(f, f) for f in factor_feats]
        return compute_product(df, factor_cols)
    
    raise ValueError(f"Unhandled derivation type: {deriv_type}")