"""
Unit Tests - Feature Engineering
================================

Tests for feature builder and definitions.
"""

import pytest
import numpy as np

from src.features.feature_builder import FeatureBuilder
from src.features.feature_definitions import (
    FeatureRegistry,
    FeatureStatus,
    get_feature_registry,
)


class TestFeatureRegistry:
    """Tests for feature registry."""
    
    def test_registry_initialization(self):
        """Registry should initialize with features."""
        registry = get_feature_registry()
        assert len(registry.list_features()) > 0
    
    def test_get_feature(self):
        """Should retrieve feature definition."""
        registry = get_feature_registry()
        feature = registry.get_feature("iris_sepal_length_cm_v1")
        assert feature.name == "iris_sepal_length_cm_v1"
        assert feature.domain == "iris"
    
    def test_get_feature_group(self):
        """Should retrieve feature group."""
        registry = get_feature_registry()
        group = registry.get_feature_group("base_v1")
        assert len(group.features) == 4
    
    def test_feature_status(self):
        """Features should have valid status."""
        registry = get_feature_registry()
        feature = registry.get_feature("iris_sepal_length_cm_v1")
        assert isinstance(feature.status, FeatureStatus)
    
    def test_version_property(self):
        """Registry should have version."""
        registry = get_feature_registry()
        assert registry.version is not None


class TestFeatureBuilder:
    """Tests for feature builder."""
    
    def test_initialization(self):
        """Should initialize with feature group."""
        builder = FeatureBuilder(feature_group="base_v1")
        assert builder.feature_group_name == "base_v1"
    
    def test_fit_transform_shapes(self, iris_data):
        """Should return correct array shapes."""
        builder = FeatureBuilder(feature_group="base_v1")
        X_train, X_test, y_train, y_test = builder.fit_transform(
            iris_data, test_size=0.2, random_state=42
        )
        
        assert X_train.shape[0] == 120  # 80% of 150
        assert X_test.shape[0] == 30   # 20% of 150
        assert X_train.shape[1] == 4   # 4 base features
        assert len(y_train) == 120
        assert len(y_test) == 30
    
    def test_feature_names_after_fit(self, iris_data):
        """Feature names should be accessible after fitting."""
        builder = FeatureBuilder(feature_group="base_v1")
        builder.fit_transform(iris_data)
        
        assert len(builder.feature_names) == 4
        assert "iris_sepal_length_cm_v1" in builder.feature_names
    
    def test_scaling_applied(self, iris_data):
        """Scaled features should have ~mean 0, std 1."""
        builder = FeatureBuilder(
            feature_group="base_v1",
            scaling_method="standard",
        )
        X_train, _, _, _ = builder.fit_transform(iris_data)
        
        # Scaled data should have mean close to 0
        assert abs(X_train.mean()) < 0.1
    
    def test_no_scaling(self, iris_data):
        """Without scaling, features retain original scale."""
        builder = FeatureBuilder(
            feature_group="base_v1",
            scaling_method="none",
        )
        X_train, _, _, _ = builder.fit_transform(iris_data)
        
        # Original data has mean > 1
        assert X_train.mean() > 1
    
    def test_stratified_split(self, iris_data):
        """Stratified split should preserve class proportions."""
        builder = FeatureBuilder(feature_group="base_v1")
        _, _, y_train, y_test = builder.fit_transform(
            iris_data, test_size=0.2, random_state=42, stratify=True
        )
        
        train_counts = np.bincount(y_train)
        test_counts = np.bincount(y_test)
        
        # Each class should be equally represented
        assert len(set(train_counts)) == 1
        assert len(set(test_counts)) == 1
    
    def test_metadata(self, iris_data):
        """Should return valid metadata."""
        builder = FeatureBuilder(feature_group="base_v1")
        builder.fit_transform(iris_data)
        
        metadata = builder.get_metadata()
        
        assert "feature_group" in metadata
        assert "feature_version" in metadata
        assert "feature_count" in metadata
        assert metadata["is_fitted"] is True
    
    def test_unfitted_raises(self, iris_data):
        """Accessing feature_names before fit should raise."""
        builder = FeatureBuilder(feature_group="base_v1")
        
        with pytest.raises(RuntimeError):
            _ = builder.feature_names