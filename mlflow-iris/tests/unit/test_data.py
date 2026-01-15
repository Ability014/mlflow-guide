"""
Unit Tests - Data Loading
=========================

Tests for data loading functionality.
"""

import pytest
import pandas as pd


class TestLoadIrisLocal:
    """Tests for local Iris data loading."""
    
    def test_returns_dataframe(self, iris_data):
        """Should return a pandas DataFrame."""
        assert isinstance(iris_data, pd.DataFrame)
    
    def test_correct_shape(self, iris_data):
        """Should have 150 rows and 6 columns."""
        assert iris_data.shape == (150, 6)
    
    def test_has_required_columns(self, iris_data):
        """Should contain all required columns."""
        required = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "species",
            "species_encoded",
        ]
        for col in required:
            assert col in iris_data.columns, f"Missing column: {col}"
    
    def test_target_values(self, iris_data):
        """Target should be 0, 1, or 2."""
        assert set(iris_data["species_encoded"].unique()) == {0, 1, 2}
    
    def test_species_names(self, iris_data):
        """Species should be setosa, versicolor, virginica."""
        expected = {"setosa", "versicolor", "virginica"}
        assert set(iris_data["species"].unique()) == expected
    
    def test_balanced_classes(self, iris_data):
        """Each class should have 50 samples."""
        class_counts = iris_data["species"].value_counts()
        assert all(count == 50 for count in class_counts)
    
    def test_no_missing_values(self, iris_data):
        """Should have no missing values."""
        assert iris_data.isnull().sum().sum() == 0
    
    def test_numeric_columns_dtype(self, iris_data):
        """Numeric columns should be float."""
        numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        for col in numeric_cols:
            assert iris_data[col].dtype in ["float64", "float32"]