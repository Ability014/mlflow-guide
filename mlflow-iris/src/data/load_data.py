"""
Data Loading Module
===================

Load training data from Unity Catalog or sklearn (local development).
"""

from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
import mlflow

from ..utils.logging import get_logger

logger = get_logger(__name__)


# For Databricks environments
try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


def load_training_data(
    catalog: str,
    schema: str,
    table: str,
    snapshot_version: Optional[int] = None,
    as_pandas: bool = True,
) -> pd.DataFrame:
    """
    Load training data from Unity Catalog.
    
    Args:
        catalog: Unity Catalog name
        schema: Schema name
        table: Table name
        snapshot_version: Optional Iceberg snapshot for reproducibility
        as_pandas: Convert to pandas DataFrame
        
    Returns:
        DataFrame with training data
    """
    if not SPARK_AVAILABLE:
        raise ImportError("PySpark required for Unity Catalog access")
    
    table_path = f"{catalog}.{schema}.{table}"
    logger.info(f"Loading data from: {table_path}")
    
    spark = SparkSession.builder.getOrCreate()
    
    if snapshot_version:
        spark_df = (
            spark.read
            .option("versionAsOf", snapshot_version)
            .table(table_path)
        )
    else:
        spark_df = spark.table(table_path)
    
    row_count = spark_df.count()
    logger.info(f"Loaded {row_count:,} rows")
    
    if as_pandas:
        return spark_df.toPandas()
    
    return spark_df


def load_iris_local() -> pd.DataFrame:
    """
    Load Iris dataset from sklearn for local development.
    
    Returns:
        pandas DataFrame with Iris data matching expected schema
    """
    from sklearn.datasets import load_iris
    
    logger.info("Loading Iris dataset from sklearn (local)")
    
    iris = load_iris()
    
    df = pd.DataFrame(
        data=iris.data,
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
    
    df["species_encoded"] = iris.target
    df["species"] = df["species_encoded"].map({
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    })
    
    logger.info(f"Loaded {len(df)} samples")
    
    return df


def log_data_info_to_mlflow(
    df: pd.DataFrame,
    source: str = "sklearn",
    catalog: str = None,
    schema: str = None,
    table: str = None,
) -> None:
    """
    Log data information to MLflow.
    
    Args:
        df: Training DataFrame
        source: Data source identifier
        catalog: Unity Catalog name (if applicable)
        schema: Schema name (if applicable)
        table: Table name (if applicable)
    """
    if mlflow.active_run() is None:
        logger.warning("No active MLflow run")
        return
    
    # Log source
    if catalog and schema and table:
        mlflow.log_params({
            "data_source": f"{catalog}.{schema}.{table}",
            "data_catalog": catalog,
            "data_schema": schema,
            "data_table": table,
        })
    else:
        mlflow.log_param("data_source", source)
    
    # Log statistics
    mlflow.log_metrics({
        "data_rows": len(df),
        "data_columns": len(df.columns),
    })
    
    logger.info("Logged data info to MLflow")