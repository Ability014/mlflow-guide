"""
Inference Pipeline
==================

Batch inference pipeline for scoring data with a registered model.

Usage:
    python -m pipelines.inference_pipeline \
        --model-name catalog.schema.iris_classifier \
        --input-path /path/to/input.csv \
        --output-path /path/to/output.csv
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.components.scoring import ScoringComponent
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


class InferencePipeline:
    """
    Batch inference pipeline.
    
    Example:
        >>> pipeline = InferencePipeline(
        ...     model_name="catalog.schema.iris_classifier",
        ...     model_alias="production"
        ... )
        >>> result_df = pipeline.run(input_df)
    """
    
    def __init__(
        self,
        model_name: str,
        model_version: str = None,
        model_alias: str = None,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.model_alias = model_alias
        
        self.scorer = None
        
        logger.info(f"InferencePipeline initialized: {model_name}")
    
    def _load_scorer(self) -> ScoringComponent:
        """Load the scoring component."""
        if self.scorer is None:
            self.scorer = ScoringComponent.from_registry(
                model_name=self.model_name,
                model_version=self.model_version,
                model_alias=self.model_alias,
            )
        return self.scorer
    
    def run(
        self,
        input_data: pd.DataFrame,
        add_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Run batch inference.
        
        Args:
            input_data: Input DataFrame with features
            add_metadata: Add inference metadata columns
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Running inference on {len(input_data)} samples")
        
        scorer = self._load_scorer()
        
        # Score batch
        result_df = scorer.score_batch(input_data, add_columns=True)
        
        # Add metadata
        if add_metadata:
            result_df["inference_timestamp"] = datetime.now().isoformat()
            result_df["model_name"] = self.model_name
            result_df["model_version"] = self.model_version or "latest"
            result_df["model_alias"] = self.model_alias or ""
        
        logger.info(f"Inference complete. {len(result_df)} predictions generated")
        
        return result_df
    
    def run_from_file(
        self,
        input_path: str,
        output_path: str,
        add_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Run inference from file to file.
        
        Args:
            input_path: Path to input CSV/Parquet
            output_path: Path to output CSV/Parquet
            add_metadata: Add inference metadata columns
            
        Returns:
            Result DataFrame
        """
        # Load input
        input_path = Path(input_path)
        if input_path.suffix == ".parquet":
            input_df = pd.read_parquet(input_path)
        else:
            input_df = pd.read_csv(input_path)
        
        logger.info(f"Loaded {len(input_df)} rows from {input_path}")
        
        # Run inference
        result_df = self.run(input_df, add_metadata=add_metadata)
        
        # Save output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == ".parquet":
            result_df.to_parquet(output_path, index=False)
        else:
            result_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(result_df)} rows to {output_path}")
        
        return result_df


def main():
    parser = argparse.ArgumentParser(description="Run batch inference")
    parser.add_argument("--model-name", type=str, required=True, help="Registered model name")
    parser.add_argument("--model-version", type=str, help="Model version")
    parser.add_argument("--model-alias", type=str, help="Model alias (dev/staging/production)")
    parser.add_argument("--input-path", type=str, required=True, help="Input file path")
    parser.add_argument("--output-path", type=str, required=True, help="Output file path")
    parser.add_argument("--log-level", type=str, default="INFO")
    
    args = parser.parse_args()
    
    setup_logging(level=args.log_level)
    
    pipeline = InferencePipeline(
        model_name=args.model_name,
        model_version=args.model_version,
        model_alias=args.model_alias,
    )
    
    result = pipeline.run_from_file(
        input_path=args.input_path,
        output_path=args.output_path,
    )
    
    print(f"\nInference complete. Output saved to: {args.output_path}")
    print(f"Total predictions: {len(result)}")


if __name__ == "__main__":
    main()