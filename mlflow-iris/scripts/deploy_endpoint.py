#!/usr/bin/env python
"""
Endpoint Deployment Script
==========================

Deploy models to Databricks Model Serving.

Usage:
    python scripts/deploy_endpoint.py --environment staging
    python scripts/deploy_endpoint.py --environment production --version 1
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import config
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def deploy_endpoint(
    environment: str,
    model_version: str = None,
):
    """
    Deploy model to Databricks Model Serving.
    
    Args:
        environment: Target environment (staging/production)
        model_version: Model version to deploy
    """
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import (
            EndpointCoreConfigInput,
            ServedEntityInput,
        )
    except ImportError:
        logger.error("databricks-sdk required. Install with: pip install databricks-sdk")
        sys.exit(1)
    
    # Get configuration
    model_name = config.get_registry_model_name()
    endpoint_name = config.get_endpoint_name()
    
    if environment == "staging":
        endpoint_name = f"{endpoint_name}-staging"
    
    logger.info(f"Deploying to endpoint: {endpoint_name}")
    logger.info(f"Model: {model_name}, Version: {model_version or 'latest'}")
    
    # Create client
    w = WorkspaceClient()
    
    # Prepare served entity config
    served_entity = ServedEntityInput(
        entity_name=model_name,
        entity_version=model_version or "1",
        workload_size="Small",
        scale_to_zero_enabled=True,
    )
    
    # Check if endpoint exists
    try:
        existing = w.serving_endpoints.get(endpoint_name)
        logger.info(f"Updating existing endpoint: {endpoint_name}")
        
        # Update endpoint
        w.serving_endpoints.update_config_and_wait(
            name=endpoint_name,
            served_entities=[served_entity],
        )
        
    except Exception:
        logger.info(f"Creating new endpoint: {endpoint_name}")
        
        # Create endpoint
        w.serving_endpoints.create_and_wait(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                served_entities=[served_entity],
            ),
        )
    
    logger.info(f"✓ Deployment complete: {endpoint_name}")
    
    # Get endpoint URL
    endpoint = w.serving_endpoints.get(endpoint_name)
    logger.info(f"Endpoint URL: https://{os.environ.get('DATABRICKS_HOST')}/serving-endpoints/{endpoint_name}/invocations")


def main():
    parser = argparse.ArgumentParser(description="Deploy model endpoint")
    parser.add_argument(
        "--environment",
        type=str,
        required=True,
        choices=["staging", "production"],
    )
    parser.add_argument("--version", type=str, help="Model version to deploy")
    parser.add_argument("--log-level", type=str, default="INFO")
    
    args = parser.parse_args()
    
    setup_logging(level=args.log_level)
    
    deploy_endpoint(
        environment=args.environment,
        model_version=args.version,
    )


if __name__ == "__main__":
    main()