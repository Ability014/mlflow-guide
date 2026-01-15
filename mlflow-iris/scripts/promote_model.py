#!/usr/bin/env python
"""
Model Promotion Script
======================

Promote models through lifecycle stages.

Usage:
    python scripts/promote_model.py --stage staging --build-id 123
    python scripts/promote_model.py --stage production --version 1 --approved-by user@company.com
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow

from src.models.model_registry import ModelRegistry, ModelStage
from src.utils.config import config


def main():
    parser = argparse.ArgumentParser(description="Promote model")
    parser.add_argument("--stage", type=str, required=True, choices=["staging", "production"])
    parser.add_argument("--version", type=str, help="Model version to promote")
    parser.add_argument("--build-id", type=str, help="Build ID (for staging)")
    parser.add_argument("--approved-by", type=str, help="Approver (for production)")
    parser.add_argument("--model-name", type=str, help="Model name (uses config if not provided)")
    
    args = parser.parse_args()
    
    # Get model name
    model_name = args.model_name or config.get_registry_model_name()
    
    print(f"Model: {model_name}")
    print(f"Promoting to: {args.stage}")
    
    registry = ModelRegistry(model_name=model_name)
    
    if args.stage == "staging":
        # Get latest dev version if not specified
        version = args.version or registry.get_latest_version(ModelStage.DEV)
        
        if not version:
            print("Error: No model version found")
            sys.exit(1)
        
        print(f"Promoting version {version} to staging...")
        
        registry.promote_to_staging(
            version=version,
            validation_results={"build_id": args.build_id} if args.build_id else None,
        )
        
        print(f"✓ Version {version} promoted to staging")
        
    elif args.stage == "production":
        if not args.version:
            # Get latest staging version
            args.version = registry.get_latest_version(ModelStage.STAGING)
        
        if not args.version:
            print("Error: No version specified or found in staging")
            sys.exit(1)
        
        if not args.approved_by:
            print("Error: --approved-by required for production promotion")
            sys.exit(1)
        
        print(f"Promoting version {args.version} to production...")
        print(f"Approved by: {args.approved_by}")
        
        registry.promote_to_production(
            version=args.version,
            approved_by=args.approved_by,
        )
        
        print(f"✓ Version {args.version} promoted to production")
    
    print("\nDone!")


if __name__ == "__main__":
    main()