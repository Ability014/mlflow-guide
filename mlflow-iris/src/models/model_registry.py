"""
Model Registry Operations
=========================

Model registration, promotion, and lifecycle management.

Promotion Workflow:
    DEV (experiments) -> STAGING (validation) -> PRODUCTION (approved)
"""

from typing import Any, Dict, List, Optional
from enum import Enum

import mlflow
from mlflow.tracking import MlflowClient

from ..utils.logging import get_logger
from ..utils.config import config

logger = get_logger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages."""
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


class ModelRegistry:
    """
    Model registry operations for enterprise MLflow.
    
    Example:
        >>> registry = ModelRegistry()
        >>> version = registry.register_model(
        ...     model_uri="runs:/abc123/model",
        ...     description="Iris classifier v1.0.0"
        ... )
        >>> registry.promote_to_staging(version)
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize registry.
        
        Args:
            model_name: Full model name (uses config if None)
        """
        self.client = MlflowClient()
        self.model_name = model_name or config.get_registry_model_name()
        
        logger.info(f"ModelRegistry initialized: {self.model_name}")
    
    def register_model(
        self,
        model_uri: str,
        description: str = None,
        tags: Dict[str, str] = None,
    ) -> str:
        """
        Register a model to the registry.
        
        Args:
            model_uri: MLflow model URI (runs:/... or models:/...)
            description: Model description
            tags: Tags to apply
            
        Returns:
            Model version string
        """
        logger.info(f"Registering model to: {self.model_name}")
        
        # Register model
        result = mlflow.register_model(
            model_uri=model_uri,
            name=self.model_name,
        )
        
        version = result.version
        logger.info(f"Registered version: {version}")
        
        # Update description
        if description:
            self.client.update_model_version(
                name=self.model_name,
                version=version,
                description=description,
            )
        
        # Set tags
        if tags:
            for key, value in tags.items():
                if value is not None:
                    self.client.set_model_version_tag(
                        name=self.model_name,
                        version=version,
                        key=key,
                        value=str(value),
                    )
        
        # Set initial alias (dev)
        self._set_alias(version, ModelStage.DEV)
        
        return version
    
    def _set_alias(self, version: str, stage: ModelStage) -> None:
        """Set model alias for a version."""
        try:
            self.client.set_registered_model_alias(
                name=self.model_name,
                alias=stage.value,
                version=version,
            )
            logger.info(f"Set alias '{stage.value}' for version {version}")
        except Exception as e:
            # Alias API may not be available in all MLflow versions
            logger.warning(f"Could not set alias: {e}")
    
    def promote_to_staging(
        self,
        version: str,
        validation_results: Dict[str, Any] = None,
    ) -> None:
        """
        Promote model version to staging.
        
        Args:
            version: Model version to promote
            validation_results: Optional validation results to log
        """
        logger.info(f"Promoting version {version} to staging")
        
        # Set staging alias
        self._set_alias(version, ModelStage.STAGING)
        
        # Log validation results
        if validation_results:
            for key, value in validation_results.items():
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=version,
                    key=f"staging_{key}",
                    value=str(value),
                )
        
        # Add promotion timestamp
        self.client.set_model_version_tag(
            name=self.model_name,
            version=version,
            key="promoted_to_staging_at",
            value=str(mlflow.utils.time.get_current_time_millis()),
        )
        
        logger.info(f"Version {version} promoted to staging")
    
    def promote_to_production(
        self,
        version: str,
        approved_by: str = None,
    ) -> None:
        """
        Promote model version to production.
        
        Args:
            version: Model version to promote
            approved_by: Approver identifier
        """
        logger.info(f"Promoting version {version} to production")
        
        # Verify it's in staging
        model_version = self.client.get_model_version(
            name=self.model_name,
            version=version,
        )
        
        # Set production alias
        self._set_alias(version, ModelStage.PRODUCTION)
        
        # Log approval
        if approved_by:
            self.client.set_model_version_tag(
                name=self.model_name,
                version=version,
                key="approved_by",
                value=approved_by,
            )
        
        self.client.set_model_version_tag(
            name=self.model_name,
            version=version,
            key="promoted_to_production_at",
            value=str(mlflow.utils.time.get_current_time_millis()),
        )
        
        logger.info(f"Version {version} promoted to production")
    
    def get_latest_version(self, stage: ModelStage = None) -> Optional[str]:
        """
        Get latest model version, optionally filtered by stage.
        
        Args:
            stage: Filter by stage (None for any)
            
        Returns:
            Version string or None
        """
        try:
            if stage:
                # Try to get by alias
                version = self.client.get_model_version_by_alias(
                    name=self.model_name,
                    alias=stage.value,
                )
                return version.version
            else:
                # Get latest overall
                versions = self.client.get_latest_versions(
                    name=self.model_name,
                )
                if versions:
                    return versions[0].version
        except Exception as e:
            logger.warning(f"Could not get latest version: {e}")
        
        return None
    
    def get_model_version_details(self, version: str) -> Dict[str, Any]:
        """
        Get detailed information about a model version.
        
        Args:
            version: Model version
            
        Returns:
            Dictionary with version details
        """
        mv = self.client.get_model_version(
            name=self.model_name,
            version=version,
        )
        
        return {
            "name": mv.name,
            "version": mv.version,
            "description": mv.description,
            "status": mv.status,
            "source": mv.source,
            "run_id": mv.run_id,
            "tags": mv.tags,
            "creation_timestamp": mv.creation_timestamp,
            "last_updated_timestamp": mv.last_updated_timestamp,
        }
    
    def list_versions(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        List model versions.
        
        Args:
            max_results: Maximum number of versions to return
            
        Returns:
            List of version details
        """
        versions = self.client.search_model_versions(
            filter_string=f"name='{self.model_name}'",
            max_results=max_results,
            order_by=["version_number DESC"],
        )
        
        return [
            {
                "version": v.version,
                "status": v.status,
                "run_id": v.run_id,
                "tags": v.tags,
            }
            for v in versions
        ]
    
    def archive_version(self, version: str) -> None:
        """
        Archive a model version.
        
        Args:
            version: Model version to archive
        """
        logger.info(f"Archiving version {version}")
        
        self.client.set_model_version_tag(
            name=self.model_name,
            version=version,
            key="archived",
            value="true",
        )
        
        self.client.set_model_version_tag(
            name=self.model_name,
            version=version,
            key="archived_at",
            value=str(mlflow.utils.time.get_current_time_millis()),
        )
        
        logger.info(f"Version {version} archived")


def register_model(
    model_uri: str,
    model_name: str = None,
    description: str = None,
    tags: Dict[str, str] = None,
) -> str:
    """
    Convenience function to register a model.
    
    Args:
        model_uri: MLflow model URI
        model_name: Model name (uses config if None)
        description: Model description
        tags: Tags to apply
        
    Returns:
        Model version string
    """
    registry = ModelRegistry(model_name=model_name)
    return registry.register_model(
        model_uri=model_uri,
        description=description,
        tags=tags,
    )