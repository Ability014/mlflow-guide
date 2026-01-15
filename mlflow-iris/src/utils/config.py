"""
Configuration Loader
Load and validate configuration files with environment variable substitution.
"""
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

def _substitute_env_vars(value: Any) -> Any:
    """
    Substitute environment variables in string values.

    Supports patterns:
    - ${VAR_NAME} - Required, raises error if not set
    - ${VAR_NAME:default} - Optional with default value
    """

    if not isinstance(value, str):
        return value

    pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

    def replacer(match):
        var_name = match.group(1)
        default = match.group(2)
        
        env_value = os.environ.get(var_name)
        
        if env_value is not None:
            return env_value
        elif default is not None:
            return default
        else:
            # Return original if no env var and no default
            return match.group(0)

    return re.sub(pattern, replacer, value)

def _process_config(config: Any) -> Any:
    """Recursively process config and substitute env vars."""
    if isinstance(config, dict):
        return {k: _process_config(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_process_config(v) for v in config]
    else:
        return _substitute_env_vars(config)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file with environment variable substitution.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Parsed configuration dictionary
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return _process_config(config)

def load_model_config(config_path: str = None) -> Dict[str, Any]:
    """Load model configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "model_config.yaml"
    return load_config(str(config_path))


def load_feature_config(config_path: str = None) -> Dict[str, Any]:
    """Load feature configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "feature_config.yaml"
    return load_config(str(config_path))


def load_environment_config(config_path: str = None) -> Dict[str, Any]:
    """Load environment configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "environment.yaml"
    return load_config(str(config_path))

class Config:
    """
    Centralized configuration access.
    
    Example:
        >>> config = Config()
        >>> print(config.model['hyperparameters']['random_forest'])
        >>> print(config.features['feature_groups']['base_v1'])
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._model_config = None
        self._feature_config = None
        self._environment_config = None
        self._initialized = True
    
    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        if self._model_config is None:
            self._model_config = load_model_config()
        return self._model_config
    
    @property
    def features(self) -> Dict[str, Any]:
        """Get feature configuration."""
        if self._feature_config is None:
            self._feature_config = load_feature_config()
        return self._feature_config
    
    @property
    def environment(self) -> Dict[str, Any]:
        """Get environment configuration."""
        if self._environment_config is None:
            self._environment_config = load_environment_config()
        return self._environment_config
    
    def get_experiment_name(self) -> str:
        """Get formatted experiment name."""
        template = self.model['experiment']['name_template']
        metadata = self.model['metadata']
        return template.format(**metadata)
    
    def get_registry_model_name(self) -> str:
        """Get formatted registry model name."""
        template = self.model['mlflow']['registry']['name_template']
        metadata = self.model['metadata']
        registry = self.model['mlflow']['registry']
        
        return template.format(
            catalog=registry['catalog'],
            schema=registry['schema'],
            team=metadata['team'],
            project=metadata['project'],
            model_name=metadata['model_name'],
        )
    
    def get_endpoint_name(self) -> str:
        """Get formatted endpoint name."""
        template = self.model['deployment']['serving']['endpoint_name_template']
        metadata = self.model['metadata']
        return template.format(**metadata)
    
    def reload(self):
        """Force reload all configurations."""
        self._model_config = None
        self._feature_config = None
        self._environment_config = None


# Global config instance
config = Config()
