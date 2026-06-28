"""
Configuration management for the Contract Risk Intelligence System.
Handles loading, validation, and access to configuration parameters.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

@dataclass
class ModelConfig:
    """Configuration for ML models."""
    embedding_model: str = "nlpaueb/legal-bert-base-uncased"
    embedding_dim: int = 768
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    max_clause_length: int = 512
    
@dataclass
class TrainingConfig:
    """Configuration for model training."""
    clause_classifier_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    })
    risk_regressor_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 150,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    })
    compliance_classifier_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    })

@dataclass
class DataConfig:
    """Configuration for data processing."""
    min_clause_words: int = 5
    clause_separators: list = field(default_factory=lambda: ["\n\n", ". ", "; "])
    min_samples_per_category: int = 10

@dataclass
class APIConfig:
    """Configuration for FastAPI."""
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_size: int = 10485760  # 10MB
    debug: bool = False

@dataclass
class PathsConfig:
    """Configuration for file paths."""
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    outputs_dir: str = "outputs"
    
    def __post_init__(self):
        """Create directories after initialization."""
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir, self.outputs_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    api: APIConfig = field(default_factory=APIConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create Config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config instance
        """
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            api=APIConfig(**config_dict.get('api', {})),
            paths=PathsConfig(**config_dict.get('paths', {}))
        )
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config instance
        """
        if not Path(config_path).exists():
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return cls()
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return cls()
    
    def to_yaml(self, config_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save configuration
        """
        config_dict = {
            'model': {
                'embedding_model': self.model.embedding_model,
                'embedding_dim': self.model.embedding_dim,
                'test_size': self.model.test_size,
                'random_state': self.model.random_state,
                'cv_folds': self.model.cv_folds,
                'max_clause_length': self.model.max_clause_length
            },
            'training': {
                'clause_classifier_params': self.training.clause_classifier_params,
                'risk_regressor_params': self.training.risk_regressor_params,
                'compliance_classifier_params': self.training.compliance_classifier_params
            },
            'data': {
                'min_clause_words': self.data.min_clause_words,
                'clause_separators': self.data.clause_separators,
                'min_samples_per_category': self.data.min_samples_per_category
            },
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'max_upload_size': self.api.max_upload_size,
                'debug': self.api.debug
            },
            'paths': {
                'data_dir': self.paths.data_dir,
                'models_dir': self.paths.models_dir,
                'logs_dir': self.paths.logs_dir,
                'outputs_dir': self.paths.outputs_dir
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        logger.info(f"Configuration saved to {config_path}")

# Default configuration instance
default_config = Config()

# Singleton config instance
_config_instance: Optional[Config] = None

def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get configuration instance (singleton).
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        if config_path and Path(config_path).exists():
            _config_instance = Config.from_yaml(config_path)
        else:
            _config_instance = Config()
    return _config_instance