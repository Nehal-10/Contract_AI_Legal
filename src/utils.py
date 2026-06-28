"""
Utility functions for the Contract Risk Intelligence System.
Provides logging, configuration loading, and common helper methods.
"""

import logging
import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger

# Configure Loguru with both file and console output
def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> None:
    """
    Configure logging system with both file and console output.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Remove default logger
    logger.remove()
    
    # Console logging
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # File logging
    log_file = log_dir_path / f"contract_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(
        str(log_file),
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level
    )
    
    logger.info(f"Logging initialized. Log file: {log_file}")

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return get_default_config()
    
    try:
        with open(config_file, 'r') as f:
            if config_file.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_file.suffix == '.json':
                config = json.load(f)
            else:
                logger.warning(f"Unsupported config format: {config_file.suffix}, using defaults")
                return get_default_config()
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values.
    
    Returns:
        Dictionary with default configuration
    """
    return {
        "model": {
            "embedding_model": "nlpaueb/legal-bert-base-uncased",
            "embedding_dim": 768,
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 5
        },
        "training": {
            "clause_classifier": {
                "model_type": "xgboost",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8
                }
            },
            "risk_regressor": {
                "model_type": "xgboost",
                "params": {
                    "n_estimators": 150,
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8
                }
            },
            "compliance_classifier": {
                "model_type": "xgboost",
                "params": {
                    "n_estimators": 150,
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8
                }
            }
        },
        "data": {
            "max_clause_length": 512,
            "min_clause_words": 5,
            "clause_separators": ["\n\n", ". ", "; "]
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "max_upload_size": 10485760  # 10MB
        },
        "paths": {
            "data_dir": "data",
            "models_dir": "models",
            "logs_dir": "logs",
            "outputs_dir": "outputs"
        }
    }

def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object of the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def save_model(model: Any, path: str) -> None:
    """
    Save a trained model using joblib.
    
    Args:
        model: Trained model object
        path: Path to save the model
    """
    import joblib
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path_obj)
    logger.info(f"Model saved to {path}")

def load_model(path: str) -> Any:
    """
    Load a trained model using joblib.
    
    Args:
        path: Path to the saved model
        
    Returns:
        Loaded model object
    """
    import joblib
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    model = joblib.load(path_obj)
    logger.info(f"Model loaded from {path}")
    return model

def compute_risk_percentage(risk_score: float) -> str:
    """
    Convert risk score to percentage string.
    
    Args:
        risk_score: Risk score between 0 and 1
        
    Returns:
        Formatted percentage string
    """
    percentage = min(max(risk_score, 0), 1) * 100
    return f"{percentage:.1f}%"

def get_risk_level(risk_score: float) -> str:
    """
    Convert risk score to risk level label.
    
    Args:
        risk_score: Risk score between 0 and 1
        
    Returns:
        Risk level string: 'Low', 'Medium', or 'High'
    """
    if risk_score < 0.33:
        return "Low"
    elif risk_score < 0.66:
        return "Medium"
    else:
        return "High"

def extract_clause_text(clause: Dict[str, Any]) -> str:
    """
    Extract text from a clause dictionary.
    
    Args:
        clause: Clause dictionary with 'text' key
        
    Returns:
        Clause text string
    """
    if isinstance(clause, dict):
        return clause.get('text', '')
    return str(clause)

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator

# Import sys for console logging
import sys

logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)