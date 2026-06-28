"""
Model definitions for the Contract Risk Intelligence System.
Contains ML model definitions for classification, regression, and training pipelines.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from loguru import logger

class ModelFactory:
    """
    Factory class for creating ML models.
    """
    
    @staticmethod
    def create_clause_classifier(model_type: str = 'xgboost', **kwargs) -> BaseEstimator:
        """
        Create a classifier for clause category classification.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional arguments for the model
            
        Returns:
            BaseEstimator instance
        """
        logger.info(f"Creating clause classifier of type: {model_type}")
        
        default_params = {
            'random_state': 42,
            'n_jobs': -1
        }
        
        params = {**default_params, **kwargs}
        
        if model_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000, **params)
        elif model_type == 'linear_svm':
            return LinearSVC(max_iter=2000, **params)
        elif model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 200),
                max_depth=params.get('max_depth', 10),
                **params
            )
        elif model_type == 'xgboost':
            return XGBClassifier(
                n_estimators=params.get('n_estimators', 200),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                random_state=params.get('random_state', 42),
                n_jobs=params.get('n_jobs', -1)
            )
        elif model_type == 'lightgbm':
            return LGBMClassifier(
                n_estimators=params.get('n_estimators', 200),
                max_depth=params.get('max_depth', -1),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                random_state=params.get('random_state', 42),
                n_jobs=params.get('n_jobs', -1)
            )
        elif model_type == 'catboost':
            return CatBoostClassifier(
                iterations=params.get('n_estimators', 200),
                depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=params.get('random_state', 42),
                thread_count=params.get('n_jobs', -1),
                verbose=False
            )
        elif model_type == 'naive_bayes':
            return GaussianNB()
        else:
            logger.warning(f"Unknown model type: {model_type}, using XGBoost")
            return XGBClassifier(**{k: v for k, v in params.items() if k != 'n_jobs'})
    
    @staticmethod
    def create_risk_regressor(model_type: str = 'xgboost', **kwargs) -> BaseEstimator:
        """
        Create a regressor for risk score prediction.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional arguments for the model
            
        Returns:
            BaseEstimator instance
        """
        logger.info(f"Creating risk regressor of type: {model_type}")
        
        default_params = {
            'random_state': 42,
            'n_jobs': -1
        }
        
        params = {**default_params, **kwargs}
        
        if model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=params.get('n_estimators', 150),
                max_depth=params.get('max_depth', 8),
                **params
            )
        elif model_type == 'xgboost':
            return XGBRegressor(
                n_estimators=params.get('n_estimators', 150),
                max_depth=params.get('max_depth', 5),
                learning_rate=params.get('learning_rate', 0.05),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                random_state=params.get('random_state', 42),
                n_jobs=params.get('n_jobs', -1)
            )
        elif model_type == 'lightgbm':
            return LGBMRegressor(
                n_estimators=params.get('n_estimators', 150),
                max_depth=params.get('max_depth', -1),
                learning_rate=params.get('learning_rate', 0.05),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                random_state=params.get('random_state', 42),
                n_jobs=params.get('n_jobs', -1)
            )
        elif model_type == 'catboost':
            return CatBoostRegressor(
                iterations=params.get('n_estimators', 150),
                depth=params.get('max_depth', 5),
                learning_rate=params.get('learning_rate', 0.05),
                random_state=params.get('random_state', 42),
                thread_count=params.get('n_jobs', -1),
                verbose=False
            )
        else:
            logger.warning(f"Unknown model type: {model_type}, using XGBoost")
            return XGBRegressor(**{k: v for k, v in params.items() if k != 'n_jobs'})
    
    @staticmethod
    def create_compliance_classifier(model_type: str = 'xgboost', **kwargs) -> BaseEstimator:
        """
        Create a classifier for compliance prediction.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional arguments for the model
            
        Returns:
            BaseEstimator instance
        """
        logger.info(f"Creating compliance classifier of type: {model_type}")
        
        default_params = {
            'random_state': 42,
            'n_jobs': -1
        }
        
        params = {**default_params, **kwargs}
        
        if model_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000, **params)
        elif model_type == 'linear_svm':
            return LinearSVC(max_iter=2000, **params)
        elif model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 150),
                max_depth=params.get('max_depth', 8),
                **params
            )
        elif model_type == 'xgboost':
            return XGBClassifier(
                n_estimators=params.get('n_estimators', 150),
                max_depth=params.get('max_depth', 4),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                random_state=params.get('random_state', 42),
                n_jobs=params.get('n_jobs', -1)
            )
        elif model_type == 'lightgbm':
            return LGBMClassifier(
                n_estimators=params.get('n_estimators', 150),
                max_depth=params.get('max_depth', -1),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                random_state=params.get('random_state', 42),
                n_jobs=params.get('n_jobs', -1)
            )
        elif model_type == 'catboost':
            return CatBoostClassifier(
                iterations=params.get('n_estimators', 150),
                depth=params.get('max_depth', 4),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=params.get('random_state', 42),
                thread_count=params.get('n_jobs', -1),
                verbose=False
            )
        else:
            logger.warning(f"Unknown model type: {model_type}, using XGBoost")
            return XGBClassifier(**{k: v for k, v in params.items() if k != 'n_jobs'})

class ModelManager:
    """
    Manages model training, saving, and loading.
    """
    
    def __init__(self, config: Any):
        """
        Initialize ModelManager with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.models = {}
        self.scaler = None
        self.training_history = {}
        
    def train_models(
        self,
        X_train: np.ndarray,
        y_clause_train: np.ndarray,
        y_risk_train: np.ndarray,
        y_compliance_train: np.ndarray,
        y_score_train: np.ndarray,
        model_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, BaseEstimator]:
        """
        Train all models on the provided data.
        
        Args:
            X_train: Training features (embeddings)
            y_clause_train: Training labels for clause classification
            y_risk_train: Training labels for risk level classification
            y_compliance_train: Training labels for compliance classification
            y_score_train: Training labels for risk score regression
            model_configs: Optional model configurations
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training all models...")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Default model configurations
        if model_configs is None:
            model_configs = {
                'clause_classifier': {
                    'type': self.config.training.clause_classifier_params.get('model_type', 'xgboost'),
                    'params': {k: v for k, v in self.config.training.clause_classifier_params.items() if k != 'model_type'}
                },
                'risk_regressor': {
                    'type': self.config.training.risk_regressor_params.get('model_type', 'xgboost'),
                    'params': {k: v for k, v in self.config.training.risk_regressor_params.items() if k != 'model_type'}
                },
                'compliance_classifier': {
                    'type': self.config.training.compliance_classifier_params.get('model_type', 'xgboost'),
                    'params': {k: v for k, v in self.config.training.compliance_classifier_params.items() if k != 'model_type'}
                }
            }
        
        # Train clause classifier
        logger.info("Training clause classifier...")
        clause_model = ModelFactory.create_clause_classifier(
            model_type=model_configs['clause_classifier']['type'],
            **model_configs['clause_classifier']['params']
        )
        clause_model.fit(X_train_scaled, y_clause_train)
        self.models['clause_classifier'] = clause_model
        self.training_history['clause_classifier'] = {'type': model_configs['clause_classifier']['type']}
        
        # Train risk regressor
        logger.info("Training risk regressor...")
        risk_model = ModelFactory.create_risk_regressor(
            model_type=model_configs['risk_regressor']['type'],
            **model_configs['risk_regressor']['params']
        )
        risk_model.fit(X_train_scaled, y_score_train)
        self.models['risk_regressor'] = risk_model
        self.training_history['risk_regressor'] = {'type': model_configs['risk_regressor']['type']}
        
        # Train compliance classifier
        logger.info("Training compliance classifier...")
        compliance_model = ModelFactory.create_compliance_classifier(
            model_type=model_configs['compliance_classifier']['type'],
            **model_configs['compliance_classifier']['params']
        )
        compliance_model.fit(X_train_scaled, y_compliance_train)
        self.models['compliance_classifier'] = compliance_model
        self.training_history['compliance_classifier'] = {'type': model_configs['compliance_classifier']['type']}
        
        logger.info("All models trained successfully")
        return self.models
    
    def predict_all(
        self,
        X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with all trained models.
        
        Args:
            X: Features for prediction
            
        Returns:
            Dictionary with prediction results
        """
        if not self.models:
            raise ValueError("No models trained. Call train_models() first.")
        
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        predictions = {}
        
        # Clause classification
        if 'clause_classifier' in self.models:
            predictions['clause_category'] = self.models['clause_classifier'].predict(X_scaled)
        
        # Risk regressor
        if 'risk_regressor' in self.models:
            risk_scores = self.models['risk_regressor'].predict(X_scaled)
            predictions['risk_score'] = np.clip(risk_scores, 0, 1)
            
            # Convert to risk levels based on thresholds
            risk_levels = np.zeros_like(risk_scores, dtype=int)
            risk_levels[risk_scores >= 0.66] = 2  # High
            risk_levels[(risk_scores >= 0.33) & (risk_scores < 0.66)] = 1  # Medium
            risk_levels[risk_scores < 0.33] = 0  # Low
            predictions['risk_level'] = risk_levels
        
        # Compliance classifier
        if 'compliance_classifier' in self.models:
            predictions['compliance'] = self.models['compliance_classifier'].predict(X_scaled)
            predictions['compliance_probability'] = self.models['compliance_classifier'].predict_proba(X_scaled)[:, 1]
        
        return predictions
    
    def save_models(self, model_dir: str) -> None:
        """
        Save all trained models to disk.
        
        Args:
            model_dir: Directory to save models
        """
        import joblib
        from pathlib import Path
        
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, model_path / f"{name}.pkl")
            logger.info(f"Saved {name} to {model_path / f'{name}.pkl'}")
        
        # Save scaler
        if self.scaler is not None:
            joblib.dump(self.scaler, model_path / "scaler.pkl")
            logger.info(f"Saved scaler to {model_path / 'scaler.pkl'}")
        
        # Save training history
        import json
        with open(model_path / "training_history.json", 'w') as f:
            json.dump(self.training_history, f)
    
    def load_models(self, model_dir: str) -> Dict[str, BaseEstimator]:
        """
        Load trained models from disk.
        
        Args:
            model_dir: Directory containing saved models
            
        Returns:
            Dictionary of loaded models
        """
        import joblib
        from pathlib import Path
        import json
        
        model_path = Path(model_dir)
        
        self.models = {}
        
        for model_file in model_path.glob("*.pkl"):
            if model_file.stem != 'scaler':
                self.models[model_file.stem] = joblib.load(model_file)
                logger.info(f"Loaded {model_file.stem} from {model_file}")
        
        # Load scaler
        scaler_file = model_path / "scaler.pkl"
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)
            logger.info(f"Loaded scaler from {scaler_file}")
        
        # Load training history
        history_file = model_path / "training_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.training_history = json.load(f)
        
        return self.models
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about trained models.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'models': {},
            'scaler_fitted': self.scaler is not None,
            'training_history': self.training_history
        }
        
        for name, model in self.models.items():
            info['models'][name] = {
                'type': type(model).__name__,
                'is_fitted': hasattr(model, 'classes_') or hasattr(model, 'feature_importances_')
            }
            
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                info['models'][name]['has_feature_importance'] = True
        
        return info