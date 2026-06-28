"""
Model evaluation module for the Contract Risk Intelligence System.
Provides comprehensive evaluation metrics and visualization tools.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_curve, auc
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from pathlib import Path

class ModelEvaluator:
    """
    Comprehensive model evaluation with multiple metrics and visualizations.
    """
    
    def __init__(self, config: Any):
        """
        Initialize ModelEvaluator with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.cv_folds = config.model.cv_folds
        
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        task_name: str = "Classification",
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate classification model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            task_name: Name of the task
            labels: List of label names
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {task_name}...")
        
        # Handle single-class or small dataset
        n_classes = len(np.unique(y_true))
        
        results = {}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Handle multi-class vs binary
        if n_classes == 2:
            # Binary classification
            results['precision_binary'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
            results['recall_binary'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
            results['f1_binary'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            if y_proba is not None:
                try:
                    results['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                except:
                    results['roc_auc'] = None
        elif n_classes > 2:
            # Multi-class classification
            try:
                if y_proba is not None:
                    results['roc_auc_macro'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                    results['roc_auc_weighted'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            except:
                results['roc_auc_macro'] = None
                results['roc_auc_weighted'] = None
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        results['labels'] = labels
        
        # Classification report
        results['classification_report'] = classification_report(
            y_true, y_pred, 
            target_names=labels,
            zero_division=0,
            output_dict=True
        )
        
        logger.info(f"{task_name} - Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")
        
        return results
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_name: str = "Regression"
    ) -> Dict[str, Any]:
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            task_name: Name of the task
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {task_name}...")
        
        results = {}
        
        # Basic metrics
        results['mae'] = mean_absolute_error(y_true, y_pred)
        results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        results['r2'] = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
        results['mape'] = mape
        
        # Error distribution
        errors = y_true - y_pred
        results['error_mean'] = np.mean(errors)
        results['error_std'] = np.std(errors)
        results['error_max'] = np.max(np.abs(errors))
        results['error_25'] = np.percentile(np.abs(errors), 25)
        results['error_75'] = np.percentile(np.abs(errors), 75)
        
        logger.info(f"{task_name} - R²: {results['r2']:.4f}, RMSE: {results['rmse']:.4f}")
        
        return results
    
    def cross_validate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = 'classification',
        cv_folds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Trained model
            X: Features
            y: Labels/targets
            task_type: 'classification' or 'regression'
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with CV results
        """
        cv_folds = cv_folds or self.cv_folds
        
        logger.info(f"Performing {cv_folds}-fold cross-validation on {type(model).__name__}...")
        
        # Select scoring metric
        if task_type == 'classification':
            scoring = 'accuracy'
        else:
            scoring = 'r2'
        
        # Choose CV splitter
        if task_type == 'classification' and len(np.unique(y)) > 1:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config.model.random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.config.model.random_state)
        
        # Perform cross-validation
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            
            results = {
                'scores': scores.tolist(),
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
            
            logger.info(f"Cross-validation {scoring}: {results['mean']:.4f} (+/- {results['std']:.4f})")
            return results
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {'error': str(e)}
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            labels: Label names
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        if labels is None:
            labels = [str(i) for i in range(cm.shape[0])]
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "ROC Curve",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 else 1
        
        if n_classes == 1:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
            
        else:
            # Multi-class: One-vs-Rest
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        title: str = "Feature Importance",
        top_k: int = 30,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot feature importance.
        
        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names
            title: Plot title
            top_k: Number of top features to show
            save_path: Path to save the plot
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return
        
        importance = model.feature_importances_
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Take top_k
        feature_importance_df = feature_importance_df.head(top_k)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=feature_importance_df,
            x='importance',
            y='feature',
            palette='viridis'
        )
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def compare_models(
        self,
        model_results: Dict[str, Dict[str, Any]],
        metric: str = 'accuracy',
        title: str = "Model Comparison",
        save_path: Optional[str] = None
    ) -> None:
        """
        Compare multiple models using bar chart.
        
        Args:
            model_results: Dictionary mapping model names to evaluation results
            metric: Metric to compare
            title: Plot title
            save_path: Path to save the plot
        """
        models = list(model_results.keys())
        scores = [model_results[m].get(metric, 0) for m in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, scores, color='steelblue', edgecolor='black')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.title(title)
        plt.xlabel('Model')
        plt.ylabel(metric.title())
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_prediction_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Prediction Distribution",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot distribution of predictions vs ground truth.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram of predictions
        axes[0].hist(y_true, bins=30, alpha=0.5, label='True', color='blue', edgecolor='black')
        axes[0].hist(y_pred, bins=30, alpha=0.5, label='Predicted', color='red', edgecolor='black')
        axes[0].set_title('Distribution Comparison')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Count')
        axes[0].legend()
        
        # Scatter plot: predicted vs true
        axes[1].scatter(y_true, y_pred, alpha=0.5, s=10)
        axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect prediction')
        axes[1].set_title('Predicted vs True')
        axes[1].set_xlabel('True')
        axes[1].set_ylabel('Predicted')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction distribution plot saved to {save_path}")
        else:
            plt.show()
        plt.close()