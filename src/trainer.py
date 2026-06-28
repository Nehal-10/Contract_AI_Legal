"""
Training pipeline for the Contract Risk Intelligence System.
Orchestrates the complete training workflow including data loading,
embedding generation, model training, and evaluation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import joblib
from loguru import logger
import time

from .config import get_config, Config
from .data_loader import DataLoader
from .embedding_generator import EmbeddingGenerator
from .models import ModelManager
from .evaluator import ModelEvaluator
from .utils import setup_logging, ensure_directory, save_model

class Trainer:
    """
    Orchestrates the complete training pipeline.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize Trainer with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.data_loader = DataLoader(self.config)
        self.embedding_generator = EmbeddingGenerator(self.config)
        self.model_manager = ModelManager(self.config)
        self.evaluator = ModelEvaluator(self.config)
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.split_data = None
        self.embeddings_train = None
        self.embeddings_test = None
        self.trained_models = None
        self.evaluation_results = None
        
    def run(self, save_models: bool = True) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            save_models: Whether to save trained models
            
        Returns:
            Dictionary with training results
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("Starting Contract Risk Intelligence System Training Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load and prepare data
        logger.info("Step 1: Loading and preparing data...")
        self.processed_data, self.split_data = self.data_loader.load_data_with_labels()
        logger.info(f"Data loaded: {len(self.processed_data)} samples")
        
        # Step 2: Generate embeddings
        logger.info("Step 2: Generating LegalBERT embeddings...")
        self.embedding_generator.initialize()
        
        # Get texts from training and test sets
        X_train_texts = self.split_data['X_train']['clause_text'].tolist()
        X_test_texts = self.split_data['X_test']['clause_text'].tolist()
        
        # Generate embeddings
        self.embeddings_train = self.embedding_generator.generate_embeddings(X_train_texts)
        self.embeddings_test = self.embedding_generator.generate_embeddings(X_test_texts)
        
        logger.info(f"Embeddings generated: Train shape {self.embeddings_train.shape}, Test shape {self.embeddings_test.shape}")
        
        # Step 3: Train models
        logger.info("Step 3: Training ML models...")
        
        y_clause_train = self.split_data['y_clause_train']
        y_risk_train = self.split_data['y_risk_train']
        y_comp_train = self.split_data['y_comp_train']
        y_score_train = self.split_data['y_score_train']
        
        self.trained_models = self.model_manager.train_models(
            X_train=self.embeddings_train,
            y_clause_train=y_clause_train,
            y_risk_train=y_risk_train,
            y_compliance_train=y_comp_train,
            y_score_train=y_score_train
        )
        
        # Step 4: Evaluate models
        logger.info("Step 4: Evaluating models...")
        self.evaluation_results = self.evaluate_models()
        
        # Step 5: Save models
        if save_models:
            logger.info("Step 5: Saving models...")
            model_dir = ensure_directory(self.config.paths.models_dir)
            self.model_manager.save_models(str(model_dir))
            
            # Save label encoders
            for name, encoder in self.data_loader.label_encoders.items():
                joblib.dump(encoder, model_dir / f"{name}_encoder.pkl")
                logger.info(f"Saved {name}_encoder to {model_dir / f'{name}_encoder.pkl'}")
            
            # Save category mapping
            import json
            category_mapping = self.data_loader.get_category_mapping()
            with open(model_dir / "category_mapping.json", 'w') as f:
                json.dump(category_mapping, f)
            
            risk_mapping = self.data_loader.get_risk_level_mapping()
            with open(model_dir / "risk_mapping.json", 'w') as f:
                json.dump(risk_mapping, f)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Training pipeline completed in {elapsed_time:.2f} seconds")
        logger.info("=" * 60)
        
        return {
            'processed_data': self.processed_data,
            'split_data': self.split_data,
            'embeddings_train': self.embeddings_train,
            'embeddings_test': self.embeddings_test,
            'trained_models': self.trained_models,
            'evaluation_results': self.evaluation_results,
            'elapsed_time': elapsed_time
        }
    
    def evaluate_models(self) -> Dict[str, Any]:
        """
        Evaluate trained models on test data.
        
        Returns:
            Dictionary with evaluation results
        """
        # Get test data
        y_clause_true = self.split_data['y_clause_test']
        y_risk_true = self.split_data['y_risk_test']
        y_comp_true = self.split_data['y_comp_test']
        y_score_true = self.split_data['y_score_test']
        
        # Get predictions
        X_test = self.embeddings_test
        predictions = self.model_manager.predict_all(X_test)
        
        results = {}
        
        # Evaluate clause classification
        if 'clause_category' in predictions:
            results['clause_classification'] = self.evaluator.evaluate_classification(
                y_true=y_clause_true,
                y_pred=predictions['clause_category'],
                task_name="Clause Classification",
                labels=self.data_loader.label_encoders['clause_category'].classes_
            )
        
        # Evaluate risk regression
        if 'risk_score' in predictions:
            results['risk_regression'] = self.evaluator.evaluate_regression(
                y_true=y_score_true,
                y_pred=predictions['risk_score'],
                task_name="Risk Score Regression"
            )
        
        # Evaluate risk level classification
        if 'risk_level' in predictions:
            results['risk_level_classification'] = self.evaluator.evaluate_classification(
                y_true=y_risk_true,
                y_pred=predictions['risk_level'],
                task_name="Risk Level Classification",
                labels=['Low', 'Medium', 'High']
            )
        
        # Evaluate compliance classification
        if 'compliance' in predictions:
            results['compliance_classification'] = self.evaluator.evaluate_classification(
                y_true=y_comp_true,
                y_pred=predictions['compliance'],
                task_name="Compliance Classification",
                labels=['Non-Compliant', 'Compliant']
            )
        
        # Cross-validation for the main models
        logger.info("Performing cross-validation on best models...")
        
        # CV for clause classifier
        if 'clause_classifier' in self.trained_models:
            cv_scores = self.evaluator.cross_validate_model(
                model=self.trained_models['clause_classifier'],
                X=self.embeddings_train,
                y=y_clause_true,
                task_type='classification'
            )
            results['clause_classifier_cv'] = cv_scores
        
        # CV for risk regressor
        if 'risk_regressor' in self.trained_models:
            cv_scores = self.evaluator.cross_validate_model(
                model=self.trained_models['risk_regressor'],
                X=self.embeddings_train,
                y=y_score_true,
                task_type='regression'
            )
            results['risk_regressor_cv'] = cv_scores
        
        return results
    
    def print_summary(self) -> None:
        """
        Print a summary of the training results.
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results available. Run training first.")
            return
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        
        # Data summary
        logger.info(f"Total samples: {len(self.processed_data)}")
        logger.info(f"Training samples: {len(self.split_data['X_train'])}")
        logger.info(f"Test samples: {len(self.split_data['X_test'])}")
        logger.info(f"Embedding dimension: {self.config.model.embedding_dim}")
        
        # Model summary
        logger.info("\n--- Model Performance ---")
        
        for task_name, metrics in self.evaluation_results.items():
            if isinstance(metrics, dict):
                if 'accuracy' in metrics:
                    logger.info(f"{task_name}: Accuracy = {metrics['accuracy']:.4f}")
                if 'r2' in metrics:
                    logger.info(f"{task_name}: R² = {metrics['r2']:.4f}")
                    logger.info(f"{task_name}: RMSE = {metrics['rmse']:.4f}")
        
        logger.info("=" * 60)
    
    def save_results(self, output_dir: str = "outputs") -> None:
        """
        Save training results to disk.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        import json
        
        # Save evaluation results
        if self.evaluation_results:
            # Convert numpy arrays to lists for JSON serialization
            eval_results = {}
            for k, v in self.evaluation_results.items():
                if isinstance(v, dict):
                    eval_results[k] = {}
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, np.ndarray):
                            eval_results[k][sub_k] = sub_v.tolist()
                        else:
                            eval_results[k][sub_k] = sub_v
                else:
                    eval_results[k] = v
            
            with open(output_path / "evaluation_results.json", 'w') as f:
                json.dump(eval_results, f, indent=2)
            logger.info(f"Evaluation results saved to {output_path / 'evaluation_results.json'}")
        
        # Save model info
        model_info = self.model_manager.get_model_info()
        with open(output_path / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        
        # Create summary report
        self._create_summary_report(output_path)
    
    def _create_summary_report(self, output_dir: Path) -> None:
        """
        Create a summary report.
        
        Args:
            output_dir: Directory to save report
        """
        report_lines = []
        report_lines.append("CONTRACT RISK INTELLIGENCE SYSTEM - TRAINING REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        report_lines.append("DATA SUMMARY:")
        report_lines.append(f"  Total samples: {len(self.processed_data)}")
        report_lines.append(f"  Training samples: {len(self.split_data['X_train'])}")
        report_lines.append(f"  Test samples: {len(self.split_data['X_test'])}")
        report_lines.append(f"  Embedding dimension: {self.config.model.embedding_dim}")
        report_lines.append("")
        
        report_lines.append("MODEL PERFORMANCE:")
        
        if self.evaluation_results:
            for task_name, metrics in self.evaluation_results.items():
                if isinstance(metrics, dict):
                    report_lines.append(f"  {task_name}:")
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            report_lines.append(f"    {metric_name}: {value:.4f}")
                        elif isinstance(value, dict):
                            for sub_k, sub_v in value.items():
                                if isinstance(sub_v, (int, float)):
                                    report_lines.append(f"    {sub_k}: {sub_v:.4f}")
        report_lines.append("")
        report_lines.append("=" * 60)
        
        # Write report
        report_file = output_dir / "training_report.txt"
        with open(report_file, 'w') as f:
            f.write("\n".join(report_lines))
        logger.info(f"Training report saved to {report_file}")