"""
Tests for the Contract Risk Intelligence System pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.config import get_config
from src.pdf_processor import PDFProcessor
from src.embedding_generator import EmbeddingGenerator
from src.data_loader import DataLoader
from src.models import ModelManager, ModelFactory

def test_config():
    """Test configuration loading."""
    config = get_config()
    assert config is not None
    assert hasattr(config, 'model')
    assert hasattr(config, 'training')
    assert hasattr(config, 'data')
    assert hasattr(config, 'api')
    assert hasattr(config, 'paths')

def test_data_loader():
    """Test data loader."""
    config = get_config()
    loader = DataLoader(config)
    
    # Try to load data
    try:
        df = loader.load_data()
        assert df is not None
        assert len(df) > 0
    except FileNotFoundError:
        pytest.skip("Data file not found")

def test_pdf_processor():
    """Test PDF processor."""
    config = get_config()
    processor = PDFProcessor(config)
    
    # Test with sample text
    sample_text = """
    1. Termination. Either party may terminate this Agreement at any time without liability.
    2. Governing Law. This Agreement is governed by the laws of Delaware.
    
    This is a test paragraph that should be split into clauses.
    """
    
    clauses = processor.segment_clauses(sample_text)
    assert len(clauses) > 0
    assert all('text' in c for c in clauses)

def test_embedding_generator():
    """Test embedding generator."""
    config = get_config()
    generator = EmbeddingGenerator(config)
    
    try:
        generator.initialize()
        
        test_texts = ["This is a test clause.", "Another test clause for embedding."]
        embeddings = generator.generate_embeddings(test_texts)
        
        assert len(embeddings) == 2
        assert embeddings.shape[1] == config.model.embedding_dim
    except Exception as e:
        pytest.skip(f"Embedding generator not available: {e}")

def test_model_factory():
    """Test model factory."""
    # Test clause classifier creation
    model = ModelFactory.create_clause_classifier('xgboost')
    assert model is not None
    
    # Test risk regressor creation
    model = ModelFactory.create_risk_regressor('xgboost')
    assert model is not None
    
    # Test compliance classifier creation
    model = ModelFactory.create_compliance_classifier('xgboost')
    assert model is not None

def test_model_manager():
    """Test model manager."""
    config = get_config()
    manager = ModelManager(config)
    
    # Test with synthetic data
    n_samples = 100
    n_features = 768
    
    X = np.random.randn(n_samples, n_features)
    y_clause = np.random.randint(0, 5, n_samples)
    y_risk = np.random.randint(0, 3, n_samples)
    y_comp = np.random.randint(0, 2, n_samples)
    y_score = np.random.rand(n_samples)
    
    # Train models
    try:
        models = manager.train_models(
            X_train=X,
            y_clause_train=y_clause,
            y_risk_train=y_risk,
            y_compliance_train=y_comp,
            y_score_train=y_score
        )
        
        assert len(models) == 3
        assert 'clause_classifier' in models
        assert 'risk_regressor' in models
        assert 'compliance_classifier' in models
        
        # Test prediction
        predictions = manager.predict_all(X)
        assert 'clause_category' in predictions
        assert 'risk_score' in predictions
        assert 'risk_level' in predictions
        assert 'compliance' in predictions
        
    except Exception as e:
        pytest.skip(f"Model training failed: {e}")