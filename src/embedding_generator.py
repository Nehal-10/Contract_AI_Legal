"""
Embedding generation module for the Contract Risk Intelligence System.
Uses LegalBERT to generate 768-dimensional dense embeddings from clause text.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModel
from loguru import logger
from tqdm import tqdm
import warnings

# Suppress tokenizer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

class EmbeddingGenerator:
    """
    Generates embeddings using LegalBERT for legal clause text.
    LegalBERT is used as a feature extractor only.
    """
    
    def __init__(self, config: Any):
        """
        Initialize EmbeddingGenerator with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model_name = config.model.embedding_model
        self.embedding_dim = config.model.embedding_dim
        self.max_length = config.model.max_clause_length
        self.device = self._get_device()
        self.tokenizer = None
        self.model = None
        self.is_initialized = False
        
    def _get_device(self) -> str:
        """
        Determine the device to use for inference.
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def initialize(self) -> None:
        """
        Initialize the tokenizer and model for embedding generation.
        Lazy initialization to avoid loading model if not needed.
        """
        if self.is_initialized:
            return
        
        logger.info(f"Initializing LegalBERT model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.is_initialized = True
            logger.info("LegalBERT model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LegalBERT model: {e}")
            raise
    
    def generate_embeddings(
        self, 
        texts: List[str],
        batch_size: int = 16,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of clause texts.
        
        Args:
            texts: List of clause text strings
            batch_size: Batch size for inference
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            logger.warning("No texts provided for embedding generation")
            return np.array([])
        
        # Ensure model is initialized
        if not self.is_initialized:
            self.initialize()
        
        # Filter out empty texts
        valid_indices = []
        valid_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)
        
        if not valid_texts:
            logger.warning("No valid texts after filtering empty strings")
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(valid_texts)} texts")
        
        # Generate embeddings in batches
        embeddings = []
        iterator = tqdm(range(0, len(valid_texts), batch_size), desc="Generating embeddings", disable=not show_progress)
        
        with torch.no_grad():
            for start_idx in iterator:
                batch_texts = valid_texts[start_idx:start_idx + batch_size]
                
                # Tokenize batch
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Mean pooling to get sentence embeddings
                batch_embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
                
                # Move to CPU and convert to numpy
                embeddings.append(batch_embeddings.cpu().numpy())
        
        # Combine all embeddings
        if embeddings:
            all_embeddings = np.vstack(embeddings)
            logger.info(f"Generated embeddings with shape: {all_embeddings.shape}")
            return all_embeddings
        else:
            logger.warning("No embeddings generated")
            return np.array([])
    
    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on token embeddings.
        
        Args:
            last_hidden_state: Last hidden state from model
            attention_mask: Attention mask for tokens
            
        Returns:
            Pooled embeddings tensor
        """
        # Expand attention mask to match hidden state dimensions
        attention_mask = attention_mask.unsqueeze(-1).float()
        
        # Apply attention mask and sum
        sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
        
        # Divide by sum of attention mask (number of tokens)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        pooled_embeddings = sum_embeddings / sum_mask
        
        return pooled_embeddings
    
    def generate_clause_embeddings(
        self, 
        clauses: List[Dict[str, Any]],
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for clauses and add to clause dictionaries.
        
        Args:
            clauses: List of clause dictionaries
            batch_size: Batch size for inference
            
        Returns:
            Updated clause dictionaries with 'embedding' key
        """
        if not clauses:
            return []
        
        # Extract texts
        texts = [c.get('text', '') for c in clauses]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts, batch_size=batch_size)
        
        # Add embeddings to clauses
        for i, clause in enumerate(clauses):
            if i < len(embeddings):
                clause['embedding'] = embeddings[i]
            else:
                # Fallback: zero vector if embedding generation failed
                clause['embedding'] = np.zeros(self.embedding_dim)
        
        logger.info(f"Added embeddings to {len(clauses)} clauses")
        return clauses
    
    def generate_embeddings_for_dataframe(
        self, 
        df: 'pd.DataFrame',
        text_column: str = 'clause_text',
        batch_size: int = 16
    ) -> np.ndarray:
        """
        Generate embeddings for texts in a DataFrame.
        
        Args:
            df: DataFrame with text column
            text_column: Name of column containing text
            batch_size: Batch size for inference
            
        Returns:
            numpy array of embeddings
        """
        texts = df[text_column].fillna('').astype(str).tolist()
        return self.generate_embeddings(texts, batch_size=batch_size)
    
    def get_embedding_dim(self) -> int:
        """
        Get the embedding dimension.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim
    
    def verify_model(self) -> bool:
        """
        Verify that the model is working correctly.
        
        Returns:
            True if model is working, False otherwise
        """
        try:
            test_texts = ["This is a test clause for embedding generation."]
            embeddings = self.generate_embeddings(test_texts)
            return embeddings.shape == (1, self.embedding_dim)
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return False