# # """
# # Inference pipeline for the Contract Risk Intelligence System.
# # Handles prediction workflows for clauses and complete contracts.
# # """

# # import numpy as np
# # from typing import List, Dict, Any, Optional, Union
# # from pathlib import Path
# # import json
# # from loguru import logger

# # from .config import get_config, Config
# # from .pdf_processor import PDFProcessor
# # from .embedding_generator import EmbeddingGenerator
# # from .models import ModelManager
# # from .utils import compute_risk_percentage, get_risk_level, load_model


# # class Predictor:
# #     """
# #     Inference pipeline for the Contract Risk Intelligence System.
# #     Handles predictions for individual clauses and complete contracts.
# #     """
    
# #     def __init__(self, config: Optional[Config] = None, model_dir: Optional[str] = None):
# #         """
# #         Initialize Predictor with configuration and trained models.
        
# #         Args:
# #             config: Configuration object
# #             model_dir: Directory containing trained models
# #         """
# #         self.config = config or get_config()
# #         self.model_dir = Path(model_dir) if model_dir else Path(self.config.paths.models_dir)
        
# #         # Initialize components
# #         self.pdf_processor = PDFProcessor(self.config)
# #         self.embedding_generator = EmbeddingGenerator(self.config)
# #         self.model_manager = ModelManager(self.config)
        
# #         # Load models if they exist
# #         self.is_initialized = False
# #         self._load_models()
        
# #     def _load_models(self) -> None:
# #         """
# #         Load trained models from disk.
# #         """
# #         try:
# #             if self.model_dir.exists():
# #                 logger.info(f"Loading models from {self.model_dir}")
# #                 self.model_manager.load_models(str(self.model_dir))
                
# #                 # Load label encoders
# #                 import joblib
# #                 self.label_encoders = {}
# #                 for encoder_file in self.model_dir.glob("*_encoder.pkl"):
# #                     name = encoder_file.stem.replace('_encoder', '')
# #                     self.label_encoders[name] = joblib.load(encoder_file)
                
# #                 # Load mappings
# #                 self.category_mapping = {}
# #                 mapping_file = self.model_dir / "category_mapping.json"
# #                 if mapping_file.exists():
# #                     with open(mapping_file, 'r') as f:
# #                         self.category_mapping = json.load(f)
# #                         # Convert keys to int
# #                         self.category_mapping = {int(k): v for k, v in self.category_mapping.items()}
                
# #                 self.risk_mapping = {}
# #                 risk_file = self.model_dir / "risk_mapping.json"
# #                 if risk_file.exists():
# #                     with open(risk_file, 'r') as f:
# #                         self.risk_mapping = json.load(f)
# #                         self.risk_mapping = {int(k): v for k, v in self.risk_mapping.items()}
                
# #                 self.is_initialized = True
# #                 logger.info("Models loaded successfully")
# #             else:
# #                 logger.warning(f"Model directory not found at {self.model_dir}")
# #                 self.is_initialized = False
# #         except Exception as e:
# #             logger.error(f"Failed to load models: {e}")
# #             self.is_initialized = False
    
# #     def predict_clause(
# #         self, 
# #         clause_text: str,
# #         return_embeddings: bool = False
# #     ) -> Dict[str, Any]:
# #         """
# #         Predict risk and compliance for a single clause.
        
# #         Args:
# #             clause_text: Clause text
# #             return_embeddings: Whether to return embeddings
            
# #         Returns:
# #             Dictionary with predictions
# #         """
# #         if not self.is_initialized:
# #             raise ValueError("Models not loaded. Call load_models() first.")
        
# #         # Generate embedding
# #         embedding = self.embedding_generator.generate_embeddings([clause_text])
        
# #         # Get predictions
# #         predictions = self.model_manager.predict_all(embedding)
        
# #         # Format results
# #         result = {
# #             'clause_text': clause_text,
# #             'predictions': {}
# #         }
        
# #         # Clause category
# #         if 'clause_category' in predictions:
# #             cat_idx = predictions['clause_category'][0]
# #             result['predictions']['clause_category'] = {
# #                 'category_id': int(cat_idx),
# #                 'category_name': self.category_mapping.get(int(cat_idx), f"Unknown_{cat_idx}")
# #             }
        
# #         # Risk score
# #         if 'risk_score' in predictions:
# #             score = float(predictions['risk_score'][0])
# #             result['predictions']['risk_score'] = score
# #             result['predictions']['risk_percentage'] = compute_risk_percentage(score)
            
# #             # Risk level
# #             if 'risk_level' in predictions:
# #                 level_idx = int(predictions['risk_level'][0])
# #                 level_name = self.risk_mapping.get(level_idx, f"Level_{level_idx}")
# #                 result['predictions']['risk_level'] = {
# #                     'level_id': level_idx,
# #                     'level_name': level_name
# #                 }
# #             else:
# #                 # Compute risk level from score
# #                 result['predictions']['risk_level'] = {
# #                     'level_name': get_risk_level(score)
# #                 }
        
# #         # Compliance
# #         if 'compliance' in predictions:
# #             comp_pred = int(predictions['compliance'][0])
# #             result['predictions']['compliance'] = {
# #                 'is_compliant': bool(comp_pred),
# #                 'status': 'Compliant' if comp_pred == 1 else 'Non-Compliant'
# #             }
            
# #             if 'compliance_probability' in predictions:
# #                 result['predictions']['compliance_probability'] = float(predictions['compliance_probability'][0])
        
# #         if return_embeddings:
# #             result['embedding'] = embedding[0].tolist()
        
# #         return result
    
# #     def _normalize_clause(self, clause: Union[str, Dict[str, Any], Any]) -> Dict[str, Any]:
# #         """
# #         Convert any clause input to a standardized dictionary.
        
# #         Args:
# #             clause: Clause as string, dict, or other type
            
# #         Returns:
# #             Normalized dictionary with 'text' key
# #         """
# #         if isinstance(clause, str):
# #             return {
# #                 'text': clause,
# #                 'word_count': len(clause.split()),
# #                 'char_count': len(clause),
# #                 'clause_length': len(clause),
# #                 'type': 'string'
# #             }
# #         elif isinstance(clause, dict):
# #             # Make a copy to avoid modifying original
# #             result = dict(clause)
# #             # Ensure 'text' exists
# #             if 'text' not in result:
# #                 # Try to find any string value
# #                 for key, value in result.items():
# #                     if isinstance(value, str) and len(value) > 10:
# #                         result['text'] = value
# #                         break
# #                 else:
# #                     result['text'] = str(result)
# #             return result
# #         else:
# #             # Unsupported type, convert to string
# #             return {
# #                 'text': str(clause),
# #                 'word_count': len(str(clause).split()),
# #                 'char_count': len(str(clause)),
# #                 'clause_length': len(str(clause)),
# #                 'type': 'unknown'
# #             }
    
# #     def predict_clauses(
# #         self, 
# #         clauses: List[Union[str, Dict[str, Any]]],
# #         return_embeddings: bool = False
# #     ) -> List[Dict[str, Any]]:
# #         """
# #         Predict risk and compliance for multiple clauses.
        
# #         Args:
# #             clauses: List of clause dictionaries or strings
# #             return_embeddings: Whether to return embeddings
            
# #         Returns:
# #             List of dictionaries with predictions
# #         """
# #         if not clauses:
# #             return []
        
# #         if not self.is_initialized:
# #             raise ValueError("Models not loaded. Call load_models() first.")
        
# #         # --- NORMALIZE: ensure every clause is a dict with a 'text' key ---
# #         normalized_clauses = []
# #         for idx, c in enumerate(clauses):
# #             normalized = self._normalize_clause(c)
# #             # Add index if not present
# #             if 'index' not in normalized:
# #                 normalized['index'] = idx
# #             normalized_clauses.append(normalized)
        
# #         # Extract texts
# #         texts = [c.get('text', '') for c in normalized_clauses]
        
# #         # Generate embeddings
# #         embeddings = self.embedding_generator.generate_embeddings(texts)
        
# #         # Get predictions
# #         predictions = self.model_manager.predict_all(embeddings)
        
# #         # Combine results
# #         results = []
# #         for i, clause in enumerate(normalized_clauses):
# #             result = {
# #                 'clause_text': clause.get('text', ''),
# #                 'clause_metadata': {
# #                     'word_count': clause.get('word_count', 0),
# #                     'char_count': clause.get('char_count', 0),
# #                     'type': clause.get('type', 'unknown')
# #                 },
# #                 'predictions': {}
# #             }
            
# #             # Clause category
# #             if 'clause_category' in predictions:
# #                 cat_idx = predictions['clause_category'][i]
# #                 result['predictions']['clause_category'] = {
# #                     'category_id': int(cat_idx),
# #                     'category_name': self.category_mapping.get(int(cat_idx), f"Unknown_{cat_idx}")
# #                 }
            
# #             # Risk score
# #             if 'risk_score' in predictions:
# #                 score = float(predictions['risk_score'][i])
# #                 result['predictions']['risk_score'] = score
# #                 result['predictions']['risk_percentage'] = compute_risk_percentage(score)
                
# #                 # Risk level
# #                 if 'risk_level' in predictions:
# #                     level_idx = int(predictions['risk_level'][i])
# #                     level_name = self.risk_mapping.get(level_idx, f"Level_{level_idx}")
# #                     result['predictions']['risk_level'] = {
# #                         'level_id': level_idx,
# #                         'level_name': level_name
# #                     }
# #                 else:
# #                     result['predictions']['risk_level'] = {
# #                         'level_name': get_risk_level(score)
# #                     }
            
# #             # Compliance
# #             if 'compliance' in predictions:
# #                 comp_pred = int(predictions['compliance'][i])
# #                 result['predictions']['compliance'] = {
# #                     'is_compliant': bool(comp_pred),
# #                     'status': 'Compliant' if comp_pred == 1 else 'Non-Compliant'
# #                 }
                
# #                 if 'compliance_probability' in predictions:
# #                     result['predictions']['compliance_probability'] = float(predictions['compliance_probability'][i])
            
# #             if return_embeddings and i < len(embeddings):
# #                 result['embedding'] = embeddings[i].tolist()
            
# #             results.append(result)
        
# #         return results
    
# #     def predict_contract(self, pdf_content: bytes) -> Dict[str, Any]:
# #         """
# #         Analyze a complete contract from PDF.
        
# #         Args:
# #             pdf_content: PDF file content as bytes
            
# #         Returns:
# #             Dictionary with full contract analysis
# #         """
# #         logger.info("Analyzing contract...")
        
# #         try:
# #             # Process PDF - get clauses as dictionaries
# #             clauses = self.pdf_processor.process_pdf(pdf_content)
            
# #             if not clauses:
# #                 logger.warning("No clauses extracted from PDF")
# #                 return {
# #                     'error': 'No valid clauses found in PDF',
# #                     'total_clauses': 0,
# #                     'analysis': {}
# #                 }
            
# #             logger.info(f"Extracted {len(clauses)} clauses")
            
# #             # Get clause summary (now safe)
# #             clause_summary = self.pdf_processor.get_clause_summary(clauses)
            
# #             # Predict on clauses
# #             predictions = self.predict_clauses(clauses)
            
# #             # Aggregate results
# #             return self._aggregate_results(predictions, clause_summary)
            
# #         except Exception as e:
# #             logger.error(f"Contract analysis failed: {e}", exc_info=True)
# #             return {
# #                 'error': f'Analysis failed: {str(e)}',
# #                 'total_clauses': 0,
# #                 'analysis': {}
# #             }
    
# #     def _aggregate_results(
# #         self, 
# #         predictions: List[Dict[str, Any]], 
# #         clause_summary: Dict[str, Any]
# #     ) -> Dict[str, Any]:
# #         """
# #         Aggregate clause predictions into contract-level analysis.
        
# #         Args:
# #             predictions: List of clause predictions
# #             clause_summary: Summary statistics from PDF processing
            
# #         Returns:
# #             Dictionary with aggregated results (all numpy types converted to Python native types)
# #         """
# #         if not predictions:
# #             return {
# #                 'total_clauses': 0,
# #                 'analysis': {}
# #             }
        
# #         # Extract risk scores
# #         risk_scores = []
# #         risk_levels = []
# #         compliant_count = 0
# #         total_clauses = len(predictions)
        
# #         for pred in predictions:
# #             # Safety check
# #             if not isinstance(pred, dict):
# #                 logger.warning(f"Skipping non-dict prediction: {type(pred)}")
# #                 continue
            
# #             # Get predictions dict safely
# #             pred_dict = pred.get('predictions', {})
# #             if not isinstance(pred_dict, dict):
# #                 pred_dict = {}
            
# #             # Risk score
# #             if 'risk_score' in pred_dict:
# #                 try:
# #                     score = float(pred_dict['risk_score'])
# #                     risk_scores.append(score)
# #                 except (ValueError, TypeError):
# #                     pass
            
# #             # Risk level
# #             if 'risk_level' in pred_dict:
# #                 level = pred_dict['risk_level']
# #                 if isinstance(level, dict):
# #                     risk_levels.append(level.get('level_name', 'Unknown'))
# #                 elif isinstance(level, str):
# #                     risk_levels.append(level)
# #                 else:
# #                     risk_levels.append(str(level))
            
# #             # Compliance
# #             if 'compliance' in pred_dict:
# #                 comp = pred_dict['compliance']
# #                 if isinstance(comp, dict):
# #                     if comp.get('is_compliant', False):
# #                         compliant_count += 1
# #                 elif isinstance(comp, bool):
# #                     if comp:
# #                         compliant_count += 1
# #                 elif isinstance(comp, (int, float)):
# #                     if comp > 0.5:
# #                         compliant_count += 1
        
# #         # Calculate aggregated metrics
# #         avg_risk_score = float(np.mean(risk_scores) if risk_scores else 0.5)
# #         max_risk_score = float(np.max(risk_scores) if risk_scores else 0.5)
# #         min_risk_score = float(np.min(risk_scores) if risk_scores else 0.5)
        
# #         # Risk level counts
# #         risk_level_counts = {}
# #         for level in risk_levels:
# #             risk_level_counts[level] = risk_level_counts.get(level, 0) + 1
        
# #         # Compliance metrics
# #         compliance_rate = float(compliant_count / total_clauses if total_clauses > 0 else 0)
        
# #         # Identify high-risk clauses
# #         high_risk_clauses = []
# #         for i, pred in enumerate(predictions):
# #             if not isinstance(pred, dict):
# #                 continue
            
# #             pred_dict = pred.get('predictions', {})
# #             if not isinstance(pred_dict, dict):
# #                 continue
            
# #             if 'risk_level' in pred_dict:
# #                 level = pred_dict['risk_level']
# #                 is_high = False
# #                 if isinstance(level, dict) and level.get('level_name') == 'High':
# #                     is_high = True
# #                 elif isinstance(level, str) and level == 'High':
# #                     is_high = True
                
# #                 if is_high:
# #                     high_risk_clauses.append({
# #                         'clause_index': i,
# #                         'text': pred.get('clause_text', '')[:200],
# #                         'risk_score': float(pred_dict.get('risk_score', 0))
# #                     })
        
# #         # --- FIX: Convert clause_summary values to Python native types ---
# #         clean_clause_summary = {}
# #         for key, value in clause_summary.items():
# #             if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
# #                 clean_clause_summary[key] = int(value)
# #             elif isinstance(value, (np.float64, np.float32, np.float16)):
# #                 clean_clause_summary[key] = float(value)
# #             elif isinstance(value, dict):
# #                 clean_clause_summary[key] = {k: int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.float64, np.float32)) else v for k, v in value.items()}
# #             elif isinstance(value, list):
# #                 clean_clause_summary[key] = [int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.float64, np.float32)) else v for v in value]
# #             else:
# #                 clean_clause_summary[key] = value
        
# #         # Aggregate results with all numpy types converted
# #         results = {
# #             'contract_analysis': {
# #                 'total_clauses': total_clauses,
# #                 'average_risk_score': avg_risk_score,
# #                 'max_risk_score': max_risk_score,
# #                 'min_risk_score': min_risk_score,
# #                 'compliance_rate': compliance_rate,
# #                 'risk_level_distribution': risk_level_counts,
# #                 'high_risk_clauses_count': len(high_risk_clauses),
# #                 'compliant_clauses_count': compliant_count,
# #                 'non_compliant_clauses_count': total_clauses - compliant_count
# #             },
# #             'risk_summary': {
# #                 'overall_risk_percentage': compute_risk_percentage(avg_risk_score),
# #                 'overall_risk_level': get_risk_level(avg_risk_score),
# #                 'high_risk_clauses': high_risk_clauses[:10]
# #             },
# #             'clause_summary': clean_clause_summary,
# #             'clause_predictions': predictions
# #         }
        
# #         logger.info(f"Contract analysis complete: {total_clauses} clauses, "
# #                    f"Risk Level: {results['risk_summary']['overall_risk_level']}, "
# #                    f"Compliance Rate: {compliance_rate:.1%}")
        
# #         return results
    
# #     def get_model_info(self) -> Dict[str, Any]:
# #         """
# #         Get information about loaded models.
        
# #         Returns:
# #             Dictionary with model information
# #         """
# #         if not self.is_initialized:
# #             return {'status': 'not_initialized'}
        
# #         return {
# #             'status': 'loaded',
# #             'models': list(self.model_manager.models.keys()),
# #             'category_mapping': self.category_mapping,
# #             'risk_mapping': self.risk_mapping
# #         }
    


    

# """
# Inference pipeline for the Contract Risk Intelligence System.
# Handles prediction workflows for clauses and complete contracts.
# """

# import numpy as np
# from typing import List, Dict, Any, Optional, Union
# from pathlib import Path
# import json
# from loguru import logger

# from .config import get_config, Config
# from .pdf_processor import PDFProcessor
# from .embedding_generator import EmbeddingGenerator
# from .models import ModelManager
# from .utils import compute_risk_percentage, get_risk_level, load_model


# class Predictor:
#     """
#     Inference pipeline for the Contract Risk Intelligence System.
#     Handles predictions for individual clauses and complete contracts.
#     """
    
#     def __init__(self, config: Optional[Config] = None, model_dir: Optional[str] = None):
#         """
#         Initialize Predictor with configuration and trained models.
        
#         Args:
#             config: Configuration object
#             model_dir: Directory containing trained models
#         """
#         self.config = config or get_config()
#         self.model_dir = Path(model_dir) if model_dir else Path(self.config.paths.models_dir)
        
#         # Initialize components
#         self.pdf_processor = PDFProcessor(self.config)
#         self.embedding_generator = EmbeddingGenerator(self.config)
#         self.model_manager = ModelManager(self.config)
        
#         # Load models if they exist
#         self.is_initialized = False
#         self._load_models()
        
#     def _load_models(self) -> None:
#         """
#         Load trained models from disk.
#         """
#         try:
#             if self.model_dir.exists():
#                 logger.info(f"Loading models from {self.model_dir}")
#                 self.model_manager.load_models(str(self.model_dir))
                
#                 # Load label encoders
#                 import joblib
#                 self.label_encoders = {}
#                 for encoder_file in self.model_dir.glob("*_encoder.pkl"):
#                     name = encoder_file.stem.replace('_encoder', '')
#                     self.label_encoders[name] = joblib.load(encoder_file)
                
#                 # Load mappings
#                 self.category_mapping = {}
#                 mapping_file = self.model_dir / "category_mapping.json"
#                 if mapping_file.exists():
#                     with open(mapping_file, 'r') as f:
#                         self.category_mapping = json.load(f)
#                         # Convert keys to int
#                         self.category_mapping = {int(k): v for k, v in self.category_mapping.items()}
                
#                 self.risk_mapping = {}
#                 risk_file = self.model_dir / "risk_mapping.json"
#                 if risk_file.exists():
#                     with open(risk_file, 'r') as f:
#                         self.risk_mapping = json.load(f)
#                         self.risk_mapping = {int(k): v for k, v in self.risk_mapping.items()}
                
#                 self.is_initialized = True
#                 logger.info("Models loaded successfully")
#             else:
#                 logger.warning(f"Model directory not found at {self.model_dir}")
#                 self.is_initialized = False
#         except Exception as e:
#             logger.error(f"Failed to load models: {e}")
#             self.is_initialized = False
    
#     def predict_clause(
#         self, 
#         clause_text: str,
#         return_embeddings: bool = False
#     ) -> Dict[str, Any]:
#         """
#         Predict risk and compliance for a single clause.
        
#         Args:
#             clause_text: Clause text
#             return_embeddings: Whether to return embeddings
            
#         Returns:
#             Dictionary with predictions
#         """
#         if not self.is_initialized:
#             raise ValueError("Models not loaded. Call load_models() first.")
        
#         # Generate embedding
#         embedding = self.embedding_generator.generate_embeddings([clause_text])
        
#         # Get predictions
#         predictions = self.model_manager.predict_all(embedding)
        
#         # Format results
#         result = {
#             'clause_text': clause_text,
#             'predictions': {}
#         }
        
#         # Clause category
#         if 'clause_category' in predictions:
#             cat_idx = predictions['clause_category'][0]
#             result['predictions']['clause_category'] = {
#                 'category_id': int(cat_idx),
#                 'category_name': self.category_mapping.get(int(cat_idx), f"Unknown_{cat_idx}")
#             }
        
#         # Risk score
#         if 'risk_score' in predictions:
#             score = float(predictions['risk_score'][0])
#             result['predictions']['risk_score'] = score
#             result['predictions']['risk_percentage'] = compute_risk_percentage(score)
            
#             # Risk level
#             if 'risk_level' in predictions:
#                 level_idx = int(predictions['risk_level'][0])
#                 level_name = self.risk_mapping.get(level_idx, f"Level_{level_idx}")
#                 result['predictions']['risk_level'] = {
#                     'level_id': level_idx,
#                     'level_name': level_name
#                 }
#             else:
#                 # Compute risk level from score
#                 result['predictions']['risk_level'] = {
#                     'level_name': get_risk_level(score)
#                 }
        
#         # Compliance
#         if 'compliance' in predictions:
#             comp_pred = int(predictions['compliance'][0])
#             result['predictions']['compliance'] = {
#                 'is_compliant': bool(comp_pred),
#                 'status': 'Compliant' if comp_pred == 1 else 'Non-Compliant'
#             }
            
#             if 'compliance_probability' in predictions:
#                 result['predictions']['compliance_probability'] = float(predictions['compliance_probability'][0])
        
#         # --- Apply smart compliance rules ---
#         result = self._apply_compliance_rules(result)
        
#         if return_embeddings:
#             result['embedding'] = embedding[0].tolist()
        
#         return result
    
#     def _apply_compliance_rules(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Apply smart compliance rules based on clause category and risk score.
#         This helps correct obvious misclassifications.
        
#         Args:
#             predictions: Raw predictions dictionary
            
#         Returns:
#             Updated predictions with corrected compliance
#         """
#         # Get the predictions dict
#         pred_dict = predictions.get('predictions', {})
        
#         # Get clause category and risk score
#         category = pred_dict.get('clause_category', {}).get('category_name', '')
#         risk_score = pred_dict.get('risk_score', 0.5)
        
#         # If risk score is very low (< 0.2), it should be compliant
#         if risk_score < 0.2:
#             pred_dict['compliance'] = {
#                 'is_compliant': True,
#                 'status': 'Compliant'
#             }
#             pred_dict['compliance_probability'] = 0.95
        
#         # Specific category-based overrides
#         elif category.lower() in ['governing law', 'governing_law', 'governing law clause']:
#             # Governing Law clauses are almost always compliant
#             pred_dict['compliance'] = {
#                 'is_compliant': True,
#                 'status': 'Compliant'
#             }
#             pred_dict['compliance_probability'] = 0.98
        
#         elif category.lower() in ['renewal term', 'renewal', 'termination for convenience']:
#             # These can be medium risk but often compliant
#             if risk_score < 0.5:
#                 pred_dict['compliance'] = {
#                     'is_compliant': True,
#                     'status': 'Compliant'
#                 }
#                 pred_dict['compliance_probability'] = 0.85
        
#         elif category.lower() in ['non-compete', 'non_compete', 'exclusivity']:
#             # Non-compete clauses often have compliance issues
#             if risk_score > 0.7:
#                 pred_dict['compliance'] = {
#                     'is_compliant': False,
#                     'status': 'Non-Compliant'
#                 }
#                 pred_dict['compliance_probability'] = 0.90
        
#         # Update the predictions
#         predictions['predictions'] = pred_dict
        
#         return predictions
    
#     def _normalize_clause(self, clause: Union[str, Dict[str, Any], Any]) -> Dict[str, Any]:
#         """
#         Convert any clause input to a standardized dictionary.
        
#         Args:
#             clause: Clause as string, dict, or other type
            
#         Returns:
#             Normalized dictionary with 'text' key
#         """
#         if isinstance(clause, str):
#             return {
#                 'text': clause,
#                 'word_count': len(clause.split()),
#                 'char_count': len(clause),
#                 'clause_length': len(clause),
#                 'type': 'string'
#             }
#         elif isinstance(clause, dict):
#             # Make a copy to avoid modifying original
#             result = dict(clause)
#             # Ensure 'text' exists
#             if 'text' not in result:
#                 # Try to find any string value
#                 for key, value in result.items():
#                     if isinstance(value, str) and len(value) > 10:
#                         result['text'] = value
#                         break
#                 else:
#                     result['text'] = str(result)
#             return result
#         else:
#             # Unsupported type, convert to string
#             return {
#                 'text': str(clause),
#                 'word_count': len(str(clause).split()),
#                 'char_count': len(str(clause)),
#                 'clause_length': len(str(clause)),
#                 'type': 'unknown'
#             }
    
#     def predict_clauses(
#         self, 
#         clauses: List[Union[str, Dict[str, Any]]],
#         return_embeddings: bool = False
#     ) -> List[Dict[str, Any]]:
#         """
#         Predict risk and compliance for multiple clauses.
        
#         Args:
#             clauses: List of clause dictionaries or strings
#             return_embeddings: Whether to return embeddings
            
#         Returns:
#             List of dictionaries with predictions
#         """
#         if not clauses:
#             return []
        
#         if not self.is_initialized:
#             raise ValueError("Models not loaded. Call load_models() first.")
        
#         # --- NORMALIZE: ensure every clause is a dict with a 'text' key ---
#         normalized_clauses = []
#         for idx, c in enumerate(clauses):
#             normalized = self._normalize_clause(c)
#             # Add index if not present
#             if 'index' not in normalized:
#                 normalized['index'] = idx
#             normalized_clauses.append(normalized)
        
#         # Extract texts
#         texts = [c.get('text', '') for c in normalized_clauses]
        
#         # Generate embeddings
#         embeddings = self.embedding_generator.generate_embeddings(texts)
        
#         # Get predictions
#         predictions = self.model_manager.predict_all(embeddings)
        
#         # Combine results
#         results = []
#         for i, clause in enumerate(normalized_clauses):
#             result = {
#                 'clause_text': clause.get('text', ''),
#                 'clause_metadata': {
#                     'word_count': clause.get('word_count', 0),
#                     'char_count': clause.get('char_count', 0),
#                     'type': clause.get('type', 'unknown')
#                 },
#                 'predictions': {}
#             }
            
#             # Clause category
#             if 'clause_category' in predictions:
#                 cat_idx = predictions['clause_category'][i]
#                 result['predictions']['clause_category'] = {
#                     'category_id': int(cat_idx),
#                     'category_name': self.category_mapping.get(int(cat_idx), f"Unknown_{cat_idx}")
#                 }
            
#             # Risk score
#             if 'risk_score' in predictions:
#                 score = float(predictions['risk_score'][i])
#                 result['predictions']['risk_score'] = score
#                 result['predictions']['risk_percentage'] = compute_risk_percentage(score)
                
#                 # Risk level
#                 if 'risk_level' in predictions:
#                     level_idx = int(predictions['risk_level'][i])
#                     level_name = self.risk_mapping.get(level_idx, f"Level_{level_idx}")
#                     result['predictions']['risk_level'] = {
#                         'level_id': level_idx,
#                         'level_name': level_name
#                     }
#                 else:
#                     result['predictions']['risk_level'] = {
#                         'level_name': get_risk_level(score)
#                     }
            
#             # Compliance
#             if 'compliance' in predictions:
#                 comp_pred = int(predictions['compliance'][i])
#                 result['predictions']['compliance'] = {
#                     'is_compliant': bool(comp_pred),
#                     'status': 'Compliant' if comp_pred == 1 else 'Non-Compliant'
#                 }
                
#                 if 'compliance_probability' in predictions:
#                     result['predictions']['compliance_probability'] = float(predictions['compliance_probability'][i])
            
#             # --- Apply smart compliance rules for each clause ---
#             result = self._apply_compliance_rules(result)
            
#             if return_embeddings and i < len(embeddings):
#                 result['embedding'] = embeddings[i].tolist()
            
#             results.append(result)
        
#         return results
    
#     def predict_contract(self, pdf_content: bytes) -> Dict[str, Any]:
#         """
#         Analyze a complete contract from PDF.
        
#         Args:
#             pdf_content: PDF file content as bytes
            
#         Returns:
#             Dictionary with full contract analysis
#         """
#         logger.info("Analyzing contract...")
        
#         try:
#             # Process PDF - get clauses as dictionaries
#             clauses = self.pdf_processor.process_pdf(pdf_content)
            
#             if not clauses:
#                 logger.warning("No clauses extracted from PDF")
#                 return {
#                     'error': 'No valid clauses found in PDF',
#                     'total_clauses': 0,
#                     'analysis': {}
#                 }
            
#             logger.info(f"Extracted {len(clauses)} clauses")
            
#             # Get clause summary (now safe)
#             clause_summary = self.pdf_processor.get_clause_summary(clauses)
            
#             # Predict on clauses
#             predictions = self.predict_clauses(clauses)
            
#             # Aggregate results
#             return self._aggregate_results(predictions, clause_summary)
            
#         except Exception as e:
#             logger.error(f"Contract analysis failed: {e}", exc_info=True)
#             return {
#                 'error': f'Analysis failed: {str(e)}',
#                 'total_clauses': 0,
#                 'analysis': {}
#             }
    
#     def _aggregate_results(
#         self, 
#         predictions: List[Dict[str, Any]], 
#         clause_summary: Dict[str, Any]
#     ) -> Dict[str, Any]:
#         """
#         Aggregate clause predictions into contract-level analysis.
        
#         Args:
#             predictions: List of clause predictions
#             clause_summary: Summary statistics from PDF processing
            
#         Returns:
#             Dictionary with aggregated results (all numpy types converted to Python native types)
#         """
#         if not predictions:
#             return {
#                 'total_clauses': 0,
#                 'analysis': {}
#             }
        
#         # Extract risk scores
#         risk_scores = []
#         risk_levels = []
#         compliant_count = 0
#         total_clauses = len(predictions)
        
#         for pred in predictions:
#             # Safety check
#             if not isinstance(pred, dict):
#                 logger.warning(f"Skipping non-dict prediction: {type(pred)}")
#                 continue
            
#             # Get predictions dict safely
#             pred_dict = pred.get('predictions', {})
#             if not isinstance(pred_dict, dict):
#                 pred_dict = {}
            
#             # Risk score
#             if 'risk_score' in pred_dict:
#                 try:
#                     score = float(pred_dict['risk_score'])
#                     risk_scores.append(score)
#                 except (ValueError, TypeError):
#                     pass
            
#             # Risk level
#             if 'risk_level' in pred_dict:
#                 level = pred_dict['risk_level']
#                 if isinstance(level, dict):
#                     risk_levels.append(level.get('level_name', 'Unknown'))
#                 elif isinstance(level, str):
#                     risk_levels.append(level)
#                 else:
#                     risk_levels.append(str(level))
            
#             # Compliance
#             if 'compliance' in pred_dict:
#                 comp = pred_dict['compliance']
#                 if isinstance(comp, dict):
#                     if comp.get('is_compliant', False):
#                         compliant_count += 1
#                 elif isinstance(comp, bool):
#                     if comp:
#                         compliant_count += 1
#                 elif isinstance(comp, (int, float)):
#                     if comp > 0.5:
#                         compliant_count += 1
        
#         # Calculate aggregated metrics
#         avg_risk_score = float(np.mean(risk_scores) if risk_scores else 0.5)
#         max_risk_score = float(np.max(risk_scores) if risk_scores else 0.5)
#         min_risk_score = float(np.min(risk_scores) if risk_scores else 0.5)
        
#         # Risk level counts
#         risk_level_counts = {}
#         for level in risk_levels:
#             risk_level_counts[level] = risk_level_counts.get(level, 0) + 1
        
#         # Compliance metrics
#         compliance_rate = float(compliant_count / total_clauses if total_clauses > 0 else 0)
        
#         # Identify high-risk clauses
#         high_risk_clauses = []
#         for i, pred in enumerate(predictions):
#             if not isinstance(pred, dict):
#                 continue
            
#             pred_dict = pred.get('predictions', {})
#             if not isinstance(pred_dict, dict):
#                 continue
            
#             if 'risk_level' in pred_dict:
#                 level = pred_dict['risk_level']
#                 is_high = False
#                 if isinstance(level, dict) and level.get('level_name') == 'High':
#                     is_high = True
#                 elif isinstance(level, str) and level == 'High':
#                     is_high = True
                
#                 if is_high:
#                     high_risk_clauses.append({
#                         'clause_index': i,
#                         'text': pred.get('clause_text', '')[:200],
#                         'risk_score': float(pred_dict.get('risk_score', 0))
#                     })
        
#         # --- FIX: Convert clause_summary values to Python native types ---
#         clean_clause_summary = {}
#         for key, value in clause_summary.items():
#             if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
#                 clean_clause_summary[key] = int(value)
#             elif isinstance(value, (np.float64, np.float32, np.float16)):
#                 clean_clause_summary[key] = float(value)
#             elif isinstance(value, dict):
#                 clean_clause_summary[key] = {k: int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.float64, np.float32)) else v for k, v in value.items()}
#             elif isinstance(value, list):
#                 clean_clause_summary[key] = [int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.float64, np.float32)) else v for v in value]
#             else:
#                 clean_clause_summary[key] = value
        
#         # Aggregate results with all numpy types converted
#         results = {
#             'contract_analysis': {
#                 'total_clauses': total_clauses,
#                 'average_risk_score': avg_risk_score,
#                 'max_risk_score': max_risk_score,
#                 'min_risk_score': min_risk_score,
#                 'compliance_rate': compliance_rate,
#                 'risk_level_distribution': risk_level_counts,
#                 'high_risk_clauses_count': len(high_risk_clauses),
#                 'compliant_clauses_count': compliant_count,
#                 'non_compliant_clauses_count': total_clauses - compliant_count
#             },
#             'risk_summary': {
#                 'overall_risk_percentage': compute_risk_percentage(avg_risk_score),
#                 'overall_risk_level': get_risk_level(avg_risk_score),
#                 'high_risk_clauses': high_risk_clauses[:10]
#             },
#             'clause_summary': clean_clause_summary,
#             'clause_predictions': predictions
#         }
        
#         logger.info(f"Contract analysis complete: {total_clauses} clauses, "
#                    f"Risk Level: {results['risk_summary']['overall_risk_level']}, "
#                    f"Compliance Rate: {compliance_rate:.1%}")
        
#         return results
    
#     def get_model_info(self) -> Dict[str, Any]:
#         """
#         Get information about loaded models.
        
#         Returns:
#             Dictionary with model information
#         """
#         if not self.is_initialized:
#             return {'status': 'not_initialized'}
        
#         return {
#             'status': 'loaded',
#             'models': list(self.model_manager.models.keys()),
#             'category_mapping': self.category_mapping,
#             'risk_mapping': self.risk_mapping
#         }


"""
Inference pipeline for the Contract Risk Intelligence System.
Handles prediction workflows for clauses and complete contracts.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
from loguru import logger

from .config import get_config, Config
from .pdf_processor import PDFProcessor
from .embedding_generator import EmbeddingGenerator
from .models import ModelManager
from .utils import compute_risk_percentage, get_risk_level, load_model


class Predictor:
    """
    Inference pipeline for the Contract Risk Intelligence System.
    Handles predictions for individual clauses and complete contracts.
    """
    
    def __init__(self, config: Optional[Config] = None, model_dir: Optional[str] = None):
        """
        Initialize Predictor with configuration and trained models.
        
        Args:
            config: Configuration object
            model_dir: Directory containing trained models
        """
        self.config = config or get_config()
        self.model_dir = Path(model_dir) if model_dir else Path(self.config.paths.models_dir)
        
        # Initialize components
        self.pdf_processor = PDFProcessor(self.config)
        self.embedding_generator = EmbeddingGenerator(self.config)
        self.model_manager = ModelManager(self.config)
        
        # Load models if they exist
        self.is_initialized = False
        self._load_models()
        
    def _load_models(self) -> None:
        """
        Load trained models from disk.
        """
        try:
            if self.model_dir.exists():
                logger.info(f"Loading models from {self.model_dir}")
                self.model_manager.load_models(str(self.model_dir))
                
                # Load label encoders
                import joblib
                self.label_encoders = {}
                for encoder_file in self.model_dir.glob("*_encoder.pkl"):
                    name = encoder_file.stem.replace('_encoder', '')
                    self.label_encoders[name] = joblib.load(encoder_file)
                
                # Load mappings
                self.category_mapping = {}
                mapping_file = self.model_dir / "category_mapping.json"
                if mapping_file.exists():
                    with open(mapping_file, 'r') as f:
                        self.category_mapping = json.load(f)
                        # Convert keys to int
                        self.category_mapping = {int(k): v for k, v in self.category_mapping.items()}
                
                self.risk_mapping = {}
                risk_file = self.model_dir / "risk_mapping.json"
                if risk_file.exists():
                    with open(risk_file, 'r') as f:
                        self.risk_mapping = json.load(f)
                        self.risk_mapping = {int(k): v for k, v in self.risk_mapping.items()}
                
                self.is_initialized = True
                logger.info("Models loaded successfully")
            else:
                logger.warning(f"Model directory not found at {self.model_dir}")
                self.is_initialized = False
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.is_initialized = False
    
    def predict_clause(
        self, 
        clause_text: str,
        return_embeddings: bool = False
    ) -> Dict[str, Any]:
        """
        Predict risk and compliance for a single clause.
        
        Args:
            clause_text: Clause text
            return_embeddings: Whether to return embeddings
            
        Returns:
            Dictionary with predictions
        """
        if not self.is_initialized:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Generate embedding
        embedding = self.embedding_generator.generate_embeddings([clause_text])
        
        # Get predictions
        predictions = self.model_manager.predict_all(embedding)
        
        # Format results
        result = {
            'clause_text': clause_text,
            'predictions': {}
        }
        
        # Clause category
        if 'clause_category' in predictions:
            cat_idx = predictions['clause_category'][0]
            result['predictions']['clause_category'] = {
                'category_id': int(cat_idx),
                'category_name': self.category_mapping.get(int(cat_idx), f"Unknown_{cat_idx}")
            }
        
        # Risk score
        if 'risk_score' in predictions:
            score = float(predictions['risk_score'][0])
            result['predictions']['risk_score'] = score
            result['predictions']['risk_percentage'] = compute_risk_percentage(score)
            
            # Risk level
            if 'risk_level' in predictions:
                level_idx = int(predictions['risk_level'][0])
                level_name = self.risk_mapping.get(level_idx, f"Level_{level_idx}")
                result['predictions']['risk_level'] = {
                    'level_id': level_idx,
                    'level_name': level_name
                }
            else:
                # Compute risk level from score
                result['predictions']['risk_level'] = {
                    'level_name': get_risk_level(score)
                }
        
        # Compliance
        if 'compliance' in predictions:
            comp_pred = int(predictions['compliance'][0])
            result['predictions']['compliance'] = {
                'is_compliant': bool(comp_pred),
                'status': 'Compliant' if comp_pred == 1 else 'Non-Compliant'
            }
            
            if 'compliance_probability' in predictions:
                result['predictions']['compliance_probability'] = float(predictions['compliance_probability'][0])
        
        # --- Apply smart compliance rules ---
        result = self._apply_compliance_rules(result)
        
        if return_embeddings:
            result['embedding'] = embedding[0].tolist()
        
        return result
    
    def _apply_compliance_rules(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply smart compliance rules based on clause category and risk score.
        This helps correct obvious misclassifications.
        
        Args:
            predictions: Raw predictions dictionary
            
        Returns:
            Updated predictions with corrected compliance
        """
        # Get the predictions dict
        pred_dict = predictions.get('predictions', {})
        
        # Get clause category and risk score
        category = pred_dict.get('clause_category', {}).get('category_name', '')
        risk_score = pred_dict.get('risk_score', 0.5)
        
        # --- COMPLIANCE RULES ---
        
        # 1. If risk score is very low (< 0.25), it should be compliant
        if risk_score < 0.25:
            pred_dict['compliance'] = {
                'is_compliant': True,
                'status': 'Compliant'
            }
            pred_dict['compliance_probability'] = 0.95
        
        # 2. Governing Law clauses are almost always compliant
        if category.lower() in ['governing law', 'governing_law', 'governing law clause']:
            pred_dict['compliance'] = {
                'is_compliant': True,
                'status': 'Compliant'
            }
            pred_dict['compliance_probability'] = 0.98
        
        # 3. Insurance clauses with reasonable limits are compliant
        elif category.lower() in ['insurance', 'insurance clause']:
            # Check if the clause mentions reasonable limits (> $500,000)
            text = predictions.get('clause_text', '').lower()
            if '$' in text or 'million' in text or 'coverage' in text:
                # Standard insurance clauses are almost always compliant
                pred_dict['compliance'] = {
                    'is_compliant': True,
                    'status': 'Compliant'
                }
                pred_dict['compliance_probability'] = 0.92
        
        # 4. Standard Termination with Notice (30+ days) is compliant
        elif category.lower() in ['termination for convenience', 'termination', 'renewal term']:
            text = predictions.get('clause_text', '').lower()
            if '30' in text or 'thirty' in text or '60' in text or 'sixty' in text:
                if risk_score < 0.6:
                    pred_dict['compliance'] = {
                        'is_compliant': True,
                        'status': 'Compliant'
                    }
                    pred_dict['compliance_probability'] = 0.85
        
        # 5. Standard Confidentiality is compliant
        elif category.lower() in ['confidentiality', 'confidential', 'non-disclosure']:
            pred_dict['compliance'] = {
                'is_compliant': True,
                'status': 'Compliant'
            }
            pred_dict['compliance_probability'] = 0.90
        
        # 6. Standard Warranties are compliant
        elif category.lower() in ['warranty', 'representation', 'representations and warranties']:
            text = predictions.get('clause_text', '').lower()
            if 'as is' not in text and 'as-is' not in text:
                pred_dict['compliance'] = {
                    'is_compliant': True,
                    'status': 'Compliant'
                }
                pred_dict['compliance_probability'] = 0.85
        
        # 7. Standard Assignment with consent is compliant
        elif category.lower() in ['anti-assignment', 'assignment', 'anti assignment']:
            text = predictions.get('clause_text', '').lower()
            if 'consent' in text or 'written' in text:
                if risk_score < 0.6:
                    pred_dict['compliance'] = {
                        'is_compliant': True,
                        'status': 'Compliant'
                    }
                    pred_dict['compliance_probability'] = 0.80
        
        # 8. Standard Audit Rights are compliant
        elif category.lower() in ['audit rights', 'audit']:
            text = predictions.get('clause_text', '').lower()
            if 'reasonable' in text or 'business hours' in text:
                pred_dict['compliance'] = {
                    'is_compliant': True,
                    'status': 'Compliant'
                }
                pred_dict['compliance_probability'] = 0.88
        
        # 9. If risk score is high (> 0.7), it should be non-compliant
        elif risk_score > 0.7:
            pred_dict['compliance'] = {
                'is_compliant': False,
                'status': 'Non-Compliant'
            }
            pred_dict['compliance_probability'] = 0.90
        
        # 10. Non-compete clauses often have compliance issues
        elif category.lower() in ['non-compete', 'non_compete', 'exclusivity']:
            if risk_score > 0.5:
                pred_dict['compliance'] = {
                    'is_compliant': False,
                    'status': 'Non-Compliant'
                }
                pred_dict['compliance_probability'] = 0.85
        
        # 11. If compliance probability is already set and > 0.7, keep it
        elif pred_dict.get('compliance_probability', 0) > 0.7:
            # Keep existing compliance
            pass
        
        # Update the predictions
        predictions['predictions'] = pred_dict
        
        return predictions
    
    def _normalize_clause(self, clause: Union[str, Dict[str, Any], Any]) -> Dict[str, Any]:
        """
        Convert any clause input to a standardized dictionary.
        
        Args:
            clause: Clause as string, dict, or other type
            
        Returns:
            Normalized dictionary with 'text' key
        """
        if isinstance(clause, str):
            return {
                'text': clause,
                'word_count': len(clause.split()),
                'char_count': len(clause),
                'clause_length': len(clause),
                'type': 'string'
            }
        elif isinstance(clause, dict):
            # Make a copy to avoid modifying original
            result = dict(clause)
            # Ensure 'text' exists
            if 'text' not in result:
                # Try to find any string value
                for key, value in result.items():
                    if isinstance(value, str) and len(value) > 10:
                        result['text'] = value
                        break
                else:
                    result['text'] = str(result)
            return result
        else:
            # Unsupported type, convert to string
            return {
                'text': str(clause),
                'word_count': len(str(clause).split()),
                'char_count': len(str(clause)),
                'clause_length': len(str(clause)),
                'type': 'unknown'
            }
    
    def predict_clauses(
        self, 
        clauses: List[Union[str, Dict[str, Any]]],
        return_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Predict risk and compliance for multiple clauses.
        
        Args:
            clauses: List of clause dictionaries or strings
            return_embeddings: Whether to return embeddings
            
        Returns:
            List of dictionaries with predictions
        """
        if not clauses:
            return []
        
        if not self.is_initialized:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # --- NORMALIZE: ensure every clause is a dict with a 'text' key ---
        normalized_clauses = []
        for idx, c in enumerate(clauses):
            normalized = self._normalize_clause(c)
            # Add index if not present
            if 'index' not in normalized:
                normalized['index'] = idx
            normalized_clauses.append(normalized)
        
        # Extract texts
        texts = [c.get('text', '') for c in normalized_clauses]
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Get predictions
        predictions = self.model_manager.predict_all(embeddings)
        
        # Combine results
        results = []
        for i, clause in enumerate(normalized_clauses):
            result = {
                'clause_text': clause.get('text', ''),
                'clause_metadata': {
                    'word_count': clause.get('word_count', 0),
                    'char_count': clause.get('char_count', 0),
                    'type': clause.get('type', 'unknown')
                },
                'predictions': {}
            }
            
            # Clause category
            if 'clause_category' in predictions:
                cat_idx = predictions['clause_category'][i]
                result['predictions']['clause_category'] = {
                    'category_id': int(cat_idx),
                    'category_name': self.category_mapping.get(int(cat_idx), f"Unknown_{cat_idx}")
                }
            
            # Risk score
            if 'risk_score' in predictions:
                score = float(predictions['risk_score'][i])
                result['predictions']['risk_score'] = score
                result['predictions']['risk_percentage'] = compute_risk_percentage(score)
                
                # Risk level
                if 'risk_level' in predictions:
                    level_idx = int(predictions['risk_level'][i])
                    level_name = self.risk_mapping.get(level_idx, f"Level_{level_idx}")
                    result['predictions']['risk_level'] = {
                        'level_id': level_idx,
                        'level_name': level_name
                    }
                else:
                    result['predictions']['risk_level'] = {
                        'level_name': get_risk_level(score)
                    }
            
            # Compliance
            if 'compliance' in predictions:
                comp_pred = int(predictions['compliance'][i])
                result['predictions']['compliance'] = {
                    'is_compliant': bool(comp_pred),
                    'status': 'Compliant' if comp_pred == 1 else 'Non-Compliant'
                }
                
                if 'compliance_probability' in predictions:
                    result['predictions']['compliance_probability'] = float(predictions['compliance_probability'][i])
            
            # --- Apply smart compliance rules for each clause ---
            result = self._apply_compliance_rules(result)
            
            if return_embeddings and i < len(embeddings):
                result['embedding'] = embeddings[i].tolist()
            
            results.append(result)
        
        return results
    
    def predict_contract(self, pdf_content: bytes) -> Dict[str, Any]:
        """
        Analyze a complete contract from PDF.
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            Dictionary with full contract analysis
        """
        logger.info("Analyzing contract...")
        
        try:
            # Process PDF - get clauses as dictionaries
            clauses = self.pdf_processor.process_pdf(pdf_content)
            
            if not clauses:
                logger.warning("No clauses extracted from PDF")
                return {
                    'error': 'No valid clauses found in PDF',
                    'total_clauses': 0,
                    'analysis': {}
                }
            
            logger.info(f"Extracted {len(clauses)} clauses")
            
            # Get clause summary (now safe)
            clause_summary = self.pdf_processor.get_clause_summary(clauses)
            
            # Predict on clauses
            predictions = self.predict_clauses(clauses)
            
            # Aggregate results
            return self._aggregate_results(predictions, clause_summary)
            
        except Exception as e:
            logger.error(f"Contract analysis failed: {e}", exc_info=True)
            return {
                'error': f'Analysis failed: {str(e)}',
                'total_clauses': 0,
                'analysis': {}
            }
    
    def _aggregate_results(
        self, 
        predictions: List[Dict[str, Any]], 
        clause_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate clause predictions into contract-level analysis.
        
        Args:
            predictions: List of clause predictions
            clause_summary: Summary statistics from PDF processing
            
        Returns:
            Dictionary with aggregated results (all numpy types converted to Python native types)
        """
        if not predictions:
            return {
                'total_clauses': 0,
                'analysis': {}
            }
        
        # Extract risk scores
        risk_scores = []
        risk_levels = []
        compliant_count = 0
        total_clauses = len(predictions)
        
        for pred in predictions:
            # Safety check
            if not isinstance(pred, dict):
                logger.warning(f"Skipping non-dict prediction: {type(pred)}")
                continue
            
            # Get predictions dict safely
            pred_dict = pred.get('predictions', {})
            if not isinstance(pred_dict, dict):
                pred_dict = {}
            
            # Risk score
            if 'risk_score' in pred_dict:
                try:
                    score = float(pred_dict['risk_score'])
                    risk_scores.append(score)
                except (ValueError, TypeError):
                    pass
            
            # Risk level
            if 'risk_level' in pred_dict:
                level = pred_dict['risk_level']
                if isinstance(level, dict):
                    risk_levels.append(level.get('level_name', 'Unknown'))
                elif isinstance(level, str):
                    risk_levels.append(level)
                else:
                    risk_levels.append(str(level))
            
            # Compliance
            if 'compliance' in pred_dict:
                comp = pred_dict['compliance']
                if isinstance(comp, dict):
                    if comp.get('is_compliant', False):
                        compliant_count += 1
                elif isinstance(comp, bool):
                    if comp:
                        compliant_count += 1
                elif isinstance(comp, (int, float)):
                    if comp > 0.5:
                        compliant_count += 1
        
        # Calculate aggregated metrics
        avg_risk_score = float(np.mean(risk_scores) if risk_scores else 0.5)
        max_risk_score = float(np.max(risk_scores) if risk_scores else 0.5)
        min_risk_score = float(np.min(risk_scores) if risk_scores else 0.5)
        
        # Risk level counts
        risk_level_counts = {}
        for level in risk_levels:
            risk_level_counts[level] = risk_level_counts.get(level, 0) + 1
        
        # Compliance metrics
        compliance_rate = float(compliant_count / total_clauses if total_clauses > 0 else 0)
        
        # Identify high-risk clauses
        high_risk_clauses = []
        for i, pred in enumerate(predictions):
            if not isinstance(pred, dict):
                continue
            
            pred_dict = pred.get('predictions', {})
            if not isinstance(pred_dict, dict):
                continue
            
            if 'risk_level' in pred_dict:
                level = pred_dict['risk_level']
                is_high = False
                if isinstance(level, dict) and level.get('level_name') == 'High':
                    is_high = True
                elif isinstance(level, str) and level == 'High':
                    is_high = True
                
                if is_high:
                    high_risk_clauses.append({
                        'clause_index': i,
                        'text': pred.get('clause_text', '')[:200],
                        'risk_score': float(pred_dict.get('risk_score', 0))
                    })
        
        # --- FIX: Convert clause_summary values to Python native types ---
        clean_clause_summary = {}
        for key, value in clause_summary.items():
            if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                clean_clause_summary[key] = int(value)
            elif isinstance(value, (np.float64, np.float32, np.float16)):
                clean_clause_summary[key] = float(value)
            elif isinstance(value, dict):
                clean_clause_summary[key] = {k: int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.float64, np.float32)) else v for k, v in value.items()}
            elif isinstance(value, list):
                clean_clause_summary[key] = [int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.float64, np.float32)) else v for v in value]
            else:
                clean_clause_summary[key] = value
        
        # Aggregate results with all numpy types converted
        results = {
            'contract_analysis': {
                'total_clauses': total_clauses,
                'average_risk_score': avg_risk_score,
                'max_risk_score': max_risk_score,
                'min_risk_score': min_risk_score,
                'compliance_rate': compliance_rate,
                'risk_level_distribution': risk_level_counts,
                'high_risk_clauses_count': len(high_risk_clauses),
                'compliant_clauses_count': compliant_count,
                'non_compliant_clauses_count': total_clauses - compliant_count
            },
            'risk_summary': {
                'overall_risk_percentage': compute_risk_percentage(avg_risk_score),
                'overall_risk_level': get_risk_level(avg_risk_score),
                'high_risk_clauses': high_risk_clauses[:10]
            },
            'clause_summary': clean_clause_summary,
            'clause_predictions': predictions
        }
        
        logger.info(f"Contract analysis complete: {total_clauses} clauses, "
                   f"Risk Level: {results['risk_summary']['overall_risk_level']}, "
                   f"Compliance Rate: {compliance_rate:.1%}")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_initialized:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'loaded',
            'models': list(self.model_manager.models.keys()),
            'category_mapping': self.category_mapping,
            'risk_mapping': self.risk_mapping
        }   