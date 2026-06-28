# """
# PDF processing module for the Contract Risk Intelligence System.
# Handles text extraction from PDF files and segmentation into legal clauses.
# """

# import re
# import numpy as np
# from pathlib import Path
# from typing import List, Dict, Any, Optional, Tuple
# from io import BytesIO
# import PyPDF2
# import pdfplumber
# from loguru import logger


# class PDFProcessor:
#     """
#     Handles PDF text extraction and clause segmentation.
#     """
    
#     def __init__(self, config: Any):
#         """
#         Initialize PDFProcessor with configuration.
        
#         Args:
#             config: Configuration object
#         """
#         self.config = config
#         self.separators = config.data.clause_separators
#         self.min_words = config.data.min_clause_words
#         self.max_length = config.model.max_clause_length
        
#     def extract_text(self, pdf_content: bytes) -> str:
#         """
#         Extract text from PDF content using multiple methods.
        
#         Args:
#             pdf_content: PDF file content as bytes
            
#         Returns:
#             Extracted text string
#         """
#         logger.info("Extracting text from PDF...")
        
#         text = ""
        
#         # Try pdfplumber first (better for complex layouts)
#         try:
#             with pdfplumber.open(BytesIO(pdf_content)) as pdf:
#                 for page in pdf.pages:
#                     page_text = page.extract_text()
#                     if page_text:
#                         text += page_text + "\n\n"
#         except Exception as e:
#             logger.warning(f"pdfplumber extraction failed: {e}")
        
#         # If pdfplumber returned empty, try PyPDF2
#         if not text.strip():
#             try:
#                 pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
#                 for page in pdf_reader.pages:
#                     page_text = page.extract_text()
#                     if page_text:
#                         text += page_text + "\n\n"
#             except Exception as e:
#                 logger.error(f"PyPDF2 extraction failed: {e}")
#                 raise ValueError("Failed to extract text from PDF")
        
#         # Clean extracted text
#         text = self._clean_text(text)
        
#         logger.info(f"Extracted {len(text)} characters from PDF")
#         return text
    
#     def _clean_text(self, text: str) -> str:
#         """
#         Clean extracted text by removing extra whitespace and formatting issues.
        
#         Args:
#             text: Raw extracted text
            
#         Returns:
#             Cleaned text
#         """
#         # Replace multiple newlines with single newline
#         text = re.sub(r'\n\s*\n', '\n\n', text)
        
#         # Remove excessive spaces
#         text = re.sub(r' +', ' ', text)
        
#         # Remove leading/trailing whitespace
#         text = text.strip()
        
#         return text
    
#     def segment_clauses(self, text: str) -> List[Dict[str, Any]]:
#         """
#         Segment legal text into individual clauses.
        
#         Uses multiple strategies:
#         1. Look for numbered sections (e.g., "1.", "2.", "Section 1", etc.)
#         2. Look for capitalized headings (e.g., "TERMINATION", "GOVERNING LAW")
#         3. Split by separators (e.g., "\n\n", ". ")
#         4. Intelligent fallback
        
#         Args:
#             text: Cleaned legal text
            
#         Returns:
#             List of clause dictionaries with text and metadata
#         """
#         logger.info("Segmenting text into clauses...")
        
#         if not text:
#             logger.warning("Empty text provided for segmentation")
#             return []
        
#         clauses = []
        
#         # Strategy 1: Numbered sections
#         numbered_pattern = r'(?:\n|^)(?:\d+\.|\d+\.\d+|\d+\.\d+\.\d+|[A-Z]\.|[\(\)\d]+)\s+'
#         numbered_segments = re.split(numbered_pattern, text)
#         numbered_matches = re.findall(numbered_pattern, text)
        
#         if len(numbered_segments) > 1:
#             for i, segment in enumerate(numbered_segments):
#                 if i < len(numbered_matches):
#                     prefix = numbered_matches[i]
#                 else:
#                     prefix = ""
                
#                 segment = segment.strip()
#                 if self._is_valid_clause(segment):
#                     clauses.append({
#                         'text': prefix + segment if prefix else segment,
#                         'type': 'numbered',
#                         'index': i
#                     })
            
#             if clauses:
#                 logger.info(f"Found {len(clauses)} clauses using numbered section strategy")
#                 return clauses
        
#         # Strategy 2: Capitalized headings
#         heading_pattern = r'(?:\n|^)(?=[A-Z][A-Z\s\-]{3,})([A-Z][A-Z\s\-]{3,}\n)'
#         heading_segments = re.split(heading_pattern, text)
        
#         if len(heading_segments) > 1:
#             clauses = []
#             current_text = ""
#             heading = ""
            
#             for i, segment in enumerate(heading_segments):
#                 if i % 2 == 1:  # This is a heading
#                     heading = segment.strip()
#                 elif segment.strip():
#                     clause_text = (heading + "\n" + segment.strip()) if heading else segment.strip()
#                     if self._is_valid_clause(clause_text):
#                         clauses.append({
#                             'text': clause_text,
#                             'type': 'heading',
#                             'heading': heading,
#                             'index': i
#                         })
#                     heading = ""
            
#             if clauses:
#                 logger.info(f"Found {len(clauses)} clauses using heading strategy")
#                 return clauses
        
#         # Strategy 3: Split by separators
#         for separator in self.separators:
#             if separator in text:
#                 raw_clauses = text.split(separator)
#                 for i, clause in enumerate(raw_clauses):
#                     clause = clause.strip()
#                     if self._is_valid_clause(clause):
#                         clauses.append({
#                             'text': clause,
#                             'type': 'separator',
#                             'separator': separator,
#                             'index': i
#                         })
                
#                 if clauses:
#                     logger.info(f"Found {len(clauses)} clauses using separator '{repr(separator)}'")
#                     return clauses
        
#         # Strategy 4: Fallback - treat entire text as one clause
#         if self._is_valid_clause(text):
#             clauses = [{
#                 'text': text,
#                 'type': 'fallback',
#                 'index': 0
#             }]
#             logger.info("Fallback: treating entire text as single clause")
#         else:
#             logger.warning("No valid clauses found in text")
        
#         return clauses
    
#     def _is_valid_clause(self, text: str) -> bool:
#         """
#         Check if a text segment is a valid clause.
        
#         Args:
#             text: Text to check
            
#         Returns:
#             True if valid clause, False otherwise
#         """
#         # Check minimum length
#         if len(text.strip()) < 10:
#             return False
        
#         # Check word count
#         word_count = len(text.split())
#         if word_count < self.min_words:
#             return False
        
#         # Check if it contains any letters
#         if not re.search(r'[a-zA-Z]', text):
#             return False
        
#         return True
    
#     def process_pdf(self, pdf_content: bytes) -> List[Dict[str, Any]]:
#         """
#         Complete PDF processing pipeline.
        
#         Args:
#             pdf_content: PDF file content as bytes
            
#         Returns:
#             List of clause dictionaries
#         """
#         # Extract text
#         text = self.extract_text(pdf_content)
        
#         # Segment into clauses
#         clauses = self.segment_clauses(text)
        
#         # Add metadata to clauses
#         for clause in clauses:
#             if isinstance(clause, dict):
#                 clause['text'] = clause.get('text', '').strip()
#                 clause['word_count'] = len(clause['text'].split())
#                 clause['char_count'] = len(clause['text'])
#                 clause['clause_length'] = clause['char_count']
        
#         logger.info(f"Processed PDF into {len(clauses)} clauses")
#         return clauses
    
#     def get_clause_summary(self, clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """
#         Generate summary statistics for extracted clauses.
        
#         Args:
#             clauses: List of clause dictionaries
            
#         Returns:
#             Dictionary with summary statistics (all numpy types converted to Python native types)
#         """
#         if not clauses:
#             return {
#                 'total_clauses': 0,
#                 'avg_word_count': 0,
#                 'avg_char_count': 0,
#                 'total_chars': 0
#             }
        
#         # --- FIX: Ensure all clauses are dictionaries ---
#         normalized_clauses = []
#         for c in clauses:
#             if isinstance(c, str):
#                 normalized_clauses.append({
#                     'text': c,
#                     'word_count': len(c.split()),
#                     'char_count': len(c),
#                     'clause_length': len(c),
#                     'type': 'string'
#                 })
#             elif isinstance(c, dict):
#                 # Ensure it has all required fields
#                 text = c.get('text', '')
#                 if not text and isinstance(c, dict):
#                     # Try to find any string value
#                     for key, value in c.items():
#                         if isinstance(value, str) and len(value) > 10:
#                             text = value
#                             break
#                 normalized_clauses.append({
#                     'text': text,
#                     'word_count': c.get('word_count', len(text.split())),
#                     'char_count': c.get('char_count', len(text)),
#                     'clause_length': c.get('clause_length', len(text)),
#                     'type': c.get('type', 'unknown')
#                 })
#             else:
#                 # Fallback: convert to string
#                 normalized_clauses.append({
#                     'text': str(c),
#                     'word_count': len(str(c).split()),
#                     'char_count': len(str(c)),
#                     'clause_length': len(str(c)),
#                     'type': 'unknown'
#                 })
        
#         # Use normalized clauses for calculations
#         word_counts = [c.get('word_count', 0) for c in normalized_clauses]
#         char_counts = [c.get('char_count', 0) for c in normalized_clauses]
        
#         # Count clause types safely
#         clause_types = {}
#         for c in normalized_clauses:
#             c_type = c.get('type', 'unknown')
#             clause_types[c_type] = clause_types.get(c_type, 0) + 1
        
#         # --- FIX: Convert all numpy types to Python native types ---
#         result = {
#             'total_clauses': int(len(normalized_clauses)),
#             'avg_word_count': float(np.mean(word_counts) if word_counts else 0),
#             'std_word_count': float(np.std(word_counts) if word_counts else 0),
#             'avg_char_count': float(np.mean(char_counts) if char_counts else 0),
#             'std_char_count': float(np.std(char_counts) if char_counts else 0),
#             'total_chars': int(sum(char_counts) if char_counts else 0),
#             'min_word_count': int(min(word_counts) if word_counts else 0),
#             'max_word_count': int(max(word_counts) if word_counts else 0),
#             'clause_types': {k: int(v) for k, v in clause_types.items()}
#         }
        
#         return result



"""
PDF processing module for the Contract Risk Intelligence System.
Handles text extraction from PDF files and segmentation into legal clauses.
"""

import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO
import PyPDF2
import pdfplumber
from loguru import logger


class PDFProcessor:
    """
    Handles PDF text extraction and clause segmentation.
    """
    
    def __init__(self, config: Any):
        """
        Initialize PDFProcessor with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.separators = config.data.clause_separators
        self.min_words = config.data.min_clause_words
        self.max_length = config.model.max_clause_length
        
    def extract_text(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF content using multiple methods.
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            Extracted text string
        """
        logger.info("Extracting text from PDF...")
        
        text = ""
        
        # Try pdfplumber first (better for complex layouts)
        try:
            with pdfplumber.open(BytesIO(pdf_content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        # If pdfplumber returned empty, try PyPDF2
        if not text.strip():
            try:
                pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            except Exception as e:
                logger.error(f"PyPDF2 extraction failed: {e}")
                raise ValueError("Failed to extract text from PDF")
        
        # Clean extracted text
        text = self._clean_text(text)
        
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and formatting issues.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Replace multiple newlines with single newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove excessive spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def segment_clauses(self, text: str) -> List[Dict[str, Any]]:
        """
        Segment legal text into individual clauses.
        
        Uses multiple strategies:
        1. Look for numbered sections (e.g., "1.", "2.", "Section 1", etc.)
        2. Look for capitalized headings (e.g., "TERMINATION", "GOVERNING LAW")
        3. Split by separators (e.g., "\n\n", ". ")
        4. Intelligent fallback
        
        Args:
            text: Cleaned legal text
            
        Returns:
            List of clause dictionaries with text and metadata
        """
        logger.info("Segmenting text into clauses...")
        
        if not text:
            logger.warning("Empty text provided for segmentation")
            return []
        
        clauses = []
        
        # Strategy 1: Numbered sections
        numbered_pattern = r'(?:\n|^)(?:\d+\.|\d+\.\d+|\d+\.\d+\.\d+|[A-Z]\.|[\(\)\d]+)\s+'
        numbered_segments = re.split(numbered_pattern, text)
        numbered_matches = re.findall(numbered_pattern, text)
        
        if len(numbered_segments) > 1:
            for i, segment in enumerate(numbered_segments):
                if i < len(numbered_matches):
                    prefix = numbered_matches[i]
                else:
                    prefix = ""
                
                segment = segment.strip()
                if self._is_valid_clause(segment):
                    clauses.append({
                        'text': prefix + segment if prefix else segment,
                        'type': 'numbered',
                        'index': i
                    })
            
            if clauses:
                logger.info(f"Found {len(clauses)} clauses using numbered section strategy")
                return clauses
        
        # Strategy 2: Capitalized headings
        heading_pattern = r'(?:\n|^)(?=[A-Z][A-Z\s\-]{3,})([A-Z][A-Z\s\-]{3,}\n)'
        heading_segments = re.split(heading_pattern, text)
        
        if len(heading_segments) > 1:
            clauses = []
            current_text = ""
            heading = ""
            
            for i, segment in enumerate(heading_segments):
                if i % 2 == 1:  # This is a heading
                    heading = segment.strip()
                elif segment.strip():
                    clause_text = (heading + "\n" + segment.strip()) if heading else segment.strip()
                    if self._is_valid_clause(clause_text):
                        clauses.append({
                            'text': clause_text,
                            'type': 'heading',
                            'heading': heading,
                            'index': i
                        })
                    heading = ""
            
            if clauses:
                logger.info(f"Found {len(clauses)} clauses using heading strategy")
                return clauses
        
        # Strategy 3: Split by separators
        for separator in self.separators:
            if separator in text:
                raw_clauses = text.split(separator)
                for i, clause in enumerate(raw_clauses):
                    clause = clause.strip()
                    if self._is_valid_clause(clause):
                        clauses.append({
                            'text': clause,
                            'type': 'separator',
                            'separator': separator,
                            'index': i
                        })
                
                if clauses:
                    logger.info(f"Found {len(clauses)} clauses using separator '{repr(separator)}'")
                    return clauses
        
        # Strategy 4: Fallback - treat entire text as one clause
        if self._is_valid_clause(text):
            clauses = [{
                'text': text,
                'type': 'fallback',
                'index': 0
            }]
            logger.info("Fallback: treating entire text as single clause")
        else:
            logger.warning("No valid clauses found in text")
        
        return clauses
    
    def _is_valid_clause(self, text: str) -> bool:
        """
        Check if a text segment is a valid clause.
        
        Args:
            text: Text to check
            
        Returns:
            True if valid clause, False otherwise
        """
        # Check minimum length
        if len(text.strip()) < 10:
            return False
        
        # Check word count
        word_count = len(text.split())
        if word_count < self.min_words:
            return False
        
        # Check if it contains any letters
        if not re.search(r'[a-zA-Z]', text):
            return False
        
        return True
    
    def process_pdf(self, pdf_content: bytes) -> List[Dict[str, Any]]:
        """
        Complete PDF processing pipeline.
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            List of clause dictionaries
        """
        # Extract text
        text = self.extract_text(pdf_content)
        
        # Segment into clauses
        clauses = self.segment_clauses(text)
        
        # Add metadata to clauses
        for clause in clauses:
            if isinstance(clause, dict):
                clause['text'] = clause.get('text', '').strip()
                clause['word_count'] = len(clause['text'].split())
                clause['char_count'] = len(clause['text'])
                clause['clause_length'] = clause['char_count']
        
        logger.info(f"Processed PDF into {len(clauses)} clauses")
        return clauses
    
    def get_clause_summary(self, clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics for extracted clauses.
        
        Args:
            clauses: List of clause dictionaries
            
        Returns:
            Dictionary with summary statistics (all numpy types converted to Python native types)
        """
        if not clauses:
            return {
                'total_clauses': 0,
                'avg_word_count': 0,
                'avg_char_count': 0,
                'total_chars': 0
            }
        
        # --- FIX: Ensure all clauses are dictionaries ---
        normalized_clauses = []
        for c in clauses:
            if isinstance(c, str):
                normalized_clauses.append({
                    'text': c,
                    'word_count': len(c.split()),
                    'char_count': len(c),
                    'clause_length': len(c),
                    'type': 'string'
                })
            elif isinstance(c, dict):
                # Ensure it has all required fields
                text = c.get('text', '')
                if not text and isinstance(c, dict):
                    # Try to find any string value
                    for key, value in c.items():
                        if isinstance(value, str) and len(value) > 10:
                            text = value
                            break
                normalized_clauses.append({
                    'text': text,
                    'word_count': c.get('word_count', len(text.split())),
                    'char_count': c.get('char_count', len(text)),
                    'clause_length': c.get('clause_length', len(text)),
                    'type': c.get('type', 'unknown')
                })
            else:
                # Fallback: convert to string
                normalized_clauses.append({
                    'text': str(c),
                    'word_count': len(str(c).split()),
                    'char_count': len(str(c)),
                    'clause_length': len(str(c)),
                    'type': 'unknown'
                })
        
        # Use normalized clauses for calculations
        word_counts = [c.get('word_count', 0) for c in normalized_clauses]
        char_counts = [c.get('char_count', 0) for c in normalized_clauses]
        
        # Count clause types safely
        clause_types = {}
        for c in normalized_clauses:
            c_type = c.get('type', 'unknown')
            clause_types[c_type] = clause_types.get(c_type, 0) + 1
        
        # --- FIX: Convert all numpy types to Python native types ---
        result = {
            'total_clauses': int(len(normalized_clauses)),
            'avg_word_count': float(np.mean(word_counts) if word_counts else 0),
            'std_word_count': float(np.std(word_counts) if word_counts else 0),
            'avg_char_count': float(np.mean(char_counts) if char_counts else 0),
            'std_char_count': float(np.std(char_counts) if char_counts else 0),
            'total_chars': int(sum(char_counts) if char_counts else 0),
            'min_word_count': int(min(word_counts) if word_counts else 0),
            'max_word_count': int(max(word_counts) if word_counts else 0),
            'clause_types': {k: int(v) for k, v in clause_types.items()}
        }
        
        return result