# Contract Risk Intelligence System

## AI-Powered Legal Contract Risk & Compliance Analysis

A production-grade Machine Learning system for analyzing legal contracts using LegalBERT embeddings and supervised ML models.

## Overview

This system provides intelligent risk and compliance analysis for legal contracts by:

1. **Extracting text** from PDF contract documents
2. **Segmenting** text into individual legal clauses
3. **Generating embeddings** using LegalBERT (nlpaueb/legal-bert-base-uncased)
4. **Predicting** clause category, risk level, compliance status, and risk score using supervised ML models
5. **Aggregating** predictions into contract-level risk assessment

## Features

- ✅ **PDF Processing**: Extract and segment text from contract PDFs
- ✅ **LegalBERT Embeddings**: Semantic representation of legal clauses
- ✅ **Multi-Task Predictions**: Clause classification, risk scoring, compliance prediction
- ✅ **Contract Analysis**: Aggregated risk assessment and high-risk clause identification
- ✅ **Production API**: FastAPI with comprehensive endpoints
- ✅ **Comprehensive Testing**: Unit tests for all components

## Architecture
