"""
FastAPI application for the Contract Risk Intelligence System.
Provides RESTful API endpoints for contract analysis and predictions.
Serves the frontend UI from the /frontend directory.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # <-- NEW IMPORT
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

from src.config import get_config
from src.predictor import Predictor
from src.utils import setup_logging

# Setup logging
setup_logging()

# Load configuration
config = get_config()

# Initialize predictor
predictor = Predictor(config=config)

# Check if models are loaded
if not predictor.is_initialized:
    logger.warning("Models not loaded. Please train models first or provide model directory.")
    logger.info("Running in limited mode: some endpoints may not work.")

# Create FastAPI app
app = FastAPI(
    title="Contract Risk Intelligence API",
    description="AI-Powered Legal Contract Risk & Compliance Analysis System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- Serve Frontend Static Files ---
# Mount the "frontend" folder at the /frontend endpoint.
# This allows you to access the UI at http://localhost:8000/frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
    logger.info(f"Frontend mounted at /frontend from {frontend_path}")
else:
    logger.warning(f"Frontend folder not found at {frontend_path}. Create it and add index.html")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ClausePredictionRequest(BaseModel):
    """Request model for clause prediction."""
    clause_text: str = Field(..., description="Text of the legal clause to analyze")
    return_embeddings: bool = Field(False, description="Whether to return embeddings")

class MultipleClausePredictionRequest(BaseModel):
    """Request model for multiple clause predictions."""
    clauses: List[Dict[str, Any]] = Field(..., description="List of clause dictionaries")
    return_embeddings: bool = Field(False, description="Whether to return embeddings")

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    status: str = Field(..., description="Status of the request")
    data: Optional[Dict[str, Any]] = Field(None, description="Prediction results")
    message: Optional[str] = Field(None, description="Additional message")

@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint providing API information."""
    return {
        "name": "Contract Risk Intelligence API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": predictor.is_initialized,
        "docs": "/docs",
        "redoc": "/redoc",
        "frontend": "/frontend"   # <-- Added convenience link
    }

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns:
        Dictionary with health status
    """
    return {
        "status": "healthy" if predictor.is_initialized else "degraded",
        "models_loaded": predictor.is_initialized,
        "embedding_model": config.model.embedding_model if hasattr(config.model, 'embedding_model') else "unknown"
    }

@app.post("/predict/clause")
async def predict_clause(request: ClausePredictionRequest) -> PredictionResponse:
    """
    Predict risk and compliance for a single clause.
    
    Args:
        request: ClausePredictionRequest object
        
    Returns:
        PredictionResponse with results
    """
    if not predictor.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please train models first."
        )
    
    try:
        if not request.clause_text or not request.clause_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Clause text cannot be empty"
            )
        
        result = predictor.predict_clause(
            clause_text=request.clause_text,
            return_embeddings=request.return_embeddings
        )
        
        return PredictionResponse(
            status="success",
            data=result
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in predict_clause: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/clauses")
async def predict_clauses(request: MultipleClausePredictionRequest) -> PredictionResponse:
    """
    Predict risk and compliance for multiple clauses.
    
    Args:
        request: MultipleClausePredictionRequest object
        
    Returns:
        PredictionResponse with results
    """
    if not predictor.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please train models first."
        )
    
    try:
        if not request.clauses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Clauses list cannot be empty"
            )
        
        results = predictor.predict_clauses(
            clauses=request.clauses,
            return_embeddings=request.return_embeddings
        )
        
        return PredictionResponse(
            status="success",
            data={
                "total_clauses": len(results),
                "results": results
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in predict_clauses: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/contract-analysis")
async def analyze_contract(
    file: UploadFile = File(..., description="PDF contract file to analyze")
) -> PredictionResponse:
    """
    Analyze a complete contract PDF.
    
    Args:
        file: Uploaded PDF file
        
    Returns:
        PredictionResponse with contract analysis
    """
    if not predictor.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please train models first."
        )
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty PDF file uploaded"
            )
        
        # Analyze contract
        result = predictor.predict_contract(content)
        
        if 'error' in result:
            return PredictionResponse(
                status="error",
                message=result.get('error', 'Analysis failed'),
                data=result
            )
        
        return PredictionResponse(
            status="success",
            data=result
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in analyze_contract: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Contract analysis failed: {str(e)}"
        )

@app.post("/predict/risk")
async def predict_risk(clause_text: str) -> PredictionResponse:
    """
    Predict risk score for a clause.
    
    Args:
        clause_text: Clause text to analyze
        
    Returns:
        PredictionResponse with risk prediction
    """
    if not predictor.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please train models first."
        )
    
    try:
        result = predictor.predict_clause(clause_text)
        
        risk_data = {
            "clause_text": clause_text,
            "risk_score": result['predictions'].get('risk_score'),
            "risk_percentage": result['predictions'].get('risk_percentage'),
            "risk_level": result['predictions'].get('risk_level')
        }
        
        return PredictionResponse(
            status="success",
            data=risk_data
        )
        
    except Exception as e:
        logger.error(f"Error in predict_risk: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk prediction failed: {str(e)}"
        )

@app.post("/predict/compliance")
async def predict_compliance(clause_text: str) -> PredictionResponse:
    """
    Predict compliance status for a clause.
    
    Args:
        clause_text: Clause text to analyze
        
    Returns:
        PredictionResponse with compliance prediction
    """
    if not predictor.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please train models first."
        )
    
    try:
        result = predictor.predict_clause(clause_text)
        
        compliance_data = {
            "clause_text": clause_text,
            "compliance": result['predictions'].get('compliance'),
            "compliance_probability": result['predictions'].get('compliance_probability')
        }
        
        return PredictionResponse(
            status="success",
            data=compliance_data
        )
        
    except Exception as e:
        logger.error(f"Error in predict_compliance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Compliance prediction failed: {str(e)}"
        )

@app.get("/models/info")
async def get_models_info() -> Dict[str, Any]:
    """
    Get information about loaded models.
    
    Returns:
        Dictionary with model information
    """
    return predictor.get_model_info()

@app.get("/metrics")
async def get_training_metrics() -> Dict[str, Any]:
    """
    Get training metrics summary.
    
    Returns:
        Dictionary with training metrics
    """
    metrics_path = Path(config.paths.outputs_dir) / "evaluation_results.json"
    
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return {"status": "success", "metrics": metrics}
    else:
        return {
            "status": "warning",
            "message": "No training metrics found. Train models first."
        }

@app.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Check if the system is ready to serve requests.
    
    Returns:
        Dictionary with readiness status
    """
    is_ready = predictor.is_initialized
    
    return {
        "ready": is_ready,
        "models_loaded": is_ready,
        "embedding_model_loaded": bool(predictor.embedding_generator.is_initialized),
        "message": "System is ready" if is_ready else "System not ready. Models not loaded."
    }

# Add exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "data": None
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": f"Internal server error: {str(exc)}",
            "data": None
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        log_level="info"
    )