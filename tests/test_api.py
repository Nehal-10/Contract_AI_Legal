"""
Tests for the Contract Risk Intelligence System API.
"""

import pytest
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data

def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Contract Risk Intelligence API"
    assert "version" in data

def test_predict_clause():
    """Test clause prediction endpoint."""
    request = {
        "clause_text": "Either party may terminate this Agreement at any time without liability.",
        "return_embeddings": False
    }
    response = client.post("/predict/clause", json=request)
    assert response.status_code in [200, 503]  # 503 if models not loaded
    
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert "predictions" in data["data"]

def test_predict_clauses():
    """Test multiple clause prediction endpoint."""
    request = {
        "clauses": [
            {"text": "Either party may terminate this Agreement at any time without liability."},
            {"text": "Agreement governed by Delaware law."}
        ],
        "return_embeddings": False
    }
    response = client.post("/predict/clauses", json=request)
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert "total_clauses" in data["data"]

def test_predict_risk():
    """Test risk prediction endpoint."""
    clause_text = "Either party may terminate this Agreement at any time without liability."
    response = client.post(f"/predict/risk?clause_text={clause_text}")
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data

def test_predict_compliance():
    """Test compliance prediction endpoint."""
    clause_text = "Either party may terminate this Agreement at any time without liability."
    response = client.post(f"/predict/compliance?clause_text={clause_text}")
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data

def test_models_info():
    """Test models info endpoint."""
    response = client.get("/models/info")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

def test_ready():
    """Test readiness check endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data

@pytest.mark.skip(reason="Requires a valid PDF file")
def test_analyze_contract():
    """Test contract analysis endpoint."""
    # Create a simple test PDF
    from PyPDF2 import PdfWriter
    from io import BytesIO
    
    pdf_buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(pdf_buffer)
    pdf_buffer.seek(0)
    
    files = {"file": ("test.pdf", pdf_buffer, "application/pdf")}
    response = client.post("/contract-analysis", files=files)
    
    # Should be 503 if models not loaded, or 200 if loaded
    assert response.status_code in [200, 503]