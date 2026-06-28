"""
Main entry point for the Contract Risk Intelligence System.
Provides command-line interface for training and inference.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any

from loguru import logger

# Import modules
from src.config import get_config, Config
from src.trainer import Trainer
from src.predictor import Predictor
from src.utils import setup_logging, ensure_directory

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Contract Risk Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models
  python main.py --mode train --config config.yaml

  # Analyze a contract
  python main.py --mode analyze --file contract.pdf --output results.json

  # Predict a single clause
  python main.py --mode predict --clause "Either party may terminate this Agreement at any time without liability."

  # Start API server
  python main.py --mode serve --port 8000
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "analyze", "predict", "serve", "info"],
        required=True,
        help="Operation mode"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="Path to PDF file for analysis"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Path to output file"
    )
    
    parser.add_argument(
        "--clause",
        type=str,
        help="Clause text for prediction"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API server"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        default="models",
        help="Directory containing trained models"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="data/cuad_clauses_project.csv",
        help="Path to data file"
    )
    
    return parser.parse_args()

def train_mode(args):
    """Run training pipeline."""
    logger.info("Starting training mode...")
    
    # Load configuration
    config = get_config(args.config) if Path(args.config).exists() else get_config()
    
    # Update data path if provided
    if args.data != "data/cuad_clauses_project.csv":
        config.paths.data_dir = str(Path(args.data).parent)
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Run training
    results = trainer.run(save_models=True)
    
    # Print summary
    trainer.print_summary()
    
    # Save results
    trainer.save_results()
    
    logger.info("Training completed successfully")
    logger.info(f"Models saved to: {config.paths.models_dir}")
    logger.info(f"Results saved to: {config.paths.outputs_dir}")

def analyze_mode(args):
    """Analyze a contract PDF."""
    logger.info("Starting contract analysis mode...")
    
    if not args.file:
        logger.error("No file specified. Use --file to specify PDF path.")
        sys.exit(1)
    
    file_path = Path(args.file)
    if not file_path.exists():
        logger.error(f"File not found: {args.file}")
        sys.exit(1)
    
    # Load configuration
    config = get_config(args.config) if Path(args.config).exists() else get_config()
    
    # Initialize predictor
    predictor = Predictor(config=config, model_dir=args.models)
    
    if not predictor.is_initialized:
        logger.error("Models not loaded. Please train models first.")
        sys.exit(1)
    
    # Read PDF
    with open(file_path, 'rb') as f:
        pdf_content = f.read()
    
    # Analyze contract
    logger.info(f"Analyzing contract: {file_path.name}")
    results = predictor.predict_contract(pdf_content)
    
    # Save results
    output_path = Path(args.output)
    ensure_directory(str(output_path.parent))
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Analysis results saved to: {args.output}")
    
    # Print summary
    if 'contract_analysis' in results:
        analysis = results['contract_analysis']
        logger.info("\n" + "=" * 60)
        logger.info("CONTRACT ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Clauses: {analysis.get('total_clauses', 0)}")
        logger.info(f"Average Risk Score: {analysis.get('average_risk_score', 0):.3f}")
        logger.info(f"Compliance Rate: {analysis.get('compliance_rate', 0):.1%}")
        logger.info(f"Risk Level: {results.get('risk_summary', {}).get('overall_risk_level', 'Unknown')}")
        logger.info(f"High-Risk Clauses: {analysis.get('high_risk_clauses_count', 0)}")
        logger.info("=" * 60)

def predict_mode(args):
    """Predict on a single clause."""
    logger.info("Starting clause prediction mode...")
    
    if not args.clause:
        logger.error("No clause provided. Use --clause to specify clause text.")
        sys.exit(1)
    
    # Load configuration
    config = get_config(args.config) if Path(args.config).exists() else get_config()
    
    # Initialize predictor
    predictor = Predictor(config=config, model_dir=args.models)
    
    if not predictor.is_initialized:
        logger.error("Models not loaded. Please train models first.")
        sys.exit(1)
    
    # Predict
    logger.info(f"Analyzing clause: {args.clause[:100]}...")
    result = predictor.predict_clause(args.clause)
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("CLAUSE ANALYSIS RESULTS")
    logger.info("=" * 60)
    
    predictions = result.get('predictions', {})
    
    if 'clause_category' in predictions:
        cat = predictions['clause_category']
        logger.info(f"Category: {cat.get('category_name', 'Unknown')} (ID: {cat.get('category_id', 'N/A')})")
    
    if 'risk_score' in predictions:
        logger.info(f"Risk Score: {predictions.get('risk_score', 0):.3f}")
        logger.info(f"Risk Percentage: {predictions.get('risk_percentage', 'N/A')}")
    
    if 'risk_level' in predictions:
        level = predictions['risk_level']
        logger.info(f"Risk Level: {level.get('level_name', 'Unknown')}")
    
    if 'compliance' in predictions:
        comp = predictions['compliance']
        logger.info(f"Compliance Status: {comp.get('status', 'Unknown')}")
        if 'compliance_probability' in predictions:
            logger.info(f"Compliance Probability: {predictions['compliance_probability']:.3f}")
    
    logger.info("=" * 60)
    
    # Save result if output specified
    if args.output and args.output != "results.json":
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Result saved to: {args.output}")

def serve_mode(args):
    """Start the API server."""
    logger.info("Starting API server...")
    
    # Load configuration
    config = get_config(args.config) if Path(args.config).exists() else get_config()
    
    # Update config with port
    config.api.port = args.port
    
    # Start server
    import uvicorn
    from api.app import app
    
    # Ensure models are loaded
    predictor = Predictor(config=config, model_dir=args.models)
    
    # Override predictor in app
    import api.app as app_module
    app_module.predictor = predictor
    
    logger.info(f"Starting server on http://{config.api.host}:{config.api.port}")
    logger.info(f"Docs available at http://{config.api.host}:{config.api.port}/docs")
    
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        log_level="info"
    )

def info_mode(args):
    """Display system information."""
    logger.info("System Information:")
    logger.info("=" * 60)
    
    # Configuration
    config = get_config(args.config) if Path(args.config).exists() else get_config()
    
    logger.info(f"Configuration: {args.config if Path(args.config).exists() else 'default'}")
    logger.info(f"Embedding Model: {config.model.embedding_model}")
    logger.info(f"Embedding Dimension: {config.model.embedding_dim}")
    
    # Check if models exist
    model_dir = Path(args.models)
    if model_dir.exists():
        model_files = list(model_dir.glob("*.pkl"))
        logger.info(f"Models Directory: {args.models} ({len(model_files)} files)")
        
        # Load model info
        try:
            import joblib
            for model_file in model_files:
                if model_file.stem != 'scaler':
                    model = joblib.load(model_file)
                    logger.info(f"  - {model_file.stem}: {type(model).__name__}")
        except:
            logger.info("  Could not load model info")
    else:
        logger.info(f"Models Directory: {args.models} (not found)")
    
    # Check data
    data_path = Path(args.data)
    if data_path.exists():
        import pandas as pd
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Data: {data_path} ({len(df)} rows, {len(df.columns)} columns)")
        except:
            logger.info(f"Data: {data_path} (could not read)")
    else:
        logger.info(f"Data: {args.data} (not found)")
    
    logger.info("=" * 60)

def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logging()
    
    # Route to appropriate mode
    if args.mode == "train":
        train_mode(args)
    elif args.mode == "analyze":
        analyze_mode(args)
    elif args.mode == "predict":
        predict_mode(args)
    elif args.mode == "serve":
        serve_mode(args)
    elif args.mode == "info":
        info_mode(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()