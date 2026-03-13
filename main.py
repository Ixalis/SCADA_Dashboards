#!/usr/bin/env python3
"""
SCADA Anomaly Detection Server
==============================

A real-time anomaly detection system for SCADA/ICS environments.

Usage:
    # Start with default settings (simulator mode)
    python main.py
    
    # Start with your trained models
    python main.py --if-model path/to/if.pkl --transformer-model path/to/transformer.keras --scaler path/to/scaler.pkl
    
    # Custom port
    python main.py --port 9000

Then open http://localhost:8000 in your browser.

Author: Ixalis @ HCMUT
Date: January 2026
"""

import argparse
import logging
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from app.api.routes import app, detector, simulator


def setup_logging(debug: bool = False):
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_models(if_path: str = None, transformer_path: str = None, scaler_path: str = None):
    """Load models at startup."""
    from app.inference.detector import get_detector
    
    det = get_detector()
    
    if if_path and os.path.exists(if_path):
        det.load_isolation_forest(if_path)
    else:
        logging.info("No IF model specified - running in simulator-only mode")
    
    if transformer_path and scaler_path:
        if os.path.exists(transformer_path) and os.path.exists(scaler_path):
            det.load_transformer(transformer_path, scaler_path)
        else:
            logging.warning(f"Transformer model files not found")
    else:
        logging.info("No Transformer model specified - IF only mode")


def main():
    parser = argparse.ArgumentParser(
        description='SCADA Anomaly Detection Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with built-in simulator (no models needed)
  python main.py
  
  # Run with your trained models
  python main.py \\
    --if-model /path/to/isolation_forest.pkl \\
    --transformer-model /path/to/transformer_ae.keras \\
    --scaler /path/to/scaler.pkl
  
  # For Mac M1 with Phase 2 models
  python main.py \\
    --if-model saved_models/isolation_forest.pkl \\
    --transformer-model /Users/ixa/Phase_2/data/Models/transformer_ae_best.keras \\
    --scaler /Users/ixa/Phase_2/data/Models/transformer_scaler.pkl

Dashboard: http://localhost:8000
API Docs:  http://localhost:8000/docs
        """
    )
    
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind (default: 8000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Model paths
    parser.add_argument('--if-model', dest='if_model', help='Path to Isolation Forest model (.pkl)')
    parser.add_argument('--transformer-model', dest='transformer_model', help='Path to Transformer model (.keras)')
    parser.add_argument('--scaler', help='Path to scaler (.pkl)')
    
    # Thresholds
    parser.add_argument('--if-threshold', type=float, default=-0.05, help='IF threshold (default: -0.05)')
    parser.add_argument('--transformer-threshold', type=float, default=0.278, help='Transformer threshold (default: 0.278)')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.debug)
    
    logging.info("="*60)
    logging.info("SCADA Anomaly Detection Server")
    logging.info("="*60)
    
    # Load models if specified
    if args.if_model or args.transformer_model:
        load_models(args.if_model, args.transformer_model, args.scaler)
    
    # Set thresholds
    from app.inference.detector import get_detector
    det = get_detector()
    det.set_thresholds(args.if_threshold, args.transformer_threshold)
    
    logging.info(f"Dashboard: http://localhost:{args.port}")
    logging.info(f"API Docs:  http://localhost:{args.port}/docs")
    logging.info("="*60)
    
    # Run server
    uvicorn.run(
        "app.api.routes:app",
        host=args.host,
        port=args.port,
        reload=args.debug,
        log_level="debug" if args.debug else "info"
    )


if __name__ == "__main__":
    main()
