import argparse
import os
import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request
import mlflow
from typing import Dict, Any, List, Optional

# To mimic using multiple Python modules, such as 'core' and 'feature_pipeline',
# we will add the './src' directory to the PYTHONPATH. This is not intended for
# production use cases but for development and educational purposes.
ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from core import logger_utils
from config import settings
from serve_mlflow_model import app, load_model

# Check if running locally or in Docker
if os.environ.get("DOCKER_CONTAINER", "false").lower() != "true":
    # Only patch localhost when running locally, not in Docker
    settings.patch_localhost()

logger = logger_utils.get_logger(__file__)
logger.info(
    f"Added the following directory to PYTHONPATH to simulate multiple modules: {ROOT_DIR}"
)
if os.environ.get("DOCKER_CONTAINER", "false").lower() != "true":
    logger.warning(
        "Patched settings to work with 'localhost' URLs. \
        This is only applied when running locally, not in Docker."
    )

def main():
    parser = argparse.ArgumentParser(description="Start the LLM inference server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--model-run-id", type=str, default="", help="MLflow run ID of the model to use")
    args = parser.parse_args()
    
    # Set MLflow tracking URI
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Load model from MLflow
    run_id = args.model_run_id
    if not run_id:
        # Use environment variable if set
        run_id = os.getenv("MODEL_RUN_ID", "")
    
    # Load the model
    model = load_model(run_id)
    
    # If no model was loaded but we're still running, we'll serve the API without model functionality
    if model is None:
        logger.warning("No model was loaded. The API will run but prediction endpoints will return errors.")
    
    # Start the server
    logger.info(f"Starting inference server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
