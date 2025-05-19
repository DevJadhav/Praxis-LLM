import argparse
import os
import sys
from pathlib import Path

# Add src to the path
ROOT_DIR = str(Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)

import mlflow
from core.config import settings
from core.logger_utils import get_logger
from core.opik_utils import configure_opik, add_to_dataset_with_sampling

logger = get_logger(__name__)

def serve_model_locally(model_name: str = "PraxisLLM-Finetuned", model_version: str = "latest", port: int = None):
    """
    Serve a model locally using MLflow's model serving functionality.
    
    Args:
        model_name: Name of the registered model in MLflow
        model_version: Version of the model to serve (default: latest)
        port: Port to serve the model on (default: from settings)
    """
    # Set up MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Configure Opik for prompt monitoring
    configure_opik()
    
    # Get port from settings if not provided
    if port is None:
        port = settings.MLFLOW_SERVE_PORT
    
    # Construct model URI
    model_uri = f"models:/{model_name}/{model_version}"
    
    logger.info(f"Starting model serving for {model_uri} on port {port}")
    
    # This will start the model server - note that this blocks until the server is stopped
    mlflow.models.serve(model_uri=model_uri, port=port, enable_mlserver=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve a model locally using MLflow")
    parser.add_argument("--model-name", type=str, default="PraxisLLM-Finetuned", 
                        help="Name of the registered model in MLflow")
    parser.add_argument("--model-version", type=str, default="latest",
                        help="Version of the model to serve")
    parser.add_argument("--port", type=int, default=None,
                        help="Port to serve the model on (default: from settings)")
    
    args = parser.parse_args()
    
    serve_model_locally(args.model_name, args.model_version, args.port) 