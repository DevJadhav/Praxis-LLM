import argparse
import subprocess
import sys
from pathlib import Path

# Add src directory to path for imports
ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from config import settings
from core import logger_utils

logger = logger_utils.get_logger(__name__)


def deploy_model_with_mlflow(run_id: str = None, port: int = 1234) -> None:
    """
    Deploy a model using MLflow's serve functionality.
    
    Args:
        run_id: The MLflow run ID of the model to deploy.
               If None, will try to find the latest run.
        port: The port on which to serve the model.
    """
    import mlflow
    
    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    
    if run_id is None:
        # Find the latest run in the experiment
        runs = mlflow.search_runs(
            experiment_names=[settings.MLFLOW_EXPERIMENT_NAME],
            order_by=["start_time DESC"]
        )
        if runs.empty:
            raise ValueError(f"No runs found in experiment {settings.MLFLOW_EXPERIMENT_NAME}")
        
        run_id = runs.iloc[0].run_id
        logger.info(f"Using latest run ID: {run_id}")
    
    # Set the environment variable for port
    settings.MLFLOW_SERVE_PORT = port
    
    # Construct the command to serve the model
    model_uri = f"runs:/{run_id}/model-finetuned"
    cmd = [
        "mlflow", "models", "serve",
        "-m", model_uri,
        "-p", str(port),
        "--enable-mlserver"
    ]
    
    logger.info(f"Starting MLflow model server with command: {' '.join(cmd)}")
    
    # This will block until the server is stopped
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Model server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting model server: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Deploy a model with MLflow")
    parser.add_argument("--run-id", help="MLflow run ID to deploy. If not provided, will use the latest run.")
    parser.add_argument("--port", type=int, default=1234, help="Port on which to serve the model. Default: 1234")
    
    args = parser.parse_args()
    
    # Update settings for use in other modules
    if not hasattr(settings, "MLFLOW_SERVE_PORT"):
        setattr(settings, "MLFLOW_SERVE_PORT", args.port)
    
    deploy_model_with_mlflow(run_id=args.run_id, port=args.port)


if __name__ == "__main__":
    main() 