import argparse
import sys
import platform
from pathlib import Path
import os

# To mimic using multiple Python modules, such as 'core' and 'feature_pipeline',
# we will add the './src' directory to the PYTHONPATH. This is not intended for
# production use cases but for development and educational purposes.
ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from core import logger_utils
import mlflow

logger = logger_utils.get_logger(__file__)

from config import settings

finetuning_dir = Path(__file__).resolve().parent


def setup_device_environment():
    """Configure the environment to use Apple NPU when available, otherwise CPU"""
    is_mac = platform.system() == "Darwin"
    is_arm = platform.machine() == "arm64"
    
    if is_mac and is_arm:
        logger.info("Detected Apple Silicon (M1/M2/M3) - Configuring PyTorch for NPU/MPS")
        # Set environment variables for Apple Neural Engine (NPU)
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        
        # Check if PyTorch can use MPS
        try:
            import torch
            if torch.backends.mps.is_available():
                logger.info("PyTorch MPS backend is available! Will use Apple NPU for acceleration")
                return "mps"
            else:
                logger.warning("PyTorch MPS backend not available despite being on Apple Silicon")
                return "cpu"
        except (ImportError, AttributeError):
            logger.warning("PyTorch not installed or doesn't support MPS backend")
            return "cpu"
    else:
        if is_mac and not is_arm:
            logger.info("Detected Intel Mac - will use CPU")
        elif not is_mac:
            logger.info(f"Detected non-Mac platform ({platform.system()}/{platform.machine()}) - will use CPU")
        return "cpu"


def run_finetuning_local(
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 3e-4,
    is_dummy: bool = False,
) -> None:
    """
    Run the fine-tuning process locally (or in a container) with MLflow tracking.
    
    Args:
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        learning_rate: Learning rate for training
        is_dummy: Whether to run in dummy mode (faster for testing)
    """
    # Setup device environment (Apple NPU or CPU)
    device = setup_device_environment()
    logger.info(f"Training will use device: {device}")
    
    # Log that we're using HuggingFace directly
    logger.info("Using HuggingFace model directly")

    assert (
        settings.MLFLOW_TRACKING_URI
    ), "MLflow tracking URI (MLFLOW_TRACKING_URI) is required. Update your .env file."
    assert (
        settings.MLFLOW_EXPERIMENT_NAME
    ), "MLflow experiment name (MLFLOW_EXPERIMENT_NAME) is required. Update your .env file."

    if not finetuning_dir.exists():
        raise FileNotFoundError(f"The directory {finetuning_dir} does not exist.")

    # Set up MLflow tracking
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    # Define command arguments
    cmd_args = [
        f"--model_name={settings.HUGGINGFACE_MODEL_ID}",
        f"--dataset_id={settings.DATASET_ID}",
        f"--num_train_epochs={num_train_epochs}",
        f"--per_device_train_batch_size={per_device_train_batch_size}",
        f"--learning_rate={learning_rate}",
        f"--device={device}"
    ]
    
    if is_dummy:
        cmd_args.append("--is_dummy")
    
    # Import finetune module and run training directly
    logger.info(f"Starting training with args: {cmd_args}")
    
    from finetune import main as finetune_main
    finetune_main(cmd_args)
    
    # Log the completion
    logger.info("Training completed successfully")
    
    # Get the latest run to retrieve the run ID
    try:
        latest_run = mlflow.search_runs(experiment_names=[settings.MLFLOW_EXPERIMENT_NAME], order_by=["start_time DESC"]).iloc[0]
        run_id = latest_run.run_id
        logger.info(f"Training completed with MLflow run ID: {run_id}")
        logger.info(f"To serve the model, run: mlflow models serve -m \"runs:/{run_id}/model-finetuned\" -p 1234")
    except Exception as e:
        logger.warning(f"Failed to retrieve MLflow run ID: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is-dummy", action="store_true", help="Run in dummy mode")
    parser.add_argument("--num-train-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per-device-train-batch-size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    logger.info(f"Is the training pipeline in DUMMY mode? '{args.is_dummy}'")

    run_finetuning_local(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        is_dummy=args.is_dummy
    ) 