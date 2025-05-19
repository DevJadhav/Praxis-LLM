"""
Script to serve a fine-tuned model from MLflow.

Usage:
    python serve_mlflow_model.py --model_name PraxisLLM-Finetuned --stage Production --port 8080
"""

import argparse
import sys
import platform
from pathlib import Path
import os
import mlflow
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

# Add parent directory to path
ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from core import logger_utils
from config import settings

logger = logger_utils.get_logger(__file__)

app = FastAPI(title="LLM Model Server")

class ChatMessage(BaseModel):
    role: str
    content: str

class InferenceRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 512

class InferenceResponse(BaseModel):
    response: str
    model_info: Dict[str, Any]

# Global model variables
MODEL = None
MODEL_INFO = {}


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


def load_model(run_id: str):
    """Load model from MLflow by run_id"""
    global MODEL, MODEL_INFO
    
    # Configure device environment
    device = setup_device_environment()
    logger.info(f"Inference will use device: {device}")
    
    if not run_id:
        # Get the latest run
        runs = mlflow.search_runs(experiment_names=[settings.MLFLOW_EXPERIMENT_NAME], order_by=["start_time DESC"])
        if runs.empty:
            logger.error(f"No runs found in experiment '{settings.MLFLOW_EXPERIMENT_NAME}'. Please train a model first or provide a valid run_id.")
            MODEL_INFO = {
                "status": "error",
                "message": f"No runs found in experiment '{settings.MLFLOW_EXPERIMENT_NAME}'"
            }
            return None
        
        latest_run = runs.iloc[0]
        run_id = latest_run.run_id
        logger.info(f"Using latest MLflow run ID: {run_id}")
    
    model_uri = f"runs:/{run_id}/model-finetuned"
    logger.info(f"Loading model from {model_uri}")
    
    # Load the model in PyFunc format
    MODEL = mlflow.pyfunc.load_model(model_uri)
    
    # Store model info
    MODEL_INFO = {
        "run_id": run_id,
        "model_uri": model_uri,
        "device": device
    }
    
    # Get additional info from the run if available
    try:
        run = mlflow.get_run(run_id)
        if run and run.data:
            if hasattr(run.data, "params"):
                MODEL_INFO["params"] = run.data.params
            if hasattr(run.data, "metrics"):
                MODEL_INFO["metrics"] = run.data.metrics
            if hasattr(run.data, "tags"):
                MODEL_INFO["tags"] = run.data.tags
    except Exception as e:
        logger.warning(f"Failed to get run details: {e}")
    
    logger.info(f"Model loaded successfully: {MODEL_INFO}")
    return MODEL


@app.get("/")
def root():
    return {"message": "LLM Inference Server", "status": "running"}


@app.get("/health")
def health():
    if MODEL is None:
        status = "not_ready"
        message = "Model not loaded"
        if MODEL_INFO and "message" in MODEL_INFO:
            message = MODEL_INFO["message"]
        return {
            "status": status, 
            "message": message, 
            "model_info": MODEL_INFO
        }
    return {"status": "ready", "model_info": MODEL_INFO}


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    if MODEL is None:
        error_message = "Model not loaded"
        if MODEL_INFO and "message" in MODEL_INFO:
            error_message = MODEL_INFO["message"]
        raise HTTPException(
            status_code=503, 
            detail=error_message
        )
    
    try:
        # Format messages for model input
        formatted_messages = []
        for msg in request.messages:
            formatted_messages.append({"role": msg.role, "content": msg.content})
        
        # Set prediction parameters
        params = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "device": MODEL_INFO.get("device", "cpu")
        }
        
        # Run inference
        response = MODEL.predict({"messages": formatted_messages, "parameters": params})
        
        if isinstance(response, dict) and "generation" in response:
            return {"response": response["generation"], "model_info": MODEL_INFO}
        elif isinstance(response, str):
            return {"response": response, "model_info": MODEL_INFO}
        else:
            logger.warning(f"Unexpected model response format: {response}")
            return {"response": str(response), "model_info": MODEL_INFO}
            
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


def main(args=None):
    parser = argparse.ArgumentParser(description="Serve a fine-tuned LLM model from MLflow")
    parser.add_argument("--run-id", type=str, default="", help="MLflow run ID of the model to serve")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    
    # Load the model
    load_model(args.run_id)
    
    # Run the server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main() 