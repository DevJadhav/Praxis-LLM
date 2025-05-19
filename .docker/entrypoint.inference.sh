#!/bin/bash

# Detect platform
PLATFORM=$(uname -s)
ARCH=$(uname -m)

echo "Detected platform: $PLATFORM, architecture: $ARCH"

# Install PyTorch based on detected platform and hardware
if [ "$ARCH" = "arm64" ]; then
    echo "ARM64 architecture detected - configuring PyTorch for Apple Silicon"
    # Install PyTorch for Apple Silicon
    uv pip install torch==2.7.0
    
    # Force enable MPS acceleration
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    export USE_MPS=1
    
    echo "PyTorch configured to use Apple Neural Engine/NPU/GPU when available"
    
    # Check for PyTorch MPS availability
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {getattr(torch.backends, \"mps\", None) and torch.backends.mps.is_available()}')"
elif command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected - installing PyTorch with CUDA support"
    # Install CUDA-enabled PyTorch
    uv pip install torch --index-url https://download.pytorch.org/whl/cu121
    
    # Check CUDA availability
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"
else
    echo "No GPU detected - installing PyTorch for CPU"
    # Install CPU-only PyTorch
    uv pip install torch
    
    echo "PyTorch installed for CPU-only usage"
fi

# Set default values for inference parameters
SERVE_UI=${SERVE_UI:-false}
MODEL_RUN_ID=${MODEL_RUN_ID:-""}
PORT=${PORT:-8000}

cd /app

if [ "$SERVE_UI" = "true" ]; then
    # Start Gradio UI
    echo "Starting Gradio UI on port $PORT"
    uv run inference_pipeline/ui.py --port $PORT
else
    # Start FastAPI service
    echo "Starting inference API on port $PORT"
    
    if [ -n "$MODEL_RUN_ID" ]; then
        # Load specific model from MLflow if run ID is provided
        uv run inference_pipeline/serve_mlflow_model.py --run-id $MODEL_RUN_ID --port $PORT
    else
        # Use default model (or model specified in environment variables)
        uv run inference_pipeline/main.py --port $PORT
    fi
fi 