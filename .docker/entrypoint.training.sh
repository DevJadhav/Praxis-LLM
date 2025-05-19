#!/bin/bash

# Detect platform
PLATFORM=$(uname -s)
ARCH=$(uname -m)

# Activate the virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

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

# Login to Hugging Face CLI if token is provided
if [ -n "$HUGGINGFACE_ACCESS_TOKEN" ]; then
    echo "Logging in to Hugging Face using provided access token..."
    huggingface-cli login --token $HUGGINGFACE_ACCESS_TOKEN
    echo "Hugging Face login successful"
else
    echo "HUGGINGFACE_ACCESS_TOKEN not provided. Skipping Hugging Face login."
fi

# Set default values for training parameters
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
LEARNING_RATE=${LEARNING_RATE:-0.0003}
IS_DUMMY=${IS_DUMMY:-false}

if [ "$IS_DUMMY" = "true" ]; then
    DUMMY_FLAG="--is-dummy"
else
    DUMMY_FLAG=""
fi

# Run the training script with uv
cd /app
uv run training_pipeline/local_training.py \
    --num-train-epochs ${NUM_TRAIN_EPOCHS} \
    --per-device-train-batch-size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    ${DUMMY_FLAG}

# Keep the container running if requested
if [ "$KEEP_CONTAINER_ALIVE" = "true" ]; then
    echo "Training completed. Container will remain running as requested."
    tail -f /dev/null
fi 