# MLflow Migration Guide

This document outlines the changes made to migrate from SageMaker to MLflow for model training and deployment.

## Overview of Changes

The codebase has been modified to:
1. Remove SageMaker dependencies for training and deployment
2. Add MLflow-based local and remote training
3. Add MLflow-based model serving
4. Keep Opik for prompt monitoring

## Training with MLflow

There are two options for training with MLflow:

### Option A: Local Training

Run the training script locally with MLflow tracking:

```bash
python src/training_pipeline/local_training.py [--is-dummy] [--num-train-epochs N] [--per-device-train-batch-size N] [--learning-rate N]
```

This script provides automatic detection for Apple Silicon and will use NPU/MPS acceleration when available.

### Option B: MLflow on Remote Machine or Cluster

If you want to run on a remote machine or cluster, you can:
1. Use Docker to run the training in a containerized environment
2. Configure your MLflow tracking server to point to a remote instance
3. Use MLflow pipelines or your own provisioning system

## Deploying with MLflow

### Option A: MLflow Serve Locally

After training, you can serve the model locally using:

```bash
python src/inference_pipeline/deploy_mlflow_model.py [--run-id YOUR_RUN_ID] [--port 1234]
```

This will start an MLflow model server on the specified port (default: 1234).

### Option B: MLflow on Remote Machine

For remote deployment:
1. Set up MLflow on your remote machine
2. Modify the MLFLOW_TRACKING_URI in your config to point to the remote server
3. Deploy using the same script but with the appropriate tracking URI

## Inference Pipeline

The inference pipeline has been updated to call the MLflow REST endpoint instead of SageMaker:

```python
import requests

response = requests.post("http://localhost:1234/invocations", 
                         json={"inputs": {"messages": messages, "parameters": {...}}})
```

## Opik for Prompt Monitoring

Opik is still used for prompt monitoring, with no changes to the monitoring part of the code:

```python
@opik.track(...)
def generate(...):
    # function implementation
```

## Configuration

The configuration in `src/core/config.py` has been updated to include:

```python
# MLflow settings
MLFLOW_TRACKING_URI: str = "http://mlflow:5000"  # Update this for your setup
MLFLOW_EXPERIMENT_NAME: str = "llm-praxis"
MLFLOW_SERVE_PORT: int = 1234  # Port for serving MLflow models
```

## Important Notes

1. Make sure you have MLflow installed: `uv pip install mlflow`
2. For local development, you may need to set up an MLflow tracking server: `mlflow server --host 0.0.0.0`
3. Ensure your environment variables are set correctly in your .env file 