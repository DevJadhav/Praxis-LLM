# Local Containerized Setup

This project has been configured to run in containerized environments on local machines, with special attention to compatibility with Apple Silicon (M1/M2/M3) and CPU-based training.

## Infrastructure Overview

The project uses Docker containers for all components:

- **MongoDB**: For data storage
- **RabbitMQ**: For message queuing
- **Qdrant**: For vector database
- **MLflow**: For experiment tracking
- **Training Pipeline**: For model training
- **Inference Pipeline**: For model serving
- **Inference UI**: Gradio UI for interacting with the model

## Requirements

- Docker & Docker Compose
- Git
- Minimum 16GB RAM recommended
- 50GB disk space
- Apple Silicon (M1/M2/M3) or Intel/AMD CPU
- UV package manager (installed automatically)

## Getting Started

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in the required values
3. Start the infrastructure:

```bash
make local-start
```

## Training a Model

To start the training pipeline:

```bash
make start-training-pipeline
```

For a quick test with dummy data:

```bash
make start-training-pipeline-dummy-mode
```

## Using the Model

Start the inference API:

```bash
make start-inference-pipeline
```

Start the Gradio UI:

```bash
make start-inference-ui
```

Then access the UI at http://localhost:8050

## Monitoring and Evaluation

View training metrics and models in MLflow:

```bash
make start-mlflow-docker
```

Then access MLflow at http://localhost:5001

Run evaluations:

```bash
make evaluate-llm
make evaluate-rag
make evaluate-llm-monitoring
```

## Apple Silicon NPU Compatibility

The Docker containers are configured to automatically detect Apple Silicon and use the Neural Processing Unit (NPU) when running on Mac:

- On Apple Silicon Macs (M1/M2/M3), the system will use the MPS (Metal Performance Shaders) backend with NPU acceleration
- On non-Mac systems, CPU-based computation will be used
- The system sets these environment variables to optimize NPU performance:
  - `PYTORCH_ENABLE_MPS_FALLBACK=1`: Enables PyTorch to use MPS backend
  - `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`: Optimizes memory usage
  - `UVM_ENABLE_LOGGING=1`: Enhances debugging support

## Package Management with UV

This project uses the UV package manager instead of pip or poetry for faster installations and better reproducibility:

- UV is automatically installed in the containers
- All script calls use `uv run` instead of `python`
- Dependencies are installed using `uv pip install`

## Command Reference

See the `Makefile` for all available commands:

```bash
make help
``` 