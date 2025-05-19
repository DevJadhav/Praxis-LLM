# Multi-Platform Execution Guide

This project now supports multiple hardware platforms:
1. Apple Silicon Macs (M1/M2/M3)
2. NVIDIA GPU-equipped systems
3. CPU-only systems

## Apple Silicon Macs

For native execution leveraging Apple's MPS (Metal Performance Shaders) acceleration:

```bash
# Start training pipeline natively
make apple-training-pipeline

# Start inference API natively
make apple-inference-pipeline

# Start Gradio UI natively
make apple-inference-ui
```

## NVIDIA GPU Systems

For containerized execution using NVIDIA GPUs:

```bash
# Start the entire infrastructure with GPU support
make gpu-start

# Start only the training pipeline with GPU support
make gpu-training-pipeline

# Start only the inference API with GPU support
make gpu-inference-pipeline

# Start only the Gradio UI with GPU support
make gpu-inference-ui
```

## CPU-Only Systems

For containerized execution on CPU:

```bash
# Start the entire infrastructure on CPU
make local-start

# Start only the training pipeline on CPU
make start-training-pipeline

# Start only the inference API on CPU
make start-inference-pipeline

# Start only the Gradio UI on CPU
make start-inference-ui
```

## Requirements

### For NVIDIA GPU support:
- NVIDIA GPU with compatible drivers
- NVIDIA Container Toolkit installed
- Docker with NVIDIA runtime

### For Apple Silicon support:
- Apple Silicon Mac (M1/M2/M3)
- PyTorch 2.0+ with MPS support

### For any platform:
- Docker and Docker Compose
- Python 3.12+ 