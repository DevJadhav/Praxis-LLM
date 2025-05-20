include .env

$(eval export $(shell sed -ne 's/ *#.*$$//; /./ s/=.*$$// p' .env))

PYTHONPATH := $(shell pwd)/src
PLATFORM := $(shell uname -s)
ARCH := $(shell uname -m)

uv-install: # Install UV if not already installed.
	curl -LsSf https://astral.sh/uv/install.sh | sh

install: # Create a local UV virtual environment and install all required Python dependencies.
	uv python install 3.12
	uv venv
	uv sync
	source .venv/bin/activate

help:
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

# ======================================
# ------- Docker Infrastructure --------
# ======================================

local-start: # Build and start your local Docker infrastructure.
	docker compose -f docker-compose.yml up --build -d

gpu-start: # Build and start your Docker infrastructure with GPU support.
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "NVIDIA GPU detected, starting with GPU support"; \
		docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d; \
	else \
		echo "NVIDIA GPU not detected, cannot use GPU support"; \
		exit 1; \
	fi

local-stop: # Stop your local Docker infrastructure.
	docker compose -f docker-compose.yml down --remove-orphans

# ======================================
# ---------- Crawling Data -------------
# ======================================

local-test-medium: # Make a call to your local container to crawl a Medium article.
	curl -X POST "http://localhost:9010/2015-03-31/functions/function/invocations" \
	  	-d '{"user": "Dev Jadhav", "link": "https://medium.com/mlwithdev/how-amazon-bedrocks-advanced-routing-is-changing-the-ai-game-8eecc88e7405"}'

local-test-github: # Make a call to your local container to crawl a Github repository.
	curl -X POST "http://localhost:9010/2015-03-31/functions/function/invocations" \
	  	-d '{"user": "Dev Jadhav", "link": "https://github.com/DevJadhav/GenerativeAI.git"}'

local-ingest-data: # Ingest all links from data/links.txt by calling your local container.
	while IFS= read -r link; do \
		echo "Processing: $$link"; \
		curl -X POST "http://localhost:9010/2015-03-31/functions/function/invocations" \
			-d "{\"user\": \"Dev Jadhav\", \"link\": \"$$link\"}"; \
		echo "\n"; \
		sleep 2; \
	done < data/links.txt

# ======================================
# -------- RAG Feature Pipeline --------
# ======================================

local-test-retriever: # Test the RAG retriever using uv
	cd src/feature_pipeline && uv run python -m retriever

local-generate-instruct-dataset: # Generate the fine-tuning instruct dataset using uv
	cd src/feature_pipeline && uv run python -m generate_dataset.generate

# ======================================
# -------- Training Pipeline ----------
# ======================================

download-instruct-dataset: # Download the fine-tuning instruct dataset.
	cd src/training_pipeline && PYTHONPATH=$(PYTHONPATH) uv run download_dataset.py

start-training-pipeline: # Start the training pipeline in a Docker container, auto-detect CPU/GPU.
	docker compose -f docker-compose.yml up training_pipeline --build

gpu-training-pipeline: # Start the training pipeline in a Docker container with NVIDIA GPU support.
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "NVIDIA GPU detected, starting with GPU support"; \
		docker compose -f docker-compose.yml -f docker-compose.gpu.yml up training_pipeline --build; \
	else \
		echo "NVIDIA GPU not detected, cannot use GPU support"; \
		exit 1; \
	fi

start-training-pipeline-dummy-mode: # Start the training pipeline in dummy mode.
	docker compose run -e IS_DUMMY=true training_pipeline

apple-training-pipeline: # Start the training pipeline natively on Apple Silicon Macs.
	@if [ "$(PLATFORM)" = "Darwin" ] && [ "$(ARCH)" = "arm64" ]; then \
		echo "Apple Silicon detected, running training pipeline natively"; \
		cd src/training_pipeline && \
		PYTHONPATH=$(PYTHONPATH) \
		PYTORCH_ENABLE_MPS_FALLBACK=1 \
		PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
		USE_MPS=1 \
		MONGO_URI="mongodb://localhost:30001,localhost:30002,localhost:30003/?replicaSet=my-replica-set" \
		MLFLOW_TRACKING_URI="http://localhost:5001" \
		QDRANT_HOST="localhost" \
		QDRANT_PORT=6333 \
		DOCKER_CONTAINER=false \
		SKIP_DB_CONNECTION=true \
		uv run -m finetune; \
	else \
		echo "This command is only for Apple Silicon Macs"; \
		exit 1; \
	fi

local-start-training-pipeline: # Start the training pipeline in your local environment.
	cd src/training_pipeline && uv run -m finetune

# ======================================
# -------- Inference Pipeline ----------
# ======================================

start-inference-pipeline: # Start the inference pipeline API in a Docker container.
	docker compose -f docker-compose.yml up inference_pipeline --build -d

gpu-inference-pipeline: # Start the inference pipeline API in a Docker container with NVIDIA GPU support.
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "NVIDIA GPU detected, starting with GPU support"; \
		docker compose -f docker-compose.yml -f docker-compose.gpu.yml up inference_pipeline --build -d; \
	else \
		echo "NVIDIA GPU not detected, cannot use GPU support"; \
		exit 1; \
	fi

start-inference-ui: # Start the Gradio UI for chatting with your LLM in a Docker container.
	docker compose -f docker-compose.yml up inference_ui --build -d

gpu-inference-ui: # Start the Gradio UI for chatting with your LLM in a Docker container with NVIDIA GPU support.
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "NVIDIA GPU detected, starting with GPU support"; \
		docker compose -f docker-compose.yml -f docker-compose.gpu.yml up inference_ui --build -d; \
	else \
		echo "NVIDIA GPU not detected, cannot use GPU support"; \
		exit 1; \
	fi

apple-inference-pipeline: # Start the inference pipeline API natively on Apple Silicon Macs.
	@if [ "$(PLATFORM)" = "Darwin" ] && [ "$(ARCH)" = "arm64" ]; then \
		echo "Apple Silicon detected, running inference pipeline natively"; \
		cd src/inference_pipeline && \
		PYTHONPATH=$(PYTHONPATH) \
		PYTORCH_ENABLE_MPS_FALLBACK=1 \
		PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
		USE_MPS=1 \
		MONGO_URI="mongodb://localhost:30001,localhost:30002,localhost:30003/?replicaSet=my-replica-set" \
		MLFLOW_TRACKING_URI="http://localhost:5001" \
		QDRANT_HOST="localhost" \
		QDRANT_PORT=6333 \
		DOCKER_CONTAINER=false \
		SKIP_DB_CONNECTION=true \
		uv run -m main --port 8000; \
	else \
		echo "This command is only for Apple Silicon Macs"; \
		exit 1; \
	fi

apple-inference-ui: # Start the Gradio UI for chatting with your LLM natively on Apple Silicon Macs.
	@if [ "$(PLATFORM)" = "Darwin" ] && [ "$(ARCH)" = "arm64" ]; then \
		echo "Apple Silicon detected, running inference UI natively"; \
		cd src/inference_pipeline && \
		PYTHONPATH=$(PYTHONPATH) \
		PYTORCH_ENABLE_MPS_FALLBACK=1 \
		PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
		USE_MPS=1 \
		MONGO_URI="mongodb://localhost:30001,localhost:30002,localhost:30003/?replicaSet=my-replica-set" \
		MLFLOW_TRACKING_URI="http://localhost:5001" \
		QDRANT_HOST="localhost" \
		QDRANT_PORT=6333 \
		DOCKER_CONTAINER=false \
		SKIP_DB_CONNECTION=true \
		SERVE_UI=true \
		uv run -m ui; \
	else \
		echo "This command is only for Apple Silicon Macs"; \
		exit 1; \
	fi

local-start-ui: # Start the Gradio UI for chatting with your LLM using your local environment.
	cd src/inference_pipeline && uv run -m ui

# ======================================
# ------------ MLflow & Evaluation -----
# ======================================

start-mlflow-ui: # Start MLflow UI locally
	mlflow ui --port 5000

serve-mlflow-model: # Serve the registered MLflow model locally
	mlflow models serve -m "models:/PraxisLLM-Finetuned/latest" -p $(MLFLOW_SERVE_PORT) --enable-mlserver

serve-local-model: # Serve the model locally using our custom script
	cd src/inference_pipeline && PYTHONPATH=$(PYTHONPATH) python serve_local.py

start-mlflow-docker: # Start MLflow UI in Docker
	docker compose up mlflow -d

evaluate-llm: # Run evaluation tests on the LLM model's performance.
	cd src/inference_pipeline && uv run -m evaluation.evaluate

evaluate-rag: # Run evaluation tests specifically on the RAG system's performance.
	cd src/inference_pipeline && uv run -m evaluation.evaluate_rag

evaluate-llm-monitoring: # Run evaluation tests for monitoring the LLM system.
	cd src/inference_pipeline && uv run -m evaluation.evaluate_monitoring

# ======================================
# ------------ Local Deployment --------
# ======================================

deploy-local-model: # Deploy the model locally using MLflow
	cd src/inference_pipeline && PYTHONPATH=$(PYTHONPATH) python deploy_mlflow_model.py

call-local-model: # Test the local MLflow model endpoint
	curl -X POST http://localhost:$(MLFLOW_SERVE_PORT)/invocations -H "Content-Type: application/json" -d '{"inputs": {"messages": [{"role": "user", "content": "Write a short post about RAG systems"}], "parameters": {"temperature": 0.7, "max_tokens": 512}}}'
