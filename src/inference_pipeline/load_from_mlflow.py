import os
import mlflow
import mlflow.pytorch
from transformers import AutoTokenizer, TextStreamer
from config import settings

def load_model_from_mlflow(model_name="PraxisLLM-Finetuned", stage="Production"):
    """
    Load a PyTorch model from MLflow model registry.
    
    Args:
        model_name: Name of the registered model in MLflow
        stage: Stage of the model (None, "Staging", "Production", or "Archived")
        
    Returns:
        Loaded model from MLflow
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI))
    
    # Construct the model URI
    model_uri = f"models:/{model_name}/{stage}"
    print(f"Loading model from MLflow: {model_uri}")
    
    # Load the model from MLflow
    loaded_model = mlflow.pytorch.load_model(model_uri)
    
    # Get the tokenizer from the same run as the model
    run_id = mlflow.register_model.get_latest_versions(model_name, stages=[stage])[0].run_id
    client = mlflow.tracking.MlflowClient()
    
    # Download the tokenizer artifacts
    artifact_path = client.download_artifacts(run_id, "tokenizer", ".")
    tokenizer = AutoTokenizer.from_pretrained(artifact_path)
    
    return loaded_model, tokenizer

def inference_with_mlflow_model(
    prompt="Write a paragraph to introduce supervised fine-tuning.",
    max_new_tokens=256,
    model_name="PraxisLLM-Finetuned",
    stage="Production"
):
    """
    Run inference using a model loaded from MLflow.
    
    Args:
        prompt: Input prompt for the model
        max_new_tokens: Maximum number of tokens to generate
        model_name: Name of the registered model in MLflow
        stage: Stage of the model (None, "Staging", "Production", or "Archived")
    """
    # Load the model and tokenizer from MLflow
    model, tokenizer = load_model_from_mlflow(model_name, stage)
    
    # Prepare input
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda" if hasattr(model, "device") and model.device.type == "cuda" else "cpu")
    
    # Set up text streamer for generation
    text_streamer = TextStreamer(tokenizer)
    
    # Generate text
    _ = model.generate(
        **inputs, streamer=text_streamer, max_new_tokens=max_new_tokens, use_cache=True
    )

if __name__ == "__main__":
    # Example usage
    inference_with_mlflow_model(
        prompt="Explain how to implement attention mechanism in transformers.",
        model_name="PraxisLLM-Finetuned",
        stage="Production"
    )
    
    # Alternative: just load the model for use in an application
    # model, tokenizer = load_model_from_mlflow() 