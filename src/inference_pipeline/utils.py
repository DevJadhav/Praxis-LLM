from config import settings
from transformers import AutoTokenizer
import os
import mlflow
import mlflow.pytorch


def compute_num_tokens(text: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID)

    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_text_to_max_tokens(text: str, max_tokens: int) -> tuple[str, int]:
    """Truncates text to not exceed max_tokens while trying to preserve complete sentences.

    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens allowed

    Returns:
        Truncated text that fits within max_tokens and the number of tokens in the truncated text.
    """

    current_tokens = compute_num_tokens(text)

    if current_tokens <= max_tokens:
        return text, current_tokens

    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID)
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Take first max_tokens tokens and decode
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens)

    # Try to end at last complete sentence
    last_period = truncated_text.rfind(".")
    if last_period > 0:
        truncated_text = truncated_text[: last_period + 1]

    truncated_tokens = compute_num_tokens(truncated_text)

    return truncated_text, truncated_tokens


def load_model_from_mlflow(model_name="PraxisLLM-Finetuned", stage="Production"):
    """
    Load a PyTorch model from MLflow model registry.
    
    Args:
        model_name: Name of the registered model in MLflow
        stage: Stage of the model (None, "Staging", "Production", or "Archived")
        
    Returns:
        Tuple of (model, tokenizer) loaded from MLflow
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
