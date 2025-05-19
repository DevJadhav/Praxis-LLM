from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = str(Path(__file__).parent.parent.parent.parent)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR, env_file_encoding="utf-8")

    # Model config
    HUGGINGFACE_API_ENDPOINT: str = "https://api-inference.huggingface.co"
    HUGGINGFACE_ACCESS_TOKEN: str | None = None
    HUGGINGFACE_MODEL_ID: str = "google/gemma-3-4b-it"  # Hugging Face model ID directly

    # Comet and Opik settings (keep for Opik prompt monitoring)
    COMET_API_KEY: str | None = None
    COMET_WORKSPACE: str | None = None
    COMET_PROJECT: str = "praxis-llm"
    
    # MLflow settings (for experiment tracking)
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"
    MLFLOW_EXPERIMENT_NAME: str = "llm-praxis"
    MLFLOW_SERVE_PORT: int = 1234  

    DATASET_ID: str = "articles-instruct-dataset"  # Comet artifact containing your fine-tuning dataset (available after generating the instruct dataset).


settings = Settings()
