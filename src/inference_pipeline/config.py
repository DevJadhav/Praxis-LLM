from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = str(Path(__file__).parent.parent.parent)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR, env_file_encoding="utf-8")

    # Embeddings config
    EMBEDDING_MODEL_ID: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_MODEL_MAX_INPUT_LENGTH: int = 512
    EMBEDDING_SIZE: int = 384
    EMBEDDING_MODEL_DEVICE: str = "cpu"
    
    def patch_localhost(self):
        """
        Patches the settings to use 'localhost' instead of container hostnames.
        This is useful for local development but should not be used in Docker.
        """
        if self.QDRANT_DATABASE_HOST == "qdrant":
            self.QDRANT_DATABASE_HOST = "localhost"
        # No need to patch Hugging Face endpoint for local development
        if self.MLFLOW_TRACKING_URI == "http://mlflow:5000":
            self.MLFLOW_TRACKING_URI = "http://localhost:5001"

    # OpenAI config
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_API_KEY: str | None = None

    # QdrantDB config
    QDRANT_DATABASE_HOST: str = "localhost"  # Or 'qdrant' if running inside Docker
    QDRANT_DATABASE_PORT: int = 6333

    USE_QDRANT_CLOUD: bool = (
        False  # if True, fill in QDRANT_CLOUD_URL and QDRANT_APIKEY
    )
    QDRANT_CLOUD_URL: str = "str"
    QDRANT_APIKEY: str | None = None

    # RAG config
    TOP_K: int = 5
    KEEP_TOP_K: int = 5
    EXPAND_N_QUERY: int = 5

    # Comet and Opik config
    COMET_API_KEY: str
    COMET_WORKSPACE: str
    COMET_PROJECT: str = "praxis-llm"
    
    # MLflow settings
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"
    MLFLOW_EXPERIMENT_NAME: str = "llm-praxis"
    MLFLOW_SERVE_PORT: int = 1234

    # LLM Model config
    HUGGINGFACE_API_ENDPOINT: str = "https://api-inference.huggingface.co"
    HUGGINGFACE_ACCESS_TOKEN: str | None = None
    MODEL_ID: str = "google/gemma-3-4b-it"  # Hugging Face model ID
    DEPLOYMENT_ENDPOINT_NAME: str = "praxis"

    MAX_INPUT_TOKENS: int = 1536  # Max length of input text.
    MAX_TOTAL_TOKENS: int = 2048  # Max length of the generation (including input text).
    MAX_BATCH_TOTAL_TOKENS: int = 2048  # Limits the number of tokens that can be processed in parallel during the generation.



settings = Settings()
