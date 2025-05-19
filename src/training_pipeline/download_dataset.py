import json
import os
from pathlib import Path
import logging

import mlflow
from config import settings
from core import get_logger
from datasets import Dataset, load_dataset  # noqa: E402

logger = get_logger(__file__)


class DatasetClient:
    def __init__(
        self,
        output_dir: Path = Path("./finetuning_dataset"),
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, dataset_id: str, split: str = "train") -> Dataset:
        """
        Download a dataset from MLflow or load from a Dataset ID.
        
        Args:
            dataset_id: Can be one of:
                - An MLflow path: experiment_id/run_id/artifact_path
                - A HuggingFace dataset ID (legacy support)
                - A local file prefix
            split: Either 'train' or 'test'
            
        Returns:
            Dataset object
        """
        assert split in ["train", "test"], "Split must be either 'train' or 'test'"

        # Set up MLflow tracking URI and experiment
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
        logger.info(f"Setting MLflow tracking URI to {mlflow_uri}")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", settings.MLFLOW_EXPERIMENT_NAME))

        # Try to load dataset based on the format of dataset_id
        if '/' in dataset_id:
            # Try to interpret as MLflow artifact path
            parts = dataset_id.split('/')
            if len(parts) >= 3:
                experiment_id, run_id, *artifact_path_parts = parts
                artifact_path = '/'.join(artifact_path_parts)
                
                try:
                    # Get MLflow client and download artifact
                    client = mlflow.tracking.MlflowClient()
                    client.download_artifacts(run_id=run_id, path=artifact_path, dst_path=str(self.output_dir))
                    
                    logger.info(f"Downloaded MLflow artifact from run {run_id} at {self.output_dir}")
                    
                    # Find the downloaded file based on split
                    split_files = list(self.output_dir.glob(f"*{split}*.json"))
                    if not split_files:
                        raise RuntimeError(f"No files matching {split} split found in downloaded artifacts")
                    
                    data_file_path = split_files[0]
                except Exception as e:
                    logger.error(f"Error downloading MLflow artifact: {str(e)}")
                    logger.info("Trying to interpret as a Dataset ID...")
                    try:
                        # Use HuggingFace for dataset loading
                        logger.info(f"Loading dataset from Hugging Face: {dataset_id}")
                        return load_dataset(dataset_id, split=split)
                    except Exception as load_error:
                        logger.error(f"Failed to load dataset: {str(load_error)}")
                        raise RuntimeError(f"Could not load dataset from {dataset_id}") from e
            else:
                # Treat as a Dataset ID
                logger.info(f"Treating {dataset_id} as a Dataset ID")
                return load_dataset(dataset_id, split=split)
        else:
            # Look for local file in output_dir
            data_file_path = self.output_dir / f"{dataset_id}_{split}.json"
            if not data_file_path.exists():
                raise FileNotFoundError(f"Dataset file {data_file_path} not found")

        # Load the dataset from the JSON file
        try:
            with open(data_file_path, "r") as file:
                data = json.load(file)
            
            # Convert to Dataset
            dataset_dict = {k: [str(d[k]) for d in data] for k in data[0].keys()}
            dataset = Dataset.from_dict(dataset_dict)
            
            logger.info(f"Successfully loaded dataset, num_samples = {len(dataset)}")
            
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise


if __name__ == "__main__":
    dataset_client = DatasetClient()
    try:
        dataset = dataset_client.download_dataset(dataset_id=settings.DATASET_ID)
        logger.info(f"Data available at '{dataset_client.output_dir}'. Dataset size: {len(dataset)}")
    except Exception as e:
        logger.error(f"Failed to download dataset: {str(e)}")
