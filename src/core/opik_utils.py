import json
import os
import random
import tempfile
from pathlib import Path

from tqdm import tqdm

from core.config import settings
from core.logger_utils import get_logger

logger = get_logger(__name__)

# Try to import Opik if it's installed, otherwise log a warning
try:
    import opik
    from opik.configurator.configure import OpikConfigurator
except ImportError:
    logger.info("Could not import Opik. Prompt monitoring will be disabled.")
    opik = None
    OpikConfigurator = None

# Try to import MLflow
try:
    import mlflow
except ImportError:
    logger.warning("Could not import MLflow. Experiment tracking will be disabled.")
    mlflow = None


def configure_opik() -> None:
    """
    Configure Opik for prompt monitoring using the API key from settings
    """
    if settings.COMET_API_KEY and settings.COMET_PROJECT:
        if settings.COMET_WORKSPACE:
            default_workspace = settings.COMET_WORKSPACE
        else:
            try:
                client = OpikConfigurator(api_key=settings.COMET_API_KEY)
                default_workspace = client._get_default_workspace()
            except Exception:
                logger.warning(
                    "Default workspace not found. Setting workspace to None and enabling interactive mode."
                )
                default_workspace = None

        os.environ["OPIK_PROJECT_NAME"] = settings.COMET_PROJECT

        opik.configure(
            api_key=settings.COMET_API_KEY,
            workspace=default_workspace,
            use_local=False,
            force=True,
        )
        logger.info("Opik configured successfully for prompt monitoring.")
    else:
        logger.warning(
            "COMET_API_KEY and COMET_PROJECT are not set. Set them to enable prompt monitoring with Opik."
        )


def create_dataset_from_artifacts(
    dataset_name: str, artifact_names: list[str]
) -> opik.Dataset | None:
    """
    Creates an Opik dataset from MLflow artifacts
    """
    if not (opik and mlflow):
        logger.warning("Opik or MLflow not available. Cannot create dataset from artifacts.")
        return None
    
    client = opik.Opik()
    try:
        dataset = client.get_dataset(name=dataset_name)
        logger.warning(f"Dataset '{dataset_name}' already exists. Skipping dataset creation.")
        return dataset
    except Exception:
        dataset = None

    # Use MLflow to get artifacts
    mlflow_client = mlflow.tracking.MlflowClient()
    
    dataset_items = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for artifact_name in tqdm(artifact_names):
            artifact_dir = Path(tmp_dir) / artifact_name
            artifact_dir.mkdir(exist_ok=True, parents=True)
            
            try:
                # Get run_id based on artifact name or other criteria
                # This is a simplification - you'll need to implement how to find the right run
                runs = mlflow.search_runs(filter_string=f"tags.artifact_name='{artifact_name}'")
                if len(runs) == 0:
                    logger.warning(f"No runs found for artifact {artifact_name}")
                    continue
                    
                run_id = runs.iloc[0].run_id
                mlflow_client.download_artifacts(run_id, artifact_name, str(artifact_dir))
                logger.info(f"Successfully downloaded '{artifact_name}' at location '{tmp_dir}'")
            except Exception as e:
                logger.error(f"Error retrieving artifact: {str(e)}")
                continue

            testing_artifact_file = list(artifact_dir.glob("*_testing.json"))
            if not testing_artifact_file:
                logger.warning(f"No testing artifact file found for {artifact_name}")
                continue
                
            testing_artifact_file = testing_artifact_file[0]
            logger.info(f"Loading testing data from: {testing_artifact_file}")
            
            with open(testing_artifact_file, "r") as file:
                items = json.load(file)

            enhanced_items = [
                {**item, "artifact_name": artifact_name} for item in items
            ]
            dataset_items.extend(enhanced_items)

    if len(dataset_items) == 0:
        logger.warning("No items found in the artifacts. Dataset creation skipped.")
        return None

    dataset = create_dataset(
        name=dataset_name,
        description="Dataset created from artifacts",
        items=dataset_items,
    )

    return dataset


def create_dataset(name: str, description: str, items: list[dict]) -> opik.Dataset | None:
    """
    Create an Opik dataset for prompt monitoring
    """
    if not opik:
        logger.warning("Opik not available. Cannot create dataset.")
        return None
        
    client = opik.Opik()
    dataset = client.get_or_create_dataset(name=name, description=description)
    dataset.insert(items)
    dataset = client.get_dataset(name=name)
    return dataset


def add_to_dataset_with_sampling(item: dict, dataset_name: str) -> bool:
    """
    Add an item to an Opik dataset with random sampling
    """
    if not opik:
        logger.warning("Opik not available. Cannot add to dataset.")
        return False
        
    if "1" in random.choices(["0", "1"], weights=[0.3, 0.7]):
        client = opik.Opik()
        dataset = client.get_or_create_dataset(name=dataset_name)
        dataset.insert([item])
        return True

    return False
