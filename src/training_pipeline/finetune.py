import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any, List, Optional  # noqa: E402

import torch  # noqa
import mlflow
import transformers
from datasets import Dataset, concatenate_datasets, load_dataset  # noqa: E402
from transformers import (
    AutoModelForCausalLM,  # Changed Gemma3ForCausalLM to AutoModelForCausalLM
    AutoTokenizer,
    TextStreamer,
    TrainingArguments,
    BitsAndBytesConfig,
    Gemma3ForCausalLM,
)  # noqa: E402
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)  # noqa: E402
from trl import SFTTrainer  # noqa: E402

from core.config import settings
from core.opik_utils import configure_opik

# Initialize MLflow for experiment tracking
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", settings.MLFLOW_EXPERIMENT_NAME))

# Configure Opik for prompt monitoring if available
configure_opik()

ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""


class DatasetClient:
    def __init__(
        self,
        output_dir: Path = Path("./finetuning_dataset"),
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up MLflow tracking URI and experiment
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", settings.MLFLOW_EXPERIMENT_NAME))
        self.client = mlflow.tracking.MlflowClient()

    def download_dataset(self, dataset_id: str, split: str = "train") -> Dataset:
        """
        Download and prepare dataset for training.
        
        Args:
            dataset_id: Can be one of:
                - MLflow artifact path: "run_id/artifact_path"
                - MLflow registered dataset name: "registered:dataset_name"
                - HuggingFace dataset ID
                - Local dataset file name (without .json extension)
            split: Dataset split to use ('train' or 'test')
            
        Returns:
            Dataset: The loaded dataset
        """
        assert split in ["train", "test"], "Split must be either 'train' or 'test'"

        # Log the dataset source for reproducibility
        if mlflow.active_run():
            mlflow.log_param("dataset_id", dataset_id)
            mlflow.log_param("dataset_split", split)
        
        # Case 1: MLflow registered dataset (format: "registered:dataset_name")
        if dataset_id.startswith("registered:"):
            dataset_name = dataset_id.split(":", 1)[1]
            try:
                # Get the latest version of the registered model
                latest_version = self.client.get_latest_versions(dataset_name, stages=["Production"])[0]
                run_id = latest_version.run_id
                # Download the artifact
                artifact_uri = f"runs:/{run_id}/dataset"
                local_path = self.client.download_artifacts(run_id, "dataset", str(self.output_dir))
                print(f"Downloaded registered dataset '{dataset_name}' from run {run_id}")
                
                # Look for the split file
                split_files = list(Path(local_path).glob(f"*{split}*.json"))
                if not split_files:
                    raise RuntimeError(f"No files matching {split} split found in downloaded artifacts")
                
                data_file_path = split_files[0]
            except Exception as e:
                raise RuntimeError(f"Failed to download registered dataset '{dataset_name}': {e}")
        
        # Case 2: MLflow run artifact (format: "run_id/artifact_path")
        elif '/' in dataset_id:
            parts = dataset_id.split('/')
            if len(parts) >= 2:
                run_id, *artifact_path_parts = parts
                artifact_path = '/'.join(artifact_path_parts) if artifact_path_parts else "dataset"
                
                try:
                    # Download the artifact
                    local_path = self.client.download_artifacts(run_id, artifact_path, str(self.output_dir))
                    print(f"Downloaded MLflow artifact from run {run_id} at {local_path}")
                    
                    # Find the downloaded file based on split
                    split_files = list(Path(local_path).glob(f"*{split}*.json"))
                    if not split_files:
                        raise RuntimeError(f"No files matching {split} split found in downloaded artifacts")
                    
                    data_file_path = split_files[0]
                except Exception as e:
                    # If MLflow download fails, try HuggingFace dataset
                    print(f"Failed to download from MLflow, trying as HuggingFace dataset: {e}")
                    try:
                        return load_dataset(dataset_id, split=split)
                    except Exception:
                        raise RuntimeError(f"Failed to load dataset '{dataset_id}' from both MLflow and HuggingFace")
            else:
                # Try as HuggingFace dataset
                print(f"Treating {dataset_id} as a Dataset ID")
                return load_dataset(dataset_id, split=split)
        else:
            # Look for local file in output_dir
            data_file_path = self.output_dir / f"{dataset_id}_{split}.json"
            if not data_file_path.exists():
                # Try as HuggingFace dataset
                try:
                    return load_dataset(dataset_id, split=split)
                except Exception:
                    raise FileNotFoundError(f"Dataset file {data_file_path} not found and '{dataset_id}' not found on HuggingFace")

        # Load the dataset from the JSON file
        with open(data_file_path, "r") as file:
            data = json.load(file)
        
        # Convert to Dataset
        dataset_dict = {k: [str(d[k]) for d in data] for k in data[0].keys()}
        dataset = Dataset.from_dict(dataset_dict)
        
        # Log dataset stats to MLflow
        if mlflow.active_run():
            mlflow.log_metrics({
                "dataset_size": len(dataset),
                "num_columns": len(dataset_dict)
            })
            
            # Log dataset sample as artifact
            sample_path = str(self.output_dir / "dataset_sample.json") 
            with open(sample_path, "w") as f:
                json.dump(data[:5] if len(data) > 5 else data, f, indent=2)
            mlflow.log_artifact(sample_path, "dataset_samples")
        
        print(f"Successfully loaded dataset, num_samples = {len(dataset)}")
        
        return dataset


def load_model(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    chat_template: str,
) -> tuple:
    # Set up quantization config if needed
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    
    # Always use gemma-3-4b-it directly, regardless of input model name
    hf_model_name = "google/gemma-3-4b-it"  # Keep this as per existing code
    print(f"Using HuggingFace model {hf_model_name}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise
    
    # Set pad token if it doesn't exist
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = ""
    
    # Apply chat template
    if chat_template == "chatml":
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'assistant' %}\n<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'system' %}\n<|im_start|>system\n{{ message['content'] }}<|im_end|>\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt %}\n<|im_start|>assistant\n{% endif %}"
    
    # Load model with quantization if specified
    try:
        if quantization_config:
            model = Gemma3ForCausalLM.from_pretrained(  # Changed AutoModelForCausalLM to Gemma3ForCausalLM
                hf_model_name,
                quantization_config=quantization_config,
                device_map={"": "cpu"},  # Use CPU for all model parts instead of auto
                trust_remote_code=True,
            )
        else:
            model = Gemma3ForCausalLM.from_pretrained(  # Changed AutoModelForCausalLM to Gemma3ForCausalLM
                hf_model_name,
                device_map={"": "cpu"},  # Use CPU for all model parts instead of auto
                trust_remote_code=True,
            )
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Fix model padding for Apple Silicon compatibility
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Setup for 4-bit training if needed
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure PEFT (LoRA) for efficient fine-tuning
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Set model length
    if hasattr(model.config, "max_position_embeddings"):
        context_length = min(max_seq_length, model.config.max_position_embeddings)
    else:
        context_length = max_seq_length
    
    tokenizer.model_max_length = context_length
    
    return model, tokenizer


def finetune(
    model_name: str,
    output_dir: str,
    dataset_id: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "up_proj",
        "down_proj",
        "o_proj",
        "gate_proj",
    ],  # noqa: B006
    chat_template: str = "chatml",
    learning_rate: float = 3e-4,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    is_dummy: bool = True,
) -> tuple:
    # Start an MLflow run for experiment tracking
    with mlflow.start_run(run_name=f"finetune-{model_name}"):
        # Log parameters
        mlflow.log_params({
            "model_name": model_name,
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "chat_template": chat_template,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "is_dummy": is_dummy
        })
        
        model, tokenizer = load_model(
            model_name,
            max_seq_length,
            load_in_4bit,
            lora_rank,
            lora_alpha,
            lora_dropout,
            target_modules,
            chat_template,
        )
        EOS_TOKEN = tokenizer.eos_token
        print(f"Setting EOS_TOKEN to {EOS_TOKEN}")  # noqa

        if is_dummy is True:
            num_train_epochs = 1
            print(
                f"Training in dummy mode. Setting num_train_epochs to '{num_train_epochs}'"
            )  # noqa
            print(f"Training in dummy mode. Reducing dataset size to '400'.")  # noqa

        def format_samples_sft(examples):
            text = []
            for instruction, output in zip(
                examples["instruction"], examples["content"], strict=False
            ):
                message = ALPACA_TEMPLATE.format(instruction, output) + EOS_TOKEN
                text.append(message)

            return {"text": text}

        dataset_client = DatasetClient()
        custom_dataset = dataset_client.download_dataset(dataset_id=dataset_id)
        static_dataset = load_dataset(
            "mlabonne/FineTome-Alpaca-100k", split="train[:10000]"
        )
        dataset = concatenate_datasets([custom_dataset, static_dataset])
        if is_dummy:
            dataset = dataset.select(range(400))
        print(f"Loaded dataset with {len(dataset)} samples.")  # noqa

        dataset = dataset.map(
            format_samples_sft, batched=True, remove_columns=dataset.column_names
        )
        dataset = dataset.train_test_split(test_size=0.05)

        print("Training dataset example:")  # noqa
        print(dataset["train"][0])  # noqa

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=True,
            args=TrainingArguments(
                output_dir=output_dir,
                learning_rate=learning_rate,
                lr_scheduler_type="cosine",
                num_train_epochs=num_train_epochs,
                weight_decay=0.01,
                warmup_ratio=0.03,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                logging_steps=30,
                fp16=not load_in_4bit and torch.cuda.is_available(),
                bf16=not load_in_4bit and torch.cuda.is_bf16_supported(),
                eval_steps=30,
                logging_first_step=True,
                evaluation_strategy="steps",
                # Use MLflow for reporting instead of Comet ML
                report_to="mlflow",
            ),
        )

        # Train the model
        train_result = trainer.train()
        
        # Evaluate the model
        eval_results = trainer.evaluate()
        
        # Log metrics to MLflow
        metrics = {
            "train_runtime": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"],
            "train_loss": train_result.metrics.get("train_loss", 0),
            "eval_loss": eval_results.get("eval_loss", 0),
        }
        mlflow.log_metrics(metrics)
        
        # Save training results and evaluation as artifacts
        with open(os.path.join(output_dir, "train_results.json"), "w") as f:
            json.dump(train_result.metrics, f, indent=2)
        with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
            
        mlflow.log_artifact(os.path.join(output_dir, "train_results.json"), "metrics")
        mlflow.log_artifact(os.path.join(output_dir, "eval_results.json"), "metrics")
        
        # Log sample inference results
        sample_prompts = [
            "Explain the concept of machine learning to a 5-year old.",
            "What are the benefits of using MLflow for ML projects?",
        ]
        inference_results = {}
        
        # Generate sample outputs from the fine-tuned model
        model.eval()
        for i, prompt in enumerate(sample_prompts):
            message = ALPACA_TEMPLATE.format(prompt, "")
            inputs = tokenizer([message], return_tensors="pt").to(model.device)
            
            # Generate output
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=100, do_sample=True, 
                    temperature=0.7, top_k=50, top_p=0.95
                )
            
            # Decode output
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            inference_results[f"sample_{i+1}"] = {
                "prompt": prompt,
                "output": output_text
            }
            
        # Save and log sample inference results
        with open(os.path.join(output_dir, "sample_inferences.json"), "w") as f:
            json.dump(inference_results, f, indent=2)
        mlflow.log_artifact(os.path.join(output_dir, "sample_inferences.json"), "samples")
        
        # Log the model and config
        mlflow.pytorch.log_model(
            model, 
            "model", 
            registered_model_name="PraxisLLM-Finetuned",
            code_paths=[__file__],  # Include this file in the model artifacts
            signature=None  # You could define an input/output signature here
        )
        
        # Log model config separately to make it easier to access
        with open(os.path.join(output_dir, "model_config.json"), "w") as f:
            model_config = {
                "model_name": model_name,
                "max_seq_length": max_seq_length,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "target_modules": target_modules,
                "chat_template": chat_template
            }
            json.dump(model_config, f, indent=2)
        mlflow.log_artifact(os.path.join(output_dir, "model_config.json"), "config")
        
        # Save and log tokenizer
        tokenizer_path = os.path.join(output_dir, "tokenizer")
        tokenizer.save_pretrained(tokenizer_path)
        mlflow.log_artifact(tokenizer_path, "tokenizer")
        
        # Set model tags for easier filtering
        run_id = mlflow.active_run().info.run_id
        mlflow.set_tag("model_type", "PraxisLLM")
        mlflow.set_tag("base_model", model_name)
        mlflow.set_tag("fine_tuning_method", "LoRA")
        
        # Log execution environment
        mlflow.log_param("python_version", platform.python_version())
        mlflow.log_param("torch_version", torch.__version__)
        mlflow.log_param("transformers_version", transformers.__version__)
        
        print(f"Model training complete! MLflow run ID: {run_id}")
        print(f"To serve the model: mlflow models serve -m 'runs:/{run_id}/model' -p 1234")
        
        return model, tokenizer


def inference(
    model: Any,
    tokenizer: Any,
    prompt: str = "Write a paragraph to introduce supervised fine-tuning.",
    max_new_tokens: int = 256,
) -> None:
    # Ensure model is in evaluation mode
    model.eval()
    
    message = ALPACA_TEMPLATE.format(prompt, "")
    inputs = tokenizer([message], return_tensors="pt").to(model.device)

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        **inputs, streamer=text_streamer, max_new_tokens=max_new_tokens, use_cache=True
    )


def save_model(
    model: Any,
    tokenizer: Any,
    output_dir: str,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
) -> None:
    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model and tokenizer saved to {output_dir}")

    # Log model to MLflow regardless of push_to_hub flag
    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
    
    if run_id:
        # Log model to MLflow
        mlflow.pytorch.log_model(
            model, 
            "model",
            registered_model_name=repo_id if repo_id else "PraxisLLM-Finetuned"
        )
        
        # Log tokenizer as an artifact
        mlflow.log_artifact(output_dir, "tokenizer")
        
        print(f"Model and tokenizer logged to MLflow run {run_id}")
    else:
        print("No active MLflow run found. Model not logged to MLflow.")


# Add main function to be imported by local_training.py
def main(args_list=None):
    """
    Main entry point for the fine-tuning script that can be imported by other modules.
    
    Args:
        args_list: Optional list of command-line arguments
    """
    parser = argparse.ArgumentParser(description="Fine-tune a language model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the fine-tuned model")
    parser.add_argument("--dataset_id", type=str, required=True, help="Dataset ID to use for fine-tuning")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,up_proj,down_proj,o_proj,gate_proj", 
                       help="Comma-separated list of target modules for LoRA")
    parser.add_argument("--chat_template", type=str, default="chatml", help="Chat template to use")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--is_dummy", action="store_true", help="Run in dummy mode (faster for testing)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training (cpu, mps, cuda, auto)")
    
    # Parse arguments
    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()
        
    # Convert target_modules from comma-separated string to list
    target_modules = args.target_modules.split(",")
    
    # Call the finetune function with parsed arguments
    model, tokenizer = finetune(
        model_name=args.model_name,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        chat_template=args.chat_template,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        is_dummy=args.is_dummy
    )
    
    return model, tokenizer


if __name__ == "__main__":
    main()
