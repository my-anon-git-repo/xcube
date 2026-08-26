# =============================================================================
# Imports
# =============================================================================
# generic imports
import warnings
import argparse
import os
from pathlib import Path
import logging
from datetime import datetime

# imports from transformers 
from transformers import (
    AdamW,
    get_scheduler,
    AutoConfig,
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    pipeline, 
    BitsAndBytesConfig, 
    TrainerCallback, 
    TrainerState, 
    TrainerControl
)
import torch
import sys

from datasets import Dataset, DatasetDict
from accelerate import Accelerator
from huggingface_hub import login, HfApi

# imports from my library xcube
from xcube.imports import *
from xcube.data.all import *
from tqdm import tqdm
import pandas as pd


# =============================================================================
# Constants
# =============================================================================
global base_dir
base_dir = None


# =============================================================================
# Helper Functions
# =============================================================================
def log_section(message, char="=", length=80):
    """
    Creates a visually appealing log message with a box around it.
    
    Args:
        message (str): The message to display
        char (str): Character to use for the box (default: "=")
        length (int): Width of the box (default: 80)
    """
    border = char * length
    padding = (length - len(message) - 4) // 2
    left_padding = " " * padding
    right_padding = " " * (length - len(message) - 4 - padding)
    
    # Choose emoji based on border character
    section_emoji = "📌"
    if char == "=":
        section_emoji = "📌"
    elif char == "*":
        section_emoji = "🌟"
    elif char == "+":
        section_emoji = "✨"
    elif char == "-":
        section_emoji = "📋"
    elif char == "!":
        section_emoji = "⚠️"
    
    logging.info(border)
    logging.info(f"{char}{left_padding} {section_emoji} {message} {right_padding}{char}")
    logging.info(border)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetuning Large Language Model"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output path for the finetuned LLM model."
    )
    parser.add_argument(
        "--source_url",
        type=str,
        default="XURLs.MIMIC4",
        help="Source url"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="mimic4_icd10_full",
        help="Filename of the raw data"
    )
    parser.add_argument(
        '--data_l2r', 
        type=str, 
        required=True,
        help='Data source identifier (e.g., mimic4_icd10_tok_lbl)'
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="training_l2r_log",
        help="Filename to write logs to"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Path to the base directory that will contain intermediate results and models."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="your_username/mistral-7b-medical-lora",
        help="Hugging Face hub model id (replace with your username/model name)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Model name (e.g., mistralai/Mistral-7B-Instruct-v0.3)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum length (default: 512). Adjust as needed (e.g., 1024 if your summaries are longer and 48GB VRAM allows)"
    )
    parser.add_argument(
        "--do_fresh_training",
        action="store_true",
        default=False,
        help="If set, perform fresh training (default: False)"
    )
    parser.add_argument(
        "--load_from_checkpoint", 
        action="store_true", 
        default=False,
        help="Load model and tokenizer from Hub before training"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate for the optimizer (default: 2e-4)"
    )
    parser.add_argument(
        "--warmup_steps", 
        type=int, 
        default=500,
        help="Number of warmup steps for training (default: 500)"
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="ndcg@k",
        choices=[],
        help="Metric to determine the best model (default: ndcg@k)."
    )
    parser.add_argument(
        "--l2r_boot_dir",
        type=str,
        default="mimic4_l2rboot",
        help="Directory for l2r boot files (default: mimic4_l2rboot)"
    )
    
    return parser.parse_args()


def setup_logging(**kwargs):
    """
    Configure logging for the application.

    Sets up both console and file logging based on the quiet flag in args.
    Logs include detailed information such as function name, line number,
    and timestamp. Log files are saved in a 'logs' directory within base_dir
    with a timestamp in their filename.
    """
    # Access the 'base_dir' and 'logfile' arguments from kwargs.
    quiet = kwargs.get("quiet")
    logfile = kwargs.get("logfile")
    base_dir = kwargs.get("base_dir")

    # Configure the root logger based on the quiet flag
    if quiet:
        logging.basicConfig(level=logging.CRITICAL)  # Suppresses all logging below CRITICAL
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d:%(funcName)s]"
        )

    # Set up file logging globally
    log_base_dir = Path(base_dir) / "logs"
    log_base_dir.mkdir(exist_ok=True, parents=True)

    # Format the current date and time for the filename
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = (
        f"{logfile}_{current_time}.log"
        if logfile
        else f"default_logfile_{current_time}.log"
    )
    log_filepath = log_base_dir / log_filename

    # Create file handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)  # Set logging level for the file handler

    # Create formatter and add it to the file handler
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d:%(funcName)s]"
    )
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"📝 Logging initialized. Log file created at: {log_filepath}")

    return log_filepath


def print_args(**kwargs):
    """
    Logs and prints the provided keyword arguments in a formatted manner.
    
    Args:
        **kwargs: Arbitrary keyword arguments representing command-line parameters.
    """
    # Create a formatted string of key-value pairs.
    formatted_args = "\n".join(f"  🔹 {key}: {value}" for key, value in kwargs.items())
    
    # Log and print the formatted arguments.
    logging.info("🛠️ Command-line Arguments:")
    logging.info("\n" + formatted_args)
    logging.info("➖" * 30)


# Function to Authenticate with Hugging Face
def authenticate_hf(token):
    """Authenticate with Hugging Face using a token."""
    login(token=token)
    logging.info("🔑 Successfully authenticated with Hugging Face Hub")


# Function to Get Data from Source-Dir
def get_data(source, l2r_boot_dir, data, data_l2r):
    """
    Load CSV data and L2R data into pandas DataFrames.
    
    Constructs the file paths using the provided parameters,
    reads the CSV file and L2R feather file, and ensures that 
    the 'text' and 'labels' columns are strings.

    Args:
        source (str): The source directory or identifier used to locate the CSV file.
        l2r_boot_dir (str): Directory containing the L2R feather file.
        data (str): The base filename (without extension) of the CSV data.
        data_l2r (str): The base filename (without extension) of the L2R feather data.

    Returns:
        tuple: (df_data, df_l2r) containing the main data and L2R data DataFrames.
    """
    # Construct the file path with a '.csv' extension for the main data.
    filepath = join_path_file(data, source, ext=".csv")
    logging.info(f"📂 Loading main data from: {filepath}")
    
    # Read the CSV file, selecting only the necessary columns,
    # and explicitly define their data types.
    df_data = pd.read_csv(
        filepath,
        header=0,
        usecols=["text", "labels", "is_valid"],
        dtype={"text": str, "labels": str, "is_valid": bool}
    )
    
    # Convert the selected columns to string type (if needed).
    df_data[["text", "labels"]] = df_data[["text", "labels"]].astype(str)
    logging.info(f"✅ Successfully loaded main data: {len(df_data)} rows")
    
    # Construct the file path for the L2R feather file
    l2r_filepath = os.path.join(l2r_boot_dir, f"{data_l2r}.ft")
    logging.info(f"📂 Loading L2R data from: {l2r_filepath}")
    
    try:
        # Read the feather file for L2R data
        df_l2r = pd.read_feather(l2r_filepath)
        logging.info(f"✅ Successfully loaded L2R data: {len(df_l2r)} rows")
    except Exception as e:
        logging.error(f"❌ Failed to load L2R data: {e}")
        raise RuntimeError(f"Failed to load L2R data from {l2r_filepath}: {e}")
    
    logging.info(f"🔄 Total data loaded: {len(df_data)} main rows, {len(df_l2r)} L2R rows")
    
    return df_data, df_l2r


def load_datasets(source, l2r_boot_dir, data, data_l2r):
    """
    Load main and L2R datasets with comprehensive logging.
    
    Args:
        source (str): Source directory for the main dataset
        l2r_boot_dir (str): Directory containing the L2R data
        data (str): Filename for the main dataset
        data_l2r (str): Filename for the L2R dataset
        
    Returns:
        tuple: (df_data, df_l2r) containing the main data and L2R data DataFrames
    """
    try:
        df_data, df_l2r = get_data(source, l2r_boot_dir, data, data_l2r)
        logging.info(f"✨ Successfully loaded both datasets:")
        logging.info(f"   - 📄 Main dataset: {len(df_data)} records")
        logging.info(f"   - 📄 L2R dataset: {len(df_l2r)} records")
        
        # Log data sample for verification
        logging.debug("🔍 Sample of main dataset (first 2 rows):")
        logging.debug(f"\n{df_data.head(2)}")
        logging.debug("🔍 Sample of L2R dataset (first 2 rows):")
        logging.debug(f"\n{df_l2r.head(2)}")
        
        logging.info("✅ Data loading completed successfully")
    except Exception as e:
        logging.error(f"❌ Failed to load datasets: {e}")
        raise RuntimeError(f"Failed to load datasets: {e}")
    
    return df_data, df_l2r


def check_model_existence(output_dir, task_hub_model_id, hf_token=None):
    """
    Check if a model exists locally and on Hugging Face Hub.
    
    Args:
        output_dir (str): Directory where the model would be saved locally
        task_hub_model_id (str): Hugging Face model ID to check
        hf_token (str, optional): Hugging Face authentication token
        
    Returns:
        tuple: (model_exists_locally, model_exists_on_hub)
    """
    logging.info("🔍 Checking model existence locally and on Hugging Face Hub...")
    
    # Check local existence
    model_exists_locally = os.path.exists(os.path.join(output_dir, "config.json"))
    if model_exists_locally:
        logging.info(f"✅ Model exists locally at: {output_dir}")
    else:
        logging.info(f"❌ Model not found locally at: {output_dir}")
    
    # Check Hub existence
    model_exists_on_hub = False
    api = HfApi()
    try:
        model_exists_on_hub = api.model_info(task_hub_model_id, token=hf_token).id is not None if hf_token else False
        if model_exists_on_hub:
            logging.info(f"✅ Model exists on Hugging Face Hub with ID: {task_hub_model_id}")
        else:
            logging.info(f"❌ Model not found on Hugging Face Hub with ID: {task_hub_model_id}")
    except Exception as e:
        # Check if it's a 404 error (Repository Not Found)
        if "404" in str(e) and "Repository Not Found" in str(e):
            logging.info(f"❌ Model repository does not exist on Hugging Face Hub: {task_hub_model_id}")
            logging.debug(f"🔍 Detail: {e}")
        else:
            # For other API errors
            logging.warning(f"❓ Failed to check model on Hub due to API error: {type(e).__name__}")
            logging.warning(f"🔍 Error details: {e}")
        
        # Set the flag to false regardless of error type
        model_exists_on_hub = False
    
    # Summarize model existence status
    if model_exists_locally or model_exists_on_hub:
        logging.info("📁 Model exists either locally or on Hub")
    else:
        logging.info("🔄 Model doesn't exist yet - will need to train from scratch")
        
    return model_exists_locally, model_exists_on_hub


# =============================================================================
# Main Function
# =============================================================================
def main(
    output_dir: str = "task2_output",
    source_url: str = "XURLs.MIMIC4",
    data: str = "mimic4_icd10_full",
    data_l2r: str = "mimic4_icd10_tok_lbl",
    logfile: str = "classification_log",
    base_dir: str = None,
    l2r_boot_dir: str = "mimic4_l2rboot",
    hub_model_id: str = "deb101/mistral-7b-mimic4",  # Task 1 Domain Adaptation model
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    max_length: int = 512,
    do_fresh_training: bool = False,
    load_from_checkpoint: bool = False,
    task: str = "l2r",
    num_train_epochs: int = 3,
    metric_for_best_model: str = "ndcg@k",
    learning_rate: float = 2e-4,
    warmup_steps: int = 500
):
    # -----------------------------------------------------------------------------
    # Data and File Setup
    # -----------------------------------------------------------------------------
    # Setup logger
    logfile_path = setup_logging(**locals())
    log_section("INITIALIZING TRAINING ENVIRONMENT", "=")

    logging.info("🚀 Setting up data paths and environment variables...")
    
    source = untar_xxx(eval(source_url))
    base_dir = base_dir + "/" + source_url.split(".")[1]
    output_dir = str(Path(base_dir) / output_dir)
    l2r_boot_dir = str(Path(base_dir) / l2r_boot_dir)
    
    # print arguments
    print_args(**locals())

    # Check if directory exists
    if not os.path.exists(l2r_boot_dir):
        logging.error(f"❌ Directory not found: {l2r_boot_dir}")
        raise FileNotFoundError(f"Directory not found: {l2r_boot_dir}")
    logging.info(f"📁 Using L2R boot directory: {l2r_boot_dir}")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"📂 Using output directory: {output_dir}")

    log_section("LOADING DATASETS", "+")
    logging.info("📊 Loading main dataset and L2R dataset...")
    
    # Load datasets
    df_data, df_l2r = load_datasets(source, l2r_boot_dir, data, data_l2r)

    log_section("STARTING LEARNING TO RANK MODEL TRAINING", "*")

    # Hardcode your HF token (or load from environment variable)
    # hf_token = "your_hf_token_here"  # Replace or use os.environ["HF_TOKEN"]
    hf_token = os.environ["HF_TOKEN"]
    logging.info("🔐 Loaded authentication token from environment")

    # Create task specific hub model id
    task_hub_model_id = f"{hub_model_id}-{task}"
    logging.info(f"🏷️ Hub Model ID for this Learning to Rank task: {task_hub_model_id}")

    log_section("MODEL EXISTENCE CHECK", "-")

    # Check if model exists locally or on Hub
    model_exists_locally, model_exists_on_hub = check_model_existence(
        output_dir,
        task_hub_model_id,
        hf_token
    )   
    
    # Train if model is not found locally or on Hub, or if do_fresh_training is True
    if do_fresh_training or not (model_exists_locally or model_exists_on_hub):
        try:
            log_section("STARTING FRESH TRAINING", "+")
            logging.info("🔄 Starting fresh training (either forced or model not found)...")
            authenticate_hf(hf_token)
            
            if load_from_checkpoint:
                log_section("LOADING FROM CHECKPOINT", "+")
                logging.info("📥 Loading existing model from checkpoint...")
                # Load the custom task specific model and the tokenizer
                custom_model, tokenizer = load_custom_task_model_and_tokenizer(
                    output_dir, 
                    hub_model_id, 
                    task_hub_model_id, 
                    model_exists_locally, 
                    do_fresh_training, 
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    hf_token=hf_token
                )
                logging.info(f"✅ Successfully loaded from checkpoint, will now continue training for {num_train_epochs} epochs.")
                custom_model.train()
                
                log_section("DATA PREPROCESSING", "+")
                logging.info("🔄 Loading and preprocessing training data...")
                # Preprocess data
                dataset, icd_codes, mlb, rare_labels, not_rare_labels = preprocess_data(
                    df, tokenizer, base_dir=base_dir, max_length=max_length
                )
                num_labels = len(icd_codes)
                logging.info(f"🏷️ Number of unique ICD-10 codes: {num_labels}")
            else:
                log_section("LOADING BASE MODEL", "+")
                logging.info("📥 Loading pretrained model and tokenizer...")
                # Load Task 1 (Domain Adapted) model and tokenizer
                base_model, tokenizer = load_base_model_and_tokenizer(
                    model_name, hub_model_id, task_hub_model_id, hf_token
                )
                
                log_section("DATA PREPROCESSING", "+")
                logging.info("🔄 Loading and preprocessing training data...")
                # Preprocess data
                dataset, icd_codes, mlb, rare_labels, not_rare_labels = preprocess_data(
                    df, tokenizer, base_dir=base_dir, max_length=max_length
                )
                num_labels = len(icd_codes)
                logging.info(f"🏷️ Number of unique ICD-10 codes: {num_labels}")
                
                log_section("MODEL INITIALIZATION", "+")
                logging.info("🧠 Initializing custom model for ICD-10 classification...")
                # Define custom model
                custom_model = define_model(base_model, num_labels)
            
            log_section("TRAINING PREPARATION", "+")    
            logging.info("⚙️ Preparing training components and optimizers...")
            # Prepare model and datasets with Accelerator and also the TrainingArguments 
            custom_model, train_dataset, eval_dataset, training_args, accelerator = prepare_training_components(
                custom_model,
                dataset,
                output_dir,
                task_hub_model_id,
                hf_token,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                metric_for_best_model=metric_for_best_model
            )
            
            log_section("MODEL TRAINING", "+")
            logging.info("🏋️ Starting model training process...")
            # Train
            trainer = train_model(
                custom_model, 
                training_args, 
                train_dataset, 
                eval_dataset, 
                rare_labels=rare_labels, 
                not_rare_labels=not_rare_labels,
            )
            
            log_section("MODEL SAVING", "+")
            logging.info("💾 Saving trained model and pushing to Hugging Face Hub...")
            # Save and push
            if accelerator.is_main_process:
                save_and_push(trainer, tokenizer, output_dir)
            
            # Done Training and Evaluation, now exit
            log_section("TRAINING COMPLETE", "=")
            logging.info("🎉 Model Training and Evaluation successfully completed.")
        except Exception as e:
            log_section("ERROR OCCURRED", "!", 80)
            logging.error(f"❌ An error occurred during model training: {e}")
            sys.exit(1)
        finally:
            sys.exit(0)
    else:
        log_section("TRAINING SKIPPED", "-")
        logging.info("⏭️ Model found locally or on Hub, skipping training...")

    
    # -----------------------------------------------------------------------------
    # Cleanup Before Evaluation
    # -----------------------------------------------------------------------------
    log_section("CLEANUP BEFORE EVALUATION", "-")
    logging.info("🧹 Cleaning up GPU memory before evaluation...")
    cleanup_gpu_memory(accelerator if 'accelerator' in locals() else None)

    # -----------------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------------
    log_section("STARTING MODEL EVALUATION", "*")
    logging.info("📊 Running comprehensive model evaluation...")
    
    # Load the custom task specific model and the tokenizer
    logging.info("📥 Loading model for evaluation...")
    custom_model, tokenizer = load_custom_task_model_and_tokenizer(
        output_dir, 
        hub_model_id, 
        task_hub_model_id, 
        model_exists_locally, 
        do_fresh_training, 
        device="cuda" if torch.cuda.is_available() else "cpu",
        hf_token=hf_token
    )
    
    # Preprocess data
    log_section("PREPARING EVALUATION DATA", "+")
    logging.info("🔄 Loading and preprocessing evaluation data...")
    dataset, icd_codes, mlb, rare_labels, not_rare_labels = preprocess_data(
        df, tokenizer, base_dir=base_dir, max_length=max_length
    )
    num_labels = len(icd_codes)
    logging.info(f"🏷️ Number of unique ICD-10 codes for evaluation: {num_labels}")
    
    # Prepare model and datasets with Accelerator and also the TrainingArguments
    log_section("SETTING UP EVALUATION ENVIRONMENT", "+") 
    logging.info("⚙️ Preparing model components for evaluation...")
    custom_model, train_dataset, eval_dataset, training_args, accelerator = prepare_training_components(
        custom_model,
        dataset,
        output_dir,
        task_hub_model_id,
        hf_token
    )
    
    # Evaluate with train_model
    log_section("RUNNING EVALUATION", "+")
    logging.info("📏 Evaluating model performance on test dataset...")
    logging.info("📊 Using k values of 5, 8, and 15 for evaluation metrics...")
    trainer = train_model(
        custom_model,
        training_args,
        train_dataset,
        eval_dataset,
        k_values=[5, 8, 15],
        rare_labels=rare_labels,
        not_rare_labels=not_rare_labels,
        eval=True
    )
    
    log_section("EVALUATION COMPLETE", "=")
    logging.info("🏆 Model evaluation completed successfully.")
        
    sys.exit(0)

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    # Convert the Namespace to a dictionary and pass to main via dictionary unpacking.
    main(**vars(args))