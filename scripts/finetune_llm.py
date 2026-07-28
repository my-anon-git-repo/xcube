# =============================================================================
# Imports
# =============================================================================
import warnings
from pathlib import Path
import logging
from datetime import datetime

import torch
import pandas as pd
import wandb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    pipeline,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import get_peft_model, LoraConfig, PeftModel
from datasets import Dataset
from accelerate import Accelerator
from huggingface_hub import login, HfApi

from xcube.imports import *
from xcube.data.all import *

# =============================================================================
# Constants
# =============================================================================
global base_dir
base_dir = None

# =============================================================================
# Warnings and Configuration
# =============================================================================
warnings.filterwarnings(action="ignore")
torch.backends.cudnn.benchmark = True


# =============================================================================
# Helper Functions
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Finetuning Large Language Model")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output path for the finetuned LLM model.",
    )
    parser.add_argument(
        "--source_url", type=str, default="XURLs.MIMIC4", help="Source url"
    )
    parser.add_argument(
        "--data", type=str, default="mimic4_icd10_full", help="Filename of the raw data"
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="finetuning_LLM_log",
        help="Filename to write logs to",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Path to the base directory that will contain intermediate results and models.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="your_username/mistral-7b-medical-lora",
        help="Hugging Face hub model id (replace with your username/model name)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Model name (e.g., mistralai/Mistral-7B-Instruct-v0.3)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum length (default: 512). Adjust as needed (e.g., 1024 if your summaries are longer and 48GB VRAM allows)",
    )
    parser.add_argument(
        "--do_fresh_training",
        action="store_true",
        default=False,
        help="If set, perform fresh training (default: False)",
    )
    parser.add_argument(
        "--load_from_checkpoint",
        action="store_true",
        default=False,
        help="Load model and tokenizer from Hub before training",
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
        logging.basicConfig(
            level=logging.CRITICAL
        )  # Suppresses all logging below CRITICAL
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d:%(funcName)s]",
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

    # Optionally, configure a specific third-party logger (e.g., 'bertopic')
    # third_party_logger = logging.getLogger('bertopic')
    # third_party_logger.setLevel(logging.DEBUG)  # Capture all messages from this library
    # third_party_logger.addHandler(file_handler)
    # Uncomment the following line if you wish to prevent propagation to the root logger:
    # third_party_logger.propagate = False


def print_args(**kwargs):
    """
    Logs and prints the provided keyword arguments in a formatted manner.

    Args:
        **kwargs: Arbitrary keyword arguments representing command-line parameters.
    """
    # Create a formatted string of key-value pairs.
    formatted_args = "\n".join(f"{key}: {value}" for key, value in kwargs.items())

    # Log and print the formatted arguments.
    logging.info("Command-line Arguments:")
    logging.info("\n" + formatted_args)


# Function to Authenticate with Hugging Face
def authenticate_hf(token):
    """Authenticate with Hugging Face using a token."""
    login(token=token)
    print("Authenticated with Hugging Face Hub")


# Function to Get Data from Source-Dir
def get_data(source, data):
    """
    Load CSV data into a pandas DataFrame.

    Constructs the file path using the provided `data` and `source` parameters,
    reads the CSV file, and ensures that the 'text' and 'labels' columns are strings.

    Args:
        source (str): The source directory or identifier used to locate the CSV file.
        data (str): The base filename (without extension) of the CSV data.

    Returns:
        pd.DataFrame: A DataFrame containing the CSV data with 'text' and 'labels' columns.
    """
    # Construct the file path with a '.csv' extension.
    filepath = join_path_file(data, source, ext=".csv")

    # Read the CSV file, selecting only the 'text' and 'labels' columns,
    # and explicitly define their data types as strings.
    df = pd.read_csv(
        filepath,
        header=0,
        usecols=["text", "labels"],
        dtype={"text": str, "labels": str},
    )

    # Convert the selected columns to string type (if needed).
    df[["text", "labels"]] = df[["text", "labels"]].astype(str)

    return df


# Function to Load Data
def load_data(df):
    """Load text data from DataFrame."""
    texts = df["text"].tolist()
    return texts


# Function to Load Model and Tokenizer with LoRA
def load_model_and_tokenizer(model_name="mistralai/Mistral-7B-Instruct-v0.3"):
    """Load the base model and tokenizer, apply LoRA."""

    # quantization_config = BitsAndBytesConfig()
    # Define 4-bit quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Enable 4-bit quantization
        bnb_4bit_quant_type="nf4",  # Use NF4 (Normal Float 4) quantization
        bnb_4bit_use_double_quant=True,  # Use double quantization for better accuracy
        bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bfloat16 for efficiency
    )

    # # Define 8-bit quantization configuration
    # quantization_config = BitsAndBytesConfig(
    #     load_in_8bit=True,                  # Enable 8-bit quantization
    #     bnb_8bit_compute_dtype=torch.bfloat16  # Compute in bfloat16
    # )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with appropriate quantization
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     device_map="auto",
    #     trust_remote_code=True,
    # )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=quantization_config,  # Apply quantization
        device_map="auto",  # Automatically map to available devices
        trust_remote_code=True,  # Required for some models
    )

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# Function to Tokenize Dataset
def tokenize_dataset(texts, tokenizer, max_length=512):
    """Tokenize the text data and return a Dataset."""
    dataset = Dataset.from_dict({"text": texts})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

    # Optional: Split into train/validation
    # tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.001)
    # return tokenized_dataset
    # Ensure dataset has at least 20 samples
    if len(tokenized_dataset) < 20:
        raise ValueError(
            "Dataset must contain at least 20 samples for a 20-sample test set."
        )

    # Randomly sample 20 indices for the test set
    test_size = 20
    test_indices = random.sample(range(len(tokenized_dataset)), test_size)

    # Create train and test splits
    train_indices = [i for i in range(len(tokenized_dataset)) if i not in test_indices]
    train_dataset = tokenized_dataset.select(train_indices)
    test_dataset = tokenized_dataset.select(test_indices)

    # Return as a dictionary with train and test splits
    return {"train": train_dataset, "test": test_dataset}


# Function to Create Data Collator
def create_data_collator(tokenizer):
    """Create a data collator for causal language modeling."""
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# Function to Define Training Arguments
def define_training_args(
    output_dir="./mistral_mimic4",
    hub_model_id="deb101/mistral-7b-mimic4",
    hub_token=None,
):
    """Define training arguments for the Trainer."""
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        fp16=True,
        # Save and evaluate at the end of each epoch
        save_strategy="epoch",
        evaluation_strategy="epoch",
        # Save only the best model based on a metric
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  # Monitor accuracy (higher is better)
        greater_is_better=True,  # For accuracy, higher values are better
        logging_strategy="steps",  # Log training loss every 10 steps (no eval),
        logging_steps=10,
        report_to="none",
        push_to_hub=True,
        hub_model_id=hub_model_id,
        hub_strategy="end",
        hub_token=hub_token,
    )


# Function to Train the Model
def train_model(model, training_args, tokenized_dataset, data_collator):
    """Initialize Trainer and fine-tune the model."""

    def compute_metrics(eval_pred):
        """Compute accuracy and perplexity for language modeling."""
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = torch.tensor(logits)
        labels = torch.tensor(labels)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        perplexity = torch.exp(loss)
        predictions = shift_logits.argmax(dim=-1)
        correct = (predictions == shift_labels).masked_fill(shift_labels == -100, 0)
        accuracy = correct.sum().float() / (shift_labels != -100).sum().float()

        return {"accuracy": accuracy.item(), "perplexity": perplexity.item()}

    # Custom callback for logging
    class LoggingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                log_message = "Training metrics: " + ", ".join(
                    f"{k}: {v}" for k, v in logs.items()
                )
                logging.info(log_message)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is not None:
                log_message = "Evaluation metrics: " + ", ".join(
                    f"{k}: {v}" for k, v in metrics.items()
                )
                logging.info(log_message)

    class BaselineCallback(TrainerCallback):
        def __init__(self, baseline_metric_name, baseline_metric_value):
            self.baseline_metric_name = baseline_metric_name
            self.baseline_metric_value = baseline_metric_value
            logging.info(
                f"Initialized baseline {baseline_metric_name} = {baseline_metric_value}"
            )

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics:
                logging.warning("No metrics provided during evaluation")
                return
            metric_key = f"eval_{self.baseline_metric_name}"
            if metric_key not in metrics:
                logging.warning(f"Metric {metric_key} not found in evaluation metrics")
                return
            current_metric_value = metrics[metric_key]
            logging.info(
                f"Current {self.baseline_metric_name} = {current_metric_value}, Baseline = {self.baseline_metric_value}"
            )
            if current_metric_value > self.baseline_metric_value:
                self.baseline_metric_value = current_metric_value
                control.should_save = True
                logging.info("Metric improved, saving checkpoint")
            else:
                control.should_save = False
                logging.info("Metric not improved, skipping save")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[LoggingCallback()],  # Add the custom callback
    )

    # Compute baseline and add it with a BaselineCallback
    baseline_metrics = trainer.evaluate()
    baseline_metric_name = training_args.metric_for_best_model
    baseline_metric_value = baseline_metrics[f"eval_{baseline_metric_name}"]
    # Add callback with baseline
    trainer.add_callback(BaselineCallback(baseline_metric_name, baseline_metric_value))

    trainer.train()
    return trainer


def push_to_hub(
    trainer,
    tokenizer,
    commit_message="Fine-tuned Mistral-7B on medical discharge summaries with LoRA",
):
    """Push the fine-tuned model and tokenizer to Hugging Face Hub."""
    trainer.push_to_hub(commit_message=commit_message)
    tokenizer.push_to_hub(trainer.args.hub_model_id, token=trainer.args.hub_token)
    print(f"Model and tokenizer pushed to Hub: {trainer.args.hub_model_id}")


def cleanup_gpu_memory(accelerator=None):
    """
    Clean up GPU memory and training objects from device 0 before evaluation.

    Args:
        accelerator (Accelerator, optional): The Accelerator object to free memory from, if available.
    """
    logging.info("Cleaning up GPU memory before evaluation...")

    # Delete training-related objects if they exist in locals
    for var_name in [
        "trainer",
        "model",
        "tokenized_dataset",
        "train_dataset",
        "eval_dataset",
        "data_collator",
    ]:
        if var_name in locals():
            del locals()[var_name]

    # Free Accelerator memory if provided
    if accelerator is not None:
        accelerator.free_memory()
        del accelerator

    # Clear GPU cache
    torch.cuda.empty_cache()

    logging.info("GPU memory cleaned up.")


def test_model_tasks(generator, clinical_note, output_dir):
    """
    Test generator on various clinical tasks and save results to Markdown.
    """
    # 📅 Get current date
    timestamp_display = datetime.now().strftime(
        "%m/%d/%y %H:%M:%S"
    )  # for Markdown content
    timestamp_filename = datetime.now().strftime("%m-%d-%y_%H-%M-%S")  # for filename

    # Markdown output path
    os.makedirs(output_dir, exist_ok=True)
    md_filename = f"test_results_{timestamp_filename}.md"
    md_path = os.path.join(output_dir, md_filename)

    # 🔍 Extract model name
    model_name = generator.model.config._name_or_path
    logging.info(f"🧠 Model used: {model_name}")

    # Define prompts and task descriptions
    task_data = [
        {
            "title": "💊 Medication Extraction",
            "prompt": f"""
--- Example ---
Clinical note: Patient was given acetaminophen and ibuprofen.
Q: List all medications mentioned in the clinical note.
A:
- acetaminophen
- ibuprofen

--- Task ---
Clinical note: {clinical_note}

Q: List all medications mentioned in the clinical note.
A:
""",
        },
        {
            "title": "🩺 Diagnosis Summarization",
            "prompt": f"""
--- Example ---
Discharge summary: Patient admitted for preeclampsia. Diagnosed with hypertension and mild proteinuria. No signs of eclampsia.
Q: What are the primary and secondary diagnoses?
A:
Primary: Preeclampsia
Secondary: Hypertension, mild proteinuria

--- Task ---
Discharge summary: {clinical_note}

Q: What are the primary and secondary diagnoses?
A:
""",
        },
        {
            "title": "📄 Structured Discharge Summary",
            "prompt": f"""
--- Example ---
Patient note: Presented with shortness of breath. Diagnosed with pneumonia. Treated with azithromycin. Discharged with oxygen therapy and follow-up in 2 days.
Q: Write a structured discharge summary with diagnosis, treatment, and follow-up.
A:
Diagnosis: Pneumonia
Treatment: Azithromycin
Follow-up: Oxygen therapy, follow-up in 2 days

--- Task ---
Patient note: {clinical_note}

Q: Write a structured discharge summary with diagnosis, treatment, and follow-up.
A:
""",
        },
        {
            "title": "✍️ Freeform Text Continuation",
            "prompt": f"{clinical_note.strip()}\n\nFollow-up plan includes",
        },
    ]

    # 📘 Initialize markdown content
    markdown_lines = [
        "# 🧪 Finetuned Model Task Results",
        f"**Generated on**: {timestamp_display}",
        f"**Model**: `{model_name}`",
        "",
    ]

    for idx, task in enumerate(task_data, start=1):
        logging.info("\n" + "=" * 90)
        logging.info(f"{task['title']} (Task {idx})")
        logging.info("-" * 90)

        if "Continuation" in task["title"]:
            output = generator(
                task["prompt"],
                max_length=512 * 3,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        else:
            output = generator(
                task["prompt"],
                max_length=512 * 3,
                do_sample=False,
                temperature=0.0,
                top_p=0.95,
            )
        result = output[0]["generated_text"].strip()

        # Console logging
        logging.info(f"📤 Output:\n{result}\n")

        # Append to markdown
        markdown_lines.append(f"## {task['title']} (Task {idx})\n")
        markdown_lines.append("```text")
        markdown_lines.append(result)
        markdown_lines.append("```\n")

    # Save to Markdown file
    with open(md_path, "w") as f:
        f.write("\n".join(markdown_lines))

    logging.info(f"✅ Results saved to: `{md_path}`")


# =============================================================================
# Main Function
# =============================================================================
def main(
    output_dir: str = "output",
    source_url: str = "XURLs.MIMIC4",
    data: str = "mimic4_icd10_full",
    logfile: str = "finetuning_LLM_log",
    base_dir: str = None,
    hub_model_id: str = "deb101/mistral-7b-medical-lora",
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    max_length: int = 512,
    do_fresh_training: bool = False,
    load_from_checkpoint: bool = False,
):
    """
    Fine-tune Large Language Model if not found or forced, then evaluate
    """
    # -----------------------------------------------------------------------------
    # Data and File Setup
    # -----------------------------------------------------------------------------
    source = untar_xxx(eval(source_url))
    base_dir = base_dir + "/" + source_url.split(".")[1]
    output_dir = str(Path(base_dir) / output_dir)

    # Setup logger and print arguments
    setup_logging(**locals())
    print_args(**locals())

    logging.info("****Program started****")

    # Hardcode your HF token (or load from environment variable)
    # hf_token = "your_hf_token_here"  # Replace or use os.environ["HF_TOKEN"]
    hf_token = os.environ["HF_TOKEN"]

    # Check if model exists locally or on Hub (unless do_fresh_training is True)
    model_exists_locally = os.path.exists(
        os.path.join(output_dir, "adapter_config.json")
    )
    api = HfApi()
    try:
        model_exists_on_hub = (
            api.model_info(hub_model_id, token=hf_token).id is not None
            if hf_token
            else False
        )
    except Exception as e:
        logging.warning(f"Failed to check model on Hub: {e}")
        model_exists_on_hub = False

    # Train if model is not found locally or on Hub, or if do_fresh_training is True
    if do_fresh_training or not (model_exists_locally or model_exists_on_hub):
        logging.info("Starting fresh training (either forced or model not found)...")
        authenticate_hf(hf_token)

        accelerator = Accelerator()

        # Load and prepare data
        df = get_data(source, data)
        texts = load_data(df)

        if load_from_checkpoint:
            logging.info(
                f"Loading model and tokenizer from checkpoint on Hub: {hub_model_id}"
            )
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", trust_remote_code=True
            )
            # Load LoRA adapters from Hub
            model = PeftModel.from_pretrained(base_model, hub_model_id, token=hf_token)
            tokenizer = AutoTokenizer.from_pretrained(hub_model_id, token=hf_token)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            # Ensure LoRA layers are trainable
            for name, param in model.named_parameters():
                if "lora" in name.lower():  # Target LoRA parameters
                    param.requires_grad = True
            model.train()  # Set model to training mode
        else:
            logging.info("Starting fresh training from base model...")
            model, tokenizer = load_model_and_tokenizer(model_name=model_name)
        # model, tokenizer = load_model_and_tokenizer(model_name=model_name)

        tokenized_dataset = tokenize_dataset(texts, tokenizer, max_length=max_length)

        # Prepare model and datasets with Accelerator
        model = accelerator.prepare(model)
        train_dataset = accelerator.prepare(tokenized_dataset["train"])
        eval_dataset = accelerator.prepare(tokenized_dataset["test"])
        tokenized_dataset = {"train": train_dataset, "test": eval_dataset}

        data_collator = create_data_collator(tokenizer)
        training_args = define_training_args(
            output_dir=output_dir, hub_model_id=hub_model_id, hub_token=hf_token
        )

        trainer = train_model(model, training_args, tokenized_dataset, data_collator)

        if accelerator.is_main_process:
            # Save model locally (already done by trainer.train(), but explicit for clarity)
            trainer.save_model(output_dir)
            # Save tokenizer locally
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model and tokenizer saved locally to: {output_dir}")
            # Push to Hub
            push_to_hub(trainer, tokenizer)
    else:
        logging.info("Model found locally or on Hub, skipping training...")

    # -----------------------------------------------------------------------------
    # Cleanup Before Evaluation
    # -----------------------------------------------------------------------------
    cleanup_gpu_memory(accelerator if "accelerator" in locals() else None)

    # -----------------------------------------------------------------------------
    # Evaluation (always runs)
    # -----------------------------------------------------------------------------
    logging.info("Running evaluation...")

    # Determine where to load the model from
    if model_exists_locally and not do_fresh_training:
        model_path = output_dir
        logging.info(f"Loading model from local path: {model_path}")
    else:
        model_path = hub_model_id
        logging.info(f"Loading model from Hub: {model_path}")

    # just checking
    # model_path = "meta-llama/Llama-3.2-3B-Instruct"
    # just checking

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generator = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        token=hf_token,
    )

    # 📝 Sample MIMIC-IV Text
    # clinical_note = """name unit no admission date discharge date date of birth sex f service obstetrics gynecology allergies penicillins attending chief complaint concern for concealed abruption ... discharged home and will have close outpatient follow up with twice weekly fetal testing"""
    # clinical_note = """name unit no admission date discharge date date of birth sex f service obstetrics gynecology allergies penicillins attending chief complaint concern for concealed abruption on outpatient ultrasound major surgical or invasive procedure none history of present illness pt is a g1 with ivf di di twins with recent demise of twin a on who presents for admission from the due to a x 4mm heterogenous area noted within the intertwin membrane and placentas concerning for concealed abruption pt recented admitted to after demise of twin a diagnosed and received a course of bmz pt reports feeling crampy and denies any vaginal bleeding or leaking of fluid reports active fm pt concerned about this new finding and hoping to be delivered soon past medical history pnc by ivf labs a ab rhig rprnr ri hbsag hiv gbsunk abn gtt issues gdma2 followed by first tri spotting rh negative s p rhogam demise of twin a dx d s p bmz ob hx g1 gyn hx endometriosis ivf pmh hx thyroid nodule negative biopsy recent u s reassuring sma and short chain acyl coa dehydrogenase deficiency carrier psh laparoscopies for endo pilonidal cyst i d breast implants wisdom teeth r shoulder surgery social history family history non contributory physical exam on admission vss gen alert oriented anxious pulm breathing comfortably on room air abd soft gravid non tender ext non tender no asymmetric swelling pertinent results wbc rbc hgb hct mcv plt yo g1 with di di twins with recent demise of twin a diagnosed on also with gdma2 admitted with placental clot concerning for concealed abruption on admission she was hemodynamically stable without any vaginal bleeding fetal testing was twin a was reassuring on hd she had a repeat ultrasound in the maternal fetal medicine which was not concerning for an abruption the fluid seen near the membranes was felt to be extravasation of fluid from the demised twin given there was no evidence of abruption discharge was recommended she was discharged home and will have close outpatient follow up with twice weekly fetal testing met with her while she was here and increased her nph to units at night medications on admission pnv nph units at night discharge medications nph units bedtimemax dose override reason per recs lorazepam mg po q8h prn anxiety rx lorazepam mg tab by mouth every eight hours disp tablet refills prenatal vitamins tab po daily discharge disposition home discharge diagnosis di di twin pregnancy at 28w4d s p recent demise of twin a gdma2 discharge condition stable discharge instructions you were admitted to the antepartum floor for observation due to concern for abruption on ultrasound however on re evaluation it was felt that the fluid near the membrane was not concerning for abruption the fluid seen was felt to be related to the demised twin you will continue to get twice weekly testing in the maternal fetal medicine testing for twin a was reassuring while you were here please continue checking your fingersticks and taking insulin as directed by followup instructions"""
    clinical_note = """A gravida 1 patient with a dichorionic diamniotic twin pregnancy conceived via IVF was admitted at 28 weeks and 4 days gestation due to a recent demise of Twin A and concern for concealed placental abruption based on outpatient ultrasound findings of a 4mm heterogeneous area between the intertwin membrane and placentas. The patient had previously received a course of betamethasone. She has a medical history of gestational diabetes mellitus (A2), Rh-negative status managed with RhoGAM, endometriosis, and is a carrier of short-chain acyl-CoA dehydrogenase deficiency. Surgical history includes laparoscopies for endometriosis, pilonidal cyst I&D, breast implants, and shoulder surgery. On admission, the patient was hemodynamically stable, denied vaginal bleeding or fluid leakage, reported cramping and ongoing fetal movement, and appeared anxious but alert and oriented. Physical exam was unremarkable. Laboratory results and vital signs were within normal limits. A repeat maternal-fetal medicine ultrasound did not confirm abruption; the fluid near the membranes was attributed to extravasation from the demised twin. Fetal testing for the surviving twin was reassuring. The patient was discharged home in stable condition with instructions to continue twice-weekly fetal testing, NPH insulin, and prenatal vitamins. She was advised to monitor fingersticks and follow her diabetes management plan. Lorazepam was prescribed as needed for anxiety. Discharge diagnoses included di-di twin pregnancy at 28w4d, recent intrauterine demise of Twin A, and gestational diabetes mellitus A2. She will follow up closely with outpatient monitoring."""

    # 🚀 Run Tasks and Save Results
    test_model_tasks(generator, clinical_note, output_dir)


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    # Convert the Namespace to a dictionary and pass to main via dictionary unpacking.
    main(**vars(args))
