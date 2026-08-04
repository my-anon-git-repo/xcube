# =============================================================================
# Imports
# =============================================================================
import json
import warnings
from pathlib import Path
import logging
from datetime import datetime

import torch
import pandas as pd
import wandb

from accelerate.utils import DistributedDataParallelKwargs
from fastcore.script import *
from fastai.distributed import *
from fastai.text.all import *
from fastai.metrics import accuracy  # note: there's also an 'accuracy' in xcube
from fastai.callback.wandb import *
from fastprogress import fastprogress
from xcube.text.all import *

# =============================================================================
# Constants
# =============================================================================
global base_dir
base_dir = None

# =============================================================================
# Warnings and Configuration
# =============================================================================
warnings.filterwarnings(action='ignore')
torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

# Global list to keep track of files generated during preprocessing
generated_files = []


# =============================================================================
# Helper Functions
# =============================================================================
def pr(s):
    "Print only on the main distributed rank."
    if rank_distrib() == 0:
        print(s)


@patch
def after_batch(self: ProgressCallback):
    "Update the progress bar after each batch."
    self.pbar.update(self.iter + 1)
    mets = ('_valid_mets', '_train_mets')[self.training]
    self.pbar.comment = ' '.join(
        [f'{met.name} = {met.value.item():.4f}' for met in getattr(self.recorder, mets)]
    )


def splitter(df):
    "Split a DataFrame into training and validation indices based on 'is_valid' column."
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train, valid


def get_dls(source, data, bs=384, sl=80, bwd=False, workers=None):
    """
    Create dataloaders for language modeling from a CSV file.
    
    Args:
        source (str): Source directory path.
        data (str): Filename of the CSV data (without extension).
        bs (int, optional): Batch size. Defaults to 384.
        sl (int, optional): Sequence length. Defaults to 80.
        bwd (bool, optional): Flag to indicate if training should be backwards. Defaults to False.
        workers (int, optional): Number of workers. Defaults to None.
    
    Returns:
        tuple: (DataLoaders, vocab)
    """
    workers = ifnone(workers, min(8, num_cpus()))
    data_path = join_path_file(data, source, ext='.csv')
    df = pd.read_csv(
        data_path,
        header=0,
        usecols=['text', 'labels'],
        dtype={'text': str, 'labels': str}
    )
    df[['text', 'labels']] = df[['text', 'labels']].astype(str)
    import pdb; pdb.set_trace()
    
    dls_lm = DataBlock(
        blocks=TextBlock.from_df('text', is_lm=True, backwards=bwd),
        get_x=ColReader('text'),
        splitter=RandomSplitter(0.1)
    ).dataloaders(df, bs=bs, seq_len=sl)
    
    return dls_lm, dls_lm.vocab


def setup_logging(**kwargs):
    """
    Configure logging for the application.

    Sets up both console and file logging based on the quiet flag in args.
    Logs include detailed information such as function name, line number,
    and timestamp. Log files are saved in a 'logs' directory within base_dir
    with a timestamp in their filename.
    """

    # Access the 'base_dir' and 'logfile' arguments from kwargs.
    quiet = kwargs.get('quiet')
    logfile = kwargs.get('logfile')
    base_dir = kwargs.get('base_dir')

    # Configure the root logger based on the quiet flag
    if quiet:
        logging.basicConfig(level=logging.CRITICAL)  # Suppresses all logging below CRITICAL
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d:%(funcName)s]'
        )

    # Set up file logging globally
    log_base_dir = Path(base_dir) / 'logs'
    log_base_dir.mkdir(exist_ok=True, parents=True)

    # Format the current date and time for the filename
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = (
        f'{logfile}_{current_time}.log'
        if logfile
        else f'default_logfile_{current_time}.log'
    )
    log_filepath = log_base_dir / log_filename

    # Create file handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)  # Set logging level for the file handler

    # Create formatter and add it to the file handler
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d:%(funcName)s]'
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


# =============================================================================
# Main Training Function
# =============================================================================
@call_parse
def main(
    output_dir: Param("Output path for the finetuned LLM model.", str) = "output",
    source_url: Param("Source url", str) = "XURLs.MIMIC4",
    data: Param("Filename of the raw data", str) = "mimic4_icd10_full",
    bs: Param("Batch size", int) = 16,
    epochs: Param("Number of epochs", str) = "[1, 15]",
    lrs: Param("Learning rates for gradual unfreezing", str) = "[1e-3, 1e-6]",
    fp16: Param("Use mixed precision training", store_true) = False,
    dump: Param("Print model; don't train", int) = 0,
    bwd: Param("Train the backward LM", store_true) = False,
    runs: Param("Number of times to repeat training", int) = 1,
    track_train: Param("Record training metrics", store_true) = False,
    wandblog: Param("Experiment tracking in wandb.ai", store_true) = False,
    log: Param("Log loss and metrics after each epoch", store_true) = False,
    workers: Param("Number of workers", int) = None,
    save_model: Param("Save model on improvement after each epoch", store_true) = False,
    root_dir: Param("Root directory for saving models", str) = "..",
    fname: Param("Save model file", str) = "mimic4",
    infer: Param("Don't train, just validate", int) = 0,
    train_from_ckpt: Param("Load the most recent model and train", store_true) = False,
    logfile: Param("Filename to write logs to", str) = "finetuning_LLM_log",
    base_dir: Param("Path to the base directory that will contain intermediate results and models.", str) = None,
):
    """
    Training of AWD-LSTM language model.
    
    This function sets up the dataloaders, learner, and training loops with options
    for logging, mixed precision, and distributed training.
    """
    # -----------------------------------------------------------------------------
    # Data and File Setup
    # -----------------------------------------------------------------------------
    source = rank0_first(untar_xxx, eval(source_url))
    
    # Create temporary directory for models and dataloaders
    # tmp = Path(root_dir) / 'tmp/models'
    # tmp.mkdir(exist_ok=True, parents=True)
    # tmp = tmp.parent
    # Setting the base directory
    base_dir = base_dir + '/' + source_url.split('.')[1]

    # This is where the model is saved
    output_dir = str(Path(base_dir)/output_dir)

    # Setup logger and print arguments
    setup_logging(**locals())
    print_args(**locals())

    logging.info("****Program started****")

    # Define file paths for dataloaders and vocabulary
    dls_suffix = '_dls_lm_bwd' if bwd else '_dls_lm'
    vocab_suffix = '_dls_lm_vocab_bwd' if bwd else '_dls_lm_vocab'
    dls_file = join_path_file(data + dls_suffix, Path(base_dir), ext='.pkl')
    vocab_file = join_path_file(data + vocab_suffix, Path(base_dir), ext='.pkl')

    # Load or create dataloaders and vocabulary
    if dls_file.exists():
        dls_lm = torch.load(dls_file, map_location=torch.device('cpu'))
    else:
        dls_lm, vocab = get_dls(source, data, bs, bwd=bwd, workers=workers)
        torch.save(dls_lm, dls_file)
        torch.save(vocab, vocab_file)
        generated_files.extend([dls_file, vocab_file])

    epochs = json.loads(epochs)
    lrs = json.loads(lrs)

    if bwd:
        fname = fname + '_bwd'

    # -----------------------------------------------------------------------------
    # Callbacks Setup
    # -----------------------------------------------------------------------------
    cbs = SaveModelCallback(
        monitor='accuracy', fname=fname, with_opt=True, reset_on_fit=False
    ) if save_model else None

    if not infer and log:
        logfname = join_path_file(fname, tmp, ext='.csv')
        # Remove log file if not resuming from checkpoint
        if not train_from_ckpt and logfname.exists():
            logfname.unlink()
        cbs = L(cbs) + L(CSVLogger(fname=logfname, append=True)) if cbs else L(CSVLogger(fname=logfname, append=True))
    
    if wandblog:
        cbs = L(cbs) + L(WandbCallback(log_preds=False, log_model=True, model_name=fname)) if cbs else L(WandbCallback(log_preds=False, log_model=True, model_name=fname))
    
    # -----------------------------------------------------------------------------
    # Learner Setup
    # -----------------------------------------------------------------------------
    learn = language_model_learner(
        dls_lm, AWD_LSTM, drop_mult=0.3, backwards=bwd,
        metrics=[accuracy, Perplexity()], cbs=cbs, path=tmp
    )
    
    if track_train:
        # Ensure the Recorder callback is in place before setting train_metrics
        assert learn.cbs[1].__class__ is Recorder
        setattr(learn.cbs[1], 'train_metrics', True)
    
    if dump:
        pr(learn.model)
        exit()
    
    if fp16:
        learn = learn.to_fp16()
    
    if infer:
        try:
            learn = learn.load(learn.save_model.fname)
            vals = validate(learn)
        except FileNotFoundError as e:
            print("Exception:", e)
            print("Model checkpoint not found!")
        finally:
            exit()

    # -----------------------------------------------------------------------------
    # Distributed Training Context
    # -----------------------------------------------------------------------------
    # Workaround for PyTorch 2.0.1: set find_unused_parameters=True in distributed training
    ddp_scaler = DistributedDataParallelKwargs(bucket_cap_mb=15, find_unused_parameters=True)
    cms = learn.distrib_ctx(kwargs_handlers=[ddp_scaler])
    if wandblog:
        cms += L(wandb.init())
    
    with ContextManagers(cms):
        if not train_from_ckpt:
            # Learning rate finding and initial training
            lr_min, lr_steep, lr_valley, lr_slide = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
            ic(lr_min, lr_steep, lr_valley, lr_slide)
            print(f"Training with {lr_min}")
            learn.fit_one_cycle(epochs[0], lr_min)
            
            # Fine-tuning with unfreezing
            learn.unfreeze()
            learn.fit(epochs[1], lrs[1])
        else:
            # Load checkpoint and continue training
            learn = learn.load(learn.save_model.fname)
            validate(learn)
            print(f"We are monitoring {learn.save_model.monitor}")
            best = input(f"Input the best {learn.save_model.monitor} so far: ")
            learn.save_model.best = float(best)
            epochs_cont = int(input("How many epochs do you want to train: "))
            learn.unfreeze()
            learn.fit(epochs_cont, lrs[1])

    # -----------------------------------------------------------------------------
    # Save the Final Models
    # -----------------------------------------------------------------------------
    learn.save_encoder(fname + '_finetuned')
    generated_files.append(join_path_file(fname + '_finetuned', learn.path / learn.model_dir, ext='.pth'))
    
    learn.save_decoder(fname + '_decoder')
    generated_files.append(join_path_file(fname + '_decoder', learn.path / learn.model_dir, ext='.pth'))

    print("Generated Files:")
    for file_path in generated_files:
        print(file_path)
    