# utils.py
import os
import random
import numpy as np
import torch
import errno


def set_random_seed(seed: int):
    """
    Set the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Parameters:
    seed (int): The seed value to set for all random number generators.
    """
    # Set environment variable for hash seed (Python)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set random seeds for Python, NumPy, and PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # For CPU
    torch.cuda.manual_seed_all(seed)  # For all GPU devices
    torch.backends.cudnn.deterministic = True  # Ensures deterministic results for GPU
    torch.backends.cudnn.benchmark = (
        False  # Disables the auto-tuner for performance tuning
    )


def load_empty_model_assests(ensemble_space):
    cpdnet_init = [None] * ensemble_space
    Data = [None] * ensemble_space
    cpdnet = [None] * ensemble_space
    cpdnet_tensorboard = [None] * ensemble_space
    return cpdnet_init, Data, cpdnet, cpdnet_tensorboard


def check_path(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


import logging
import sys


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    logger = logging.getLogger("ALACPD")

    # Define a function to log uncaught exceptions
    def log_uncaught_exceptions(exctype, value, traceback):
        logger.error("Uncaught exception", exc_info=(exctype, value, traceback))

    # Assign the function to sys.excepthook
    sys.excepthook = log_uncaught_exceptions
