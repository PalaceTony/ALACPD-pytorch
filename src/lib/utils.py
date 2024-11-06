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

    print(f"Random seed {seed} set for reproducibility.")


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
