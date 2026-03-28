import logging
import os
import random

import numpy as np
import torch


def get_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Creates a logger."""
    logging.basicConfig(
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    return logger


def seed_everything(seed: int = 314159, torch_deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(torch_deterministic)
