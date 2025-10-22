import os
import random
import numpy as np


def set_seed(seed: int = 0, deterministic: bool = True):
    try:
        import torch
    except Exception:
        torch = None

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


