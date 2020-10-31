import torch
import random
import numpy as np
from pathlib import Path


def set_seed_everywhere(seed, cuda):
    print('Setting seed')
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath):
    Path(dirpath).mkdir(parents=True, exist_ok=True)
