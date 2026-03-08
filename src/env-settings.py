#@title 🔧 Environment & helpers
import os, time, math, gc, random, warnings
from collections import deque
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert device == 'cuda', "Switch Colab to GPU (T4). Runtime → Change runtime type → GPU."

# Determinism-ish + perf
torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def peak_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    return 0.0