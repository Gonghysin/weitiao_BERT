import torch
import random
import numpy as np

def set_seed(seed=42):
    """设置所有随机种子以保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False