import copy
import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path
import numpy as np
import torch

def set_seed(seed:int)->None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
