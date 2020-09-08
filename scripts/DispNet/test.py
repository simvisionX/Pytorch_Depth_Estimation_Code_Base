import torch
import numpy as np
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dataloader
from dataloader import transforms
from model_zoo import DispNet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

torch.manual_seed(233)
torch.cuda.manual_seed(233)
np.random.seed(233)

torch.backends.cudnn.benchmark = True

device = torch.device("cuda:1")

