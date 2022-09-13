import os
import sys
import math
import time
from typing import List, Tuple, Dict, OrderedDict, Optional, Union
import itertools
from collections import OrderedDict

import cv2
import numpy as np

import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

import yaml
import hydra
from omegaconf import DictConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("You are on " + str(torch.cuda.get_device_name(device)))
else:
    print("You are on " + str(device).upper())


Numpy = np.array
Tensor = torch.Tensor


checkpoints = {
    'efficientnet_b0_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b0_backbone.pth',
    'efficientnet_b1_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b1_backbone.pth',
    'efficientnet_b2_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b2_backbone.pth',
    'efficientnet_b3_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b3_backbone.pth',
    'efficientnet_b4_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b4_backbone.pth',
    'efficientnet_b5_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b5_backbone.pth',
    'efficientnet_b6_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b6_backbone.pth',
    'efficientnet_b7_backbone': 'https://github.com/hotcouscous1/TensorBricks/releases/download/checkpoints/efficientnet_b7_backbone.pth',
}
