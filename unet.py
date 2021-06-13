import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torchvision
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm

def choose_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(device)

def 
