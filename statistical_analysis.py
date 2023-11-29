import torch
import itertools
import sys
import random
import numpy as np
from utils import train_model, train_model_topology

# 1. Set seeds for reproducibility
SEED = 42
random.seed(SEED)          # For Python's random module
np.random.seed(SEED)       # For NumPy
torch.manual_seed(SEED)    # For PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataname = 'MNIST'

saved_model_location = f"./experiments/{dataname}/trained_models/"
batch_size = 64
n_classes = 10
topology_loss = 'PD1' # or 'PD0' or 'CE'
regular_simplex = True
balanced = False

base_config = {
    'topology_lr': 1e-5, # default is 1e-5
    'n_statistical_analysis': 1,
    'epochs': 200,
    'additional_epochs' : 5,
}

# train_model(dataname, device, n_classes = 10, balanced = True, batch_size = 64, n_statistical_analysis=1, epochs=10)


# Update the config dictionary for the current combination
config = {
    **base_config,
    'topology_loss': topology_loss,
    'regular_simplex': regular_simplex,
    'n_classes': n_classes,
    'balanced': balanced,
    'batch_size': batch_size,
}

train_model_topology(config=config, dataname=dataname, device=device, save_results=True)







