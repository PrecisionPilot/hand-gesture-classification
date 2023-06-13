#!/usr/bin/env python3

import numpy as np  # to load dataset
import math
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.optim
import os

block_size = 24
columns = 20

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device in use: {device}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(block_size * columns, 256), # input
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

print(f"input size {block_size * columns}")


# replace variable with local path in linux
PATH = "classifier.pt"

### Load Existing Model
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(PATH))
model.eval() # must set dropout and batch normalization layers to evaluation mode

print("finished running")