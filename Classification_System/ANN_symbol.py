import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import Data_Process


class modifiedset(Dataset):
    def __init__(self, features, label):
        self.datapoints = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(label)

    def __len__(self):
        return self.datapoints.size(0)

    def __getitem__(self, idx):
        x = self.datapoints[idx]
        y = self.targets[idx]
        return (x, y)


class ANNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(ANNet, self).__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features, 70)
        self.fc2 = nn.Linear(70, 5)
        #self.fc3 = nn.Linear(30, 5)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        # x = F.softmax(x)
        return x