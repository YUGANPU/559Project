import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

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


class rbfNet(nn.Module):
    def __init__(self, out_features, in_features):
        super(rbfNet, self).__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
        self.fc = nn.Linear(out_features, 5)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, inputs):
        size = (inputs.size(0), self.out_features, self.in_features)
        x = inputs.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        out = self.basis(distances)
        out = self.fc(out)
        return out

    def basis(self, x):
        out = torch.exp(-1*x.pow(2))
        return out

def load_data(feature_train, label_train):
    train_modified = modifiedset(feature_train, label_train)
    train_loader = DataLoader(train_modified, batch_size=20, shuffle=True)
    return train_loader

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    running_loss = 0
    for i, data in enumerate(train_loader):
        inputs, targets = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    #sys.stdout.write("Epoch no.", epoch + 1, "|loss: ", running_loss/5)
    train_acc = accuracy(model, train_loader)
    sys.stdout.write('\rEpoch: %d, Loss: %f      , Train_acc: %f' % \
                                 (epoch, running_loss, train_acc))

def accuracy(model, x, neg=False):
    with torch.no_grad():
        correct = 0
        total = 0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        for data in x:
            images, targets = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        return (100 * correct / total)


def fit(train_loader):
    net = rbfNet(60, 30)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.1)
    for epoch in range(5000):
        train(train_loader, net, criterion, optimizer, epoch)



def kmeans(data):
    process_data = KMeans(n_clusters=5, random_state=10).fit(data)
    centroids = process_data.cluster_centers_
    return centroids









if __name__ == "__main__":
    DataTrain = pd.read_csv("./student_performance_train.csv")
    DataTest = pd.read_csv("./student_performance_test.csv")
    DataTrain = Data_Process.binary(DataTrain)
    DataTrain = Data_Process.Convert2Label(DataTrain)
    DataTrain = DataTrain.values
    train_set = DataTrain[:, :-2]
    feature_train, label_train = train_set[:, :-1], train_set[:, -1]
    # Load data
    train_loader = load_data(feature_train, label_train)
    fit(train_loader)
    # get centroids of training features
    centroid = kmeans(feature_train)
    print(0)