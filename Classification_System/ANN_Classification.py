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
        self.fc1 = nn.Linear(in_features, 60)
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 5)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = F.softmax(x)
        return x

def load_data(feature_train, label_train, batch_size=200):
    train_modified = modifiedset(feature_train, label_train)
    train_loader = DataLoader(train_modified, batch_size=batch_size, shuffle=True)
    return train_loader

def train(train_loader, test_loader, model, criterion, optimizer, epoch):
    model.train()
    running_loss = 0
    for i, data in enumerate(train_loader, 0):
        inputs, targets = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    #sys.stdout.write("Epoch no.", epoch + 1, "|loss: ", running_loss/5)
    train_acc = accuracy(model, train_loader)
    test_acc = accuracy(model, test_loader)
    sys.stdout.write('\rEpoch: %d, Loss: %f      , Train_acc: %f' % \
                                 (epoch, running_loss, train_acc))
    return train_acc, test_acc

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


def fit(train_loader, test_loader):
    net = ANNet(30, 5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    train_acc = []
    test_acc = []
    for epoch in range(10000):
        train_acc_t, test_acc_t = train(train_loader, test_loader, net, criterion, optimizer, epoch)
        train_acc.append(train_acc_t)
        test_acc.append(test_acc_t)
    display(train_acc, test_acc)


def kmeans(data):
    process_data = KMeans(n_clusters=5, random_state=10).fit(data)
    centroids = process_data.cluster_centers_
    return centroids

def display(train_acc,test_acc):
    fig,ax=plt.subplots()
    ax.plot(range(1,len(train_acc)+1),train_acc,color='r',label='train_acc')
    ax.plot(range(1,len(test_acc)+1),test_acc,color='b',label='test_acc')
    ax.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    DataTrain = pd.read_csv("../student_performance_train.csv")
    DataTest = pd.read_csv("../student_performance_test.csv")
    DataTrain, DataTest = Data_Process.binary(DataTrain), Data_Process.binary(DataTest)
    DataTrain, DataTest = Data_Process.Convert2Label(DataTrain), Data_Process.Convert2Label(DataTest)
    DataTrain, DataTest = DataTrain.values, DataTest.values
    train_set, test_set = DataTrain[:, :-2], DataTest[:, :-2]
    feature_train, label_train = train_set[:, :-1], train_set[:, -1]
    feature_test, label_test = test_set[:, :-1], test_set[:, -1]
    #feature_train = Data_Process.standardize(feature_train)
    # Load data
    train_loader = load_data(feature_train, label_train)
    test_loader = load_data(feature_test, label_test, batch_size=len(test_set))
    fit(train_loader, test_loader)
    # get centroids of training features
    centroid = kmeans(feature_train)
    print(0)