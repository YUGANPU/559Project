import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import Data_Process
from ANN_symbol import ANNet, modifiedset


class ANN_classifier:
    def __init__(self, x_train, y_train, x_test, y_test, batch_size=200, lr=0.005, epoch=500,
                 title="Misssion_1", silent=False):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.title = title
        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch
        self.silent = silent
        self.infeatures = self.x_train.shape[1]
        self.train_loader = self.load_data(self.x_train, self.y_train, self.batch_size, True)
        self.test_loader = self.load_data(self.x_test, self.y_test, len(self.y_test), False)

    def load_data(self, x, y, batch_size, shuffle=True):
        data_modified = modifiedset(x, y)
        data_loader = DataLoader(data_modified, batch_size=batch_size, shuffle=shuffle)
        return data_loader

    def train(self, criterion, optimizer, epoch):
        self.net.train()
        running_loss = 0
        for i, data in enumerate(self.train_loader, 0):
            inputs, targets = data
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # sys.stdout.write("Epoch no.", epoch + 1, "|loss: ", running_loss/5)
        train_acc, _, _ = self.accuracy(self.net, self.train_loader)
        test_acc, _, _ = self.accuracy(self.net, self.test_loader)
        sys.stdout.write('\rEpoch: %d, Loss: %3f, Train_acc: %3f, Test_acc: %3f' % \
                         (epoch, running_loss, train_acc, test_acc))
        sys.stdout.flush()
        return train_acc, test_acc

    def accuracy(self, model, x):
        with torch.no_grad():
            correct = 0
            total = 0
            for data in x:
                images, targets = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            return (100 * correct / total), predicted, targets

    def fit(self):
        self.net = ANNet(self.infeatures, 5)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.train_acc_list = []
        self.test_acc_list = []
        for epoch in range(self.epoch):
            train_acc_t, test_acc_t = self.train(criterion, optimizer, epoch)
            self.train_acc_list.append(train_acc_t)
            self.test_acc_list.append(test_acc_t)
        if not self.silent:
            self.display(self.train_acc_list, self.test_acc_list)
        self.save_result()

    def display(self, train_acc, test_acc):
        fig, ax = plt.subplots()
        fig.suptitle(self.title)
        ax.plot(range(1, len(train_acc) + 1), train_acc, color='r', label='train_acc')
        ax.plot(range(1, len(test_acc) + 1), test_acc, color='b', label='test_acc')
        ax.legend(loc='lower right')
        plt.show()

    def save_result(self):
        self.train_acc, _, _ = self.accuracy(self.net, self.train_loader)
        self.test_acc, self.preds, self.targets = self.accuracy(self.net, self.test_loader)
        self.test_f1 = f1_score(self.targets, self.preds, average='macro')
        if not self.silent:
            Data_Process.confusionMatrix(self.targets, self.preds, ['1', '2', '3', '4', '5'], self.title)
            print(self.title,
                  ":\n| final train acc is", self.train_acc, "%",
                  " | final test acc is", self.test_acc, "%",
                  " | final test f1_score is", self.test_f1)




def model_selection(features, labels, title):
    learning_rate_list = [0.001, 0.002, 0.005]
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
    best_acc, best_lr = 0.0, 0.0
    for lr in learning_rate_list:
        modelSelect = ANN_classifier(X_train, y_train, X_test, y_test, batch_size=200, lr=lr, epoch=5000, title="MS", silent=True)
        modelSelect.fit()
        if modelSelect.test_acc > best_acc:
            best_acc = modelSelect.test_acc
            best_lr = lr
    print("\nThe best lr for", title, "is", best_lr, "test_acc:", best_acc)
    return best_lr





if __name__ == "__main__":
    ## Mission 1 for G1
    print("====> Mission 1 ANN:\n")
    DataTrain1 = pd.read_csv("../student_performance_train.csv")
    DataTest1 = pd.read_csv("../student_performance_test.csv")
    DataTrain1 = Data_Process.process(DataTrain1, onehot=True, labels=['G1', 'G2', 'G3'])
    DataTest1 = Data_Process.process(DataTest1, onehot=True, labels=['G1', 'G2', 'G3'])
    DataTrain1, DataTest1 = DataTrain1.values, DataTest1.values
    train_set1, test_set1 = DataTrain1[:, :-2], DataTest1[:, :-2]
    feature_train1, label_train1 = train_set1[:, :-1], train_set1[:, -1]
    feature_test1, label_test1 = test_set1[:, :-1], test_set1[:, -1]
    transfer = StandardScaler()
    feature_train1 = transfer.fit_transform(feature_train1)
    feature_test1 = transfer.transform(feature_test1)
    best_lr1 = model_selection(feature_train1, label_train1, "Mission_1")
    mission1 = ANN_classifier(feature_train1, label_train1, feature_test1, label_test1, batch_size=200, lr=best_lr1, epoch=5000, title="Misssion_1_ANN")
    mission1.fit()
    print("\n\n")

    ## Mission 2 for G3
    print("====> Mission 2 ANN:\n")
    feature_train2, label_train2 = feature_train1, DataTrain1[:, -1]
    feature_test2, label_test2 = feature_test1, DataTest1[:, -1]
    transfer = StandardScaler()
    feature_train2 = transfer.fit_transform(feature_train2)
    feature_test2 = transfer.transform(feature_test2)
    best_lr2 = model_selection(feature_train2, label_train2, "Mission_2")
    mission2 = ANN_classifier(feature_train2, label_train2, feature_test2, label_test2, batch_size=200, lr=best_lr2, epoch=5000, title="Mission_2_ANN")
    mission2.fit()
    print("\n\n")

    ## Mission 3 for G3
    print("====> Mission 3 ANN:\n")
    DataTrain3 = pd.read_csv("../student_performance_train.csv")
    DataTest3 = pd.read_csv("../student_performance_test.csv")
    DataTrain3 = Data_Process.process(DataTrain3, onehot=True, labels=['G3'])
    DataTest3 = Data_Process.process(DataTest3, onehot=True, labels=['G3'])
    DataTrain3, DataTest3 = DataTrain3.values, DataTest3.values
    feature_train3, label_train3 = DataTrain3[:, :-1], DataTrain3[:, -1]
    feature_test3, label_test3 = DataTest3[:, :-1], DataTest3[:, -1]
    transfer = StandardScaler()
    feature_train3 = transfer.fit_transform(feature_train3)
    feature_test3 = transfer.transform(feature_test3)
    best_lr3 = model_selection(feature_train3, label_train3, "Mission_3")
    mission3 = ANN_classifier(feature_train3, label_train3, feature_test3, label_test3, batch_size=200, lr=best_lr3, epoch=5000, title="Mission_3_ANN")
    mission3.fit()
    print("\n\n")
