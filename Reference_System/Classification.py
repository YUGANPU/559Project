import numpy as np
import pandas as pd
import Data_Process
from collections import Counter
from random import random

class Baseline:
    def train(self, features, labels):
        class_index = [[] for _ in range(5)]
        for i in range(labels.shape[0]):
            temp_label = labels[i]
            class_index[temp_label].append(i)
        class_mean = []
        for i in range(5):
            temp_class_set = features[class_index[i], :]
            class_mean.append(np.mean(temp_class_set, axis=0))
        self.class_mean = class_mean
        return class_mean

    def test(self, features, labels):
        self.predict = np.zeros(shape=labels.shape)
        for i in range(features.shape[0]):
            temp_feature = features[i]
            distance = []
            for j in range(5):
                temp_distance = self.euclidean_distance(temp_feature, self.class_mean[j])
                distance.append(temp_distance)
            pred_label = distance.index(min(distance))
            self.predict[i] = int(pred_label)
        acc = (pred_label == labels)
        correct = np.count_nonzero(acc)
        print("Accuracy for Nearest Mean is", (correct/labels.shape[0])*100, "%")
        return self.predict

    def euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2, ord=2)


class Trivial:
    def get_prob(self, label_train):
        c = Counter(label_train)
        result = []
        for i in range(5):
            prob = c[i]/len(label_train)
            result.append(prob)
        self.prob_list = result
        return result

    def generate_wp(self):
        rnd = random()
        prob = 0
        for i in range(5):
            prob+=self.prob_list[i]
            if rnd <= prob:
                return i

    def test(self, label_test):
        predict_result = []
        for i in range(len(label_test)):
            temp = self.generate_wp()
            predict_result.append(temp)
        correct = np.count_nonzero(predict_result == label_test)
        self.predict = predict_result
        #print("Accuracy for Probability output is", (correct/label_test.shape[0])*100, "%")
        return (correct/label_test.shape[0])*100

    def testTimes(self, label_test, times):
        sum = 0
        for i in range(times):
            sum+=self.test(label_test)
        return sum/times


if __name__ == "__main__":
    ## For Mission 1
    DataTrain = pd.read_csv("../student_performance_train.csv")
    DataTest = pd.read_csv("../student_performance_test.csv")
    DataTrain = Data_Process.process(DataTrain, onehot=False, labels=['G1', 'G2', 'G3'])
    DataTest = Data_Process.process(DataTest, onehot=False, labels=['G1', 'G2', 'G3'])
    four_Cate = ['Mjob', 'Fjob', 'reason', 'guardian']
    DataTrain, DataTest = DataTrain.drop(four_Cate, axis=1), DataTest.drop(four_Cate, axis=1)
    DataTrain, DataTest = DataTrain.values, DataTest.values
    train_set, test_set = DataTrain[:, :-2], DataTest[:, :-2]
    feature_train, label_train = train_set[:, :-1], train_set[:, -1]
    feature_test, label_test = test_set[:, :-1], test_set[:, -1]
    ## Baseline
    baseline = Baseline()
    mean = baseline.train(feature_train, label_train)
    baseline.test(feature_test, label_test)
    ## Trivial
    trivial = Trivial()
    prob_list = trivial.get_prob(label_train)
    res = trivial.testTimes(label_test, 10)
    print(0)