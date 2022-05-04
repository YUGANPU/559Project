import numpy as np
import pandas as pd
import Data_Process
from collections import Counter
from random import random
from sklearn.metrics import f1_score

class Baseline:
    def __init__(self, x_train, y_train, x_test, y_test, title="Misssion_1"):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.title = title

    def train(self):
        class_index = [[] for _ in range(5)]
        for i in range(self.y_train.shape[0]):
            temp_label = self.y_train[i]
            class_index[temp_label].append(i)
        class_mean = []
        for i in range(5):
            temp_class_set = self.x_train[class_index[i], :]
            class_mean.append(np.mean(temp_class_set, axis=0))
        self.class_mean = class_mean
        return class_mean

    def test(self):
        self.predict = np.zeros(shape=self.y_test.shape)
        for i in range(self.x_test.shape[0]):
            temp_feature = self.x_test[i]
            distance = []
            for j in range(5):
                temp_distance = self.euclidean_distance(temp_feature, self.class_mean[j])
                distance.append(temp_distance)
            pred_label = distance.index(min(distance))
            self.predict[i] = int(pred_label)
        acc = (self.predict == self.y_test)
        correct = np.count_nonzero(acc)
        print(self.title, ": \n| Accuracy for Nearest Mean is", (correct/self.y_test.shape[0])*100, "%",
              " | f1-macro for Nearest Mean is", f1_score(self.y_test, self.predict, average='macro'))
        Data_Process.confusionMatrix(self.y_test, self.predict,['1', '2', '3', '4', '5'], self.title)
        return self.predict

    def euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2, ord=2)


class Trivial:
    def __init__(self, x_train, y_train, x_test, y_test, title="Misssion_1"):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.title = title
        c = Counter(self.y_train)
        result = []
        for i in range(5):
            prob = c[i] / len(self.y_train)
            result.append(prob)
        self.prob_list = result


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

    def testTimes(self, times):
        sum = 0
        for i in range(times):
            sum+=self.test(self.y_test)
        print("| Accuracy of trivial system:", sum/times, "%")


if __name__ == "__main__":
    ## ----- For Mission 1
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
    # Baseline
    baseline1 = Baseline(feature_train, label_train, feature_test, label_test, title="Mission_1_base")
    mean = baseline1.train()
    baseline1.test()
    # Trivial
    trivial1 = Trivial(feature_train, label_train, feature_test, label_test, title="Mission_1_trivial")
    res = trivial1.testTimes(10)
    print("\n\n")

    ## ----- For Mission 2
    feature_train2, label_train2 = feature_train, DataTrain[:, -1]
    feature_test2, label_test2 = feature_test, DataTest[:, -1]
    # Baseline
    baseline2 = Baseline(feature_train2, label_train2, feature_test2, label_test2, title="Mission_2_base")
    baseline2.train()
    baseline2.test()
    # Trivial
    trivial2 = Trivial(feature_train2, label_train2, feature_test2, label_test2, title="Mission_2_trivial")
    trivial2.testTimes(10)
    print("\n\n")

    ## ----- For Mission 3
    DataTrain3 = pd.read_csv("../student_performance_train.csv")
    DataTest3 = pd.read_csv("../student_performance_test.csv")
    DataTrain3 = Data_Process.process(DataTrain3, onehot=False, labels=['G3'])
    DataTest3 = Data_Process.process(DataTest3, onehot=False, labels=['G3'])
    four_Cate = ['Mjob', 'Fjob', 'reason', 'guardian']
    DataTrain3, DataTest3 = DataTrain3.drop(four_Cate, axis=1), DataTest3.drop(four_Cate, axis=1)
    DataTrain3, DataTest3 = DataTrain3.values, DataTest3.values
    feature_train3, label_train3 = DataTrain3[:, :-1], DataTrain3[:, -1]
    feature_test3, label_test3 = DataTest3[:, :-1], DataTest3[:, -1]
    # transfer = StandardScaler()
    # feature_train3 = transfer.fit_transform(feature_train3)
    # feature_test3 = transfer.transform(feature_test3)
    baseline3 = Baseline(feature_train3, label_train3, feature_test3, label_test3, title="Mission_3_base")
    baseline3.train()
    baseline3.test()
    # Trivial
    trivial3 = Trivial(feature_train3, label_train3, feature_test3, label_test3, title="Mission_3_trivial")
    trivial3.testTimes(10)