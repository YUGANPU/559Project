import numpy as np
import pandas as pd
import Data_Process

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




if __name__ == "__main__":
    ## For Mission 1
    DataTrain = pd.read_csv("../student_performance_train.csv")
    DataTest = pd.read_csv("../student_performance_test.csv")
    DataTrain, DataTest = Data_Process.binary(DataTrain), Data_Process.binary(DataTest)
    DataTrain, DataTest = Data_Process.Convert2Label(DataTrain), Data_Process.Convert2Label(DataTest)
    DataTrain, DataTest = DataTrain.values, DataTest.values
    train_set, test_set = DataTrain[:, :-2], DataTest[:, :-2]
    feature_train, label_train = train_set[:, :-1], train_set[:, -1]
    feature_test, label_test = test_set[:, :-1], test_set[:, -1]
    ## Baseline
    baseline = Baseline()
    mean = baseline.train(feature_train, label_train)
    baseline.test(feature_test, label_test)
    print(0)