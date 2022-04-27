import Data_Process
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib


if __name__ == "__main__":

    DataTrain = pd.read_csv("../student_performance_train.csv")
    DataTest = pd.read_csv("../student_performance_test.csv")
    Data_Process.binary(DataTrain)
    Data_Process.binary(DataTest)
    DataTrain_Label = Data_Process.Convert2Label(DataTrain)
    DataTest_Label = Data_Process.Convert2Label(DataTest)

    #print(DataTest)
    #print(DataTrain_Label)
    #print(DataTest_Label)

    TrainSet = DataTrain_Label.values

    TrainSet_feature = TrainSet[:, :-3]
    TrainSet_label = TrainSet[:, -3]

    TestSet = DataTest_Label.values
    TestSet_feature = TestSet[:, :-3]
    TestSet_label = TestSet[:, -3]
    # print(TrainSet)
    # print(TrainSet_feature)
    # print(TrainSet_label)
    transfer = StandardScaler()

    TrainSet_feature = transfer.fit_transform(TrainSet_feature)
    TestSet_feature = transfer.fit_transform(TestSet_feature)

    estimator = KNeighborsClassifier(n_neighbors=20)
    estimator.fit(TrainSet_feature, TrainSet_label)

    predict = estimator.predict(TestSet_feature)


    print(TestSet_label)
    print(predict)

    score1 = estimator.score(TrainSet_feature,TrainSet_label)
    score2 = estimator.score(TestSet_feature,TestSet_label)

    print("TrainSet accuracy is %s" %(score1))
    print("TestSet accuracy is %s" %(score2))
