import Data_Process
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

if __name__ == "__main__":

    DataTrain = pd.read_csv("../student_performance_train.csv")
    DataTest = pd.read_csv("../student_performance_test.csv")
    Data_Process.binary(DataTrain)
    Data_Process.binary(DataTest)
    # 按等级转化
    # DataTrain_Label = Data_Process.Convert2Label(DataTrain)
    # DataTest_Label = Data_Process.Convert2Label(DataTest)
    # 不按等级转化
    DataTrain_Label = DataTrain
    DataTest_Label = DataTest

    #print(DataTest)
    #print(DataTrain_Label)
    #print(DataTest_Label)


    TrainSet = DataTrain_Label.values

    tmp = np.array([0,2,4,12,13,14,17,30,31])
    #tmp = np.array([12, 13, 14, 17, 30, 31])

    TrainSet_feature = TrainSet[:, :-1]
    TrainSet_label = TrainSet[:, -1]

    TestSet = DataTest_Label.values

    TestSet_feature = TestSet[:, :-1]
    TestSet_label = TestSet[:, -1]
    # print(TrainSet)
    # print(TrainSet_feature)
    # print(TrainSet_label)
    transfer = StandardScaler()

    TrainSet_feature = transfer.fit_transform(TrainSet_feature)
    TestSet_feature = transfer.fit_transform(TestSet_feature)

    estimator = KNeighborsRegressor(n_neighbors=20)
    estimator.fit(TrainSet_feature, TrainSet_label)

    predict = estimator.predict(TestSet_feature)



    # print(predict)
    # print(TestSet_label)



    print("the R^2 is ",estimator.score(TestSet_feature,TestSet_label))
    print("rmse is", sqrt(mean_squared_error(TestSet_label, predict)))
    print("mean_absolute_error is", mean_absolute_error(TestSet_label, predict))