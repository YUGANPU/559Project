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

    tmp = np.array([0, 1,  3, 4, 5, 6, 7, 8,10,
                    12, 13, 14, 15, 17, 18, 19, 20,
                    21, 22, 27, 29])
    #tmp = np.array([0,2,4,12,13,14,17])
    #tmp = np.array([12, 13, 14,17])
    #tmp = np.array([9])

    TrainSet_feature = TrainSet[:,tmp]
    TrainSet_label = TrainSet[:, -3]

    TestSet = DataTest_Label.values

    TestSet_feature = TestSet[:, tmp]
    TestSet_label = TestSet[:, -3]
    # print(TrainSet)
    # print(TrainSet_feature)
    # print(TrainSet_label)
    transfer = StandardScaler()

    TrainSet_feature = transfer.fit_transform(TrainSet_feature)
    TestSet_feature = transfer.fit_transform(TestSet_feature)

    estimator = KNeighborsRegressor(n_neighbors=16)
    estimator.fit(TrainSet_feature, TrainSet_label)

    predict = estimator.predict(TestSet_feature)

    # print(predict)
    # print(TestSet_label)


    #print("outcome of Mission1：",num/len(predict))
    print("the R^2 is ",estimator.score(TestSet_feature,TestSet_label))
    print("rmse is",sqrt(mean_squared_error(TestSet_label,predict)))
    print("mean_absolute_error is",mean_absolute_error(TestSet_label,predict))
