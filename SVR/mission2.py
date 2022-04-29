import numpy as np
import pandas as pd
from sklearn.svm import SVR
import Data_Process
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


if __name__=="__main__":
    DataTrain = pd.read_csv("../student_performance_train.csv")
    DataTest = pd.read_csv("../student_performance_test.csv")
    Data_Process.binary(DataTrain)
    Data_Process.binary(DataTest)

    DataTrain_Label = DataTrain
    DataTest_Label = DataTest

    TrainSet = DataTrain_Label.values
    TrainSet_feature = TrainSet[:, :-3]
    TrainSet_label = TrainSet[:, -1]


    TestSet = DataTest_Label.values
    TestSet_feature = TestSet[:, :-3]
    TestSet_label = TestSet[:, -1]

    transfer = StandardScaler()
    TrainSet_feature = transfer.fit_transform(TrainSet_feature)
    TestSet_feature = transfer.transform(TestSet_feature)

    regressor = SVR(kernel='rbf')
    regressor.fit(TrainSet_feature,TrainSet_label)

    predict = regressor.predict(TestSet_feature)

    print("R^2 score is",r2_score(TestSet_label,predict))
    print("rmse is", sqrt(mean_squared_error(TestSet_label, predict)))
    print("mean_absolute_error is", mean_absolute_error(TestSet_label, predict))