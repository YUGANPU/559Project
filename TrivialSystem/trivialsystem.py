import Data_Process
import pandas as pd
import numpy as np
import sklearn
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

    TrainSet = DataTrain.values
    TrainSet_feature = TrainSet[: , :-3]

    TestSet = DataTest.values


    #mission1
    G1 = TrainSet[:,-3]
    mean_G1 = np.mean(G1)
    predict1 = np.array([mean_G1 for i in range(TestSet.shape[0])])
    #print(predict1)
    Test_G1 = TestSet[:,-3]

    print("For mission1:")

    print("R^2 score is",r2_score(Test_G1,predict1))
    print("rmse is", sqrt(mean_squared_error(Test_G1, predict1)))
    print("mean_absolute_error is", mean_absolute_error(Test_G1, predict1))

    print("--------------------------------------------------------------------------")


    #misiion 2,3
    G3 = TrainSet[:,-1]
    mean_G3 = np.mean(G3)
    predict3 = np.array([mean_G3 for i in range(TestSet.shape[0])])
    Test_G3 = TestSet[:,-1]
    #print(predict3)
    Test_G3 = TestSet[:, -1]

    print("For mission2,3:")
    print("R^2 score is", r2_score(Test_G3, predict3))
    print("rmse is", sqrt(mean_squared_error(Test_G3, predict3)))
    print("mean_absolute_error is", mean_absolute_error(Test_G3, predict3))

