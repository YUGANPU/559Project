import Data_Process
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
import matplotlib
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score




if __name__ == "__main__":
    DataTrain = pd.read_csv("../student_performance_train.csv")
    DataTest = pd.read_csv("../student_performance_test.csv")

    DataTrain = Data_Process.process(DataTrain, onehot=False)
    DataTest = Data_Process.process(DataTest, onehot=False)

    ###### mission1

    TrainSet = DataTrain.values
    tmp1 = np.array([0, 1, 2, 3, 4, 5, 6, 7,
                    12, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 26, 27, 28, 29])
    TrainSet_feature1 = TrainSet[:, tmp1]
    TrainSet_label1 = TrainSet[:, -3]

    TestSet = DataTest.values
    TestSet_feature1 = TestSet[:, tmp1]
    TestSet_label1 = TestSet[:, -3]

    estimator1 = KNeighborsRegressor(n_neighbors=1)
    estimator1.fit(TrainSet_feature1,TrainSet_label1)

    predict1 = estimator1.predict(TestSet_feature1)

    print("for mission1:")
    print("R^2 score is", r2_score(TestSet_label1, predict1))
    print("rmse is", sqrt(mean_squared_error(TestSet_label1, predict1)))
    print("mean_absolute_error is", mean_absolute_error(TestSet_label1, predict1))

    ##### mission2:
    TrainSet = DataTrain.values
    tmp2 = np.array([0, 1, 2, 3, 4, 5, 6, 7,
                     12, 13, 14, 15, 16, 17, 18, 19, 20,
                     21, 22, 23, 24, 25, 26, 27, 28, 29])
    TrainSet_feature2 = TrainSet[:, tmp2]
    TrainSet_label2 = TrainSet[:, -1]

    TestSet = DataTest.values
    TestSet_feature2 = TestSet[:, tmp2]
    TestSet_label2 = TestSet[:, -1]

    estimator2 = KNeighborsRegressor(n_neighbors=1)
    estimator2.fit(TrainSet_feature2, TrainSet_label2)

    predict2 = estimator2.predict(TestSet_feature2)

    print("-----------------------------------------------------------------")
    print("for mission2:")
    print("R^2 score is", r2_score(TestSet_label2, predict2))
    print("rmse is", sqrt(mean_squared_error(TestSet_label2, predict2)))
    print("mean_absolute_error is", mean_absolute_error(TestSet_label2, predict2))

    #### fro mission3:
    TrainSet = DataTrain.values
    tmp3 = np.array([0, 1, 2, 3, 4, 5, 6, 7,
                     12, 13, 14, 15, 16, 17, 18, 19, 20,
                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
    TrainSet_feature3 = TrainSet[:, tmp3]
    TrainSet_label3 = TrainSet[:, -1]

    TestSet = DataTest.values
    TestSet_feature3 = TestSet[:, tmp3]
    TestSet_label3 = TestSet[:, -1]

    estimator3 = KNeighborsRegressor(n_neighbors=1)
    estimator3.fit(TrainSet_feature3, TrainSet_label3)

    predict3 = estimator3.predict(TestSet_feature3)

    print("-----------------------------------------------------------------")
    print("for mission3:")
    print("R^2 score is", r2_score(TestSet_label3, predict3))
    print("rmse is", sqrt(mean_squared_error(TestSet_label3, predict3)))
    print("mean_absolute_error is", mean_absolute_error(TestSet_label3, predict3))

