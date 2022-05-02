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
    Data_Process.binary(DataTrain)
    Data_Process.binary(DataTest)

    TrainSet = DataTrain.values
    tmp = np.array([0, 1, 2, 3, 4, 5, 6, 7,
                    12, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
    TrainSet_feature = TrainSet[:, tmp]
    TrainSet_label = TrainSet[:, -1]

    TestSet = DataTest.values
    TestSet_feature = TestSet[:, tmp]
    TestSet_label = TestSet[:, -1]

    estimator = KNeighborsRegressor(n_neighbors=1)
    estimator.fit(TrainSet_feature,TrainSet_label)

    predict = estimator.predict(TestSet_feature)

    print("R^2 score is", r2_score(TestSet_label, predict))
    print("rmse is", sqrt(mean_squared_error(TestSet_label, predict)))
    print("mean_absolute_error is", mean_absolute_error(TestSet_label, predict))