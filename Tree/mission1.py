import numpy as np
import pandas as pd
from sklearn.svm import SVR
import Data_Process
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.tree import DecisionTreeRegressor


if __name__ == "__main__":
    DataTrain = pd.read_csv("../student_performance_train.csv")
    DataTest = pd.read_csv("../student_performance_test.csv")
    Data_Process.binary(DataTrain)
    Data_Process.binary(DataTest)
    tmp = np.array([0, 1, 3, 4, 5, 6, 7, 8, 10,
                    12, 13, 14, 15, 17, 18, 19, 20,
                    21, 22, 27, 29])
    tmp2 = np.array([0, 1, 3, 4, 5, 6, 7, 8, 10,
                    12, 13, 14, 15, 17, 18, 19, 20,
                    21, 22, 27, 29, 30, 31])

    TrainSet = DataTrain.values
    Train_feature = TrainSet[:, tmp]
    Train_label = TrainSet[:, -3]

    TestSet = DataTest.values
    Test_feature = TestSet[:, tmp]
    Test_label = TestSet[:, -3]

    transfer = StandardScaler()
    TrainSet_feature = transfer.fit_transform(Train_feature)
    TestSet_feature = transfer.transform(Test_feature)

    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(Train_feature,Train_label)

    predict = tree.predict(Test_feature)

    print("R^2 score is", r2_score(Test_label, predict))
    print("rmse is", sqrt(mean_squared_error(Test_label, predict)))
    print("mean_absolute_error is", mean_absolute_error(Test_label, predict))



