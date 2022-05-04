import numpy as np
import pandas as pd
from sklearn.svm import SVR
import Data_Process
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

class svrset():
    def __init__(self, x_train, y_train, x_test, y_test, title="Mission_1"):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.title = title

    def model_selection(self):
        print("-------------model selection------------")

        parameter = {
            "kernel": ['rbf', 'linear', 'poly']
        }

        estimator = SVR(kernel='rbf')

        ms = GridSearchCV(estimator, param_grid=parameter, cv=5)
        ms.fit(self.x_train, self.y_train)
        bestresult = ms.best_score_
        bestpara = ms.best_params_
        self.best_params = ms.best_params_
        print("==>", self.title,
              ": \n| best R^2 for train is", bestresult,
              "| best parameters for validation is", bestpara)

    def test(self):
        print("-------------Training Start------------")
        estimator = SVR(kernel=self.best_params["kernel"])
        estimator.fit(self.x_train, self.y_train)
        train_pred = estimator.predict(self.x_train)
        test_pred = estimator.predict(self.x_test)
        self.test_pred = test_pred

        self.train_r2 = r2_score(self.y_train, train_pred)
        self.train_RMSE = sqrt(mean_squared_error(self.y_train, train_pred))
        self.train_MAE = mean_absolute_error(self.y_train, train_pred)

        self.test_r2 = r2_score(self.y_test, test_pred)
        self.test_RMSE = sqrt(mean_squared_error(self.y_test, test_pred))
        self.test_MAE = mean_absolute_error(self.y_test, test_pred)

        print("the R^2 of trainset is", self.train_r2)
        print("the RMSE of trainset is", self.train_RMSE)
        print("the MAE of trainset is", self.train_MAE)
        print("-----------------------------------------------")
        print("the R^2 of testset is", self.test_r2)
        print("the RMSE of testset is", self.test_RMSE)
        print("the MAE of testset is", self.test_MAE)




if __name__=="__main__":
    DataTrain = pd.read_csv("../student_performance_train.csv")
    DataTest = pd.read_csv("../student_performance_test.csv")
    DataTrain = Data_Process.process(DataTrain, onehot=True)
    DataTest = Data_Process.process(DataTest, onehot=True)


    # tmp = np.array([0, 1, 3, 4, 5, 6, 7, 8, 10,
    #                 12, 13, 14, 15, 17, 18, 19, 20,
    #                 21, 22, 27, 29])

    tmp = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10,
                    11, 12, 15, 18, 19,
                    22, 23, 24, 25, 26, 27, 28, 29, 30,
                    31, 32, 33, 34, 36, 37, 38,
                    41])

    tmp2 = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10,
                     11, 12, 15, 18, 19,
                     22, 23, 24, 25, 26, 27, 28, 29, 30,
                     31, 32, 33, 34, 36, 37, 38,
                     41, 42, 43, 44])

    TrainSet = DataTrain.values
    TrainSet_feature1 = TrainSet[:, tmp]
    TrainSet_label1 = TrainSet[:, -3]


    TestSet = DataTest.values
    TestSet_feature1 = TestSet[:, tmp]
    TestSet_label1 = TestSet[:, -3]

    transfer = StandardScaler()
    TrainSet_feature1 = transfer.fit_transform(TrainSet_feature1)
    TestSet_feature1 = transfer.transform(TestSet_feature1)

    mission1 = svrset(TrainSet_feature1,TrainSet_label1,TestSet_feature1,TestSet_label1,title="Mission_1")
    mission1.model_selection()
    mission1.test()

    ###############
    TrainSet_feature2, TrainSet_labels2 = TrainSet[:, tmp], TrainSet[:, -1]
    TestSet_feature2, TestSet_labels2 = TestSet[:, tmp], TestSet[:, -1]
    TrainSet_feature2 = transfer.fit_transform(TrainSet_feature2)
    TestSet_feature2 = transfer.fit_transform(TestSet_feature2)

    transfer = StandardScaler()
    TrainSet_feature2 = transfer.fit_transform(TrainSet_feature2)
    TestSet_feature2 = transfer.transform(TestSet_feature2)

    mission2 = svrset(TrainSet_feature2,TrainSet_labels2,TestSet_feature2,TestSet_labels2,title="Mission_2")
    mission2.model_selection()
    mission2.test()

    ###############
    TrainSet_feature3, TrainSet_labels3 = TrainSet[:, tmp2], TrainSet[:, -1]
    TestSet_feature3, TestSet_labels3 = TestSet[:, tmp2], TestSet[:, -1]
    TrainSet_feature3 = transfer.fit_transform(TrainSet_feature3)
    TestSet_feature3 = transfer.fit_transform(TestSet_feature3)

    transfer = StandardScaler()
    TrainSet_feature3 = transfer.fit_transform(TrainSet_feature3)
    TestSet_feature3 = transfer.transform(TestSet_feature3)

    mission3 = svrset(TrainSet_feature3, TrainSet_labels3, TestSet_feature3, TestSet_labels3, title="Mission_3")
    mission3.model_selection()
    mission3.test()


    # TrainSet = DataTrain.values
    # TrainSet_feature1 = TrainSet[:, :-3]
    # TrainSet_label1 = TrainSet[:, -3]
    #
    #
    # TestSet = DataTest.values
    # TestSet_feature1 = TestSet[:, :-3]
    # TestSet_label1 = TestSet[:, -3]
    #
    # transfer = StandardScaler()
    # TrainSet_feature1 = transfer.fit_transform(TrainSet_feature1)
    # TestSet_feature1 = transfer.transform(TestSet_feature1)
    #
    # mission1 = svrset(TrainSet_feature1,TrainSet_label1,TestSet_feature1,TestSet_label1,title="Mission_1")
    # mission1.model_selection()
    # mission1.test()
    #
    # ###############
    # TrainSet_feature2, TrainSet_labels2 = TrainSet[:, :-3], TrainSet[:, -1]
    # TestSet_feature2, TestSet_labels2 = TestSet[:, :-3], TestSet[:, -1]
    # TrainSet_feature2 = transfer.fit_transform(TrainSet_feature2)
    # TestSet_feature2 = transfer.fit_transform(TestSet_feature2)
    #
    # transfer = StandardScaler()
    # TrainSet_feature2 = transfer.fit_transform(TrainSet_feature2)
    # TestSet_feature2 = transfer.transform(TestSet_feature2)
    #
    # mission2 = svrset(TrainSet_feature2,TrainSet_labels2,TestSet_feature2,TestSet_labels2,title="Mission_2")
    # mission2.model_selection()
    # mission2.test()
    #
    # ###############
    # TrainSet_feature3, TrainSet_labels3 = TrainSet[:, :-1], TrainSet[:, -1]
    # TestSet_feature3, TestSet_labels3 = TestSet[:, :-1], TestSet[:, -1]
    # TrainSet_feature3 = transfer.fit_transform(TrainSet_feature3)
    # TestSet_feature3 = transfer.fit_transform(TestSet_feature3)
    #
    # transfer = StandardScaler()
    # TrainSet_feature3 = transfer.fit_transform(TrainSet_feature3)
    # TestSet_feature3 = transfer.transform(TestSet_feature3)
    #
    # mission3 = svrset(TrainSet_feature3, TrainSet_labels3, TestSet_feature3, TestSet_labels3, title="Mission_3")
    # mission3.model_selection()
    # mission3.test()
    #

