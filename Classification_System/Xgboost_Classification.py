import xgboost as xgb
import numpy as np
import Data_Process
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

import warnings


class xgbset():
    def __init__(self, x_train, y_train, x_test, y_test, title="Misssion_1"):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.title = title
        print("==>", self.title)

    def model_selection(self):
        print("=========>Model Selection<=========")
        parameters = {
            'max_depth': [6, 10, 15],
            'learning_rate': [0.01, 0.02, 0.1, 0.15],
            'subsample': [0.6, 0.8, 0.85, 0.95]
        }
        clf = xgb.XGBClassifier(n_jobs=-1,
                                objective='multi:softprob',
                                # tree_method='gpu_hist', gpu_id=None,
                                max_depth=6, n_estimators=100,
                                min_child_weight=5, gamma=5,
                                subsample=0.8, learning_rate=0.1,
                                nthread=8, colsample_bytree=1.0)
        ms = GridSearchCV(clf, param_grid=parameters, scoring='accuracy', cv=3)
        ms.fit(self.x_train, self.y_train)
        bestacc = ms.best_score_
        bestpara = ms.best_params_
        self.best_params = ms.best_params_
        print(": \n| best acc for train is", bestacc,
              "| best parameters for validation is", bestpara)

    def test(self):
        print("=========>Training Start<=========")
        ## initial parameters thanks to EE569 HW6
        clf = xgb.XGBClassifier(n_jobs=-1,
                                objective='multi:softprob',
                                # tree_method='gpu_hist', gpu_id=None,
                                max_depth=self.best_params['max_depth'],
                                n_estimators=100, min_child_weight=5,
                                gamma=5, subsample=self.best_params['subsample'],
                                learning_rate=self.best_params['learning_rate'],
                                nthread=8, colsample_bytree=1.0)
        clf.fit(self.x_train, self.y_train)
        train_pred = clf.predict(self.x_train)
        test_pred = clf.predict(self.x_test)
        self.test_pred = test_pred
        self.train_acc = accuracy_score(self.y_train, train_pred)
        self.train_f1 = f1_score(self.y_train, train_pred, average='macro')
        self.test_acc = accuracy_score(self.y_test, test_pred)
        self.test_f1 = f1_score(self.y_test, test_pred, average='macro')
        Data_Process.confusionMatrix(self.y_test, test_pred,['1', '2', '3', '4', '5'], self.title)
        print("| train accuracy:", self.train_acc*100, "%",
              "| train f1 score:", self.train_f1)
        print("| test accuracy:", self.test_acc * 100, "%",
              "| test f1 score:", self.test_f1)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    xgb.set_config(verbosity=0)
    ## Mission 1 for G1
    DataTrain1 = pd.read_csv("../student_performance_train.csv")
    DataTest1 = pd.read_csv("../student_performance_test.csv")
    DataTrain1 = Data_Process.process(DataTrain1, onehot=True, labels=['G1', 'G2', 'G3'])
    DataTest1 = Data_Process.process(DataTest1, onehot=True, labels=['G1', 'G2', 'G3'])
    DataTrain1, DataTest1 = DataTrain1.values, DataTest1.values
    train_set1, test_set1 = DataTrain1[:, :-2], DataTest1[:, :-2]
    feature_train1, label_train1 = train_set1[:, :-1], train_set1[:, -1]
    feature_test1, label_test1 = test_set1[:, :-1], test_set1[:, -1]
    #feature_train1, feature_test1 = Data_Process.featuresReduction(feature_train1, feature_test1)
    mission1 = xgbset(feature_train1, label_train1, feature_test1, label_test1, title="Mission_1_XGB")
    mission1.model_selection()
    mission1.test()
    print("\n\n")

    ## Mission 2 for G3
    feature_train2, label_train2 = feature_train1, DataTrain1[:, -1]
    feature_test2, label_test2 = feature_test1, DataTest1[:, -1]
    #feature_train2, feature_test2 = Data_Process.featuresReduction(feature_train2, feature_test2)
    mission2 = xgbset(feature_train2, label_train2, feature_test2, label_test2, title="Mission_2_XGB")
    mission2.model_selection()
    mission2.test()
    print("\n\n")

    ## Mission 3 for G3
    DataTrain3 = pd.read_csv("../student_performance_train.csv")
    DataTest3 = pd.read_csv("../student_performance_test.csv")
    DataTrain3 = Data_Process.process(DataTrain3, onehot=True, labels=['G3'])
    DataTest3 = Data_Process.process(DataTest3, onehot=True, labels=['G3'])
    DataTrain3, DataTest3 = DataTrain3.values, DataTest3.values
    feature_train3, label_train3 = DataTrain3[:, :-1], DataTrain3[:, -1]
    feature_test3, label_test3 = DataTest3[:, :-1], DataTest3[:, -1]
    #feature_train3, feature_test3 = Data_Process.featuresReduction(feature_train3, feature_test3)
    # transfer = StandardScaler()
    # feature_train3 = transfer.fit_transform(feature_train3)
    # feature_test3 = transfer.transform(feature_test3)
    mission3 = xgbset(feature_train3, label_train3, feature_test3, label_test3, title="Mission_3_XGB")
    mission3.model_selection()
    mission3.test()
    print("\n\n")
