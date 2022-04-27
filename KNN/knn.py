import Data_Process
import pandas as pd
import numpy as np
import sklearn
import matplotlib






if __name__ == "__main__":

    DataTrain = pd.read_csv("../student_performance_train.csv")
    DataTest = pd.read_csv("../student_performance_test.csv")
    Data_Process.binary(DataTrain)
    Data_Process.binary(DataTest)

    print(DataTest)
