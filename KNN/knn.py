from .. import Data_Process
import pandas as pd


DataTrain = pd.read_csv("./student_performance_train.csv")
DataTest = pd.read_csv("./student_performance_test.csv")
Data_Process.binary(DataTrain)
DataTest.binary(DataTest)

print(DataTest)
